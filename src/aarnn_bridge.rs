use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};

use crate::{
    aer::{encode_spikes, payload_hex},
    config::{AarnnBridgeConfig, GailConfig, SpecialistProfile},
    models::{
        AarnnBridgeStatus, AarnnMirrorCandidate, AarnnMirrorDirection, AarnnMirrorInvocationTrace,
        AarnnMirrorRequest, AarnnMirrorResponse, AarnnMirrorTrace, AarnnResponsePreference,
    },
    specialists::SpecialistEngine,
};

const AARNN_MIRROR_PATH: &str = "/api/llm/mirror";
const AARNN_RESPONSE_MODEL: &str = "aarnn-snn-aer-bridge";

#[derive(Clone, Debug)]
pub struct AarnnMirrorExchange {
    pub request_id: String,
    pub conversation_id: String,
    pub workflow: String,
    pub role: String,
    pub direction: AarnnMirrorDirection,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub request_category: Option<String>,
    pub system: Option<String>,
    pub prompt_text: Option<String>,
    pub text: String,
    pub message_roles: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AarnnMirrorClient {
    client: Client,
    endpoint: String,
    access_token: Option<String>,
    timeout: Duration,
    mirror_input: bool,
    mirror_output: bool,
    request_candidate_reply: bool,
    response_preference: AarnnResponsePreference,
    candidate_confidence_threshold: f64,
    candidate_min_reply_chars: usize,
    network_id: Option<String>,
    node_id: Option<String>,
    max_text_chars: usize,
    sensory_size: usize,
    aer_sensory_base: u32,
    aer_output_base: u32,
}

impl AarnnMirrorClient {
    pub fn from_config(
        config: &GailConfig,
        client: Client,
        specialists: &[SpecialistEngine],
    ) -> Option<Self> {
        let bridge = &config.aarnn_bridge;
        if !bridge.enabled {
            return None;
        }
        let endpoint = resolve_endpoint(bridge, specialists, &config.specialists)?;
        let transport = resolve_transport_profile(specialists, &config.specialists);
        Some(Self {
            client,
            endpoint,
            access_token: bridge.access_token.clone(),
            timeout: Duration::from_secs_f64(bridge.timeout_seconds.max(0.2)),
            mirror_input: bridge.mirror_input,
            mirror_output: bridge.mirror_output,
            request_candidate_reply: bridge.request_candidate_reply,
            response_preference: bridge.response_preference.clone(),
            candidate_confidence_threshold: bridge.candidate_confidence_threshold.clamp(0.0, 1.0),
            candidate_min_reply_chars: bridge.candidate_min_reply_chars.max(1),
            network_id: bridge.network_id.clone(),
            node_id: bridge.node_id.clone(),
            max_text_chars: bridge.max_text_chars.clamp(128, 65_536),
            sensory_size: transport.sensory_size,
            aer_sensory_base: transport.aer_sensory_base,
            aer_output_base: transport.aer_output_base,
        })
    }

    pub fn status(config: &GailConfig, specialists: &[SpecialistEngine]) -> AarnnBridgeStatus {
        let bridge = &config.aarnn_bridge;
        let transport = resolve_transport_profile(specialists, &config.specialists);
        let endpoint = resolve_endpoint(bridge, specialists, &config.specialists);
        let reason = if !bridge.enabled {
            Some("AARNN mirrored LLM exchange support is disabled.".to_string())
        } else if endpoint.is_none() {
            Some("No AARNN HTTP endpoint is configured for mirrored LLM exchanges.".to_string())
        } else {
            None
        };
        AarnnBridgeStatus {
            enabled: bridge.enabled,
            available: bridge.enabled && endpoint.is_some(),
            endpoint,
            timeout_seconds: bridge.timeout_seconds.max(0.2),
            mirror_input: bridge.mirror_input,
            mirror_output: bridge.mirror_output,
            request_candidate_reply: bridge.request_candidate_reply,
            response_preference: bridge.response_preference.clone(),
            candidate_confidence_threshold: bridge.candidate_confidence_threshold.clamp(0.0, 1.0),
            candidate_min_reply_chars: bridge.candidate_min_reply_chars.max(1),
            network_id: bridge.network_id.clone(),
            node_id: bridge.node_id.clone(),
            sensory_size: transport.sensory_size,
            output_size: transport.output_size,
            aer_sensory_base: transport.aer_sensory_base,
            aer_output_base: transport.aer_output_base,
            max_text_chars: bridge.max_text_chars.clamp(128, 65_536),
            reason,
        }
    }

    pub fn should_mirror_input(&self) -> bool {
        self.mirror_input
    }

    pub fn should_mirror_output(&self) -> bool {
        self.mirror_output
    }

    pub fn response_model(&self) -> &'static str {
        AARNN_RESPONSE_MODEL
    }

    pub fn response_preference(&self) -> &AarnnResponsePreference {
        &self.response_preference
    }

    pub fn endpoint(&self) -> &str {
        self.endpoint.as_str()
    }

    pub fn candidate_confidence_threshold(&self) -> f64 {
        self.candidate_confidence_threshold
    }

    pub fn candidate_min_reply_chars(&self) -> usize {
        self.candidate_min_reply_chars
    }

    pub fn build_trace(
        &self,
        input: Option<AarnnMirrorInvocationTrace>,
        output: Option<AarnnMirrorInvocationTrace>,
    ) -> AarnnMirrorTrace {
        AarnnMirrorTrace {
            enabled: true,
            endpoint: self.endpoint.clone(),
            mirror_input: self.mirror_input,
            mirror_output: self.mirror_output,
            response_preference: self.response_preference.clone(),
            candidate_confidence_threshold: self.candidate_confidence_threshold,
            candidate_min_reply_chars: self.candidate_min_reply_chars,
            input,
            output,
        }
    }

    pub fn should_promote_candidate(
        &self,
        trace: &AarnnMirrorInvocationTrace,
        llm_text: &str,
    ) -> bool {
        if self.response_preference != AarnnResponsePreference::PreferAarnnWhenConfident {
            return false;
        }
        let Some(candidate) = trace.candidate.as_ref() else {
            return false;
        };
        if !candidate.usable {
            return false;
        }
        let Some(reply_text) = candidate
            .reply_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return false;
        };
        if reply_text.chars().count() < self.candidate_min_reply_chars {
            return false;
        }
        if candidate.confidence.unwrap_or(0.0) < self.candidate_confidence_threshold {
            return false;
        }
        normalise_for_compare(reply_text) != normalise_for_compare(llm_text)
    }

    pub fn promoted_reply(&self, trace: &AarnnMirrorInvocationTrace) -> Option<String> {
        trace
            .candidate
            .as_ref()?
            .reply_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    pub async fn mirror(&self, exchange: AarnnMirrorExchange) -> AarnnMirrorInvocationTrace {
        let started = Instant::now();
        let text_chars = exchange.text.chars().count();
        let request = self.build_request(exchange);
        let spike_count = request
            .sensory_spikes
            .iter()
            .filter(|value| **value > 0)
            .count();
        match self.mirror_once(&request).await {
            Ok(response) => AarnnMirrorInvocationTrace {
                direction: request.direction.clone(),
                accepted: response.accepted,
                endpoint: self.endpoint.clone(),
                latency_ms: started.elapsed().as_millis() as u64,
                text_chars: response.text_chars.max(text_chars),
                spike_count: response.spike_count.max(spike_count),
                candidate: response
                    .candidate
                    .map(|candidate| sanitize_candidate(candidate, self.max_text_chars)),
                stimulation: response.stimulation,
                error: None,
            },
            Err(error) => AarnnMirrorInvocationTrace {
                direction: request.direction,
                accepted: false,
                endpoint: self.endpoint.clone(),
                latency_ms: started.elapsed().as_millis() as u64,
                text_chars,
                spike_count,
                candidate: None,
                stimulation: None,
                error: Some(error),
            },
        }
    }

    fn build_request(&self, exchange: AarnnMirrorExchange) -> AarnnMirrorRequest {
        let text = truncate_chars(&compact_text(&exchange.text), self.max_text_chars);
        let system = exchange
            .system
            .as_deref()
            .map(compact_text)
            .map(|value| truncate_chars(&value, self.max_text_chars));
        let prompt_text = exchange
            .prompt_text
            .as_deref()
            .map(compact_text)
            .map(|value| truncate_chars(&value, self.max_text_chars));
        let sensory_spikes = text_to_spikes(text.as_str(), self.sensory_size);
        let aer_payload = encode_spikes(now_ts_us(), self.aer_sensory_base, &sensory_spikes);
        AarnnMirrorRequest {
            request_id: exchange.request_id,
            conversation_id: exchange.conversation_id,
            workflow: exchange.workflow,
            role: exchange.role,
            direction: exchange.direction.clone(),
            provider: exchange.provider,
            model: exchange.model,
            request_category: exchange.request_category,
            system,
            prompt_text,
            text,
            message_roles: exchange.message_roles,
            aer_base: self.aer_sensory_base,
            output_base: self.aer_output_base,
            aer_payload_hex: payload_hex(&aer_payload),
            sensory_spikes,
            network_id: self.network_id.clone(),
            node_id: self.node_id.clone(),
            request_candidate_reply: self.request_candidate_reply
                && matches!(exchange.direction, AarnnMirrorDirection::Output),
        }
    }

    async fn mirror_once(
        &self,
        request: &AarnnMirrorRequest,
    ) -> Result<AarnnMirrorResponse, String> {
        let url = format!("{}{}", self.endpoint, AARNN_MIRROR_PATH);
        let response = self
            .client
            .post(url)
            .headers(self.headers()?)
            .timeout(self.timeout)
            .json(request)
            .send()
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let message = if body.trim().is_empty() {
                status.to_string()
            } else {
                format!("{status}: {body}")
            };
            return Err(message);
        }
        response
            .json::<AarnnMirrorResponse>()
            .await
            .map_err(|error| error.to_string())
    }

    fn headers(&self) -> Result<HeaderMap, String> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(token) = self.access_token.as_deref() {
            let value = format!("Bearer {token}");
            let header = HeaderValue::from_str(&value).map_err(|error| error.to_string())?;
            headers.insert(AUTHORIZATION, header);
        }
        Ok(headers)
    }
}

#[derive(Clone, Debug)]
struct TransportProfile {
    sensory_size: usize,
    output_size: usize,
    aer_sensory_base: u32,
    aer_output_base: u32,
}

fn resolve_endpoint(
    bridge: &AarnnBridgeConfig,
    active_specialists: &[SpecialistEngine],
    configured_specialists: &[SpecialistProfile],
) -> Option<String> {
    bridge
        .endpoint
        .clone()
        .or_else(|| {
            active_specialists
                .iter()
                .find(|engine| engine.engine_type().eq_ignore_ascii_case("aarnn"))
                .and_then(|engine| engine.profile().endpoint.clone())
        })
        .or_else(|| {
            configured_specialists
                .iter()
                .find(|profile| profile.engine_type.eq_ignore_ascii_case("aarnn"))
                .and_then(|profile| profile.endpoint.clone())
        })
        .or_else(|| legacy_env("GAIL_AARNN_ENDPOINT"))
        .and_then(|value| normalized_url(value.as_str()))
}

fn resolve_transport_profile(
    active_specialists: &[SpecialistEngine],
    configured_specialists: &[SpecialistProfile],
) -> TransportProfile {
    let profile = active_specialists
        .iter()
        .find(|engine| engine.engine_type().eq_ignore_ascii_case("aarnn"))
        .map(|engine| engine.profile().clone())
        .or_else(|| {
            configured_specialists
                .iter()
                .find(|profile| profile.engine_type.eq_ignore_ascii_case("aarnn"))
                .cloned()
        })
        .unwrap_or_default();
    TransportProfile {
        sensory_size: profile.sensory_size.max(8),
        output_size: profile.output_size.max(8),
        aer_sensory_base: profile.aer_sensory_base,
        aer_output_base: profile.aer_output_base,
    }
}

fn text_to_spikes(text: &str, sensory_size: usize) -> Vec<u8> {
    let sensory_size = sensory_size.max(8);
    let mut spikes = vec![0u8; sensory_size];
    let compact = compact_text(text).to_ascii_lowercase();
    let bytes = compact.as_bytes();
    let len = bytes.len().max(1);
    for (index, byte) in bytes.iter().enumerate() {
        let primary = ((*byte as usize) + index * 17 + len * 31) % sensory_size;
        spikes[primary] = 1;
        if index > 0 {
            let previous = bytes[index - 1] as usize;
            let secondary =
                ((*byte as usize) * 31 + previous + index * 13 + len * 7) % sensory_size;
            spikes[secondary] = 1;
        }
    }
    spikes
}

fn sanitize_candidate(
    mut candidate: AarnnMirrorCandidate,
    max_text_chars: usize,
) -> AarnnMirrorCandidate {
    candidate.reply_text = candidate
        .reply_text
        .as_deref()
        .map(compact_text)
        .map(|value| truncate_chars(&value, max_text_chars))
        .filter(|value| !value.is_empty());
    if candidate.reply_text.is_none() {
        candidate.usable = false;
    }
    candidate
}

fn compact_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit.max(1)).collect()
}

fn normalise_for_compare(value: &str) -> String {
    compact_text(value).to_ascii_lowercase()
}

fn normalized_url(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if value.contains("://") {
        Some(value.trim_end_matches('/').to_string())
    } else {
        Some(format!("http://{}", value.trim_end_matches('/')))
    }
}

fn legacy_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn now_ts_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use crate::{
        config::{GailConfig, SpecialistProfile},
        models::{AarnnMirrorDirection, AarnnResponsePreference},
    };

    #[test]
    fn status_uses_specialist_endpoint_when_bridge_endpoint_is_unset() {
        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.specialists.push(SpecialistProfile {
            endpoint: Some("http://aarnn.internal:8080".to_string()),
            ..SpecialistProfile::default()
        });

        let status = AarnnMirrorClient::status(&config, &[]);
        assert!(status.enabled);
        assert!(status.available);
        assert_eq!(
            status.endpoint.as_deref(),
            Some("http://aarnn.internal:8080")
        );
    }

    #[test]
    fn candidate_promotion_requires_confident_non_duplicate_reply() {
        let client = AarnnMirrorClient {
            client: Client::builder().build().expect("client"),
            endpoint: "http://example.invalid".to_string(),
            access_token: None,
            timeout: Duration::from_secs(1),
            mirror_input: true,
            mirror_output: true,
            request_candidate_reply: true,
            response_preference: AarnnResponsePreference::PreferAarnnWhenConfident,
            candidate_confidence_threshold: 0.8,
            candidate_min_reply_chars: 10,
            network_id: None,
            node_id: None,
            max_text_chars: 2048,
            sensory_size: 32,
            aer_sensory_base: 4096,
            aer_output_base: 16384,
        };
        let promoted = AarnnMirrorInvocationTrace {
            direction: AarnnMirrorDirection::Output,
            accepted: true,
            endpoint: client.endpoint.clone(),
            latency_ms: 5,
            text_chars: 20,
            spike_count: 8,
            candidate: Some(AarnnMirrorCandidate {
                reply_text: Some("Alternative SNN answer".to_string()),
                confidence: Some(0.9),
                usable: true,
                source: Some("transport_mirror_echo".to_string()),
                output_spike_indices: vec![1, 2],
                output_aer_payload_hex: Some("41455231".to_string()),
            }),
            stimulation: None,
            error: None,
        };
        assert!(client.should_promote_candidate(&promoted, "LLM answer"));

        let duplicate = AarnnMirrorInvocationTrace {
            candidate: Some(AarnnMirrorCandidate {
                reply_text: Some("LLM answer".to_string()),
                confidence: Some(0.95),
                usable: true,
                source: Some("transport_mirror_echo".to_string()),
                output_spike_indices: vec![],
                output_aer_payload_hex: None,
            }),
            ..promoted.clone()
        };
        assert!(!client.should_promote_candidate(&duplicate, "LLM answer"));
    }

    #[tokio::test]
    async fn mirror_posts_bearer_authenticated_payload() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/llm/mirror"))
            .and(header("authorization", "Bearer bridge-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": true,
                "text_chars": 12,
                "spike_count": 4
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.aarnn_bridge.endpoint = Some(server.uri());
        config.aarnn_bridge.access_token = Some("bridge-token".to_string());
        let client = AarnnMirrorClient::from_config(
            &config,
            Client::builder().build().expect("client"),
            &[],
        )
        .expect("bridge client");

        let trace = client
            .mirror(AarnnMirrorExchange {
                request_id: "req-1".to_string(),
                conversation_id: "conv-1".to_string(),
                workflow: "assistant".to_string(),
                role: "assistant".to_string(),
                direction: AarnnMirrorDirection::Input,
                provider: Some("openai".to_string()),
                model: Some("gpt-4o-mini".to_string()),
                request_category: None,
                system: Some("Keep it concise.".to_string()),
                prompt_text: None,
                text: "Hello world".to_string(),
                message_roles: vec!["system".to_string(), "user".to_string()],
            })
            .await;

        assert!(trace.accepted);
        assert!(trace.error.is_none());
    }
}
