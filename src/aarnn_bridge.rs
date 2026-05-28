use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde_json::Value;
use tokio::{
    sync::{Semaphore, mpsc, oneshot},
    time::{sleep, timeout},
};
use tracing::{info, warn};

use crate::{
    adaptive_schema,
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

#[derive(Debug)]
struct AarnnMirrorJob {
    exchange: AarnnMirrorExchange,
    response: Option<oneshot::Sender<AarnnMirrorInvocationTrace>>,
}

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
    queue_tx: mpsc::Sender<AarnnMirrorJob>,
    queue_capacity: usize,
    worker_count: usize,
    enqueue_timeout: Duration,
    candidate_wait_timeout: Duration,
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
    request_max_attempts: usize,
    request_backoff: Duration,
    request_backoff_max: Duration,
    audit_enabled: bool,
    audit_log_llm_prompts: bool,
    audit_log_llm_responses: bool,
    audit_log_aer_payloads: bool,
    audit_max_chars: usize,
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
        let queue_capacity = bridge.queue_capacity.clamp(8, 32_768);
        let worker_count = bridge.worker_count.clamp(1, 128);
        let request_max_attempts = env_usize_clamped("GAIL_AARNN_MIRROR_MAX_ATTEMPTS", 3, 1, 10);
        let request_backoff_ms = env_u64_clamped("GAIL_AARNN_MIRROR_BACKOFF_MS", 300, 10, 30_000);
        let request_backoff_max_ms = env_u64_clamped(
            "GAIL_AARNN_MIRROR_BACKOFF_MAX_MS",
            request_backoff_ms.saturating_mul(8),
            request_backoff_ms,
            120_000,
        );
        let (queue_tx, queue_rx) = mpsc::channel(queue_capacity);
        let mirror = Self {
            client,
            endpoint,
            access_token: bridge.access_token.clone(),
            timeout: Duration::from_secs_f64(bridge.timeout_seconds.max(0.2)),
            queue_tx,
            queue_capacity,
            worker_count,
            enqueue_timeout: Duration::from_millis(bridge.enqueue_timeout_ms.clamp(1, 10_000)),
            candidate_wait_timeout: Duration::from_millis(
                bridge.candidate_wait_timeout_ms.min(30_000),
            ),
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
            request_max_attempts,
            request_backoff: Duration::from_millis(request_backoff_ms),
            request_backoff_max: Duration::from_millis(request_backoff_max_ms),
            audit_enabled: config.audit_logging.enabled,
            audit_log_llm_prompts: config.audit_logging.log_llm_prompts,
            audit_log_llm_responses: config.audit_logging.log_llm_responses,
            audit_log_aer_payloads: config.audit_logging.log_aer_payloads,
            audit_max_chars: config.audit_logging.max_chars.clamp(1, 262_144),
        };
        mirror.start_worker_bus(queue_rx);
        Some(mirror)
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
            queue_capacity: bridge.queue_capacity.clamp(8, 32_768),
            worker_count: bridge.worker_count.clamp(1, 128),
            enqueue_timeout_ms: bridge.enqueue_timeout_ms.clamp(1, 10_000),
            candidate_wait_timeout_ms: bridge.candidate_wait_timeout_ms.min(30_000),
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

    pub fn candidate_wait_timeout(&self) -> Duration {
        self.candidate_wait_timeout
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

    fn truncate_audit_text(&self, value: &str) -> String {
        truncate_chars(value, self.audit_max_chars.max(1))
    }

    fn optional_audit_text(&self, value: Option<&str>) -> Option<String> {
        value
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| self.truncate_audit_text(item))
    }

    fn audit_text_for_direction(
        &self,
        direction: &AarnnMirrorDirection,
        value: Option<&str>,
    ) -> Option<String> {
        match direction {
            AarnnMirrorDirection::Input if self.audit_log_llm_prompts => {
                self.optional_audit_text(value)
            }
            AarnnMirrorDirection::Output if self.audit_log_llm_responses => {
                self.optional_audit_text(value)
            }
            _ => None,
        }
    }

    fn log_mirror_request_audit(&self, request: &AarnnMirrorRequest, spike_count: usize) {
        if !self.audit_enabled {
            return;
        }
        let text = self.audit_text_for_direction(&request.direction, Some(request.text.as_str()));
        let prompt_text =
            self.audit_text_for_direction(&request.direction, request.prompt_text.as_deref());
        let system_text = if self.audit_log_llm_prompts {
            self.optional_audit_text(request.system.as_deref())
        } else {
            None
        };
        let payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(Some(request.aer_payload_hex.as_str()))
        } else {
            None
        };
        let active_spike_indices = if self.audit_log_aer_payloads {
            Some(
                request
                    .sensory_spikes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, spike)| (*spike > 0).then_some(index as u32))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        info!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            conversation_id = %request.conversation_id,
            workflow = %request.workflow,
            role = %request.role,
            provider = ?request.provider,
            model = ?request.model,
            request_category = ?request.request_category,
            system_prompt = ?system_text,
            prompt_text = ?prompt_text,
            text = ?text,
            aer_base = request.aer_base,
            output_base = request.output_base,
            aer_payload_hex = ?payload_hex,
            sensory_spike_count = spike_count,
            sensory_spike_indices = ?active_spike_indices,
            "GAIL_AUDIT_AARNN_MIRROR_REQUEST"
        );
    }

    fn log_mirror_response_audit(
        &self,
        request: &AarnnMirrorRequest,
        response: &AarnnMirrorResponse,
        text_chars: usize,
        spike_count: usize,
    ) {
        if !self.audit_enabled {
            return;
        }
        let candidate_reply_text = if self.audit_log_llm_responses {
            self.optional_audit_text(
                response
                    .candidate
                    .as_ref()
                    .and_then(|candidate| candidate.reply_text.as_deref()),
            )
        } else {
            None
        };
        let response_payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(response.aer_payload_hex.as_deref())
        } else {
            None
        };
        let candidate_payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(
                response
                    .candidate
                    .as_ref()
                    .and_then(|candidate| candidate.output_aer_payload_hex.as_deref()),
            )
        } else {
            None
        };
        let candidate_spike_indices = if self.audit_log_aer_payloads {
            response
                .candidate
                .as_ref()
                .map(|candidate| candidate.output_spike_indices.clone())
        } else {
            None
        };
        info!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            accepted = response.accepted,
            endpoint = %self.endpoint,
            text_chars = response.text_chars.max(text_chars),
            spike_count = response.spike_count.max(spike_count),
            aer_payload_hex = ?response_payload_hex,
            candidate_usable = ?response.candidate.as_ref().map(|candidate| candidate.usable),
            candidate_confidence = ?response.candidate.as_ref().and_then(|candidate| candidate.confidence),
            candidate_source = ?response.candidate.as_ref().and_then(|candidate| candidate.source.as_deref()),
            candidate_reply_text = ?candidate_reply_text,
            candidate_output_aer_payload_hex = ?candidate_payload_hex,
            candidate_output_spike_indices = ?candidate_spike_indices,
            stimulation = ?response.stimulation,
            "GAIL_AUDIT_AARNN_MIRROR_RESPONSE"
        );
    }

    fn log_mirror_error_audit(
        &self,
        request: &AarnnMirrorRequest,
        error: &str,
        text_chars: usize,
        spike_count: usize,
    ) {
        if !self.audit_enabled {
            return;
        }
        let text = self.audit_text_for_direction(&request.direction, Some(request.text.as_str()));
        let payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(Some(request.aer_payload_hex.as_str()))
        } else {
            None
        };
        warn!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            endpoint = %self.endpoint,
            text_chars,
            spike_count,
            text = ?text,
            aer_payload_hex = ?payload_hex,
            error = %self.truncate_audit_text(error),
            "GAIL_AUDIT_AARNN_MIRROR_ERROR"
        );
    }

    fn start_worker_bus(&self, mut rx: mpsc::Receiver<AarnnMirrorJob>) {
        let worker = self.clone();
        let concurrency = self.worker_count.max(1);
        tokio::spawn(async move {
            let permits = Arc::new(Semaphore::new(concurrency));
            while let Some(job) = rx.recv().await {
                let permit = match permits.clone().acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => break,
                };
                let worker = worker.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    let trace = worker.mirror(job.exchange).await;
                    if let Some(response) = job.response {
                        let _ = response.send(trace);
                    }
                });
            }
        });
    }

    pub async fn enqueue(
        &self,
        exchange: AarnnMirrorExchange,
        wait_for_trace: bool,
    ) -> Option<oneshot::Receiver<AarnnMirrorInvocationTrace>> {
        let (response_tx, response_rx) = if wait_for_trace {
            let (tx, rx) = oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let direction = exchange.direction.clone();
        let send = self.queue_tx.send(AarnnMirrorJob {
            exchange,
            response: response_tx,
        });
        match timeout(self.enqueue_timeout, send).await {
            Ok(Ok(())) => response_rx,
            Ok(Err(error)) => {
                warn!(
                    endpoint = %self.endpoint,
                    direction = ?direction,
                    error = %error,
                    "AARNN mirror queue is closed; dropping exchange"
                );
                None
            }
            Err(_) => {
                warn!(
                    endpoint = %self.endpoint,
                    direction = ?direction,
                    queue_capacity = self.queue_capacity,
                    enqueue_timeout_ms = self.enqueue_timeout.as_millis(),
                    "AARNN mirror queue is saturated; dropping exchange"
                );
                None
            }
        }
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
        self.log_mirror_request_audit(&request, spike_count);
        match self.mirror_once(&request).await {
            Ok(response) => {
                self.log_mirror_response_audit(&request, &response, text_chars, spike_count);
                AarnnMirrorInvocationTrace {
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
                }
            }
            Err(error) => {
                self.log_mirror_error_audit(&request, error.as_str(), text_chars, spike_count);
                AarnnMirrorInvocationTrace {
                    direction: request.direction,
                    accepted: false,
                    endpoint: self.endpoint.clone(),
                    latency_ms: started.elapsed().as_millis() as u64,
                    text_chars,
                    spike_count,
                    candidate: None,
                    stimulation: None,
                    error: Some(error),
                }
            }
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
        let max_attempts = self.request_max_attempts.max(1);
        let mut last_error = String::new();
        for attempt in 1..=max_attempts {
            match self.mirror_once_attempt(request).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    last_error = error.message.clone();
                    let retryable = error.retryable && attempt < max_attempts;
                    if !retryable {
                        return Err(error.message);
                    }
                    let delay = retry_delay(
                        self.request_backoff,
                        self.request_backoff_max,
                        attempt.saturating_sub(1),
                    );
                    warn!(
                        endpoint = %self.endpoint,
                        attempt,
                        max_attempts,
                        backoff_ms = delay.as_millis() as u64,
                        error = %error.message,
                        "AARNN mirror request failed; retrying"
                    );
                    sleep(delay).await;
                }
            }
        }
        Err(last_error)
    }

    async fn mirror_once_attempt(
        &self,
        request: &AarnnMirrorRequest,
    ) -> Result<AarnnMirrorResponse, MirrorHttpError> {
        let url = format!("{}{}", self.endpoint, AARNN_MIRROR_PATH);
        let response = match self
            .client
            .post(url)
            .headers(
                self.headers()
                    .map_err(|error| MirrorHttpError::non_retryable(error.as_str()))?,
            )
            .timeout(self.timeout)
            .json(request)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                adaptive_schema::observe_failure(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    None,
                    &message,
                )
                .await;
                return Err(MirrorHttpError::retryable(message));
            }
        };
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let message = if body.trim().is_empty() {
                status.to_string()
            } else {
                format!("{status}: {body}")
            };
            let retryable = mirror_status_retryable(status.as_u16());
            adaptive_schema::observe_failure(
                "aarnn_bridge",
                "POST",
                AARNN_MIRROR_PATH,
                "mirror",
                Some(status.as_u16()),
                &message,
            )
            .await;
            return Err(if retryable {
                MirrorHttpError::retryable(message)
            } else {
                MirrorHttpError::non_retryable(message.as_str())
            });
        }
        match response.json::<AarnnMirrorResponse>().await {
            Ok(parsed) => {
                let body = serde_json::to_value(&parsed).unwrap_or(Value::Null);
                adaptive_schema::observe_success(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    &body,
                )
                .await;
                Ok(parsed)
            }
            Err(error) => {
                adaptive_schema::observe_failure(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    Some(status.as_u16()),
                    &error.to_string(),
                )
                .await;
                Err(MirrorHttpError::retryable(error.to_string()))
            }
        }
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

#[derive(Debug)]
struct MirrorHttpError {
    message: String,
    retryable: bool,
}

impl MirrorHttpError {
    fn retryable(message: String) -> Self {
        Self {
            message,
            retryable: true,
        }
    }

    fn non_retryable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            retryable: false,
        }
    }
}

#[derive(Clone, Debug)]
struct TransportProfile {
    sensory_size: usize,
    output_size: usize,
    aer_sensory_base: u32,
    aer_output_base: u32,
}

fn mirror_status_retryable(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

fn retry_delay(base: Duration, max: Duration, attempt_index: usize) -> Duration {
    let shift = attempt_index.min(8) as u32;
    let factor = 2_u32.saturating_pow(shift).max(1);
    let expanded = base.saturating_mul(factor);
    if expanded > max { max } else { expanded }
}

fn env_usize_clamped(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

fn env_u64_clamped(name: &str, default: u64, min: u64, max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
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
        let (queue_tx, _queue_rx) = mpsc::channel(1);
        let client = AarnnMirrorClient {
            client: Client::builder().build().expect("client"),
            endpoint: "http://example.invalid".to_string(),
            access_token: None,
            timeout: Duration::from_secs(1),
            queue_tx,
            queue_capacity: 1,
            worker_count: 1,
            enqueue_timeout: Duration::from_millis(10),
            candidate_wait_timeout: Duration::from_millis(100),
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
            request_max_attempts: 1,
            request_backoff: Duration::from_millis(10),
            request_backoff_max: Duration::from_millis(50),
            audit_enabled: false,
            audit_log_llm_prompts: true,
            audit_log_llm_responses: true,
            audit_log_aer_payloads: true,
            audit_max_chars: 2048,
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

    #[tokio::test]
    async fn enqueue_dispatches_non_blocking_worker_job() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/llm/mirror"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": true,
                "text_chars": 16,
                "spike_count": 4
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.aarnn_bridge.endpoint = Some(server.uri());
        config.aarnn_bridge.queue_capacity = 8;
        config.aarnn_bridge.worker_count = 2;
        config.aarnn_bridge.enqueue_timeout_ms = 25;
        let client = AarnnMirrorClient::from_config(
            &config,
            Client::builder().build().expect("client"),
            &[],
        )
        .expect("bridge client");
        let receiver = client
            .enqueue(
                AarnnMirrorExchange {
                    request_id: "req-2".to_string(),
                    conversation_id: "conv-2".to_string(),
                    workflow: "assistant".to_string(),
                    role: "assistant".to_string(),
                    direction: AarnnMirrorDirection::Input,
                    provider: Some("openai".to_string()),
                    model: Some("gpt-4o-mini".to_string()),
                    request_category: None,
                    system: None,
                    prompt_text: None,
                    text: "queued hello".to_string(),
                    message_roles: vec!["user".to_string()],
                },
                true,
            )
            .await
            .expect("queued receiver");
        let trace = timeout(Duration::from_secs(1), receiver)
            .await
            .expect("worker timeout")
            .expect("worker trace");
        assert!(trace.accepted);
        assert!(trace.error.is_none());
    }
}
