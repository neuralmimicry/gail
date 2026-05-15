use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use reqwest::{
    Client,
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::config::GailConfig;

const NMC_TRACEY_ADAPTIVE_PATH: &str = "/tracey/adaptive";

#[derive(Clone, Debug, Default)]
pub struct NmcAgentSignal {
    pub agent_id: String,
    pub host: String,
    pub status: String,
    pub mode: String,
    pub optimize_status: String,
    pub placement_score: f64,
    pub headroom_pct: f64,
    pub compromise_risk: f64,
    pub stale: bool,
    pub constrained: bool,
    pub pressure_ratio: f64,
}

#[derive(Clone, Debug, Default)]
struct NmcSnapshot {
    generated_epoch_ms: Option<i64>,
    summary_mode: Option<String>,
    agents_by_id: HashMap<String, NmcAgentSignal>,
    agents_by_host: HashMap<String, NmcAgentSignal>,
}

#[derive(Clone, Debug, Default)]
struct NmcCache {
    snapshot: Option<NmcSnapshot>,
    fetched_at: Option<Instant>,
    last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NmcTelemetryStatus {
    pub enabled: bool,
    pub available: bool,
    pub base_url: Option<String>,
    pub timeout_seconds: f64,
    pub cache_ttl_seconds: f64,
    pub stale_after_seconds: u64,
    pub adaptive_policy: Option<String>,
    pub cache_populated: bool,
    pub cache_age_seconds: Option<f64>,
    pub generated_epoch_ms: Option<i64>,
    pub summary_mode: Option<String>,
    pub agent_count: usize,
    pub last_error: Option<String>,
    pub reason: Option<String>,
}

#[derive(Clone)]
pub struct NmcTelemetryClient {
    client: Client,
    base_url: String,
    access_token: Option<String>,
    timeout: Duration,
    cache_ttl: Duration,
    stale_after: Duration,
    adaptive_policy: Option<String>,
    cache: Arc<Mutex<NmcCache>>,
}

impl NmcTelemetryClient {
    pub fn from_config(config: &GailConfig, client: Client) -> Option<Self> {
        let nmc = &config.nmc_telemetry;
        if !nmc.enabled {
            return None;
        }
        let base_url = nmc
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.trim_end_matches('/').to_string())?;
        Some(Self {
            client,
            base_url,
            access_token: nmc.access_token.clone(),
            timeout: Duration::from_secs_f64(nmc.timeout_seconds.max(0.2)),
            cache_ttl: Duration::from_secs_f64(nmc.cache_ttl_seconds.max(0.2)),
            stale_after: Duration::from_secs(nmc.stale_after_seconds.clamp(5, 86_400)),
            adaptive_policy: nmc.adaptive_policy.clone(),
            cache: Arc::new(Mutex::new(NmcCache::default())),
        })
    }

    pub fn status_from_config(config: &GailConfig) -> NmcTelemetryStatus {
        let nmc = &config.nmc_telemetry;
        let base_url = nmc.base_url.clone();
        let available = nmc.enabled && base_url.is_some();
        let reason = if !nmc.enabled {
            Some("NMC/Tracey telemetry integration is disabled.".to_string())
        } else if base_url.is_none() {
            Some("NMC telemetry base URL is not configured.".to_string())
        } else {
            None
        };
        NmcTelemetryStatus {
            enabled: nmc.enabled,
            available,
            base_url,
            timeout_seconds: nmc.timeout_seconds.max(0.2),
            cache_ttl_seconds: nmc.cache_ttl_seconds.max(0.2),
            stale_after_seconds: nmc.stale_after_seconds.clamp(5, 86_400),
            adaptive_policy: nmc.adaptive_policy.clone(),
            cache_populated: false,
            cache_age_seconds: None,
            generated_epoch_ms: None,
            summary_mode: None,
            agent_count: 0,
            last_error: None,
            reason,
        }
    }

    pub async fn status(&self) -> NmcTelemetryStatus {
        let cache = self.cache.lock().await;
        let cache_age_seconds = cache
            .fetched_at
            .map(|value| value.elapsed().as_secs_f64())
            .map(|value| value.max(0.0));
        let (generated_epoch_ms, summary_mode, agent_count) = cache
            .snapshot
            .as_ref()
            .map(|snapshot| {
                (
                    snapshot.generated_epoch_ms,
                    snapshot.summary_mode.clone(),
                    snapshot.agents_by_id.len(),
                )
            })
            .unwrap_or((None, None, 0));
        NmcTelemetryStatus {
            enabled: true,
            available: true,
            base_url: Some(self.base_url.clone()),
            timeout_seconds: self.timeout.as_secs_f64(),
            cache_ttl_seconds: self.cache_ttl.as_secs_f64(),
            stale_after_seconds: self.stale_after.as_secs(),
            adaptive_policy: self.adaptive_policy.clone(),
            cache_populated: cache.snapshot.is_some(),
            cache_age_seconds,
            generated_epoch_ms,
            summary_mode,
            agent_count,
            last_error: cache.last_error.clone(),
            reason: None,
        }
    }

    pub async fn signal(
        &self,
        nmc_agent_id: Option<&str>,
        nmc_host: Option<&str>,
        host_group: Option<&str>,
    ) -> Option<NmcAgentSignal> {
        let snapshot = self.snapshot().await?;
        let by_agent = normalize_key(nmc_agent_id)
            .and_then(|key| snapshot.agents_by_id.get(key.as_str()).cloned());
        if by_agent.is_some() {
            return by_agent;
        }
        let by_host = normalize_key(nmc_host)
            .and_then(|key| snapshot.agents_by_host.get(key.as_str()).cloned());
        if by_host.is_some() {
            return by_host;
        }
        normalize_key(host_group).and_then(|key| snapshot.agents_by_host.get(key.as_str()).cloned())
    }

    async fn snapshot(&self) -> Option<NmcSnapshot> {
        {
            let cache = self.cache.lock().await;
            if cache
                .fetched_at
                .is_some_and(|value| value.elapsed() <= self.cache_ttl)
            {
                return cache.snapshot.clone();
            }
        }
        let fetched = self.fetch_snapshot().await;
        let mut cache = self.cache.lock().await;
        cache.fetched_at = Some(Instant::now());
        match fetched {
            Ok(snapshot) => {
                cache.last_error = None;
                cache.snapshot = Some(snapshot.clone());
                Some(snapshot)
            }
            Err(error) => {
                tracing::warn!(error = %error, "NMC telemetry fetch failed");
                cache.last_error = Some(error);
                cache.snapshot.clone()
            }
        }
    }

    async fn fetch_snapshot(&self) -> Result<NmcSnapshot, String> {
        let mut request = self
            .client
            .get(format!("{}{}", self.base_url, NMC_TRACEY_ADAPTIVE_PATH))
            .headers(self.headers()?)
            .timeout(self.timeout);
        if let Some(policy) = self.adaptive_policy.as_deref() {
            request = request.query(&[("policy", policy)]);
        }
        let response = request.send().await.map_err(|error| error.to_string())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if body.trim().is_empty() {
                return Err(status.to_string());
            }
            return Err(format!("{status}: {body}"));
        }
        let payload = response
            .json::<Value>()
            .await
            .map_err(|error| error.to_string())?;
        let root = if payload.is_object()
            && payload
                .get("data")
                .is_some_and(|value| value.is_object() || value.is_array())
        {
            payload.get("data").unwrap_or(&payload)
        } else {
            &payload
        };
        let generated_epoch_ms = int_field(root, &["generated_epoch_ms"]);
        let source_stale = generated_epoch_ms.is_some_and(|generated| {
            let age_ms = now_epoch_ms().saturating_sub(generated);
            age_ms > self.stale_after.as_millis() as i64
        });
        let summary_mode = string_field(root, &["summary", "mode"])
            .or_else(|| string_field(root, &["mode"]))
            .map(|value| value.to_ascii_lowercase());
        let mut snapshot = NmcSnapshot {
            generated_epoch_ms,
            summary_mode,
            agents_by_id: HashMap::new(),
            agents_by_host: HashMap::new(),
        };
        let agents = root
            .get("agents")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for entry in agents {
            let Some(signal) = parse_agent_signal(&entry, source_stale) else {
                continue;
            };
            if let Some(agent_id) = normalize_key(Some(signal.agent_id.as_str())) {
                upsert_worst_signal(&mut snapshot.agents_by_id, agent_id, signal.clone());
            }
            if let Some(host) = normalize_key(Some(signal.host.as_str())) {
                upsert_worst_signal(&mut snapshot.agents_by_host, host, signal);
            }
        }
        Ok(snapshot)
    }

    fn headers(&self) -> Result<HeaderMap, String> {
        let mut headers = HeaderMap::new();
        if let Some(token) = self.access_token.as_deref() {
            let value = format!("Bearer {token}");
            let header = HeaderValue::from_str(&value).map_err(|error| error.to_string())?;
            headers.insert(AUTHORIZATION, header);
        }
        Ok(headers)
    }
}

fn upsert_worst_signal(
    signals: &mut HashMap<String, NmcAgentSignal>,
    key: String,
    signal: NmcAgentSignal,
) {
    match signals.get(key.as_str()) {
        Some(existing) if existing.pressure_ratio >= signal.pressure_ratio => {}
        _ => {
            signals.insert(key, signal);
        }
    }
}

fn parse_agent_signal(entry: &Value, source_stale: bool) -> Option<NmcAgentSignal> {
    let agent_id = string_field(entry, &["agent_id"]).unwrap_or_default();
    let host = string_field(entry, &["host"]).unwrap_or_default();
    if agent_id.trim().is_empty() && host.trim().is_empty() {
        return None;
    }
    let status = string_field(entry, &["status"])
        .unwrap_or_else(|| "unknown".to_string())
        .to_ascii_lowercase();
    let mode = string_field(entry, &["mode"])
        .unwrap_or_else(|| "unknown".to_string())
        .to_ascii_lowercase();
    let optimize_status = string_field(entry, &["optimize_status"])
        .unwrap_or_else(|| "unknown".to_string())
        .to_ascii_lowercase();
    let placement_score = float_field(entry, &["placement_score"]).clamp(0.0, 1.0);
    let headroom_pct = float_field(entry, &["headroom_pct"]).clamp(0.0, 100.0);
    let compromise_risk = float_field(entry, &["compromise_risk"]).clamp(0.0, 1.0);
    let stale = bool_field(entry, &["stale"])
        || source_stale
        || string_field(entry, &["repeat_status"])
            .map(|value| value.eq_ignore_ascii_case("stale"))
            .unwrap_or(false);
    let unavailable = status_indicates_unavailable(status.as_str());
    let constrained = unavailable
        || mode.eq_ignore_ascii_case("constrained")
        || optimize_status.eq_ignore_ascii_case("avoid")
        || compromise_risk >= 0.80;
    let pressure_ratio = pressure_ratio(
        placement_score,
        headroom_pct,
        compromise_risk,
        status.as_str(),
        mode.as_str(),
        optimize_status.as_str(),
        stale,
    );
    Some(NmcAgentSignal {
        agent_id,
        host,
        status,
        mode,
        optimize_status,
        placement_score,
        headroom_pct,
        compromise_risk,
        stale,
        constrained,
        pressure_ratio,
    })
}

fn pressure_ratio(
    placement_score: f64,
    headroom_pct: f64,
    compromise_risk: f64,
    status: &str,
    mode: &str,
    optimize_status: &str,
    stale: bool,
) -> f64 {
    let headroom_pressure = ((100.0 - headroom_pct) / 100.0).clamp(0.0, 1.5);
    let placement_pressure = (1.0 - placement_score).clamp(0.0, 1.5);
    let risk_pressure = compromise_risk.clamp(0.0, 1.5);
    let stale_pressure = if stale { 0.85 } else { 0.0 };
    let status_pressure = if status_indicates_unavailable(status) {
        1.35
    } else if status.contains("degraded") {
        0.85
    } else {
        0.0
    };
    let mode_pressure = if mode.eq_ignore_ascii_case("constrained") {
        1.25
    } else if mode.eq_ignore_ascii_case("degraded") {
        0.80
    } else {
        0.0
    };
    let optimize_pressure = if optimize_status.eq_ignore_ascii_case("avoid") {
        1.25
    } else if optimize_status.eq_ignore_ascii_case("tight") {
        0.85
    } else {
        0.0
    };
    headroom_pressure
        .max(placement_pressure)
        .max(risk_pressure)
        .max(stale_pressure)
        .max(status_pressure)
        .max(mode_pressure)
        .max(optimize_pressure)
}

fn status_indicates_unavailable(status: &str) -> bool {
    let lowered = status.to_ascii_lowercase();
    lowered.contains("offline")
        || lowered.contains("unreachable")
        || lowered.contains("error")
        || lowered.contains("down")
}

fn bool_field(value: &Value, path: &[&str]) -> bool {
    resolve_path(value, path)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn int_field(value: &Value, path: &[&str]) -> Option<i64> {
    resolve_path(value, path).and_then(|item| {
        item.as_i64()
            .or_else(|| item.as_u64().map(|raw| raw as i64))
    })
}

fn float_field(value: &Value, path: &[&str]) -> f64 {
    resolve_path(value, path)
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

fn string_field(value: &Value, path: &[&str]) -> Option<String> {
    resolve_path(value, path)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
}

fn resolve_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn normalize_key(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_ascii_lowercase())
}

fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path, query_param},
    };

    #[tokio::test]
    async fn signal_parses_adaptive_payload_and_resolves_agent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tracey/adaptive"))
            .and(query_param("policy", "balanced"))
            .and(header("authorization", "Bearer nmc-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "success": true,
                "data": {
                    "generated_epoch_ms": now_epoch_ms(),
                    "summary": { "mode": "balanced" },
                    "agents": [
                        {
                            "agent_id": "tracey-1",
                            "host": "node-a",
                            "status": "healthy",
                            "mode": "balanced",
                            "optimize_status": "open",
                            "placement_score": 0.82,
                            "headroom_pct": 43.0,
                            "compromise_risk": 0.12
                        }
                    ]
                }
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.nmc_telemetry.enabled = true;
        config.nmc_telemetry.base_url = Some(server.uri());
        config.nmc_telemetry.access_token = Some("nmc-token".to_string());
        config.nmc_telemetry.adaptive_policy = Some("balanced".to_string());
        let client = NmcTelemetryClient::from_config(
            &config,
            Client::builder().build().expect("http client"),
        )
        .expect("nmc client");

        let signal = client
            .signal(Some("tracey-1"), None, None)
            .await
            .expect("agent signal");
        assert_eq!(signal.host, "node-a");
        assert!(!signal.constrained);
        assert!(signal.placement_score > 0.75);
        assert!(signal.pressure_ratio < 0.65);
    }

    #[tokio::test]
    async fn stale_generated_payload_marks_signal_stale() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tracey/adaptive"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "generated_epoch_ms": 1,
                "agents": [
                    {
                        "agent_id": "tracey-2",
                        "host": "node-b",
                        "status": "healthy",
                        "mode": "balanced",
                        "optimize_status": "open",
                        "placement_score": 0.95,
                        "headroom_pct": 70.0,
                        "compromise_risk": 0.05
                    }
                ]
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.nmc_telemetry.enabled = true;
        config.nmc_telemetry.base_url = Some(server.uri());
        config.nmc_telemetry.stale_after_seconds = 5;
        let client = NmcTelemetryClient::from_config(
            &config,
            Client::builder().build().expect("http client"),
        )
        .expect("nmc client");

        let signal = client
            .signal(Some("tracey-2"), None, None)
            .await
            .expect("agent signal");
        assert!(signal.stale);
        assert!(signal.pressure_ratio >= 0.8);
    }
}
