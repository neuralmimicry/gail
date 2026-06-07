//! Adaptive remote API schema tracking.
//!
//! Gail talks to APIs that are useful but not always static: LLM gateways,
//! local model servers, Refiner, AARNN, OctoBot, and specialist services.  This
//! module records endpoint health, coarse response shape, semantic hints from
//! logs/errors, and bounded numeric hints so call sites can adapt without each
//! integration growing its own schema cache.

use std::{collections::BTreeMap, path::PathBuf};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{fs, sync::Mutex};
use url::Url;

const MAX_RECENT_ADJUSTMENTS: usize = 100;
const STRUCTURAL_SKIP_SECONDS: f64 = 15.0 * 60.0;
const RATE_LIMIT_SKIP_SECONDS: f64 = 2.0 * 60.0;
const TRANSIENT_SKIP_SECONDS: f64 = 60.0;
const SAVE_MIN_INTERVAL_SECONDS: f64 = 5.0;
const SAVE_MAX_DIRTY_OBSERVATIONS: u64 = 25;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AdaptiveApiRegistry {
    pub version: u64,
    pub updated_at: Option<f64>,
    pub apis: BTreeMap<String, AdaptiveApiSchema>,
    pub recent_adjustments: Vec<ApiAdjustment>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AdaptiveApiSchema {
    pub version: u64,
    pub updated_at: Option<f64>,
    pub endpoints: BTreeMap<String, AdaptiveEndpointSchema>,
    pub semantic_hints: BTreeMap<String, AdaptiveSemanticHint>,
    pub numeric_hints: BTreeMap<String, f64>,
    pub recent_adjustments: Vec<ApiAdjustment>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AdaptiveEndpointSchema {
    pub method: String,
    pub path_template: String,
    pub label: String,
    pub success_count: u64,
    pub failure_count: u64,
    pub last_observed_at: Option<f64>,
    pub last_success_at: Option<f64>,
    pub last_failure_at: Option<f64>,
    pub last_status: Option<u16>,
    pub last_error: Option<String>,
    pub response_kind: Option<String>,
    pub response_keys: Vec<String>,
    pub degraded: bool,
    pub skip_until: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AdaptiveSemanticHint {
    pub key: String,
    pub summary: String,
    pub evidence: String,
    pub count: u64,
    pub first_seen_at: Option<f64>,
    pub last_seen_at: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiAdjustment {
    pub ts: f64,
    pub api: Option<String>,
    pub kind: String,
    pub summary: String,
    pub evidence: String,
}

#[derive(Debug, Default)]
struct GlobalStore {
    registry: AdaptiveApiRegistry,
    path: Option<PathBuf>,
    dirty_observations: u64,
    last_save_at: f64,
}

static GLOBAL_STORE: Lazy<Mutex<GlobalStore>> = Lazy::new(|| Mutex::new(GlobalStore::default()));

impl AdaptiveApiRegistry {
    pub fn api(&self, api: &str) -> Option<&AdaptiveApiSchema> {
        self.apis.get(api)
    }

    pub fn api_or_default(&self, api: &str) -> AdaptiveApiSchema {
        self.apis.get(api).cloned().unwrap_or_default()
    }

    pub fn observe_success(
        &mut self,
        api: &str,
        method: &str,
        path: &str,
        label: &str,
        body: &Value,
    ) {
        let now = now_ts();
        self.apis
            .entry(normalize_api_name(api))
            .or_default()
            .observe_success(method, path, label, body);
        self.bump(now);
    }

    pub fn observe_failure(
        &mut self,
        api: &str,
        method: &str,
        path: &str,
        label: &str,
        status: Option<u16>,
        error: &str,
    ) {
        let now = now_ts();
        let api = normalize_api_name(api);
        self.apis
            .entry(api.clone())
            .or_default()
            .observe_failure(method, path, label, status, error);
        self.record_adjustment(
            Some(api),
            "api_failure",
            format!(
                "{} {} observed failure",
                method.to_ascii_uppercase(),
                normalize_path_template(path)
            ),
            error,
        );
        self.bump(now);
    }

    pub fn observe_log_entry(&mut self, api: &str, level: &str, source: &str, message: &str) {
        let now = now_ts();
        self.apis
            .entry(normalize_api_name(api))
            .or_default()
            .observe_log_entry(level, source, message);
        self.bump(now);
    }

    pub fn should_skip(&self, api: &str, method: &str, path: &str) -> bool {
        self.apis
            .get(&normalize_api_name(api))
            .is_some_and(|schema| schema.should_skip(method, path))
    }

    pub fn numeric_hint(&self, api: &str, key: &str) -> Option<f64> {
        self.apis
            .get(&normalize_api_name(api))
            .and_then(|schema| schema.numeric_hints.get(key).copied())
    }

    pub fn merge(&mut self, other: AdaptiveApiRegistry) {
        for (api, schema) in other.apis {
            self.apis.entry(api).or_default().merge(schema);
        }
        self.recent_adjustments.extend(other.recent_adjustments);
        trim_adjustments(&mut self.recent_adjustments);
        self.bump(now_ts());
    }

    fn record_adjustment(
        &mut self,
        api: Option<String>,
        kind: impl Into<String>,
        summary: impl Into<String>,
        evidence: &str,
    ) {
        self.recent_adjustments.push(ApiAdjustment {
            ts: now_ts(),
            api,
            kind: kind.into(),
            summary: summary.into(),
            evidence: truncate(evidence, 700),
        });
        trim_adjustments(&mut self.recent_adjustments);
    }

    fn bump(&mut self, now: f64) {
        self.version = self.version.saturating_add(1);
        self.updated_at = Some(now);
    }
}

impl AdaptiveApiSchema {
    pub fn observe_success(&mut self, method: &str, path: &str, label: &str, body: &Value) {
        let now = now_ts();
        let key = endpoint_key(method, path);
        let shape = response_shape(body);
        let endpoint = self.endpoints.entry(key).or_insert_with(|| {
            AdaptiveEndpointSchema::new(method, normalize_path_template(path), label)
        });
        endpoint.label = label.to_string();
        endpoint.success_count = endpoint.success_count.saturating_add(1);
        endpoint.last_observed_at = Some(now);
        endpoint.last_success_at = Some(now);
        endpoint.last_status = Some(200);
        endpoint.last_error = None;
        endpoint.response_kind = Some(shape.kind);
        endpoint.response_keys = shape.keys;
        endpoint.degraded = false;
        endpoint.skip_until = None;
        self.bump(now);
    }

    pub fn observe_failure(
        &mut self,
        method: &str,
        path: &str,
        label: &str,
        status: Option<u16>,
        error: &str,
    ) {
        let now = now_ts();
        let key = endpoint_key(method, path);
        let endpoint = self.endpoints.entry(key).or_insert_with(|| {
            AdaptiveEndpointSchema::new(method, normalize_path_template(path), label)
        });
        endpoint.label = label.to_string();
        endpoint.failure_count = endpoint.failure_count.saturating_add(1);
        endpoint.last_observed_at = Some(now);
        endpoint.last_failure_at = Some(now);
        endpoint.last_status = status;
        endpoint.last_error = Some(truncate(error, 500));
        if let Some(skip_seconds) = skip_seconds_for_failure(status, error) {
            endpoint.degraded = true;
            endpoint.skip_until = Some(now + skip_seconds);
            self.record_adjustment(
                "endpoint_degraded",
                format!(
                    "{} {} temporarily skipped after {}",
                    method.to_ascii_uppercase(),
                    normalize_path_template(path),
                    status
                        .map(|value| format!("HTTP {value}"))
                        .unwrap_or_else(|| "client error".to_string())
                ),
                error,
            );
        }
        self.observe_error_semantics(error);
        self.bump(now);
    }

    pub fn should_skip(&self, method: &str, path: &str) -> bool {
        let now = now_ts();
        self.endpoints
            .get(&endpoint_key(method, path))
            .and_then(|endpoint| endpoint.skip_until)
            .is_some_and(|skip_until| skip_until > now)
    }

    pub fn observe_log_entry(&mut self, level: &str, source: &str, message: &str) {
        self.observe_message_semantics(level, source, message);
    }

    pub fn upsert_semantic_hint(&mut self, key: &str, summary: &str, source: &str, message: &str) {
        let now = now_ts();
        let evidence = truncate(&format!("{source}: {message}"), 700);
        let changed = {
            let hint = self
                .semantic_hints
                .entry(key.to_string())
                .or_insert_with(|| AdaptiveSemanticHint {
                    key: key.to_string(),
                    summary: summary.to_string(),
                    evidence: String::new(),
                    count: 0,
                    first_seen_at: Some(now),
                    last_seen_at: None,
                });
            let changed = hint.summary != summary || hint.evidence != evidence;
            hint.summary = summary.to_string();
            hint.evidence = evidence;
            hint.count += 1;
            hint.last_seen_at = Some(now);
            changed
        };
        if changed {
            self.record_adjustment("semantic_hint", summary.to_string(), message);
        }
        self.bump(now);
    }

    pub fn observe_numeric_hint(
        &mut self,
        key: &str,
        value: f64,
        summary: impl Into<String>,
        evidence: &str,
    ) {
        if !value.is_finite() {
            return;
        }
        let previous = self.numeric_hints.get(key).copied();
        if previous.is_some_and(|item| item >= value) {
            return;
        }
        self.numeric_hints.insert(key.to_string(), value);
        self.record_adjustment("numeric_hint", summary.into(), evidence);
        self.bump(now_ts());
    }

    pub fn merge(&mut self, other: AdaptiveApiSchema) {
        for (key, endpoint) in other.endpoints {
            self.endpoints
                .entry(key)
                .and_modify(|existing| {
                    let success_count = existing
                        .success_count
                        .saturating_add(endpoint.success_count);
                    let failure_count = existing
                        .failure_count
                        .saturating_add(endpoint.failure_count);
                    if endpoint.last_observed_at > existing.last_observed_at {
                        *existing = endpoint.clone();
                    }
                    existing.success_count = success_count;
                    existing.failure_count = failure_count;
                })
                .or_insert(endpoint);
        }
        for (key, hint) in other.semantic_hints {
            self.semantic_hints
                .entry(key)
                .and_modify(|existing| {
                    existing.count = existing.count.saturating_add(hint.count);
                    if hint.last_seen_at > existing.last_seen_at {
                        existing.summary = hint.summary.clone();
                        existing.evidence = hint.evidence.clone();
                        existing.last_seen_at = hint.last_seen_at;
                    }
                })
                .or_insert(hint);
        }
        for (key, value) in other.numeric_hints {
            self.numeric_hints
                .entry(key)
                .and_modify(|existing| *existing = existing.max(value))
                .or_insert(value);
        }
        self.recent_adjustments.extend(other.recent_adjustments);
        trim_adjustments(&mut self.recent_adjustments);
        self.bump(now_ts());
    }

    fn observe_error_semantics(&mut self, error: &str) {
        let source = "error";
        self.observe_message_semantics("error", source, error);
    }

    fn observe_message_semantics(&mut self, level: &str, source: &str, message: &str) {
        let lowered = message.to_ascii_lowercase();
        if lowered.contains("managertoolcall")
            && lowered.contains("tool_name")
            && lowered.contains("arguments")
        {
            self.upsert_semantic_hint(
                "manager_tool_call_shape",
                "Manager tool calls require {\"tool_name\":\"...\",\"arguments\":{...}} rather than bare tool arguments.",
                &format!("{level} {source}"),
                message,
            );
        }

        if lowered.contains("missingminimalexchangetradevolume")
            || lowered.contains("required order size is not compatible")
        {
            self.upsert_semantic_hint(
                "exchange_minimum_trade_volume",
                "Exchange minimum trade-volume constraints must be applied before requesting trades.",
                &format!("{level} {source}"),
                message,
            );
        }

        if let Some(cost_min) = extract_exchange_min_trade_cost_usd(message) {
            self.observe_numeric_hint(
                "micro_trade_min_usd",
                cost_min,
                format!(
                    "Recommended micro-trade minimum raised to ${cost_min:.2} from exchange requirements"
                ),
                message,
            );
        }

        if lowered.contains("coingecko") && lowered.contains("429") {
            self.upsert_semantic_hint(
                "market_metadata_rate_limit",
                "Market metadata lookups are being rate limited; prefer cached/local exchange data where possible.",
                &format!("{level} {source}"),
                message,
            );
        }

        if lowered.contains("upstream error")
            || lowered.contains("bad gateway")
            || lowered.contains("gateway timeout")
            || lowered.contains("error sending request")
            || lowered.contains("connection reset")
            || lowered.contains("connection closed")
            || lowered.contains("http 502")
            || lowered.contains("http 503")
            || lowered.contains("http 504")
            || lowered.contains("status 502")
            || lowered.contains("status 503")
            || lowered.contains("status 504")
        {
            self.upsert_semantic_hint(
                "transient_upstream_failure",
                "Remote upstream is transiently failing; prefer alternate provider families or cached/fallback paths until health recovers.",
                &format!("{level} {source}"),
                message,
            );
        }
    }

    fn record_adjustment(
        &mut self,
        kind: impl Into<String>,
        summary: impl Into<String>,
        evidence: &str,
    ) {
        self.recent_adjustments.push(ApiAdjustment {
            ts: now_ts(),
            api: None,
            kind: kind.into(),
            summary: summary.into(),
            evidence: truncate(evidence, 700),
        });
        trim_adjustments(&mut self.recent_adjustments);
    }

    fn bump(&mut self, now: f64) {
        self.version = self.version.saturating_add(1);
        self.updated_at = Some(now);
    }
}

impl AdaptiveEndpointSchema {
    fn new(method: &str, path_template: String, label: &str) -> Self {
        Self {
            method: method.to_ascii_uppercase(),
            path_template,
            label: label.to_string(),
            ..Self::default()
        }
    }
}

pub async fn configure_persistence(path: impl Into<PathBuf>) {
    let path = path.into();
    let loaded = match fs::read_to_string(&path).await {
        Ok(raw) => serde_json::from_str::<AdaptiveApiRegistry>(&raw).ok(),
        Err(_) => None,
    };
    let mut store = GLOBAL_STORE.lock().await;
    store.path = Some(path);
    if let Some(loaded) = loaded {
        store.registry.merge(loaded);
    }
}

pub async fn snapshot() -> AdaptiveApiRegistry {
    GLOBAL_STORE.lock().await.registry.clone()
}

pub async fn api_snapshot(api: &str) -> AdaptiveApiSchema {
    GLOBAL_STORE.lock().await.registry.api_or_default(api)
}

pub async fn merge_snapshot(snapshot: AdaptiveApiRegistry) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store.registry.merge(snapshot);
        store.dirty_observations = store
            .dirty_observations
            .saturating_add(SAVE_MAX_DIRTY_OBSERVATIONS);
        save_request_if_due(&mut store)
    };
    spawn_save(save);
}

pub async fn observe_success(api: &str, method: &str, path: &str, label: &str, body: &Value) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store
            .registry
            .observe_success(api, method, path, label, body);
        mark_dirty_and_maybe_save(&mut store)
    };
    spawn_save(save);
}

pub async fn observe_failure(
    api: &str,
    method: &str,
    path: &str,
    label: &str,
    status: Option<u16>,
    error: &str,
) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store
            .registry
            .observe_failure(api, method, path, label, status, error);
        mark_dirty_and_maybe_save(&mut store)
    };
    spawn_save(save);
}

pub async fn observe_log_entry(api: &str, level: &str, source: &str, message: &str) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store
            .registry
            .observe_log_entry(api, level, source, message);
        mark_dirty_and_maybe_save(&mut store)
    };
    spawn_save(save);
}

pub async fn should_skip(api: &str, method: &str, path: &str) -> bool {
    GLOBAL_STORE
        .lock()
        .await
        .registry
        .should_skip(api, method, path)
}

pub async fn numeric_hint(api: &str, key: &str) -> Option<f64> {
    GLOBAL_STORE.lock().await.registry.numeric_hint(api, key)
}

#[cfg(test)]
pub async fn reset_for_tests() {
    let mut store = GLOBAL_STORE.lock().await;
    *store = GlobalStore::default();
}

struct SaveRequest {
    path: PathBuf,
    snapshot: AdaptiveApiRegistry,
}

fn mark_dirty_and_maybe_save(store: &mut GlobalStore) -> Option<SaveRequest> {
    store.dirty_observations = store.dirty_observations.saturating_add(1);
    save_request_if_due(store)
}

fn save_request_if_due(store: &mut GlobalStore) -> Option<SaveRequest> {
    let path = store.path.clone()?;
    let now = now_ts();
    let interval_due =
        store.last_save_at <= 0.0 || (now - store.last_save_at) >= SAVE_MIN_INTERVAL_SECONDS;
    let volume_due = store.dirty_observations >= SAVE_MAX_DIRTY_OBSERVATIONS;
    if !interval_due && !volume_due {
        return None;
    }
    store.dirty_observations = 0;
    store.last_save_at = now;
    Some(SaveRequest {
        path,
        snapshot: store.registry.clone(),
    })
}

fn spawn_save(save: Option<SaveRequest>) {
    let Some(save) = save else {
        return;
    };
    tokio::spawn(async move {
        if let Some(parent) = save.path.parent()
            && let Err(error) = fs::create_dir_all(parent).await
        {
            tracing::warn!(
                path = %save.path.display(),
                error = %error,
                "failed to create adaptive API schema directory"
            );
            return;
        }
        match serde_json::to_string_pretty(&save.snapshot) {
            Ok(rendered) => {
                if let Err(error) = fs::write(&save.path, rendered).await {
                    tracing::warn!(
                        path = %save.path.display(),
                        error = %error,
                        "failed to persist adaptive API schema"
                    );
                }
            }
            Err(error) => {
                tracing::warn!(error = %error, "failed to serialise adaptive API schema");
            }
        }
    });
}

struct ResponseShape {
    kind: String,
    keys: Vec<String>,
}

fn response_shape(body: &Value) -> ResponseShape {
    match body {
        Value::Object(object) => ResponseShape {
            kind: "object".to_string(),
            keys: sorted_limited_keys(object.keys().cloned()),
        },
        Value::Array(items) => {
            let keys = items
                .iter()
                .find_map(Value::as_object)
                .map(|object| sorted_limited_keys(object.keys().cloned()))
                .unwrap_or_default();
            ResponseShape {
                kind: "array".to_string(),
                keys,
            }
        }
        Value::String(_) => ResponseShape {
            kind: "string".to_string(),
            keys: Vec::new(),
        },
        Value::Number(_) => ResponseShape {
            kind: "number".to_string(),
            keys: Vec::new(),
        },
        Value::Bool(_) => ResponseShape {
            kind: "bool".to_string(),
            keys: Vec::new(),
        },
        Value::Null => ResponseShape {
            kind: "null".to_string(),
            keys: Vec::new(),
        },
    }
}

fn sorted_limited_keys(keys: impl Iterator<Item = String>) -> Vec<String> {
    let mut keys = keys.collect::<Vec<_>>();
    keys.sort();
    keys.truncate(30);
    keys
}

fn skip_seconds_for_failure(status: Option<u16>, error: &str) -> Option<f64> {
    match status {
        Some(404 | 405) => Some(STRUCTURAL_SKIP_SECONDS),
        Some(429) => Some(RATE_LIMIT_SKIP_SECONDS),
        Some(value) if value >= 500 => Some(TRANSIENT_SKIP_SECONDS),
        _ if error.to_ascii_lowercase().contains("parse failed") => Some(TRANSIENT_SKIP_SECONDS),
        _ => None,
    }
}

fn endpoint_key(method: &str, path: &str) -> String {
    format!(
        "{} {}",
        method.to_ascii_uppercase(),
        normalize_path_template(path)
    )
}

fn normalize_path_template(path: &str) -> String {
    let url_path = Url::parse(path)
        .ok()
        .map(|url| {
            if let Some(query) = url.query() {
                format!("{}?{query}", url.path())
            } else {
                url.path().to_string()
            }
        })
        .unwrap_or_else(|| path.to_string());
    let path_only = url_path.split('?').next().unwrap_or(url_path.as_str());

    if path_only == "/api/market/ticker" {
        return "/api/market/ticker".to_string();
    }
    if path_only.starts_with("/dashboard/watched_symbol/") {
        return "/dashboard/watched_symbol/{symbol}".to_string();
    }
    if path_only.starts_with("/dashboard/currency_price_graph_update/") {
        return "/dashboard/currency_price_graph_update/{exchange_id}/{symbol}/{time_frame}/live"
            .to_string();
    }
    if path_only == "/backtesting" {
        return match url_path.as_str() {
            item if item.contains("start_backtesting") => {
                "/backtesting?action_type=start_backtesting".to_string()
            }
            item if item.contains("backtesting_report") => {
                "/backtesting?update_type=backtesting_report".to_string()
            }
            item if item.contains("backtesting_data_files") => {
                "/backtesting?update_type=backtesting_data_files".to_string()
            }
            _ => "/backtesting".to_string(),
        };
    }

    path_only
        .split('/')
        .map(|segment| {
            if segment.is_empty() {
                String::new()
            } else if looks_dynamic_segment(segment) {
                "{id}".to_string()
            } else {
                segment.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn looks_dynamic_segment(segment: &str) -> bool {
    let alnum = segment
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .count();
    let digit = segment.chars().filter(|ch| ch.is_ascii_digit()).count();
    segment.len() >= 12 && (digit >= 6 || alnum >= 12)
}

pub fn extract_exchange_min_trade_cost_usd(message: &str) -> Option<f64> {
    static COST_MIN_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"["']cost["']\s*:\s*\{[^}\n]*["']min["']\s*:\s*([0-9]+(?:\.[0-9]+)?)"#)
            .expect("cost min regex")
    });
    static TEXT_MIN_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r#"(?i)(?:cost|notional|trade volume)[^0-9]{0,60}min[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)"#,
        )
        .expect("text min regex")
    });

    COST_MIN_RE
        .captures(message)
        .or_else(|| TEXT_MIN_RE.captures(message))
        .and_then(|captures| captures.get(1))
        .and_then(|item| item.as_str().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn normalize_api_name(api: &str) -> String {
    let cleaned = api.trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
    }
}

fn trim_adjustments(adjustments: &mut Vec<ApiAdjustment>) {
    if adjustments.len() > MAX_RECENT_ADJUSTMENTS {
        let remove = adjustments.len() - MAX_RECENT_ADJUSTMENTS;
        adjustments.drain(0..remove);
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

pub fn endpoint_status_body(status: u16) -> Value {
    json!({ "http_status": status })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_exchange_min_trade_cost_from_log() {
        let message = "order size is not compatible with BNB/USDT exchange requirements: {'cost': {'min': 5.0, 'max': 9000000.0}}";
        assert_eq!(extract_exchange_min_trade_cost_usd(message), Some(5.0));
    }

    #[test]
    fn records_manager_tool_call_semantic_hint() {
        let mut schema = AdaptiveApiSchema::default();
        schema.observe_log_entry(
            "WARNING",
            "AIToolsTeamManagerAgentProducer",
            "3 validation errors for ManagerToolCall tool_name Field required arguments Field required agent_name Extra inputs",
        );
        assert!(
            schema
                .semantic_hints
                .contains_key("manager_tool_call_shape")
        );
    }

    #[test]
    fn normalizes_remote_provider_url_to_endpoint_shape() {
        let mut schema = AdaptiveApiSchema::default();
        schema.observe_success(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            "chat completions",
            &json!({"choices": []}),
        );
        assert!(schema.endpoints.contains_key("POST /v1/chat/completions"));
    }

    #[test]
    fn merge_preserves_endpoint_counts_when_latest_shape_wins() {
        let mut original = AdaptiveApiSchema::default();
        original.observe_success("GET", "/api/ping", "ping", &json!({"old": true}));
        let original_seen_at = original.endpoints["GET /api/ping"].last_observed_at;

        let mut incoming = AdaptiveApiSchema::default();
        incoming.observe_success("GET", "/api/ping", "ping", &json!({"new": true}));
        incoming
            .endpoints
            .get_mut("GET /api/ping")
            .expect("incoming endpoint")
            .last_observed_at = original_seen_at.map(|ts| ts + 1.0);

        original.merge(incoming);

        let endpoint = original
            .endpoints
            .get("GET /api/ping")
            .expect("merged endpoint");
        assert_eq!(endpoint.success_count, 2);
        assert_eq!(endpoint.response_keys, vec!["new"]);
    }
}
