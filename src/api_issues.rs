//! Persistent API issue and mitigation tracking.
//!
//! This store is deliberately local to Gail.  It records failures, backoffs,
//! and recoveries without asking Refiner to diagnose Gail while Refiner is
//! blocked on Gail, which avoids the retry loop that originally surfaced this
//! class of issue.

use std::{collections::BTreeMap, path::PathBuf};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{fs, sync::Mutex};
use tokio_postgres::NoTls;

const MAX_ACTIONS_PER_ISSUE: usize = 30;
const MAX_RECENT_EVENTS: usize = 120;
const SAVE_MIN_INTERVAL_SECONDS: f64 = 2.0;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ApiIssueRegistry {
    pub version: u64,
    pub updated_at: Option<f64>,
    pub storage: ApiIssueStorageStatus,
    pub summary: ApiIssueSummary,
    pub issues: BTreeMap<String, ApiIssue>,
    pub recent_events: Vec<ApiIssueAction>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ApiIssueStorageStatus {
    pub file_path: Option<String>,
    pub postgres_configured: bool,
    pub last_file_save_at: Option<f64>,
    pub last_postgres_save_at: Option<f64>,
    pub last_postgres_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ApiIssueSummary {
    pub active: usize,
    pub mitigated: usize,
    pub resolved: usize,
    pub critical: usize,
    pub warning: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiIssue {
    pub id: String,
    pub api: String,
    pub provider: Option<String>,
    pub endpoint: String,
    pub category: String,
    pub severity: String,
    pub status: String,
    pub summary: String,
    pub latest_error: Option<String>,
    pub occurrences: u64,
    pub first_seen_at: Option<f64>,
    pub last_seen_at: Option<f64>,
    pub resolved_at: Option<f64>,
    pub next_retry_at: Option<f64>,
    pub active: bool,
    pub mitigation: Option<String>,
    pub context: Value,
    pub actions: Vec<ApiIssueAction>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiIssueAction {
    pub ts: f64,
    pub issue_id: Option<String>,
    pub kind: String,
    pub summary: String,
    pub details: Value,
}

#[derive(Debug, Default)]
struct GlobalStore {
    registry: ApiIssueRegistry,
    path: Option<PathBuf>,
    postgres_dsn: Option<String>,
    dirty: bool,
    last_save_at: f64,
}

struct SaveRequest {
    path: Option<PathBuf>,
    postgres_dsn: Option<String>,
    snapshot: ApiIssueRegistry,
}

static GLOBAL_STORE: Lazy<Mutex<GlobalStore>> = Lazy::new(|| Mutex::new(GlobalStore::default()));

impl Default for ApiIssue {
    fn default() -> Self {
        Self {
            id: String::new(),
            api: String::new(),
            provider: None,
            endpoint: String::new(),
            category: "unknown".to_string(),
            severity: "warning".to_string(),
            status: "active".to_string(),
            summary: String::new(),
            latest_error: None,
            occurrences: 0,
            first_seen_at: None,
            last_seen_at: None,
            resolved_at: None,
            next_retry_at: None,
            active: true,
            mitigation: None,
            context: Value::Null,
            actions: Vec::new(),
        }
    }
}

impl Default for ApiIssueAction {
    fn default() -> Self {
        Self {
            ts: 0.0,
            issue_id: None,
            kind: String::new(),
            summary: String::new(),
            details: Value::Null,
        }
    }
}

impl ApiIssueRegistry {
    fn observe_failure(&mut self, observation: IssueObservation) {
        let now = now_ts();
        let id = issue_id(
            &observation.api,
            observation.provider.as_deref(),
            &observation.endpoint,
            &observation.category,
        );
        let mitigation = mitigation_for(&observation.category, observation.retry_ttl_seconds);
        let next_retry_at = observation
            .retry_ttl_seconds
            .filter(|ttl| *ttl > 0.0)
            .map(|ttl| now + ttl);
        let action = ApiIssueAction {
            ts: now,
            issue_id: Some(id.clone()),
            kind: "mitigation_applied".to_string(),
            summary: mitigation.clone(),
            details: json!({
                "workflow": observation.workflow,
                "role": observation.role,
                "provider": observation.provider,
                "endpoint": observation.endpoint,
                "category": observation.category,
                "retry_ttl_seconds": observation.retry_ttl_seconds,
            }),
        };
        let issue = self.issues.entry(id.clone()).or_insert_with(|| ApiIssue {
            id: id.clone(),
            api: observation.api.clone(),
            provider: observation.provider.clone(),
            endpoint: observation.endpoint.clone(),
            category: observation.category.clone(),
            severity: observation.severity.clone(),
            status: "active".to_string(),
            summary: observation.summary.clone(),
            latest_error: None,
            occurrences: 0,
            first_seen_at: Some(now),
            last_seen_at: None,
            resolved_at: None,
            next_retry_at: None,
            active: true,
            mitigation: None,
            context: Value::Null,
            actions: Vec::new(),
        });
        issue.api = observation.api;
        issue.provider = observation.provider;
        issue.endpoint = observation.endpoint;
        issue.category = observation.category;
        issue.severity = observation.severity;
        issue.status = "mitigating".to_string();
        issue.summary = observation.summary;
        issue.latest_error = Some(truncate(&observation.error, 900));
        issue.occurrences = issue.occurrences.saturating_add(1);
        issue.last_seen_at = Some(now);
        issue.resolved_at = None;
        issue.next_retry_at = next_retry_at;
        issue.active = true;
        issue.mitigation = Some(mitigation);
        issue.context = observation.context;
        issue.actions.push(action.clone());
        trim_actions(&mut issue.actions);
        self.record_event(action);
        self.bump(now);
    }

    fn observe_recovery(
        &mut self,
        api: &str,
        provider: Option<&str>,
        endpoint: &str,
        summary: &str,
    ) -> bool {
        let now = now_ts();
        let mut recovered = Vec::new();
        for issue in self.issues.values_mut() {
            if !issue.active {
                continue;
            }
            if !issue.api.eq_ignore_ascii_case(api) {
                continue;
            }
            if issue.endpoint != endpoint {
                continue;
            }
            if let Some(provider) = provider
                && !issue
                    .provider
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case(provider))
            {
                continue;
            }
            issue.active = false;
            issue.status = "resolved".to_string();
            issue.resolved_at = Some(now);
            issue.next_retry_at = None;
            let action = ApiIssueAction {
                ts: now,
                issue_id: Some(issue.id.clone()),
                kind: "recovered".to_string(),
                summary: summary.to_string(),
                details: json!({
                    "api": api,
                    "provider": provider,
                    "endpoint": endpoint,
                }),
            };
            issue.actions.push(action.clone());
            trim_actions(&mut issue.actions);
            recovered.push(action);
        }
        if !recovered.is_empty() {
            for action in recovered {
                self.record_event(action);
            }
            self.bump(now);
            true
        } else {
            false
        }
    }

    fn record_event(&mut self, event: ApiIssueAction) {
        self.recent_events.push(event);
        if self.recent_events.len() > MAX_RECENT_EVENTS {
            let remove = self.recent_events.len() - MAX_RECENT_EVENTS;
            self.recent_events.drain(0..remove);
        }
    }

    fn bump(&mut self, now: f64) {
        self.version = self.version.saturating_add(1);
        self.updated_at = Some(now);
        self.refresh_summary();
    }

    fn refresh_summary(&mut self) {
        let mut summary = ApiIssueSummary::default();
        for issue in self.issues.values() {
            if issue.active {
                summary.active += 1;
            } else if issue.status == "resolved" {
                summary.resolved += 1;
            }
            if issue.status == "mitigating" {
                summary.mitigated += 1;
            }
            if issue.severity == "critical" {
                summary.critical += 1;
            } else if issue.severity == "warning" {
                summary.warning += 1;
            }
        }
        self.summary = summary;
    }
}

struct IssueObservation {
    api: String,
    provider: Option<String>,
    endpoint: String,
    category: String,
    severity: String,
    summary: String,
    error: String,
    workflow: Option<String>,
    role: Option<String>,
    retry_ttl_seconds: Option<f64>,
    context: Value,
}

pub async fn configure_persistence(path: impl Into<PathBuf>, postgres_dsn: Option<String>) {
    let path = path.into();
    let loaded = match fs::read_to_string(&path).await {
        Ok(raw) => serde_json::from_str::<ApiIssueRegistry>(&raw).ok(),
        Err(_) => None,
    };
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store.path = Some(path.clone());
        store.postgres_dsn = postgres_dsn
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let mut registry = loaded.unwrap_or_default();
        registry.storage.file_path = Some(path.to_string_lossy().to_string());
        registry.storage.postgres_configured = store.postgres_dsn.is_some();
        registry.refresh_summary();
        store.registry = registry;
        store.dirty = true;
        save_request_if_due(&mut store, true)
    };
    spawn_save(save);
}

pub async fn snapshot() -> ApiIssueRegistry {
    let mut snapshot = GLOBAL_STORE.lock().await.registry.clone();
    snapshot.refresh_summary();
    snapshot
}

pub async fn observe_provider_failure(
    provider: &str,
    model: &str,
    workflow: &str,
    role: &str,
    category: &str,
    severity: &str,
    error: &str,
    retry_ttl_seconds: Option<f64>,
) {
    observe_failure(IssueObservation {
        api: "provider".to_string(),
        provider: Some(normalize(provider)),
        endpoint: provider_endpoint(provider, model),
        category: normalize(category),
        severity: normalize_severity(severity),
        summary: format!(
            "{} provider failure while serving {workflow}/{role}",
            provider.trim()
        ),
        error: error.to_string(),
        workflow: Some(workflow.to_string()),
        role: Some(role.to_string()),
        retry_ttl_seconds,
        context: json!({
            "model": model,
            "source": "runtime_completion",
        }),
    })
    .await;
}

pub async fn observe_provider_recovery(provider: &str, model: &str) {
    observe_recovery(
        "provider",
        Some(provider),
        &provider_endpoint(provider, model),
        "Provider completed successfully; adaptive backoff can relax.",
    )
    .await;
}

pub async fn observe_orchestration_failure(
    workflow: &str,
    role: &str,
    error: &str,
    context: Value,
) {
    observe_failure(IssueObservation {
        api: "gail".to_string(),
        provider: None,
        endpoint: "orchestration".to_string(),
        category: "orchestration_exhausted".to_string(),
        severity: "critical".to_string(),
        summary: format!("Gail orchestration exhausted all candidates for {workflow}/{role}"),
        error: error.to_string(),
        workflow: Some(workflow.to_string()),
        role: Some(role.to_string()),
        retry_ttl_seconds: Some(60.0),
        context,
    })
    .await;
}

pub async fn observe_api_failure(
    api: &str,
    method: &str,
    path: &str,
    label: &str,
    status: Option<u16>,
    error: &str,
) {
    let category = classify_api_issue(status, error);
    let severity = severity_for_api_issue(status, &category);
    observe_failure(IssueObservation {
        api: normalize(api),
        provider: None,
        endpoint: api_endpoint(method, path),
        category: category.clone(),
        severity: severity.to_string(),
        summary: format!("{} API failure while calling {label}", api.trim()),
        error: error.to_string(),
        workflow: Some("trading".to_string()),
        role: Some(label.to_string()),
        retry_ttl_seconds: retry_ttl_for_api_issue(status, &category),
        context: json!({
            "method": method,
            "path": path,
            "status": status,
            "label": label,
        }),
    })
    .await;
}

pub async fn observe_api_recovery(api: &str, method: &str, path: &str, label: &str) {
    observe_recovery(
        &normalize(api),
        None,
        &api_endpoint(method, path),
        &format!("{} {label} API call completed successfully.", api.trim()),
    )
    .await;
}

async fn observe_failure(observation: IssueObservation) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        store.registry.observe_failure(observation);
        store.dirty = true;
        save_request_if_due(&mut store, true)
    };
    spawn_save(save);
}

async fn observe_recovery(api: &str, provider: Option<&str>, endpoint: &str, summary: &str) {
    let save = {
        let mut store = GLOBAL_STORE.lock().await;
        let changed = store
            .registry
            .observe_recovery(api, provider, endpoint, summary);
        if changed {
            store.dirty = true;
            save_request_if_due(&mut store, true)
        } else {
            None
        }
    };
    spawn_save(save);
}

pub async fn prometheus_metrics() -> String {
    let snapshot = snapshot().await;
    render_prometheus_metrics(&snapshot)
}

fn render_prometheus_metrics(snapshot: &ApiIssueRegistry) -> String {
    let mut out = String::new();
    out.push_str("# HELP gail_api_issues_active Active Gail API issues.\n");
    out.push_str("# TYPE gail_api_issues_active gauge\n");
    out.push_str(&format!(
        "gail_api_issues_active {}\n",
        snapshot.summary.active
    ));
    out.push_str("# HELP gail_api_issue_occurrences_total Gail API issue occurrence count.\n");
    out.push_str("# TYPE gail_api_issue_occurrences_total counter\n");
    for issue in snapshot.issues.values() {
        let labels = format!(
            "api=\"{}\",provider=\"{}\",category=\"{}\",severity=\"{}\",status=\"{}\"",
            escape_label(&issue.api),
            escape_label(issue.provider.as_deref().unwrap_or("")),
            escape_label(&issue.category),
            escape_label(&issue.severity),
            escape_label(&issue.status),
        );
        out.push_str(&format!(
            "gail_api_issue_active{{{labels}}} {}\n",
            if issue.active { 1 } else { 0 }
        ));
        out.push_str(&format!(
            "gail_api_issue_occurrences_total{{{labels}}} {}\n",
            issue.occurrences
        ));
    }
    out
}

fn save_request_if_due(store: &mut GlobalStore, force: bool) -> Option<SaveRequest> {
    if !store.dirty && !force {
        return None;
    }
    let now = now_ts();
    if !force && store.last_save_at > 0.0 && (now - store.last_save_at) < SAVE_MIN_INTERVAL_SECONDS
    {
        return None;
    }
    store.dirty = false;
    store.last_save_at = now;
    store.registry.storage.file_path = store
        .path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());
    store.registry.storage.postgres_configured = store.postgres_dsn.is_some();
    Some(SaveRequest {
        path: store.path.clone(),
        postgres_dsn: store.postgres_dsn.clone(),
        snapshot: store.registry.clone(),
    })
}

fn spawn_save(save: Option<SaveRequest>) {
    let Some(save) = save else {
        return;
    };
    tokio::spawn(async move {
        let file_result = if let Some(path) = save.path.as_ref() {
            persist_file(path, &save.snapshot).await
        } else {
            Ok(None)
        };
        if let Err(error) = file_result {
            tracing::warn!(error = %error, "failed to persist Gail API issue registry");
        }
        if let Some(dsn) = save.postgres_dsn.as_deref() {
            if let Err(error) = persist_postgres(dsn, &save.snapshot).await {
                tracing::warn!(error = %error, "failed to persist Gail API issues to Postgres");
                let mut store = GLOBAL_STORE.lock().await;
                store.registry.storage.last_postgres_error =
                    Some(truncate(&error.to_string(), 700));
            } else {
                let mut store = GLOBAL_STORE.lock().await;
                store.registry.storage.last_postgres_save_at = Some(now_ts());
                store.registry.storage.last_postgres_error = None;
            }
        }
    });
}

async fn persist_file(path: &PathBuf, snapshot: &ApiIssueRegistry) -> std::io::Result<Option<f64>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let saved_at = now_ts();
    let mut snapshot = snapshot.clone();
    snapshot.storage.last_file_save_at = Some(saved_at);
    let mut rendered = serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "{}".to_string());
    rendered.push('\n');
    fs::write(path, rendered).await?;
    let mut store = GLOBAL_STORE.lock().await;
    store.registry.storage.last_file_save_at = Some(
        store
            .registry
            .storage
            .last_file_save_at
            .unwrap_or(0.0)
            .max(saved_at),
    );
    Ok(store.registry.storage.last_file_save_at)
}

async fn persist_postgres(
    dsn: &str,
    snapshot: &ApiIssueRegistry,
) -> Result<(), tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            tracing::debug!(error = %error, "Gail API issue Postgres connection ended");
        }
    });
    client
        .batch_execute(
            r#"
            SET client_min_messages TO WARNING;
            CREATE TABLE IF NOT EXISTS gail_api_issue_snapshots (
                id BIGSERIAL PRIMARY KEY,
                captured_at DOUBLE PRECISION NOT NULL,
                payload JSONB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gail_api_issues (
                id TEXT PRIMARY KEY,
                api TEXT NOT NULL,
                provider TEXT,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                active BOOLEAN NOT NULL,
                last_seen_at DOUBLE PRECISION,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            "#,
        )
        .await?;
    let payload = serde_json::to_value(snapshot).unwrap_or_else(|_| json!({}));
    client
        .execute(
            "INSERT INTO gail_api_issue_snapshots (captured_at, payload) VALUES ($1, $2)",
            &[&now_ts(), &payload],
        )
        .await?;
    for issue in snapshot.issues.values() {
        let payload = serde_json::to_value(issue).unwrap_or_else(|_| json!({}));
        client
            .execute(
                r#"
                INSERT INTO gail_api_issues
                    (id, api, provider, category, severity, active, last_seen_at, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    api = EXCLUDED.api,
                    provider = EXCLUDED.provider,
                    category = EXCLUDED.category,
                    severity = EXCLUDED.severity,
                    active = EXCLUDED.active,
                    last_seen_at = EXCLUDED.last_seen_at,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                "#,
                &[
                    &issue.id,
                    &issue.api,
                    &issue.provider,
                    &issue.category,
                    &issue.severity,
                    &issue.active,
                    &issue.last_seen_at,
                    &payload,
                ],
            )
            .await?;
    }
    Ok(())
}

fn issue_id(api: &str, provider: Option<&str>, endpoint: &str, category: &str) -> String {
    format!(
        "{}:{}:{}:{}",
        normalize(api),
        provider
            .map(normalize)
            .unwrap_or_else(|| "none".to_string()),
        normalize(endpoint),
        normalize(category)
    )
}

fn provider_endpoint(provider: &str, model: &str) -> String {
    let model = model.trim();
    if model.is_empty() || model.eq_ignore_ascii_case("default") {
        format!("{}/completion", normalize(provider))
    } else {
        format!("{}/{}", normalize(provider), model)
    }
}

fn api_endpoint(method: &str, path: &str) -> String {
    format!("{} {}", normalize(method), path.trim())
}

fn mitigation_for(category: &str, retry_ttl_seconds: Option<f64>) -> String {
    let ttl = retry_ttl_seconds.unwrap_or_default();
    let ttl_text = if ttl > 0.0 {
        format!(" for about {}s", ttl.round() as u64)
    } else {
        String::new()
    };
    match normalize(category).as_str() {
        "quota" => format!(
            "Provider family backoff is active{ttl_text}; routing should prefer alternate provider families."
        ),
        "upstream" => format!(
            "Transient upstream backoff is active{ttl_text}; Gail will retry later and use other working providers where available."
        ),
        "timeout" => format!(
            "Timeout backoff is active{ttl_text}; Gail will prefer lower-latency working providers."
        ),
        "auth" => "Authentication or authorization failed; check service credentials or ingress policy.".to_string(),
        "missing_endpoint" => format!(
            "Endpoint backoff is active{ttl_text}; adaptive schema tracking will prefer known working alternatives."
        ),
        "schema_mismatch" => format!(
            "Response schema drift was detected{ttl_text}; adaptive schema tracking will keep using the latest successful shapes."
        ),
        "api_error" => format!(
            "Remote API error backoff is active{ttl_text}; Gail will retry later and continue with degraded context where possible."
        ),
        "unconfigured" => {
            "Provider is not currently usable; check credentials/base URL or rely on configured alternatives.".to_string()
        }
        "orchestration_exhausted" => {
            "All candidates failed or were in backoff; add/repair a working provider or wait for backoff to expire.".to_string()
        }
        _ => "Issue recorded for operator visibility; Gail will keep using successful alternatives.".to_string(),
    }
}

fn trim_actions(actions: &mut Vec<ApiIssueAction>) {
    if actions.len() > MAX_ACTIONS_PER_ISSUE {
        let remove = actions.len() - MAX_ACTIONS_PER_ISSUE;
        actions.drain(0..remove);
    }
}

fn normalize(value: &str) -> String {
    let cleaned = value.trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '/' | '.') {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    }
}

fn normalize_severity(value: &str) -> String {
    match normalize(value).as_str() {
        "critical" | "warning" | "info" => normalize(value),
        _ => "warning".to_string(),
    }
}

fn classify_api_issue(status: Option<u16>, error: &str) -> String {
    let lowered = error.to_ascii_lowercase();
    match status {
        Some(401 | 403) => "auth".to_string(),
        Some(404 | 405) => "missing_endpoint".to_string(),
        Some(429) => "quota".to_string(),
        Some(502..=504) => "upstream".to_string(),
        _ if lowered.contains("timeout") || lowered.contains("timed out") => "timeout".to_string(),
        _ if lowered.contains("too many requests")
            || lowered.contains("rate limit")
            || lowered.contains("status\":429") =>
        {
            "quota".to_string()
        }
        _ if lowered.contains("parse failed")
            || lowered.contains("decode")
            || lowered.contains("schema") =>
        {
            "schema_mismatch".to_string()
        }
        _ => "api_error".to_string(),
    }
}

fn severity_for_api_issue(status: Option<u16>, category: &str) -> &'static str {
    if matches!(status, Some(401 | 403)) || matches!(category, "auth") {
        "critical"
    } else {
        "warning"
    }
}

fn retry_ttl_for_api_issue(status: Option<u16>, category: &str) -> Option<f64> {
    match (status, category) {
        (Some(401 | 403), _) => Some(900.0),
        (_, "quota") => Some(300.0),
        (_, "upstream" | "timeout") => Some(120.0),
        (_, "missing_endpoint" | "schema_mismatch") => Some(300.0),
        _ => Some(60.0),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
pub async fn reset_for_tests() {
    let mut store = GLOBAL_STORE.lock().await;
    *store = GlobalStore::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_provider_failure_and_recovery() {
        let mut registry = ApiIssueRegistry::default();
        registry.observe_failure(IssueObservation {
            api: "provider".to_string(),
            provider: Some("nvidia".to_string()),
            endpoint: provider_endpoint("nvidia", "moonshot"),
            category: "quota".to_string(),
            severity: "warning".to_string(),
            summary: "nvidia provider failure while serving project_solver/planner".to_string(),
            error: r#"nvidia upstream error: {"status":429}"#.to_string(),
            workflow: Some("project_solver".to_string()),
            role: Some("planner".to_string()),
            retry_ttl_seconds: Some(120.0),
            context: json!({}),
        });
        assert_eq!(registry.summary.active, 1);
        assert_eq!(registry.summary.mitigated, 1);

        registry.observe_recovery(
            "provider",
            Some("nvidia"),
            &provider_endpoint("nvidia", "moonshot"),
            "recovered",
        );
        assert_eq!(registry.summary.active, 0);
        assert_eq!(registry.summary.resolved, 1);
    }

    #[test]
    fn recovery_without_matching_issue_is_noop() {
        let mut registry = ApiIssueRegistry::default();
        let changed = registry.observe_recovery(
            "provider",
            Some("nvidia"),
            &provider_endpoint("nvidia", "moonshot"),
            "recovered",
        );
        assert!(!changed);
        assert_eq!(registry.version, 0);
        assert_eq!(registry.summary.active, 0);
    }

    #[test]
    fn records_generic_api_failure_and_recovery() {
        let mut registry = ApiIssueRegistry::default();
        registry.observe_failure(IssueObservation {
            api: "octobot".to_string(),
            provider: None,
            endpoint: api_endpoint("GET", "/api/portfolio"),
            category: classify_api_issue(Some(502), "bad gateway"),
            severity: severity_for_api_issue(Some(502), "upstream").to_string(),
            summary: "OctoBot API failure while calling portfolio".to_string(),
            error: "bad gateway".to_string(),
            workflow: Some("trading".to_string()),
            role: Some("portfolio".to_string()),
            retry_ttl_seconds: retry_ttl_for_api_issue(Some(502), "upstream"),
            context: json!({}),
        });
        assert_eq!(registry.summary.active, 1);
        assert_eq!(
            registry.issues.values().next().unwrap().category.as_str(),
            "upstream"
        );

        let changed = registry.observe_recovery(
            "octobot",
            None,
            &api_endpoint("GET", "/api/portfolio"),
            "recovered",
        );
        assert!(changed);
        assert_eq!(registry.summary.active, 0);
        assert_eq!(registry.summary.resolved, 1);
    }

    #[test]
    fn prometheus_output_contains_issue_counts() {
        let mut registry = ApiIssueRegistry::default();
        registry.observe_failure(IssueObservation {
            api: "gail".to_string(),
            provider: None,
            endpoint: "orchestration".to_string(),
            category: "orchestration_exhausted".to_string(),
            severity: "critical".to_string(),
            summary: "Gail orchestration exhausted all candidates".to_string(),
            error: "no candidates".to_string(),
            workflow: Some("project_solver".to_string()),
            role: Some("planner".to_string()),
            retry_ttl_seconds: Some(60.0),
            context: json!({}),
        });
        let rendered = render_prometheus_metrics(&registry);
        assert!(rendered.contains("gail_api_issues_active 1"));
        assert!(rendered.contains("gail_api_issue_occurrences_total"));
    }
}
