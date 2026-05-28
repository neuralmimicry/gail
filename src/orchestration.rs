use std::{
    collections::{HashMap, HashSet},
    env,
    sync::Arc,
    time::Duration,
};

use axum::http::{HeaderMap, StatusCode, header::AUTHORIZATION};
use reqwest::Client;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit, Semaphore, oneshot},
    task::JoinSet,
    time::{Instant, sleep},
};
use tracing::info;
use uuid::Uuid;

use crate::{
    aarnn_bridge::{AarnnMirrorClient, AarnnMirrorExchange},
    adaptive_schema, aer, api_issues,
    config::{ApiTokenConfig, AuditLoggingConfig, GailConfig, ProviderProfile},
    errors::{GailError, Result, message_indicates_quota},
    hardware::{detect_hardware, log_hardware_profile},
    llm_ledger::{LlmLedger, LlmLedgerRecord},
    metrics::{HealthBucket, LocalUsageTelemetry, MetricsStore},
    models::{
        AarnnMirrorDirection, AerDecodeRequest, AerDecodeResponse, AerEncodeRequest,
        AerEncodeResponse, AuthContext, CandidateInvocationSummary, CandidateSummary,
        CompletionRequest, CompletionResponse, CompletionTrace, HealthResponse,
        NeuromorphicAnalyzeRequest, NeuromorphicPredictRequest, NeuromorphicPredictResponse,
        ProviderCompletionRequest, SelectionMode, SpecialistAnalysisResponse,
        TranscriptionResponse,
    },
    nmc_telemetry::{NmcAgentSignal, NmcTelemetryClient},
    providers::{
        ProviderHealth, ProviderInvocationResponse, TranscriptionInput, build_adapter,
        normalize_provider_type, provider_request_from_profile,
    },
    routing::{default_routing_profiles, resolve_routing_profiles_path},
    specialists::{
        SpecialistEngine, analyze_specialist_engines, build_specialist_engines,
        specialist_engine_summaries,
    },
    trading::{TradingBridge, TradingBridgeHandle},
};

const PROVIDER_HEALTH_TIMEOUT_SECONDS: u64 = 4;

#[derive(Clone)]
pub struct GailService {
    inner: Arc<GailServiceInner>,
}

struct GailServiceInner {
    config: GailConfig,
    client: Client,
    metrics: MetricsStore,
    llm_ledger: Option<LlmLedger>,
    specialists: Vec<SpecialistEngine>,
    aarnn_bridge: Option<AarnnMirrorClient>,
    nmc_telemetry: Option<NmcTelemetryClient>,
    trading_bridge: Option<TradingBridge>,
    _trading_bridge_handle: Option<TradingBridgeHandle>,
    load_tracker: Arc<Mutex<LoadTracker>>,
    interactive_pool: Arc<Semaphore>,
    solver_pool: Arc<Semaphore>,
}

#[derive(Clone, Debug)]
struct ProviderCandidate {
    profile: ProviderProfile,
    source: String,
    provider_type: String,
    configured_model: String,
    preferred: bool,
    weight: f64,
    specialties: HashSet<String>,
    roles: HashSet<String>,
    host_group: Option<String>,
    priority_bias: f64,
    usage_penalty_decay_seconds: f64,
    max_concurrent_requests: Option<usize>,
    resource_cost_cpu: f64,
    resource_cost_ram_mb: u64,
    resource_cost_vram_mb: u64,
    host_cpu_budget: Option<f64>,
    host_ram_budget_mb: Option<u64>,
    host_vram_budget_mb: Option<u64>,
    nmc_agent_id: Option<String>,
    nmc_host: Option<String>,
}

#[derive(Debug)]
struct InvocationResult {
    candidate: ProviderCandidate,
    response: Option<ProviderInvocationResponse>,
    error: Option<String>,
    latency_ms: Option<u64>,
    quality: f64,
    score: f64,
}

#[derive(Clone, Debug)]
struct RankedCandidate {
    candidate: ProviderCandidate,
    score: f64,
    health_ok: bool,
    health_mode: Option<String>,
}

#[derive(Default)]
struct LoadTracker {
    candidate_in_flight: HashMap<String, usize>,
    host_usage: HashMap<String, HostLoad>,
}

#[derive(Clone, Debug, Default)]
struct HostLoad {
    requests: usize,
    cpu: f64,
    ram_mb: u64,
    vram_mb: u64,
}

#[derive(Clone, Debug, Default)]
struct CandidateLoadSnapshot {
    candidate_limit_ratio: f64,
    candidate_limit_reached: bool,
    host_budget_ratio: f64,
    host_budget_reached: bool,
}

#[derive(Clone, Debug)]
struct LoadReservation {
    candidate_id: String,
    host_group: Option<String>,
    resource_cost_cpu: f64,
    resource_cost_ram_mb: u64,
    resource_cost_vram_mb: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkloadClass {
    Interactive,
    Solver,
}

impl WorkloadClass {
    fn label(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Solver => "solver",
        }
    }
}

impl GailService {
    pub async fn new(config: GailConfig) -> Result<Self> {
        adaptive_schema::configure_persistence(config.storage.adaptive_schema_path.clone()).await;
        api_issues::configure_persistence(
            config.storage.api_issues_path.clone(),
            config.storage.postgres_dsn.clone(),
        )
        .await;
        let client = Client::builder()
            .use_rustls_tls()
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Duration::from_secs(30))
            .user_agent(format!("gail/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        let hardware = detect_hardware().await;
        log_hardware_profile("api_service", &hardware);
        let llm_ledger = LlmLedger::from_config(&config).await;
        let metrics = MetricsStore::new(config.storage.metrics_path.clone()).await?;
        let specialists = build_specialist_engines(&config, client.clone());
        let aarnn_bridge = AarnnMirrorClient::from_config(&config, client.clone(), &specialists);
        let nmc_telemetry = NmcTelemetryClient::from_config(&config, client.clone());
        let load_tracker = Arc::new(Mutex::new(LoadTracker::default()));
        let suggested_interactive_pool = suggested_pool_size(
            hardware.cpu_cores,
            config.orchestration.interactive_pool_max_in_flight,
            2,
        );
        let suggested_solver_pool = suggested_pool_size(
            hardware.cpu_cores,
            config.orchestration.solver_pool_max_in_flight,
            3,
        );
        let interactive_pool = Arc::new(Semaphore::new(
            env_int_any(
                &[
                    "GAIL_INTERACTIVE_POOL_MAX_IN_FLIGHT",
                    "REFINER_AI_INTERACTIVE_POOL_MAX_IN_FLIGHT",
                ],
                suggested_interactive_pool as u64,
            )
            .max(1) as usize,
        ));
        let solver_pool = Arc::new(Semaphore::new(
            env_int_any(
                &[
                    "GAIL_SOLVER_POOL_MAX_IN_FLIGHT",
                    "REFINER_AI_SOLVER_POOL_MAX_IN_FLIGHT",
                ],
                suggested_solver_pool as u64,
            )
            .max(1) as usize,
        ));
        tracing::info!(
            interactive_pool_size = interactive_pool.available_permits(),
            solver_pool_size = solver_pool.available_permits(),
            "configured workload pool capacities"
        );

        // Construct a preliminary service (without trading) to pass into the trading bridge.
        let preliminary = Self {
            inner: Arc::new(GailServiceInner {
                config: config.clone(),
                client: client.clone(),
                metrics: metrics.clone(),
                llm_ledger: llm_ledger.clone(),
                specialists: specialists.clone(),
                aarnn_bridge: aarnn_bridge.clone(),
                nmc_telemetry: nmc_telemetry.clone(),
                trading_bridge: None,
                _trading_bridge_handle: None,
                load_tracker: load_tracker.clone(),
                interactive_pool: interactive_pool.clone(),
                solver_pool: solver_pool.clone(),
            }),
        };

        // Start trading bridge if configured.
        let (trading_bridge, trading_bridge_handle) = if config.trading.is_viable() {
            tracing::info!("trading: bridge is enabled — starting background loop");
            let trading_cfg = config.trading.clone();
            let (bridge, handle) = TradingBridge::start(trading_cfg, preliminary).await;
            (Some(bridge), Some(handle))
        } else {
            (None, None)
        };

        Ok(Self {
            inner: Arc::new(GailServiceInner {
                config,
                client,
                metrics,
                llm_ledger,
                specialists,
                aarnn_bridge,
                nmc_telemetry,
                trading_bridge,
                _trading_bridge_handle: trading_bridge_handle,
                load_tracker,
                interactive_pool,
                solver_pool,
            }),
        })
    }

    pub fn config(&self) -> &GailConfig {
        &self.inner.config
    }

    fn aarnn_bridge(&self) -> Option<&AarnnMirrorClient> {
        self.inner.aarnn_bridge.as_ref()
    }

    fn llm_ledger(&self) -> Option<&LlmLedger> {
        self.inner.llm_ledger.as_ref()
    }

    fn nmc_telemetry(&self) -> Option<&NmcTelemetryClient> {
        self.inner.nmc_telemetry.as_ref()
    }

    fn audit_logging(&self) -> &AuditLoggingConfig {
        &self.inner.config.audit_logging
    }

    fn audit_max_chars(&self) -> usize {
        self.audit_logging().max_chars.max(1)
    }

    fn truncate_audit_text(&self, value: &str) -> String {
        value.chars().take(self.audit_max_chars()).collect()
    }

    fn optional_audit_text(&self, value: Option<&str>) -> Option<String> {
        value
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| self.truncate_audit_text(item))
    }

    fn json_for_audit<T: Serialize>(&self, value: &T) -> String {
        match serde_json::to_string(value) {
            Ok(serialized) => self.truncate_audit_text(serialized.as_str()),
            Err(error) => self.truncate_audit_text(format!("<<serialize-error:{error}>>").as_str()),
        }
    }

    fn log_llm_audit_record(&self, record: &LlmLedgerRecord) {
        let audit = self.audit_logging();
        if !audit.enabled {
            return;
        }
        let prompt_text = if audit.log_llm_prompts {
            self.optional_audit_text(Some(record.prompt_text.as_str()))
        } else {
            None
        };
        let response_text = if audit.log_llm_responses {
            self.optional_audit_text(record.response_text.as_deref())
        } else {
            None
        };
        let system_prompt = if audit.log_llm_prompts {
            self.optional_audit_text(record.system_prompt.as_deref())
        } else {
            None
        };
        info!(
            audit_stream = "llm",
            request_id = %record.request_id,
            conversation_id = %record.conversation_id,
            workflow = %record.workflow,
            role = %record.role,
            status = %record.status,
            request_category = ?record.request_category,
            provider_requested = ?record.provider_requested,
            model_requested = ?record.model_requested,
            provider_resolved = ?record.provider_resolved,
            model_resolved = ?record.model_resolved,
            latency_ms = ?record.latency_ms,
            error_text = ?record.error_text,
            system_prompt = ?system_prompt,
            prompt_text = ?prompt_text,
            response_text = ?response_text,
            "GAIL_AUDIT_LLM_INTERACTION"
        );
    }

    fn log_aer_encode_audit(
        &self,
        ts_us: u64,
        base_addr: u32,
        request_events: Option<&[aer::AerEvent]>,
        request_spikes: Option<&[u8]>,
        encoded_events: &[aer::AerEvent],
        payload_hex: &str,
    ) {
        let audit = self.audit_logging();
        if !(audit.enabled && audit.log_aer_payloads) {
            return;
        }
        let request_events_json = request_events.map(|items| self.json_for_audit(&items));
        let request_spikes_json = request_spikes.map(|items| self.json_for_audit(&items));
        let encoded_events_json = self.json_for_audit(&encoded_events);
        info!(
            audit_stream = "aer",
            direction = "encode",
            ts_us,
            base_addr,
            request_events = ?request_events_json,
            request_spikes = ?request_spikes_json,
            encoded_events = %encoded_events_json,
            payload_hex = %self.truncate_audit_text(payload_hex),
            payload_bytes = payload_hex.len() / 2,
            event_count = encoded_events.len(),
            "GAIL_AUDIT_AER_ENCODE"
        );
    }

    fn log_aer_decode_audit(
        &self,
        payload_hex: &str,
        base_addr: Option<u32>,
        length: Option<usize>,
        events: &[aer::AerEvent],
        spikes: &[u8],
    ) {
        let audit = self.audit_logging();
        if !(audit.enabled && audit.log_aer_payloads) {
            return;
        }
        let events_json = self.json_for_audit(&events);
        let spikes_json = self.json_for_audit(&spikes);
        info!(
            audit_stream = "aer",
            direction = "decode",
            payload_hex = %self.truncate_audit_text(payload_hex),
            payload_bytes = payload_hex.len() / 2,
            base_addr = ?base_addr,
            length = ?length,
            events = %events_json,
            spikes = %spikes_json,
            event_count = events.len(),
            active_spikes = spikes.iter().filter(|value| **value > 0).count(),
            "GAIL_AUDIT_AER_DECODE"
        );
    }

    pub fn trading_bridge(&self) -> Option<&TradingBridge> {
        self.inner.trading_bridge.as_ref()
    }

    pub fn authorize(&self, headers: &HeaderMap, required_scope: &str) -> Result<AuthContext> {
        let Some(token_config) = self.matching_token(headers, required_scope) else {
            return Err(GailError::unauthorized());
        };
        Ok(AuthContext {
            client_id: Some(token_config.client_id.clone()),
        })
    }

    pub fn can_access_health_unauthenticated(&self) -> bool {
        self.inner.config.security.allow_unauthenticated_health
    }

    pub fn can_access_metrics_unauthenticated(&self) -> bool {
        self.inner.config.security.allow_unauthenticated_metrics
    }

    pub async fn health(&self) -> HealthResponse {
        HealthResponse {
            ok: true,
            service: "gail".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub async fn provider_prometheus_metrics(&self) -> String {
        self.inner.metrics.prometheus_metrics().await
    }

    pub async fn direct_complete(
        &self,
        request: ProviderCompletionRequest,
    ) -> Result<CompletionResponse> {
        let mut effective_request = request.clone();
        if effective_request.workflow.is_none() {
            effective_request.workflow = Some("direct".to_string());
        }
        if effective_request.role.is_none() {
            effective_request.role = Some("assistant".to_string());
        }
        if effective_request.min_model_size_b.is_none() {
            effective_request.min_model_size_b = self.model_floor_b(WorkloadClass::Interactive);
        }
        if effective_request.strict_no_downgrade.is_none() {
            effective_request.strict_no_downgrade = Some(self.strict_no_downgrade());
        }
        let request_id = Uuid::new_v4().to_string();
        let prompt_text = flatten_prompt_text(
            &effective_request.messages,
            effective_request.system.as_deref(),
        );
        let expected_json = expected_json(
            &effective_request.messages,
            effective_request.system.as_deref(),
        );
        let mut profile = ProviderProfile::default();
        profile.name = effective_request.provider.clone();
        profile.provider_type = effective_request.provider.clone();
        profile.model = effective_request.model.clone();
        profile.api_key = effective_request.api_key.clone();
        profile.access_token = effective_request.access_token.clone();
        profile.base_url = effective_request.base_url.clone();
        profile.source = Some("request_direct".to_string());
        let candidate = ProviderCandidate::from_profile(profile.clone());
        let workload_permit = match self
            .acquire_workload_permit(WorkloadClass::Interactive)
            .await
        {
            Some(permit) => permit,
            None => {
                return Err(GailError::upstream(
                    "gail",
                    Some(StatusCode::SERVICE_UNAVAILABLE),
                    format!(
                        "interactive workload pool is saturated; retry after {}ms",
                        self.workload_pool_wait_timeout_ms()
                    ),
                ));
            }
        };
        let mirror_input = self
            .spawn_aarnn_mirror(self.build_aarnn_exchange(
                request_id.as_str(),
                request_id.as_str(),
                "direct",
                "assistant",
                AarnnMirrorDirection::Input,
                Some(effective_request.provider.as_str()),
                effective_request.model.as_deref(),
                effective_request.request_category.as_deref(),
                effective_request.system.as_deref(),
                None,
                prompt_text.as_str(),
                &effective_request.messages,
            ))
            .await;
        let adapter = build_adapter(self.inner.client.clone(), &profile)?;
        let response = match adapter.complete(&effective_request).await {
            Ok(response) => response,
            Err(error) => {
                drop(workload_permit);
                let category = runtime_failure_health_bucket(Some(&error.to_string()), None)
                    .mode
                    .unwrap_or_else(|| "runtime_error".to_string());
                api_issues::observe_provider_failure(
                    candidate.provider_type.as_str(),
                    candidate.configured_model.as_str(),
                    "direct",
                    "assistant",
                    category.as_str(),
                    "warning",
                    &error.to_string(),
                    Some(self.health_ttl_seconds()),
                )
                .await;
                self.record_llm_interaction(LlmLedgerRecord {
                    request_id: request_id.clone(),
                    conversation_id: request_id.clone(),
                    workflow: "direct".to_string(),
                    role: "assistant".to_string(),
                    provider_requested: Some(effective_request.provider.clone()),
                    model_requested: effective_request.model.clone(),
                    provider_resolved: None,
                    model_resolved: None,
                    request_category: effective_request.request_category.clone(),
                    system_prompt: effective_request.system.clone(),
                    prompt_text: prompt_text.clone(),
                    response_text: None,
                    message_roles: effective_request
                        .messages
                        .iter()
                        .map(|message| message.role.clone())
                        .collect(),
                    status: "error".to_string(),
                    error_text: Some(error.to_string()),
                    latency_ms: None,
                    usage: None,
                    raw: None,
                    metadata: Some(json!({
                        "source": "direct_complete",
                    })),
                    created_ts: current_ts(),
                })
                .await;
                return Err(error);
            }
        };
        drop(workload_permit);
        api_issues::observe_provider_recovery(
            candidate.provider_type.as_str(),
            candidate.configured_model.as_str(),
        )
        .await;
        let quality = quality_score(response.text.as_str(), expected_json);
        let mirror_output = self
            .run_aarnn_output_mirror(
                request_id.as_str(),
                request_id.as_str(),
                "direct",
                "assistant",
                Some(response.provider.as_str()),
                Some(response.model.as_str()),
                effective_request.request_category.as_deref(),
                effective_request.system.as_deref(),
                Some(prompt_text.as_str()),
                response.text.as_str(),
                &effective_request.messages,
            )
            .await;
        let mirror_input = self.await_aarnn_mirror_task(mirror_input).await;
        let mut text = response.text.clone();
        let mut provider = response.provider.clone();
        let mut model = response.model.clone();
        let mut latency_ms = response.latency_ms;
        let mut usage = response.usage.clone();
        let mut raw = response.raw.clone();
        let mut final_source = "llm".to_string();
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref())
            && bridge.should_promote_candidate(output_trace, response.text.as_str())
            && let Some(reply_text) = bridge.promoted_reply(output_trace)
        {
            text = reply_text;
            provider = "aarnn".to_string();
            model = bridge.response_model().to_string();
            latency_ms = latency_ms.saturating_add(output_trace.latency_ms);
            usage = None;
            raw = Some(json!({
                "selected_source": "aarnn",
                "aarnn_candidate": output_trace.candidate.clone(),
                "llm_provider": response.provider,
                "llm_model": response.model,
                "llm_raw": response.raw,
            }));
            final_source = "aarnn".to_string();
        }
        let trace = if mirror_input.is_some() || mirror_output.is_some() {
            Some(CompletionTrace {
                workflow: "direct".to_string(),
                role: "assistant".to_string(),
                task_tags: vec!["direct".to_string()],
                selection_mode: SelectionMode::Fastest,
                returned_early: false,
                early_success_enabled: false,
                early_success_settle_seconds: 0.0,
                selected: candidate.summary(Some(response.model.as_str())),
                candidates: vec![CandidateInvocationSummary {
                    summary: candidate.summary(Some(response.model.as_str())),
                    latency_ms: Some(response.latency_ms),
                    quality,
                    score: quality,
                    status: "ok".to_string(),
                    error: None,
                }],
                metrics_store_path: self.inner.metrics.path(),
                specialist_engines: None,
                final_source,
                final_provider: provider.clone(),
                final_model: model.clone(),
                aarnn_mirroring: self
                    .aarnn_bridge()
                    .map(|bridge| bridge.build_trace(mirror_input.clone(), mirror_output.clone())),
            })
        } else {
            None
        };
        let completion_response = CompletionResponse {
            request_id,
            text,
            provider,
            model,
            latency_ms,
            usage,
            trace,
            raw,
        };
        self.record_llm_interaction(LlmLedgerRecord {
            request_id: completion_response.request_id.clone(),
            conversation_id: completion_response.request_id.clone(),
            workflow: "direct".to_string(),
            role: "assistant".to_string(),
            provider_requested: Some(effective_request.provider),
            model_requested: effective_request.model,
            provider_resolved: Some(completion_response.provider.clone()),
            model_resolved: Some(completion_response.model.clone()),
            request_category: effective_request.request_category,
            system_prompt: effective_request.system,
            prompt_text,
            response_text: Some(completion_response.text.clone()),
            message_roles: effective_request
                .messages
                .iter()
                .map(|message| message.role.clone())
                .collect(),
            status: "ok".to_string(),
            error_text: None,
            latency_ms: Some(completion_response.latency_ms),
            usage: completion_response
                .usage
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            raw: completion_response.raw.clone(),
            metadata: Some(json!({
                "source": "direct_complete",
                "final_source": completion_response
                    .trace
                    .as_ref()
                    .map(|trace| trace.final_source.clone()),
            })),
            created_ts: current_ts(),
        })
        .await;
        Ok(completion_response)
    }

    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let request_id = Uuid::new_v4().to_string();
        let workflow = normalize_key(request.workflow.as_deref().unwrap_or("general"), "general");
        let role = normalize_key(request.role.as_deref().unwrap_or("general"), "general");
        let workload_class = classify_workload(workflow.as_str(), role.as_str());
        let model_floor_b = self.model_floor_b(workload_class);
        let strict_no_downgrade = self.strict_no_downgrade();
        let selection_mode = request
            .selection_mode
            .clone()
            .unwrap_or_else(|| self.selection_mode());
        let include_configured = request
            .include_configured
            .unwrap_or_else(|| self.include_configured_candidates());
        let max_candidates = request
            .max_candidates
            .unwrap_or_else(|| self.max_parallel_candidates());
        let early_success_enabled = self.early_success_enabled(&workflow, &role, &selection_mode);
        let early_success_settle_seconds =
            self.early_success_settle_seconds(&workflow, &role, &selection_mode);
        let early_success_min_quality = self.early_success_min_quality();

        let mut provider_request = ProviderCompletionRequest {
            provider: request
                .preferred_provider
                .clone()
                .unwrap_or_else(|| "openai".to_string()),
            model: request.preferred_model.clone(),
            api_key: request.preferred_api_key.clone(),
            access_token: request.preferred_access_token.clone(),
            base_url: request.base_url.clone(),
            messages: request.messages.clone(),
            system: request.system.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            timeout_seconds: request.timeout_seconds,
            reasoning_effort: request.reasoning_effort.clone(),
            request_category: request.request_category.clone(),
            workflow: Some(workflow.clone()),
            role: Some(role.clone()),
            min_model_size_b: model_floor_b,
            strict_no_downgrade: Some(strict_no_downgrade),
        };

        let prompt_text = flatten_prompt_text(
            &provider_request.messages,
            provider_request.system.as_deref(),
        );
        let mut task_tags = workflow_tags(&workflow, &role, &prompt_text);
        if let Some(category) = request
            .request_category
            .as_deref()
            .or(provider_request.request_category.as_deref())
        {
            for tag in category.split(|ch: char| !ch.is_ascii_alphanumeric()) {
                let normalized = normalize_key(tag, "");
                if !normalized.is_empty() {
                    task_tags.insert(normalized);
                }
            }
        }
        let mut specialist_meta = None;
        if !self.inner.specialists.is_empty()
            && (task_tags.contains("neuromorphic") || self.always_route_specialists())
        {
            let analyze_request = NeuromorphicAnalyzeRequest {
                text: prompt_text.clone(),
                workflow: Some(workflow.clone()),
                role: Some(role.clone()),
            };
            let analysis =
                analyze_specialist_engines(&self.inner.specialists, &analyze_request).await;
            if analysis.relevant {
                task_tags.insert("neuromorphic".to_string());
                task_tags.insert("aer".to_string());
                task_tags.extend(analysis.combined_specialties.iter().cloned());
                if !analysis.context.is_empty() {
                    provider_request.system = Some(match provider_request.system {
                        Some(system) if !system.trim().is_empty() => {
                            format!("{system}\n\n{}", analysis.context)
                        }
                        _ => analysis.context.clone(),
                    });
                }
                info!(
                    workflow = %workflow,
                    role = %role,
                    engine_count = analysis.engine_count,
                    "attached neuromorphic specialist context"
                );
            }
            specialist_meta = Some(analysis);
        }
        let mirrored_prompt_text = flatten_prompt_text(
            &provider_request.messages,
            provider_request.system.as_deref(),
        );
        let mirror_input = self
            .spawn_aarnn_mirror(self.build_aarnn_exchange(
                request_id.as_str(),
                request_id.as_str(),
                workflow.as_str(),
                role.as_str(),
                AarnnMirrorDirection::Input,
                Some(provider_request.provider.as_str()),
                provider_request.model.as_deref(),
                provider_request.request_category.as_deref(),
                provider_request.system.as_deref(),
                None,
                mirrored_prompt_text.as_str(),
                &provider_request.messages,
            ))
            .await;

        let mut candidates = self.build_candidates(&request, include_configured);
        if let Some(min_model_size_b) = model_floor_b.filter(|value| *value > 0.0) {
            let before = candidates.len();
            candidates.retain(|candidate| candidate_meets_model_floor(candidate, min_model_size_b));
            let removed = before.saturating_sub(candidates.len());
            if removed > 0 {
                info!(
                    workflow = %workflow,
                    role = %role,
                    removed,
                    min_model_size_b,
                    "filtered candidates below configured model floor"
                );
            }
        }
        if candidates.is_empty() {
            return Err(GailError::bad_request(
                if let Some(min_model_size_b) = model_floor_b.filter(|value| *value > 0.0) {
                    format!(
                        "no LLM providers are configured or supplied for orchestration after enforcing model floor >= {min_model_size_b:.2}b"
                    )
                } else {
                    "no LLM providers are configured or supplied for orchestration".to_string()
                },
            ));
        }
        let mut ranked = Vec::new();
        let mut rank_join_set = JoinSet::new();
        for candidate in candidates.drain(..) {
            let service = self.clone();
            let workflow_clone = workflow.clone();
            let role_clone = role.clone();
            let task_tags_clone = task_tags.clone();
            rank_join_set.spawn(async move {
                service
                    .rank_candidate(candidate, &workflow_clone, &role_clone, &task_tags_clone)
                    .await
            });
        }
        while let Some(result) = rank_join_set.join_next().await {
            match result {
                Ok(item) => ranked.push(item),
                Err(error) => {
                    tracing::warn!(error = %error, "candidate ranking task failed");
                }
            }
        }
        ranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let expected_json = expected_json(
            &provider_request.messages,
            provider_request.system.as_deref(),
        ) || task_tags_expect_json(&task_tags);
        let timeout_cap =
            self.candidate_timeout_cap(&workflow, &role, expected_json, &task_tags, &prompt_text);
        let wave_size = max_candidates.max(1);
        let mut results = Vec::new();
        let mut attempted_candidate_ids = HashSet::new();
        let mut throttled_provider_types = HashSet::new();
        let mut returned_early = false;
        let mut wave_index = 0usize;
        loop {
            let unattempted = ranked
                .iter()
                .filter(|item| {
                    !attempted_candidate_ids.contains(&item.candidate.candidate_id())
                        && !throttled_provider_types.contains(&item.candidate.provider_type)
                })
                .cloned()
                .collect::<Vec<_>>();
            let remaining = unattempted
                .iter()
                .filter(|item| !ranked_candidate_is_in_provider_backoff(item))
                .cloned()
                .collect::<Vec<_>>();
            let mut forced_selected: Option<Vec<ProviderCandidate>> = None;
            if remaining.is_empty() && !unattempted.is_empty() {
                if results.is_empty()
                    && should_probe_transient_backoff_candidates(
                        &workflow,
                        &role,
                        expected_json,
                        &task_tags,
                        &prompt_text,
                    )
                {
                    let transient_backoff = unattempted
                        .iter()
                        .filter(|item| ranked_candidate_is_transient_backoff(item))
                        .cloned()
                        .collect::<Vec<_>>();
                    if !transient_backoff.is_empty() {
                        info!(
                            workflow = %workflow,
                            role = %role,
                            backoff_candidates = %preview_labels(
                                transient_backoff
                                    .iter()
                                    .map(|item| item.candidate.candidate_id())
                                    .collect::<Vec<_>>(),
                                6
                            ),
                            "all providers are in transient adaptive backoff; forcing a probe attempt"
                        );
                        forced_selected = Some(select_ranked_candidates(transient_backoff, 1));
                    }
                }
                if results.is_empty() {
                    let no_forced_probe =
                        forced_selected.as_ref().map(|items| items.is_empty()) != Some(false);
                    if remaining.is_empty() && no_forced_probe {
                        let message = "all suitable providers are currently in adaptive backoff; retry after the recorded mitigation window".to_string();
                        api_issues::observe_orchestration_failure(
                            &workflow,
                            &role,
                            &message,
                            json!({
                                "attempted_candidate_count": attempted_candidate_ids.len(),
                                "throttled_provider_types": sorted_strings(throttled_provider_types.clone()),
                                "backoff_candidates": unattempted
                                    .iter()
                                    .map(|item| item.candidate.candidate_id())
                                    .collect::<Vec<_>>(),
                            }),
                        )
                        .await;
                        if should_return_degraded_fallback(
                            &request,
                            include_configured,
                            &workflow,
                            &role,
                            expected_json,
                            &task_tags,
                            &prompt_text,
                        ) {
                            info!(
                                workflow = %workflow,
                                role = %role,
                                "returning Gail degraded safety fallback because every provider is in adaptive backoff"
                            );
                            let degraded = self.degraded_completion_response(
                                request_id,
                                &workflow,
                                &role,
                                &task_tags,
                                &selection_mode,
                                returned_early,
                                early_success_enabled,
                                early_success_settle_seconds,
                                expected_json,
                                &prompt_text,
                                vec![message],
                                ranked_candidate_summaries(&unattempted),
                                specialist_meta.as_ref(),
                                attempted_candidate_ids.len(),
                                sorted_strings(throttled_provider_types.clone()),
                            );
                            self.record_completion_interaction(
                                &request,
                                &provider_request,
                                mirrored_prompt_text.as_str(),
                                workflow.as_str(),
                                role.as_str(),
                                &degraded,
                                "degraded",
                            )
                            .await;
                            return Ok(degraded);
                        }
                        return Err(GailError::upstream(
                            "gail",
                            Some(StatusCode::SERVICE_UNAVAILABLE),
                            message,
                        ));
                    }
                }
            }
            let selected =
                forced_selected.unwrap_or_else(|| select_ranked_candidates(remaining, wave_size));
            if selected.is_empty() {
                if results.is_empty() {
                    return Err(GailError::bad_request(
                        "no provider candidates were selected",
                    ));
                }
                break;
            }
            wave_index += 1;
            for candidate in &selected {
                attempted_candidate_ids.insert(candidate.candidate_id());
            }

            info!(
                workflow = %workflow,
                role = %role,
                fallback_wave = wave_index,
                timeout_cap_seconds = ?timeout_cap,
                candidates = %preview_labels(selected.iter().map(|item| item.label(None)).collect::<Vec<_>>(), 6),
                throttled_providers = %preview_labels(sorted_strings(throttled_provider_types.iter().cloned()), 6),
                tags = %preview_labels(task_tags.iter().cloned().collect::<Vec<_>>(), 8),
                "dispatching Gail orchestration"
            );

            let mut wave_results = if selected.len() == 1 {
                vec![
                    self.invoke_candidate(
                        selected[0].clone(),
                        provider_request.clone(),
                        expected_json,
                        timeout_cap,
                        workload_class,
                    )
                    .await,
                ]
            } else {
                self.invoke_candidates(
                    selected.clone(),
                    provider_request.clone(),
                    expected_json,
                    selection_mode.clone(),
                    early_success_enabled,
                    early_success_settle_seconds,
                    early_success_min_quality,
                    timeout_cap,
                    workload_class,
                )
                .await?
            };

            returned_early |= wave_results.len() < selected.len() && selected.len() > 1;
            let wave_has_success = wave_results.iter().any(|result| result.response.is_some());
            let backoff_providers = wave_results
                .iter()
                .filter(|result| result.response.is_none())
                .filter_map(|result| {
                    let error = result.error.as_deref()?;
                    if error_should_backoff_provider_family(&result.candidate, error) {
                        Some(result.candidate.provider_type.clone())
                    } else {
                        None
                    }
                })
                .collect::<HashSet<_>>();
            if !backoff_providers.is_empty() {
                throttled_provider_types.extend(backoff_providers.iter().cloned());
                info!(
                    workflow = %workflow,
                    role = %role,
                    fallback_wave = wave_index,
                    throttled_providers = %preview_labels(sorted_strings(backoff_providers.into_iter()), 6),
                    "provider family in runtime backoff; trying fallback candidates"
                );
            }
            results.append(&mut wave_results);
            if wave_has_success {
                break;
            }
        }

        let mut successful = Vec::new();
        let mut failures = Vec::new();
        for result in results.iter_mut() {
            let candidate_summary = result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str()));
            if let Some(response) = result.response.as_ref() {
                let latency_penalty = response.latency_ms as f64 / 5000.0;
                let metrics_bonus = self
                    .inner
                    .metrics
                    .score_bonus(candidate_summary.candidate_id.as_str(), &workflow, &role)
                    .await;
                result.score = result.quality - latency_penalty.min(1.25) + metrics_bonus;
                let telemetry = local_usage_telemetry(response);
                self.inner
                    .metrics
                    .record_result(
                        &candidate_summary,
                        &workflow,
                        &role,
                        true,
                        Some(response.latency_ms),
                        Some(telemetry),
                        result.quality,
                        None,
                    )
                    .await?;
                self.inner
                    .metrics
                    .record_health(
                        &candidate_summary,
                        HealthBucket {
                            ok: Some(true),
                            mode: Some("runtime_completion".to_string()),
                            checked_at: None,
                            latency_ms: Some(response.latency_ms),
                            message: Some("ok".to_string()),
                        },
                    )
                    .await?;
                api_issues::observe_provider_recovery(
                    candidate_summary.provider.as_str(),
                    candidate_summary.configured_model.as_str(),
                )
                .await;
                successful.push(candidate_summary);
            } else {
                let health_bucket =
                    runtime_failure_health_bucket(result.error.as_deref(), result.latency_ms);
                let category = health_bucket
                    .mode
                    .clone()
                    .unwrap_or_else(|| "runtime_error".to_string());
                self.inner
                    .metrics
                    .record_result(
                        &candidate_summary,
                        &workflow,
                        &role,
                        false,
                        result.latency_ms,
                        None,
                        -1.0,
                        result.error.as_deref(),
                    )
                    .await?;
                self.inner
                    .metrics
                    .record_health(&candidate_summary, health_bucket)
                    .await?;
                let failure_message = result
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string());
                api_issues::observe_provider_failure(
                    candidate_summary.provider.as_str(),
                    candidate_summary.configured_model.as_str(),
                    &workflow,
                    &role,
                    &category,
                    severity_for_issue_category(&category),
                    &failure_message,
                    Some(self.health_ttl_seconds()),
                )
                .await;
                failures.push(failure_message);
            }
        }

        let Some(chosen_index) = results
            .iter()
            .enumerate()
            .filter(|(_, result)| result.response.is_some())
            .max_by(|(_, left), (_, right)| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
        else {
            let message = failures
                .last()
                .cloned()
                .unwrap_or_else(|| "LLM orchestration returned no responses".to_string());
            api_issues::observe_orchestration_failure(
                &workflow,
                &role,
                &message,
                json!({
                    "failures": failures.clone(),
                    "attempted_candidate_count": attempted_candidate_ids.len(),
                    "throttled_provider_types": sorted_strings(throttled_provider_types.clone()),
                }),
            )
            .await;
            if should_return_degraded_fallback(
                &request,
                include_configured,
                &workflow,
                &role,
                expected_json,
                &task_tags,
                &prompt_text,
            ) {
                info!(
                    workflow = %workflow,
                    role = %role,
                    "returning Gail degraded safety fallback because every provider failed"
                );
                let degraded = self.degraded_completion_response(
                    request_id,
                    &workflow,
                    &role,
                    &task_tags,
                    &selection_mode,
                    returned_early,
                    early_success_enabled,
                    early_success_settle_seconds,
                    expected_json,
                    &prompt_text,
                    failures,
                    invocation_summaries_from_results(&results),
                    specialist_meta.as_ref(),
                    attempted_candidate_ids.len(),
                    sorted_strings(throttled_provider_types.clone()),
                );
                self.record_completion_interaction(
                    &request,
                    &provider_request,
                    mirrored_prompt_text.as_str(),
                    workflow.as_str(),
                    role.as_str(),
                    &degraded,
                    "degraded",
                )
                .await;
                return Ok(degraded);
            }
            return Err(GailError::upstream(
                "gail",
                orchestration_failure_status(message.as_str()),
                message,
            ));
        };

        let chosen = results.swap_remove(chosen_index);
        let chosen_response = chosen.response.expect("chosen successful result");
        let selected_summary = chosen
            .candidate
            .summary(Some(chosen_response.model.as_str()));
        let mirror_output = self
            .run_aarnn_output_mirror(
                request_id.as_str(),
                request_id.as_str(),
                workflow.as_str(),
                role.as_str(),
                Some(chosen_response.provider.as_str()),
                Some(chosen_response.model.as_str()),
                provider_request.request_category.as_deref(),
                provider_request.system.as_deref(),
                Some(mirrored_prompt_text.as_str()),
                chosen_response.text.as_str(),
                &provider_request.messages,
            )
            .await;
        let mirror_input = self.await_aarnn_mirror_task(mirror_input).await;
        let candidate_summaries = std::iter::once((
            selected_summary.clone(),
            chosen.latency_ms,
            chosen.quality,
            chosen.score,
            chosen.error.clone(),
            true,
        ))
        .chain(results.into_iter().map(|result| {
            let summary = result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str()));
            (
                summary,
                result.latency_ms,
                result.quality,
                result.score,
                result.error,
                result.response.is_some(),
            )
        }))
        .map(
            |(summary, latency_ms, quality, score, error, ok)| CandidateInvocationSummary {
                summary,
                latency_ms,
                quality,
                score,
                status: if ok { "ok" } else { "error" }.to_string(),
                error,
            },
        )
        .collect::<Vec<_>>();

        info!(
            workflow = %workflow,
            role = %role,
            provider = %chosen_response.provider,
            model = %chosen_response.model,
            returned_early,
            "selected Gail orchestration result"
        );

        let mut text = chosen_response.text.clone();
        let mut provider = chosen_response.provider.clone();
        let mut model = chosen_response.model.clone();
        let mut latency_ms = chosen_response.latency_ms;
        let mut usage = chosen_response.usage.clone();
        let mut raw = chosen_response.raw.clone();
        let mut final_source = "llm".to_string();
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref())
            && bridge.should_promote_candidate(output_trace, chosen_response.text.as_str())
            && let Some(reply_text) = bridge.promoted_reply(output_trace)
        {
            text = reply_text;
            provider = "aarnn".to_string();
            model = bridge.response_model().to_string();
            latency_ms = latency_ms.saturating_add(output_trace.latency_ms);
            usage = None;
            raw = Some(json!({
                "selected_source": "aarnn",
                "aarnn_candidate": output_trace.candidate.clone(),
                "llm_provider": chosen_response.provider,
                "llm_model": chosen_response.model,
                "llm_raw": chosen_response.raw,
            }));
            final_source = "aarnn".to_string();
        }

        let trace = CompletionTrace {
            workflow: workflow.clone(),
            role: role.clone(),
            task_tags: sorted_strings(task_tags),
            selection_mode: selection_mode.clone(),
            returned_early,
            early_success_enabled,
            early_success_settle_seconds,
            selected: selected_summary,
            candidates: candidate_summaries,
            metrics_store_path: self.inner.metrics.path(),
            specialist_engines: specialist_meta
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            final_source,
            final_provider: provider.clone(),
            final_model: model.clone(),
            aarnn_mirroring: self
                .aarnn_bridge()
                .map(|bridge| bridge.build_trace(mirror_input.clone(), mirror_output.clone())),
        };

        let completion_response = CompletionResponse {
            request_id,
            text,
            provider,
            model,
            latency_ms,
            usage,
            trace: Some(trace),
            raw,
        };
        self.record_completion_interaction(
            &request,
            &provider_request,
            mirrored_prompt_text.as_str(),
            workflow.as_str(),
            role.as_str(),
            &completion_response,
            "ok",
        )
        .await;
        Ok(completion_response)
    }

    #[allow(clippy::too_many_arguments)]
    fn degraded_completion_response(
        &self,
        request_id: String,
        workflow: &str,
        role: &str,
        task_tags: &HashSet<String>,
        selection_mode: &SelectionMode,
        returned_early: bool,
        early_success_enabled: bool,
        early_success_settle_seconds: f64,
        expected_json: bool,
        prompt_text: &str,
        failures: Vec<String>,
        mut candidate_summaries: Vec<CandidateInvocationSummary>,
        specialist_meta: Option<&SpecialistAnalysisResponse>,
        attempted_candidate_count: usize,
        throttled_provider_types: Vec<String>,
    ) -> CompletionResponse {
        let selected_summary = degraded_candidate_summary(role);
        candidate_summaries.insert(
            0,
            CandidateInvocationSummary {
                summary: selected_summary.clone(),
                latency_ms: Some(0),
                quality: 0.0,
                score: 0.0,
                status: "ok".to_string(),
                error: None,
            },
        );
        let text = degraded_fallback_text(expected_json, workflow, role, prompt_text, &failures);
        let trace = CompletionTrace {
            workflow: workflow.to_string(),
            role: role.to_string(),
            task_tags: sorted_strings(task_tags.clone()),
            selection_mode: selection_mode.clone(),
            returned_early,
            early_success_enabled,
            early_success_settle_seconds,
            selected: selected_summary,
            candidates: candidate_summaries,
            metrics_store_path: self.inner.metrics.path(),
            specialist_engines: specialist_meta.and_then(|value| serde_json::to_value(value).ok()),
            final_source: "degraded_policy".to_string(),
            final_provider: "gail".to_string(),
            final_model: "degraded_safety".to_string(),
            aarnn_mirroring: None,
        };
        CompletionResponse {
            request_id,
            text,
            provider: "gail".to_string(),
            model: "degraded_safety".to_string(),
            latency_ms: 0,
            usage: None,
            trace: Some(trace),
            raw: Some(json!({
                "selected_source": "degraded_policy",
                "reason": "all_provider_candidates_failed",
                "attempted_candidate_count": attempted_candidate_count,
                "throttled_provider_types": throttled_provider_types,
                "failures": failures,
                "safety_action": "hold_no_trade",
            })),
        }
    }

    pub async fn transcribe(
        &self,
        provider: String,
        model: Option<String>,
        api_key: Option<String>,
        access_token: Option<String>,
        base_url: Option<String>,
        input: TranscriptionInput,
    ) -> Result<TranscriptionResponse> {
        let profile = ProviderProfile {
            name: provider.clone(),
            provider_type: provider,
            model,
            api_key,
            access_token,
            base_url,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: 0.0,
            preferred: true,
            source: Some("request_transcribe".to_string()),
            ..ProviderProfile::default()
        };
        let adapter = build_adapter(self.inner.client.clone(), &profile)?;
        let response = adapter.transcribe(&input).await?;
        Ok(TranscriptionResponse {
            request_id: Uuid::new_v4().to_string(),
            text: response.text,
            provider: response.provider,
            model: response.model,
            latency_ms: response.latency_ms,
            usage: response.usage,
        })
    }

    pub async fn analyze_neuromorphic(
        &self,
        request: NeuromorphicAnalyzeRequest,
    ) -> Result<SpecialistAnalysisResponse> {
        Ok(analyze_specialist_engines(&self.inner.specialists, &request).await)
    }

    pub async fn predict_neuromorphic(
        &self,
        request: NeuromorphicPredictRequest,
    ) -> Result<NeuromorphicPredictResponse> {
        let engine = self.select_specialist(request.engine_name.as_deref())?;
        engine.predict_request(&request).await
    }

    pub fn encode_aer(&self, request: AerEncodeRequest) -> Result<AerEncodeResponse> {
        let ts_us = request.ts_us.unwrap_or(0);
        let request_events_snapshot = request.events.clone();
        let request_spikes_snapshot = request.spikes.clone();
        let events = if let Some(events) = request.events {
            events
        } else {
            aer::spikes_to_events(
                ts_us,
                request.base_addr,
                &request.spikes.unwrap_or_default(),
            )
        };
        let payload = aer::encode_events(&events);
        let payload_hex = aer::payload_hex(&payload);
        self.log_aer_encode_audit(
            ts_us,
            request.base_addr,
            request_events_snapshot.as_deref(),
            request_spikes_snapshot.as_deref(),
            events.as_slice(),
            payload_hex.as_str(),
        );
        Ok(AerEncodeResponse {
            payload_hex,
            event_count: events.len(),
        })
    }

    pub fn decode_aer(&self, request: AerDecodeRequest) -> Result<AerDecodeResponse> {
        let payload_hex_snapshot = request.payload_hex.clone();
        let base_addr_snapshot = request.base_addr;
        let length_snapshot = request.length;
        let payload = hex::decode(request.payload_hex)
            .map_err(|error| GailError::bad_request(error.to_string()))?;
        let events = aer::decode_events(&payload)?;
        let spikes = match (request.base_addr, request.length) {
            (Some(base_addr), Some(length)) => aer::decode_spikes(&payload, base_addr, length)?,
            (Some(base_addr), None) => aer::decode_spikes_auto(&payload, base_addr)?,
            (None, Some(length)) => {
                let base_addr = events.first().map(|event| event.addr).unwrap_or_default();
                aer::decode_spikes(&payload, base_addr, length)?
            }
            (None, None) => {
                let base_addr = events.first().map(|event| event.addr).unwrap_or_default();
                aer::decode_spikes_auto(&payload, base_addr)?
            }
        };
        self.log_aer_decode_audit(
            payload_hex_snapshot.as_str(),
            base_addr_snapshot,
            length_snapshot,
            events.as_slice(),
            spikes.as_slice(),
        );
        Ok(AerDecodeResponse { events, spikes })
    }

    pub async fn orchestration_status_value(
        &self,
        candidate_limit: usize,
        probe_engines: bool,
        probe_providers: bool,
    ) -> Value {
        let providers = self.provider_summaries(probe_providers).await;
        let engines = specialist_engine_summaries(
            &self.inner.config,
            self.inner.client.clone(),
            probe_engines,
        )
        .await;
        let metrics = self.inner.metrics.summary(candidate_limit.max(1)).await;
        let api_issues = api_issues::snapshot().await;
        let model_inventory = self.first_ollama_inventory().await;
        let routing_profiles_path = resolve_routing_profiles_path(None::<&std::path::Path>)
            .ok()
            .map(|path| path.display().to_string());
        let routing_profiles_version = default_routing_profiles().version;
        let aarnn_bridge = AarnnMirrorClient::status(&self.inner.config, &self.inner.specialists);
        let nmc_telemetry = if let Some(client) = self.nmc_telemetry() {
            serde_json::to_value(client.status().await).unwrap_or(Value::Null)
        } else {
            serde_json::to_value(NmcTelemetryClient::status_from_config(&self.inner.config))
                .unwrap_or(Value::Null)
        };
        json!({
            "enabled": self.inner.config.orchestration.enabled,
            "routing_profiles_path": routing_profiles_path,
            "routing_profiles_version": routing_profiles_version,
            "selection_mode": self.selection_mode(),
            "max_parallel_candidates": self.max_parallel_candidates(),
            "interactive_pool_max_in_flight": self.inner.config.orchestration.interactive_pool_max_in_flight,
            "solver_pool_max_in_flight": self.inner.config.orchestration.solver_pool_max_in_flight,
            "workload_pool_wait_timeout_ms": self.workload_pool_wait_timeout_ms(),
            "health_ttl_seconds": self.health_ttl_seconds(),
            "interactive_model_floor_b": self.model_floor_b(WorkloadClass::Interactive),
            "solver_model_floor_b": self.model_floor_b(WorkloadClass::Solver),
            "strict_no_downgrade": self.strict_no_downgrade(),
            "provider_count": providers.len(),
            "providers": providers,
            "engine_count": engines.len(),
            "engines": engines,
            "aarnn_bridge": aarnn_bridge,
            "nmc_telemetry": nmc_telemetry,
            "metrics": metrics,
            "api_issues": api_issues,
            "model_inventory": model_inventory,
        })
    }

    async fn provider_summaries(&self, probe_health: bool) -> Vec<Value> {
        let profiles = self.inner.config.providers.clone();
        let mut join_set = JoinSet::new();
        for (index, profile) in profiles.into_iter().enumerate() {
            let client = self.inner.client.clone();
            join_set.spawn(async move {
                let provider_type = normalize_provider_type(profile.provider_type.as_str());
                let health = if probe_health {
                    match build_adapter(client, &profile) {
                        Ok(adapter) => {
                            match adapter.health(Some(PROVIDER_HEALTH_TIMEOUT_SECONDS)).await {
                                Ok(status) => json!(status),
                                Err(error) => json!({"ok": false, "message": error.to_string()}),
                            }
                        }
                        Err(error) => json!({"ok": false, "message": error.to_string()}),
                    }
                } else {
                    Value::Null
                };
                (
                    index,
                    json!({
                        "name": profile.name,
                        "provider": provider_type,
                        "model": profile.model,
                        "source": profile.source,
                        "roles": profile.roles,
                        "specialties": profile.specialties,
                        "weight": profile.weight,
                        "preferred": profile.preferred,
                        "base_url": profile.base_url,
                        "host_group": profile.host_group,
                        "max_concurrent_requests": profile.max_concurrent_requests,
                        "resource_cost_cpu": profile.resource_cost_cpu,
                        "resource_cost_ram_mb": profile.resource_cost_ram_mb,
                        "resource_cost_vram_mb": profile.resource_cost_vram_mb,
                        "host_cpu_budget": profile.host_cpu_budget,
                        "host_ram_budget_mb": profile.host_ram_budget_mb,
                        "host_vram_budget_mb": profile.host_vram_budget_mb,
                        "nmc_agent_id": profile.nmc_agent_id,
                        "nmc_host": profile.nmc_host,
                        "health": health,
                    }),
                )
            });
        }
        let mut ordered = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(item) => ordered.push(item),
                Err(error) => {
                    tracing::warn!(error = %error, "provider summary task failed");
                }
            }
        }
        ordered.sort_by_key(|(index, _)| *index);
        ordered.into_iter().map(|(_, value)| value).collect()
    }

    async fn first_ollama_inventory(&self) -> Value {
        for profile in &self.inner.config.providers {
            if normalize_provider_type(profile.provider_type.as_str()) != "ollama" {
                continue;
            }
            if let Ok(adapter) = build_adapter(self.inner.client.clone(), profile)
                && let Some(inventory) = adapter.ollama_inventory(&self.inner.config).await
            {
                return serde_json::to_value(inventory).unwrap_or(Value::Null);
            }
        }
        Value::Null
    }

    async fn spawn_aarnn_mirror(
        &self,
        exchange: AarnnMirrorExchange,
    ) -> Option<oneshot::Receiver<crate::models::AarnnMirrorInvocationTrace>> {
        let bridge = self.inner.aarnn_bridge.clone()?;
        let should_mirror = match exchange.direction {
            AarnnMirrorDirection::Input => bridge.should_mirror_input(),
            AarnnMirrorDirection::Output => bridge.should_mirror_output(),
        };
        if !should_mirror {
            return None;
        }
        bridge.enqueue(exchange, true).await
    }

    async fn await_aarnn_mirror_task(
        &self,
        task: Option<oneshot::Receiver<crate::models::AarnnMirrorInvocationTrace>>,
    ) -> Option<crate::models::AarnnMirrorInvocationTrace> {
        let task = task?;
        let wait_timeout = self
            .aarnn_bridge()
            .map(|bridge| bridge.candidate_wait_timeout())
            .unwrap_or_else(|| Duration::from_millis(0));
        if wait_timeout.is_zero() {
            return None;
        }
        match tokio::time::timeout(wait_timeout, task).await {
            Ok(Ok(trace)) => Some(trace),
            Ok(Err(error)) => {
                tracing::warn!(error = %error, "AARNN mirror receiver dropped");
                None
            }
            Err(_) => None,
        }
    }

    async fn run_aarnn_output_mirror(
        &self,
        request_id: &str,
        conversation_id: &str,
        workflow: &str,
        role: &str,
        provider: Option<&str>,
        model: Option<&str>,
        request_category: Option<&str>,
        system: Option<&str>,
        prompt_text: Option<&str>,
        text: &str,
        messages: &[crate::models::ChatMessage],
    ) -> Option<crate::models::AarnnMirrorInvocationTrace> {
        self.await_aarnn_mirror_task(
            self.spawn_aarnn_mirror(self.build_aarnn_exchange(
                request_id,
                conversation_id,
                workflow,
                role,
                AarnnMirrorDirection::Output,
                provider,
                model,
                request_category,
                system,
                prompt_text,
                text,
                messages,
            ))
            .await,
        )
        .await
    }

    fn build_aarnn_exchange(
        &self,
        request_id: &str,
        conversation_id: &str,
        workflow: &str,
        role: &str,
        direction: AarnnMirrorDirection,
        provider: Option<&str>,
        model: Option<&str>,
        request_category: Option<&str>,
        system: Option<&str>,
        prompt_text: Option<&str>,
        text: &str,
        messages: &[crate::models::ChatMessage],
    ) -> AarnnMirrorExchange {
        AarnnMirrorExchange {
            request_id: request_id.to_string(),
            conversation_id: conversation_id.to_string(),
            workflow: workflow.to_string(),
            role: role.to_string(),
            direction,
            provider: provider.map(ToOwned::to_owned),
            model: model.map(ToOwned::to_owned),
            request_category: request_category.map(ToOwned::to_owned),
            system: system.map(ToOwned::to_owned),
            prompt_text: prompt_text.map(ToOwned::to_owned),
            text: text.to_string(),
            message_roles: messages
                .iter()
                .map(|message| message.role.clone())
                .collect(),
        }
    }

    async fn record_llm_interaction(&self, record: LlmLedgerRecord) {
        self.log_llm_audit_record(&record);
        if let Some(ledger) = self.llm_ledger() {
            ledger.record(record).await;
        }
    }

    async fn record_completion_interaction(
        &self,
        request: &CompletionRequest,
        provider_request: &ProviderCompletionRequest,
        prompt_text: &str,
        workflow: &str,
        role: &str,
        response: &CompletionResponse,
        status: &str,
    ) {
        let final_source = response
            .trace
            .as_ref()
            .map(|trace| trace.final_source.clone());
        self.record_llm_interaction(LlmLedgerRecord {
            request_id: response.request_id.clone(),
            conversation_id: response.request_id.clone(),
            workflow: workflow.to_string(),
            role: role.to_string(),
            provider_requested: request
                .preferred_provider
                .clone()
                .or_else(|| Some(provider_request.provider.clone())),
            model_requested: request
                .preferred_model
                .clone()
                .or_else(|| provider_request.model.clone()),
            provider_resolved: Some(response.provider.clone()),
            model_resolved: Some(response.model.clone()),
            request_category: provider_request
                .request_category
                .clone()
                .or_else(|| request.request_category.clone()),
            system_prompt: provider_request.system.clone(),
            prompt_text: prompt_text.to_string(),
            response_text: Some(response.text.clone()),
            message_roles: provider_request
                .messages
                .iter()
                .map(|message| message.role.clone())
                .collect(),
            status: status.to_string(),
            error_text: None,
            latency_ms: Some(response.latency_ms),
            usage: response
                .usage
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            raw: response.raw.clone(),
            metadata: Some(json!({
                "source": "orchestrated_complete",
                "selection_mode": response.trace.as_ref().map(|trace| trace.selection_mode.clone()),
                "final_source": final_source,
            })),
            created_ts: current_ts(),
        })
        .await;
    }

    fn matching_token<'a>(
        &'a self,
        headers: &HeaderMap,
        required_scope: &str,
    ) -> Option<&'a ApiTokenConfig> {
        let header = headers.get(AUTHORIZATION)?.to_str().ok()?;
        let token = header.strip_prefix("Bearer ")?.trim();
        if token.is_empty() {
            return None;
        }
        self.inner.config.security.api_tokens.iter().find(|config| {
            config.token == token
                && (config.scopes.is_empty()
                    || config
                        .scopes
                        .iter()
                        .any(|scope| scope == "*" || scope.eq_ignore_ascii_case(required_scope)))
        })
    }

    fn build_candidates(
        &self,
        request: &CompletionRequest,
        include_configured: bool,
    ) -> Vec<ProviderCandidate> {
        let mut candidates = Vec::new();
        if let Some(provider) = request.preferred_provider.as_ref() {
            if request_candidate_model_allowed(
                &self.inner.config,
                provider,
                request.preferred_model.as_deref(),
            ) {
                candidates.push(self.request_candidate(
                    provider,
                    request.preferred_model.clone(),
                    request.preferred_api_key.clone(),
                    request.preferred_access_token.clone(),
                    request.base_url.clone(),
                    true,
                    "request_primary",
                ));
            } else {
                tracing::warn!(
                    provider = %provider,
                    requested_model = ?request.preferred_model,
                    "ignoring unconfigured Ollama request model; using configured provider profiles"
                );
            }
        }
        if let Some(provider) = request.fallback_provider.as_ref() {
            if request_candidate_model_allowed(
                &self.inner.config,
                provider,
                request.fallback_model.as_deref(),
            ) {
                candidates.push(self.request_candidate(
                    provider,
                    request.fallback_model.clone(),
                    request.fallback_api_key.clone(),
                    request.fallback_access_token.clone(),
                    request.base_url.clone(),
                    false,
                    "request_fallback",
                ));
            } else {
                tracing::warn!(
                    provider = %provider,
                    requested_model = ?request.fallback_model,
                    "ignoring unconfigured Ollama fallback model; using configured provider profiles"
                );
            }
        }
        let include_configured_fallback = should_include_configured_candidates(
            include_configured,
            request,
            !candidates.is_empty(),
        );
        if include_configured_fallback {
            candidates.extend(
                self.inner
                    .config
                    .providers
                    .iter()
                    .cloned()
                    .map(ProviderCandidate::from_profile),
            );
            append_local_ollama_fallback_candidate(&mut candidates);
        }
        dedupe_candidates(candidates)
            .into_iter()
            .filter(provider_candidate_is_usable)
            .collect()
    }

    fn request_candidate(
        &self,
        provider: &str,
        model: Option<String>,
        api_key: Option<String>,
        access_token: Option<String>,
        base_url: Option<String>,
        preferred: bool,
        source: &str,
    ) -> ProviderCandidate {
        ProviderCandidate::from_profile(ProviderProfile {
            name: provider.trim().to_string(),
            provider_type: provider.trim().to_string(),
            model,
            api_key,
            access_token,
            base_url,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: if preferred { 0.4 } else { 0.0 },
            preferred,
            source: Some(source.to_string()),
            ..ProviderProfile::default()
        })
    }

    async fn rank_candidate(
        &self,
        candidate: ProviderCandidate,
        workflow: &str,
        role: &str,
        task_tags: &HashSet<String>,
    ) -> RankedCandidate {
        let candidate_id = candidate.candidate_id();
        let overlap = task_tags.intersection(&candidate.specialties).count() as f64;
        let role_score = if candidate.roles.is_empty() {
            0.0
        } else if candidate.roles.contains(role) {
            0.6
        } else {
            -0.9
        };
        let health = if is_ollama_candidate(&candidate)
            && self
                .inner
                .metrics
                .candidate_in_health_backoff(
                    candidate_id.as_str(),
                    &["ollama_saturated"],
                    ollama_saturation_backoff_seconds(),
                )
                .await
        {
            ProviderHealth {
                ok: false,
                status_code: None,
                latency_ms: None,
                message: Some("local Ollama is saturated; waiting before retry".to_string()),
                mode: Some("ollama_saturated".to_string()),
            }
        } else if !is_ollama_candidate(&candidate)
            && self
                .inner
                .metrics
                .provider_in_health_backoff(
                    candidate.provider_type.as_str(),
                    &["quota", "upstream", "timeout"],
                    self.health_ttl_seconds(),
                )
                .await
        {
            ProviderHealth {
                ok: false,
                status_code: None,
                latency_ms: None,
                message: Some("provider family is in cached runtime backoff".to_string()),
                mode: Some("provider_backoff".to_string()),
            }
        } else {
            self.probe_health(&candidate).await
        };
        let nmc_signal = self.nmc_signal_for_candidate(&candidate).await;
        let nmc_constrained = nmc_signal.as_ref().is_some_and(|signal| signal.constrained);
        let nmc_pressure_penalty = nmc_signal
            .as_ref()
            .map(|signal| signal.pressure_ratio.clamp(0.0, 2.5) * 1.35)
            .unwrap_or(0.0);
        let nmc_hard_limit_penalty = if nmc_constrained { 2.8 } else { 0.0 };
        let load = self.load_snapshot(&candidate).await;
        let usage_penalty = self
            .inner
            .metrics
            .recent_usage_penalty(
                candidate_id.as_str(),
                workflow,
                role,
                candidate.usage_penalty_decay_seconds,
            )
            .await;
        let resource_penalty =
            (load.candidate_limit_ratio * 1.1) + (load.host_budget_ratio.clamp(0.0, 2.0) * 1.2);
        let hard_limit_penalty = if load.candidate_limit_reached || load.host_budget_reached {
            2.4
        } else {
            0.0
        };
        let health_ok = health.ok
            && !load.candidate_limit_reached
            && !load.host_budget_reached
            && !nmc_constrained;
        let health_mode = if nmc_constrained {
            Some("nmc_constrained".to_string())
        } else if !health_ok && health.ok {
            Some("resource_saturated".to_string())
        } else {
            health.mode.clone()
        };
        let health_score = if health_ok { 0.4 } else { -1.4 };
        let preferred_score = if candidate.preferred { 0.7 } else { 0.0 };
        let metrics_bonus = self
            .inner
            .metrics
            .score_bonus(candidate_id.as_str(), workflow, role)
            .await;
        RankedCandidate {
            health_ok,
            health_mode,
            score: candidate.weight
                + candidate.priority_bias
                + (overlap * 0.85)
                + role_score
                + health_score
                + preferred_score
                + metrics_bonus
                - usage_penalty
                - resource_penalty
                - hard_limit_penalty
                - nmc_pressure_penalty
                - nmc_hard_limit_penalty,
            candidate,
        }
    }

    async fn probe_health(&self, candidate: &ProviderCandidate) -> ProviderHealth {
        let cached = self
            .inner
            .metrics
            .health_snapshot(candidate.candidate_id().as_str())
            .await;
        let health_ttl_seconds = cached_health_ttl_seconds(
            is_ollama_candidate(candidate),
            cached.mode.as_deref(),
            self.health_ttl_seconds(),
        );
        if !self
            .inner
            .metrics
            .should_probe(candidate.candidate_id().as_str(), health_ttl_seconds)
            .await
        {
            return ProviderHealth {
                ok: cached.ok.unwrap_or(false),
                status_code: None,
                latency_ms: cached.latency_ms,
                message: cached.message,
                mode: cached.mode,
            };
        }

        let health = match build_adapter(self.inner.client.clone(), &candidate.profile) {
            Ok(adapter) => adapter
                .health(Some(PROVIDER_HEALTH_TIMEOUT_SECONDS))
                .await
                .unwrap_or_else(|error| ProviderHealth {
                    ok: false,
                    status_code: None,
                    latency_ms: None,
                    message: Some(error.to_string()),
                    mode: Some("error".to_string()),
                }),
            Err(error) => ProviderHealth {
                ok: false,
                status_code: None,
                latency_ms: None,
                message: Some(error.to_string()),
                mode: Some("unconfigured".to_string()),
            },
        };
        let summary = candidate.summary(None);
        let _ = self
            .inner
            .metrics
            .record_health(
                &summary,
                HealthBucket {
                    ok: Some(health.ok),
                    mode: health.mode.clone(),
                    checked_at: None,
                    latency_ms: health.latency_ms,
                    message: health.message.clone(),
                },
            )
            .await;
        health
    }

    async fn nmc_signal_for_candidate(
        &self,
        candidate: &ProviderCandidate,
    ) -> Option<NmcAgentSignal> {
        let nmc = self.nmc_telemetry()?;
        nmc.signal(
            candidate.nmc_agent_id.as_deref(),
            candidate.nmc_host.as_deref(),
            candidate.host_group.as_deref(),
        )
        .await
    }

    async fn load_snapshot(&self, candidate: &ProviderCandidate) -> CandidateLoadSnapshot {
        let candidate_id = candidate.candidate_id();
        let tracker = self.inner.load_tracker.lock().await;
        let candidate_in_flight = tracker
            .candidate_in_flight
            .get(&candidate_id)
            .copied()
            .unwrap_or(0);
        let candidate_limit = candidate.max_concurrent_requests;
        let candidate_limit_ratio = candidate_limit
            .map(|limit| candidate_in_flight as f64 / limit.max(1) as f64)
            .unwrap_or(0.0);
        let candidate_limit_reached = candidate_limit
            .map(|limit| candidate_in_flight >= limit.max(1))
            .unwrap_or(false);
        let host_usage = candidate
            .host_group
            .as_ref()
            .map(|group| tracker.host_usage.get(group).cloned().unwrap_or_default());
        let projected_host_usage = host_usage.as_ref().map(|current| HostLoad {
            requests: current.requests.saturating_add(1),
            cpu: current.cpu + candidate.resource_cost_cpu.max(0.0),
            ram_mb: current
                .ram_mb
                .saturating_add(candidate.resource_cost_ram_mb),
            vram_mb: current
                .vram_mb
                .saturating_add(candidate.resource_cost_vram_mb),
        });
        let host_budget_ratio = projected_host_usage
            .as_ref()
            .map(|usage| host_budget_ratio(candidate, usage))
            .unwrap_or(0.0);
        let host_budget_reached = projected_host_usage
            .as_ref()
            .is_some_and(|usage| host_budget_exceeded(candidate, usage));
        CandidateLoadSnapshot {
            candidate_limit_ratio,
            candidate_limit_reached,
            host_budget_ratio,
            host_budget_reached,
        }
    }

    async fn reserve_candidate_load(
        &self,
        candidate: &ProviderCandidate,
    ) -> Option<LoadReservation> {
        let candidate_id = candidate.candidate_id();
        let mut tracker = self.inner.load_tracker.lock().await;
        let candidate_in_flight = tracker
            .candidate_in_flight
            .get(&candidate_id)
            .copied()
            .unwrap_or(0);
        if candidate
            .max_concurrent_requests
            .is_some_and(|limit| candidate_in_flight >= limit.max(1))
        {
            return None;
        }
        if let Some(host_group) = candidate.host_group.as_ref() {
            let current = tracker
                .host_usage
                .get(host_group)
                .cloned()
                .unwrap_or_default();
            let projected = HostLoad {
                requests: current.requests.saturating_add(1),
                cpu: current.cpu + candidate.resource_cost_cpu.max(0.0),
                ram_mb: current
                    .ram_mb
                    .saturating_add(candidate.resource_cost_ram_mb),
                vram_mb: current
                    .vram_mb
                    .saturating_add(candidate.resource_cost_vram_mb),
            };
            if host_budget_exceeded(candidate, &projected) {
                return None;
            }
            tracker.host_usage.insert(host_group.clone(), projected);
        }
        tracker
            .candidate_in_flight
            .entry(candidate_id.clone())
            .and_modify(|value| *value += 1)
            .or_insert(1);
        Some(LoadReservation {
            candidate_id,
            host_group: candidate.host_group.clone(),
            resource_cost_cpu: candidate.resource_cost_cpu.max(0.0),
            resource_cost_ram_mb: candidate.resource_cost_ram_mb,
            resource_cost_vram_mb: candidate.resource_cost_vram_mb,
        })
    }

    async fn release_candidate_load(&self, reservation: LoadReservation) {
        let mut tracker = self.inner.load_tracker.lock().await;
        if let Some(current) = tracker
            .candidate_in_flight
            .get(reservation.candidate_id.as_str())
            .copied()
        {
            if current <= 1 {
                tracker
                    .candidate_in_flight
                    .remove(reservation.candidate_id.as_str());
            } else {
                tracker
                    .candidate_in_flight
                    .insert(reservation.candidate_id.clone(), current - 1);
            }
        }
        if let Some(host_group) = reservation.host_group.as_ref() {
            let mut should_remove = false;
            if let Some(current) = tracker.host_usage.get_mut(host_group) {
                current.requests = current.requests.saturating_sub(1);
                current.cpu = (current.cpu - reservation.resource_cost_cpu).max(0.0);
                current.ram_mb = current
                    .ram_mb
                    .saturating_sub(reservation.resource_cost_ram_mb);
                current.vram_mb = current
                    .vram_mb
                    .saturating_sub(reservation.resource_cost_vram_mb);
                should_remove = current.requests == 0;
            }
            if should_remove {
                tracker.host_usage.remove(host_group);
            }
        }
    }

    async fn invoke_candidates(
        &self,
        selected: Vec<ProviderCandidate>,
        provider_request: ProviderCompletionRequest,
        expected_json: bool,
        selection_mode: SelectionMode,
        early_success_enabled: bool,
        early_success_settle_seconds: f64,
        early_success_min_quality: f64,
        timeout_cap: Option<u64>,
        workload_class: WorkloadClass,
    ) -> Result<Vec<InvocationResult>> {
        let mut join_set = JoinSet::new();
        for candidate in selected.iter().cloned() {
            let service = self.clone();
            let request = provider_request.clone();
            join_set.spawn(async move {
                service
                    .invoke_candidate(
                        candidate,
                        request,
                        expected_json,
                        timeout_cap,
                        workload_class,
                    )
                    .await
            });
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum DeadlineKind {
            EarlySuccess,
            HardTimeout,
        }

        let mut results = Vec::new();
        let mut pending_candidate_ids = selected
            .iter()
            .map(ProviderCandidate::candidate_id)
            .collect::<HashSet<_>>();
        let mut early_deadline: Option<Instant> = None;
        let hard_deadline =
            timeout_cap.map(|seconds| Instant::now() + Duration::from_secs(seconds.max(1)));
        while !join_set.is_empty() {
            let next_deadline = match (early_deadline, hard_deadline) {
                (Some(early), Some(hard)) if early <= hard => {
                    Some((early, DeadlineKind::EarlySuccess))
                }
                (Some(_early), Some(hard)) => Some((hard, DeadlineKind::HardTimeout)),
                (Some(early), None) => Some((early, DeadlineKind::EarlySuccess)),
                (None, Some(hard)) => Some((hard, DeadlineKind::HardTimeout)),
                (None, None) => None,
            };
            let joined = if let Some((deadline, deadline_kind)) = next_deadline {
                match tokio::time::timeout_at(deadline, join_set.join_next()).await {
                    Ok(result) => result,
                    Err(_) => {
                        join_set.abort_all();
                        if deadline_kind == DeadlineKind::HardTimeout {
                            let timeout_seconds = timeout_cap.unwrap_or_default().max(1);
                            for candidate in selected
                                .iter()
                                .filter(|&candidate| {
                                    pending_candidate_ids.contains(&candidate.candidate_id())
                                })
                                .cloned()
                            {
                                results.push(InvocationResult {
                                    candidate,
                                    response: None,
                                    error: Some(format!(
                                        "candidate timed out after {timeout_seconds}s"
                                    )),
                                    latency_ms: Some(timeout_seconds * 1000),
                                    quality: -1.0,
                                    score: f64::NEG_INFINITY,
                                });
                            }
                        }
                        break;
                    }
                }
            } else {
                join_set.join_next().await
            };
            let Some(joined) = joined else {
                break;
            };
            let result = match joined {
                Ok(result) => result,
                Err(error) => InvocationResult {
                    candidate: ProviderCandidate::from_profile(ProviderProfile {
                        name: "join_error".to_string(),
                        provider_type: "join_error".to_string(),
                        source: Some("internal".to_string()),
                        ..ProviderProfile::default()
                    }),
                    response: None,
                    error: Some(error.to_string()),
                    latency_ms: None,
                    quality: -1.0,
                    score: f64::NEG_INFINITY,
                },
            };
            info!(candidate = %result.candidate.label(result.response.as_ref().map(|value| value.model.as_str())), status = if result.response.is_some() { "ok" } else { "error" }, latency_ms = ?result.latency_ms, quality = result.quality, error = ?result.error, "Gail candidate completed");
            let accepts_early =
                result.response.is_some() && result.quality >= early_success_min_quality;
            pending_candidate_ids.remove(&result.candidate.candidate_id());
            results.push(result);
            if !early_success_enabled || !accepts_early {
                continue;
            }
            if selection_mode == SelectionMode::Fastest {
                join_set.abort_all();
                break;
            }
            if early_deadline.is_none() {
                early_deadline = Some(
                    Instant::now() + Duration::from_secs_f64(early_success_settle_seconds.max(0.0)),
                );
            }
        }
        Ok(results)
    }

    async fn invoke_candidate(
        &self,
        candidate: ProviderCandidate,
        provider_request: ProviderCompletionRequest,
        expected_json: bool,
        timeout_cap: Option<u64>,
        workload_class: WorkloadClass,
    ) -> InvocationResult {
        if let Some(signal) = self.nmc_signal_for_candidate(&candidate).await
            && signal.constrained
        {
            let agent = if signal.agent_id.trim().is_empty() {
                "unknown"
            } else {
                signal.agent_id.as_str()
            };
            let host = if signal.host.trim().is_empty() {
                "unknown"
            } else {
                signal.host.as_str()
            };
            return InvocationResult {
                candidate,
                response: None,
                error: Some(format!(
                    "candidate skipped because NMC/Tracey telemetry reports constrained capacity (agent={agent}, host={host}, status={}, mode={}, optimize_status={}, pressure_ratio={:.2})",
                    signal.status, signal.mode, signal.optimize_status, signal.pressure_ratio,
                )),
                latency_ms: None,
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        }
        let Some(_workload_permit) = self.acquire_workload_permit(workload_class).await else {
            return InvocationResult {
                candidate,
                response: None,
                error: Some(format!(
                    "{} workload pool is saturated; retry after {}ms",
                    workload_class.label(),
                    self.workload_pool_wait_timeout_ms(),
                )),
                latency_ms: None,
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        };
        let Some(load_reservation) = self.reserve_candidate_load(&candidate).await else {
            return InvocationResult {
                candidate,
                response: None,
                error: Some(
                    "candidate skipped because configured concurrency/resource budget is exhausted"
                        .to_string(),
                ),
                latency_ms: None,
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        };
        let quota_retries = env_int_any(&["LLM_RATE_LIMIT_RETRIES"], 2) as usize;
        let timeout_retries = env_int_any(&["LLM_TIMEOUT_RETRIES"], 0) as usize;
        let quota_backoff_base = env_float_any(&["LLM_RATE_LIMIT_BACKOFF_BASE"], 1.0).max(0.1);
        let timeout_backoff_base = env_float_any(&["LLM_TIMEOUT_BACKOFF_BASE"], 1.0).max(0.1);
        let retry_empty = env_bool_any(
            &["REFINER_AI_RETRY_EMPTY_OUTPUT", "GAIL_RETRY_EMPTY_OUTPUT"],
            true,
        );
        let effective_timeout_seconds =
            request_timeout_with_cap(provider_request.timeout_seconds, timeout_cap);
        let timeout_window =
            effective_timeout_seconds.map(|seconds| Duration::from_secs(seconds.max(1)));
        let client = self.inner.client.clone();
        let candidate_for_invocation = candidate.clone();
        let provider_request_for_invocation = provider_request.clone();
        let invocation = async move {
            let mut quota_attempts = 0usize;
            let mut timeout_attempts = 0usize;
            let mut attempts = 0usize;
            loop {
                attempts += 1;
                let mut effective = provider_request_from_profile(
                    &candidate_for_invocation.profile,
                    &provider_request_for_invocation,
                );
                effective.timeout_seconds = effective_timeout_seconds;
                let started = std::time::Instant::now();
                let adapter = match build_adapter(client.clone(), &candidate_for_invocation.profile)
                {
                    Ok(adapter) => adapter,
                    Err(error) => {
                        return InvocationResult {
                            candidate: candidate_for_invocation.clone(),
                            response: None,
                            error: Some(error.to_string()),
                            latency_ms: None,
                            quality: -1.0,
                            score: f64::NEG_INFINITY,
                        };
                    }
                };
                match adapter.complete(&effective).await {
                    Ok(response) => {
                        let latency_ms = started.elapsed().as_millis() as u64;
                        if response.text.trim().is_empty() && retry_empty && attempts < 2 {
                            continue;
                        }
                        if violates_strict_model_policy(
                            effective.strict_no_downgrade.unwrap_or(false),
                            effective.min_model_size_b,
                            candidate_for_invocation.configured_model.as_str(),
                            response.model.as_str(),
                        ) {
                            return InvocationResult {
                                candidate: candidate_for_invocation.clone(),
                                response: None,
                                error: Some(format!(
                                    "model selection violated strict no-downgrade policy (configured={}, resolved={}, min_floor_b={})",
                                    candidate_for_invocation.configured_model,
                                    response.model,
                                    effective
                                        .min_model_size_b
                                        .map(|value| format!("{value:.2}"))
                                        .unwrap_or_else(|| "none".to_string())
                                )),
                                latency_ms: Some(latency_ms),
                                quality: -1.0,
                                score: f64::NEG_INFINITY,
                            };
                        }
                        let quality = quality_score(&response.text, expected_json);
                        return InvocationResult {
                            candidate: candidate_for_invocation.clone(),
                            response: Some(response),
                            error: None,
                            latency_ms: Some(latency_ms),
                            quality,
                            score: f64::NEG_INFINITY,
                        };
                    }
                    Err(error) => {
                        let latency_ms = started.elapsed().as_millis() as u64;
                        if error.is_quota() && quota_attempts < quota_retries {
                            let delay = Duration::from_secs_f64(
                                quota_backoff_base * 2_f64.powi(quota_attempts as i32),
                            );
                            quota_attempts += 1;
                            sleep(delay).await;
                            continue;
                        }
                        if error.is_timeout() && timeout_attempts < timeout_retries {
                            let delay = Duration::from_secs_f64(
                                timeout_backoff_base * 2_f64.powi(timeout_attempts as i32),
                            );
                            timeout_attempts += 1;
                            sleep(delay).await;
                            continue;
                        }
                        return InvocationResult {
                            candidate: candidate_for_invocation.clone(),
                            response: None,
                            error: Some(error.to_string()),
                            latency_ms: Some(latency_ms),
                            quality: -1.0,
                            score: f64::NEG_INFINITY,
                        };
                    }
                }
            }
        };
        let result = if let Some(timeout_window) = timeout_window {
            match tokio::time::timeout(timeout_window, invocation).await {
                Ok(result) => result,
                Err(_) => InvocationResult {
                    candidate,
                    response: None,
                    error: Some(format!(
                        "candidate timed out after {}s",
                        timeout_window.as_secs().max(1)
                    )),
                    latency_ms: Some(timeout_window.as_millis() as u64),
                    quality: -1.0,
                    score: f64::NEG_INFINITY,
                },
            }
        } else {
            invocation.await
        };
        self.release_candidate_load(load_reservation).await;
        result
    }

    fn select_specialist(&self, name: Option<&str>) -> Result<&SpecialistEngine> {
        if let Some(name) = name {
            self.inner
                .specialists
                .iter()
                .find(|engine| engine.matches_name(name))
                .ok_or_else(|| GailError::not_found(format!("unknown specialist engine: {name}")))
        } else {
            self.inner
                .specialists
                .first()
                .ok_or_else(|| GailError::not_found("no specialist engines are configured"))
        }
    }

    fn include_configured_candidates(&self) -> bool {
        env_bool_any(
            &[
                "GAIL_INCLUDE_CONFIGURED_CANDIDATES",
                "REFINER_AI_INCLUDE_CONFIGURED_CANDIDATES",
            ],
            self.inner
                .config
                .orchestration
                .include_configured_candidates,
        )
    }

    fn max_parallel_candidates(&self) -> usize {
        env_int_any(
            &[
                "GAIL_MAX_PARALLEL_CANDIDATES",
                "REFINER_AI_MAX_CONCURRENT_CANDIDATES",
            ],
            self.inner.config.orchestration.max_parallel_candidates as u64,
        ) as usize
    }

    fn workload_pool_wait_timeout_ms(&self) -> u64 {
        env_int_any(
            &[
                "GAIL_WORKLOAD_POOL_WAIT_TIMEOUT_MS",
                "REFINER_AI_WORKLOAD_POOL_WAIT_TIMEOUT_MS",
            ],
            self.inner
                .config
                .orchestration
                .workload_pool_wait_timeout_ms,
        )
        .clamp(1, 60_000)
    }

    fn model_floor_b(&self, workload_class: WorkloadClass) -> Option<f64> {
        let configured = match workload_class {
            WorkloadClass::Interactive => self.inner.config.orchestration.interactive_model_floor_b,
            WorkloadClass::Solver => self.inner.config.orchestration.solver_model_floor_b,
        };
        let env_floor = match workload_class {
            WorkloadClass::Interactive => env_float_any(
                &[
                    "GAIL_INTERACTIVE_MODEL_FLOOR_B",
                    "REFINER_AI_INTERACTIVE_MODEL_FLOOR_B",
                ],
                configured,
            ),
            WorkloadClass::Solver => env_float_any(
                &[
                    "GAIL_SOLVER_MODEL_FLOOR_B",
                    "REFINER_AI_SOLVER_MODEL_FLOOR_B",
                ],
                configured,
            ),
        };
        let floor = env_floor.max(0.0);
        if floor > 0.0 { Some(floor) } else { None }
    }

    fn strict_no_downgrade(&self) -> bool {
        env_bool_any(
            &["GAIL_STRICT_NO_DOWNGRADE", "REFINER_AI_STRICT_NO_DOWNGRADE"],
            self.inner.config.orchestration.strict_no_downgrade,
        )
    }

    async fn acquire_workload_permit(&self, class: WorkloadClass) -> Option<OwnedSemaphorePermit> {
        let wait_timeout = Duration::from_millis(self.workload_pool_wait_timeout_ms());
        let semaphore = match class {
            WorkloadClass::Interactive => self.inner.interactive_pool.clone(),
            WorkloadClass::Solver => self.inner.solver_pool.clone(),
        };
        match tokio::time::timeout(wait_timeout, semaphore.acquire_owned()).await {
            Ok(Ok(permit)) => Some(permit),
            _ => None,
        }
    }

    fn health_ttl_seconds(&self) -> f64 {
        env_float_any(
            &["GAIL_HEALTH_TTL_SECONDS", "REFINER_AI_HEALTH_TTL_SECONDS"],
            self.inner.config.orchestration.health_ttl_seconds,
        )
        .max(30.0)
    }

    fn selection_mode(&self) -> SelectionMode {
        let env_value = env::var("GAIL_SELECTION_MODE")
            .ok()
            .or_else(|| env::var("REFINER_AI_SELECTION_MODE").ok())
            .unwrap_or_default();
        match env_value.trim().to_ascii_lowercase().as_str() {
            "fastest" => SelectionMode::Fastest,
            "best" => SelectionMode::Best,
            _ => self.inner.config.orchestration.selection_mode.clone(),
        }
    }

    fn early_success_enabled(
        &self,
        workflow: &str,
        role: &str,
        selection_mode: &SelectionMode,
    ) -> bool {
        if *selection_mode == SelectionMode::Fastest {
            return true;
        }
        if let Ok(value) = env::var("REFINER_AI_EARLY_SUCCESS_ENABLED") {
            return parse_bool(&value, false);
        }
        if let Ok(value) = env::var("GAIL_EARLY_SUCCESS_ENABLED") {
            return parse_bool(&value, false);
        }
        if self.inner.config.orchestration.early_success_enabled {
            return true;
        }
        is_interactive_workflow(workflow, role)
    }

    fn early_success_settle_seconds(
        &self,
        workflow: &str,
        role: &str,
        selection_mode: &SelectionMode,
    ) -> f64 {
        let default = if *selection_mode == SelectionMode::Fastest {
            0.0
        } else if is_interactive_workflow(workflow, role) {
            0.75
        } else {
            0.0
        };
        env_float_any(
            &[
                "GAIL_EARLY_SUCCESS_SETTLE_SECONDS",
                "REFINER_AI_EARLY_SUCCESS_SETTLE_SECONDS",
            ],
            if self.inner.config.orchestration.early_success_settle_seconds > 0.0 {
                self.inner.config.orchestration.early_success_settle_seconds
            } else {
                default
            },
        )
        .max(0.0)
    }

    fn early_success_min_quality(&self) -> f64 {
        env_float_any(
            &[
                "GAIL_EARLY_SUCCESS_MIN_QUALITY",
                "REFINER_AI_EARLY_SUCCESS_MIN_QUALITY",
            ],
            self.inner.config.orchestration.early_success_min_quality,
        )
    }

    fn candidate_timeout_cap(
        &self,
        workflow: &str,
        role: &str,
        expected_json: bool,
        task_tags: &HashSet<String>,
        prompt_text: &str,
    ) -> Option<u64> {
        let default = if is_interactive_workflow(workflow, role) {
            45
        } else {
            self.inner
                .config
                .orchestration
                .candidate_timeout_cap_seconds
                .unwrap_or_default() as i64
        };
        let value = env_int_any(
            &[
                "GAIL_CANDIDATE_TIMEOUT_CAP_SECONDS",
                "REFINER_AI_CANDIDATE_TIMEOUT_CAP_SECONDS",
            ],
            default.max(0) as u64,
        );
        let base = (value > 0).then(|| value.max(1));
        if is_interactive_workflow(workflow, role) {
            return base;
        }
        if expected_json
            && (prompt_requests_execution_plan(prompt_text)
                || prompt_requests_manager_tool_call(prompt_text))
        {
            // Multi-agent manager planning payloads (ExecutionPlan / tool-call envelopes)
            // need a full request budget; forcing automation caps here collapses the
            // plan to degraded no-op outputs like `{"steps":[]}`.
            return base;
        }
        if !expected_json
            && !text_or_tags_indicate_automation(workflow, role, task_tags, prompt_text)
        {
            return base;
        }
        let automation_default = self
            .inner
            .config
            .orchestration
            .automation_candidate_timeout_cap_seconds
            .unwrap_or(12);
        let automation_value = env_int_any(
            &[
                "GAIL_AUTOMATION_CANDIDATE_TIMEOUT_SECONDS",
                "GAIL_AUTOMATION_CANDIDATE_TIMEOUT_CAP_SECONDS",
                "REFINER_AI_AUTOMATION_CANDIDATE_TIMEOUT_SECONDS",
                "REFINER_AI_AUTOMATION_CANDIDATE_TIMEOUT_CAP_SECONDS",
            ],
            automation_default,
        );
        if automation_value == 0 {
            base
        } else {
            Some(
                base.map(|base| base.min(automation_value.max(1)))
                    .unwrap_or_else(|| automation_value.max(1)),
            )
        }
    }

    fn always_route_specialists(&self) -> bool {
        self.inner.config.orchestration.always_route_specialists
            || env_bool_any(
                &[
                    "GAIL_ALWAYS_ROUTE_SPECIALISTS",
                    "REFINER_SPECIALIST_ENGINES_ALWAYS_ROUTE",
                    "REFINER_NEUROMORPHIC_ALWAYS_ROUTE",
                    "REFINER_AARNN_ALWAYS_ROUTE",
                ],
                false,
            )
    }
}

impl ProviderCandidate {
    fn from_profile(mut profile: ProviderProfile) -> Self {
        let provider_type = normalize_provider_type(profile.provider_type.as_str());
        if profile.name.trim().is_empty() {
            profile.name = provider_type.clone();
        }
        let configured_model = profile.model.clone().unwrap_or_default();
        let specialties = infer_specialties(
            provider_type.as_str(),
            configured_model.as_str(),
            profile.source.as_deref(),
            &profile.specialties,
        );
        let roles = profile
            .roles
            .iter()
            .map(|item| normalize_key(item, "general"))
            .collect::<HashSet<_>>();
        let source = profile
            .source
            .clone()
            .unwrap_or_else(|| "config".to_string());
        let weight = profile.weight;
        let preferred = profile.preferred;
        let host_group = profile.host_group.clone();
        let priority_bias = profile.priority_bias;
        let usage_penalty_decay_seconds = profile.usage_penalty_decay_seconds.max(30.0);
        let max_concurrent_requests = profile.max_concurrent_requests.map(|value| value.max(1));
        let resource_cost_cpu = profile.resource_cost_cpu.max(0.0);
        let resource_cost_ram_mb = profile.resource_cost_ram_mb;
        let resource_cost_vram_mb = profile.resource_cost_vram_mb;
        let host_cpu_budget = profile.host_cpu_budget.filter(|value| *value > 0.0);
        let host_ram_budget_mb = profile.host_ram_budget_mb.filter(|value| *value > 0);
        let host_vram_budget_mb = profile.host_vram_budget_mb.filter(|value| *value > 0);
        let nmc_agent_id = profile.nmc_agent_id.clone();
        let nmc_host = profile.nmc_host.clone();
        Self {
            profile,
            source,
            provider_type,
            configured_model,
            preferred,
            weight,
            specialties,
            roles,
            host_group,
            priority_bias,
            usage_penalty_decay_seconds,
            max_concurrent_requests,
            resource_cost_cpu,
            resource_cost_ram_mb,
            resource_cost_vram_mb,
            host_cpu_budget,
            host_ram_budget_mb,
            host_vram_budget_mb,
            nmc_agent_id,
            nmc_host,
        }
    }

    fn candidate_id(&self) -> String {
        let endpoint_scope = self.endpoint_scope();
        format!(
            "{}/{}{}",
            self.provider_type,
            if self.configured_model.trim().is_empty() {
                "default"
            } else {
                self.configured_model.trim()
            },
            endpoint_scope
                .map(|scope| format!("@{scope}"))
                .unwrap_or_default()
        )
    }

    fn endpoint_scope(&self) -> Option<String> {
        if !is_ollama_candidate(self) {
            return None;
        }
        let explicit_name = self.profile.name.trim();
        if !explicit_name.is_empty()
            && !explicit_name.eq_ignore_ascii_case(self.provider_type.as_str())
        {
            return Some(sanitize_candidate_scope(explicit_name, "endpoint"));
        }
        self.profile
            .base_url
            .as_deref()
            .and_then(candidate_scope_from_base_url)
    }

    fn label(&self, resolved_model: Option<&str>) -> String {
        let resolved = resolved_model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                if self.configured_model.trim().is_empty() {
                    "default"
                } else {
                    self.configured_model.trim()
                }
            });
        if !self.configured_model.trim().is_empty() && self.configured_model.trim() != resolved {
            format!(
                "{}/{} (configured {})",
                self.provider_type,
                resolved,
                self.configured_model.trim()
            )
        } else {
            format!("{}/{}", self.provider_type, resolved)
        }
    }

    fn summary(&self, resolved_model: Option<&str>) -> CandidateSummary {
        let resolved = resolved_model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                if self.configured_model.trim().is_empty() {
                    "default"
                } else {
                    self.configured_model.trim()
                }
            })
            .to_string();
        CandidateSummary {
            candidate_id: self.candidate_id(),
            provider: self.provider_type.clone(),
            model: resolved.clone(),
            configured_model: if self.configured_model.trim().is_empty() {
                resolved.clone()
            } else {
                self.configured_model.clone()
            },
            resolved_model: resolved,
            source: self.source.clone(),
            specialties: sorted_strings(self.specialties.clone()),
            roles: sorted_strings(self.roles.clone()),
        }
    }
}

fn dedupe_candidates(candidates: Vec<ProviderCandidate>) -> Vec<ProviderCandidate> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for candidate in candidates {
        let key = format!(
            "{}::{}::{}",
            candidate.provider_type,
            candidate.configured_model,
            candidate.profile.base_url.clone().unwrap_or_default()
        );
        if seen.insert(key) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn append_local_ollama_fallback_candidate(candidates: &mut Vec<ProviderCandidate>) {
    if env_bool_any(
        &[
            "GAIL_DISABLE_OLLAMA_FALLBACK",
            "REFINER_AI_DISABLE_OLLAMA_FALLBACK",
        ],
        false,
    ) {
        return;
    }
    if candidates
        .iter()
        .any(|candidate| candidate.provider_type.eq_ignore_ascii_case("ollama"))
    {
        return;
    }
    let model = env_string_any(&["GAIL_OLLAMA_MODEL", "OLLAMA_MODEL", "OLLAMA_DEFAULT_MODEL"])
        .unwrap_or_else(|| "llama3.2".to_string());
    let base_url = env_string_any(&["GAIL_OLLAMA_BASE_URL", "OLLAMA_BASE_URL", "OLLAMA_HOST"])
        .unwrap_or_else(|| "http://ollama.ollama.svc.cluster.local:11434".to_string());
    candidates.push(ProviderCandidate::from_profile(ProviderProfile {
        name: "OllamaLocalFallback".to_string(),
        provider_type: "ollama".to_string(),
        model: Some(model),
        api_key: None,
        access_token: None,
        base_url: Some(base_url),
        roles: vec![
            "general".to_string(),
            "planner".to_string(),
            "reviewer".to_string(),
            "researcher".to_string(),
            "assistant".to_string(),
        ],
        specialties: vec![
            "local".to_string(),
            "privacy".to_string(),
            "code".to_string(),
            "planning".to_string(),
            "json".to_string(),
            "review".to_string(),
            "research".to_string(),
        ],
        weight: 0.12,
        preferred: false,
        source: Some("auto_local_fallback".to_string()),
        ..ProviderProfile::default()
    }));
}

fn request_timeout_with_cap(request_timeout: Option<u64>, timeout_cap: Option<u64>) -> Option<u64> {
    match (
        request_timeout.map(|value| value.max(1)),
        timeout_cap.map(|value| value.max(1)),
    ) {
        (Some(request_timeout), Some(timeout_cap)) => Some(request_timeout.min(timeout_cap)),
        (Some(request_timeout), None) => Some(request_timeout),
        (None, Some(timeout_cap)) => Some(timeout_cap),
        (None, None) => None,
    }
}

fn provider_candidate_is_usable(candidate: &ProviderCandidate) -> bool {
    provider_profile_is_usable(&candidate.profile)
}

fn provider_profile_is_usable(profile: &ProviderProfile) -> bool {
    let provider_type = normalize_provider_type(profile.provider_type.as_str());
    if provider_type.trim().is_empty() {
        return false;
    }
    match provider_type.as_str() {
        "openai" => {
            has_usable_value(profile.api_key.as_deref())
                || env_has_usable_value(&["OPENAI_API_KEY"])
        }
        "nvidia" => {
            has_usable_value(profile.api_key.as_deref())
                || env_has_usable_value(&["NVIDIA_API_KEY"])
        }
        "gemini" => {
            has_usable_value(profile.api_key.as_deref())
                || has_usable_value(profile.access_token.as_deref())
                || env_has_usable_value(&[
                    "GEMINI_API_KEY",
                    "GEMINI_ACCESS_TOKEN",
                    "GOOGLE_ACCESS_TOKEN",
                ])
        }
        "ollama" => true,
        _ => true,
    }
}

fn has_usable_value(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .map(|value| !matches!(value.as_str(), "none" | "null" | "nil" | "undefined"))
        .unwrap_or(false)
}

fn env_has_usable_value(names: &[&str]) -> bool {
    names
        .iter()
        .any(|name| has_usable_value(env::var(name).ok().as_deref()))
}

fn select_ranked_candidates(
    ranked: Vec<RankedCandidate>,
    max_candidates: usize,
) -> Vec<ProviderCandidate> {
    let target = max_candidates.max(1);
    let mut selected = Vec::new();
    let mut selected_ids = HashSet::new();
    let mut selected_provider_types = HashSet::new();
    let local_fallback = if target >= 2 {
        best_local_fallback_candidate(&ranked)
    } else {
        None
    };

    for health_ok in [true, false] {
        for item in ranked.iter().filter(|item| item.health_ok == health_ok) {
            if !selected_provider_types.insert(item.candidate.provider_type.clone()) {
                continue;
            }
            let candidate_id = item.candidate.candidate_id();
            if selected_ids.insert(candidate_id) {
                selected.push(item.candidate.clone());
                if selected.len() == target {
                    return ensure_local_fallback_selected(selected, local_fallback, target);
                }
            }
        }
    }

    for health_ok in [true, false] {
        for item in ranked.iter().filter(|item| item.health_ok == health_ok) {
            let candidate_id = item.candidate.candidate_id();
            if selected_ids.insert(candidate_id) {
                selected.push(item.candidate.clone());
                if selected.len() == target {
                    return ensure_local_fallback_selected(selected, local_fallback, target);
                }
            }
        }
    }

    ensure_local_fallback_selected(selected, local_fallback, target)
}

fn suggested_pool_size(cpu_cores: usize, configured: usize, divisor: usize) -> usize {
    let derived = if divisor == 0 {
        cpu_cores
    } else {
        cpu_cores / divisor
    }
    .clamp(1, 4096);
    configured.max(derived).clamp(1, 4096)
}

fn current_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn best_local_fallback_candidate(ranked: &[RankedCandidate]) -> Option<ProviderCandidate> {
    ranked
        .iter()
        .filter(|item| candidate_is_local_fallback(&item.candidate))
        .max_by(|left, right| {
            left.health_ok.cmp(&right.health_ok).then_with(|| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .map(|item| item.candidate.clone())
}

fn candidate_is_local_fallback(candidate: &ProviderCandidate) -> bool {
    candidate.provider_type.eq_ignore_ascii_case("ollama")
        || candidate.specialties.iter().any(|item| item == "local")
}

fn is_ollama_candidate(candidate: &ProviderCandidate) -> bool {
    candidate.provider_type.eq_ignore_ascii_case("ollama")
}

fn ollama_saturation_backoff_seconds() -> f64 {
    env_float_any(&["GAIL_OLLAMA_SATURATION_BACKOFF_SECONDS"], 20.0).max(1.0)
}

fn ollama_transient_health_ttl_seconds() -> f64 {
    env_float_any(&["GAIL_OLLAMA_TRANSIENT_HEALTH_TTL_SECONDS"], 30.0).max(1.0)
}

fn cached_health_ttl_seconds(is_ollama: bool, mode: Option<&str>, default_ttl: f64) -> f64 {
    if !is_ollama {
        return default_ttl;
    }
    match mode.map(|value| value.trim().to_ascii_lowercase()) {
        Some(mode) if mode == "ollama_saturated" => ollama_saturation_backoff_seconds(),
        Some(mode)
            if matches!(
                mode.as_str(),
                "timeout" | "upstream" | "resource_saturated" | "runtime_error" | "error"
            ) =>
        {
            ollama_transient_health_ttl_seconds()
        }
        _ => default_ttl,
    }
}

fn host_budget_ratio(candidate: &ProviderCandidate, usage: &HostLoad) -> f64 {
    let mut ratios = Vec::new();
    if let Some(cpu_budget) = candidate.host_cpu_budget.filter(|value| *value > 0.0) {
        ratios.push(usage.cpu / cpu_budget);
    }
    if let Some(ram_budget_mb) = candidate.host_ram_budget_mb.filter(|value| *value > 0) {
        ratios.push(usage.ram_mb as f64 / ram_budget_mb as f64);
    }
    if let Some(vram_budget_mb) = candidate.host_vram_budget_mb.filter(|value| *value > 0) {
        ratios.push(usage.vram_mb as f64 / vram_budget_mb as f64);
    }
    ratios
        .into_iter()
        .fold(0.0_f64, |acc, value| acc.max(value))
        .max(0.0)
}

fn host_budget_exceeded(candidate: &ProviderCandidate, usage: &HostLoad) -> bool {
    host_budget_ratio(candidate, usage) > 1.0
}

fn ensure_local_fallback_selected(
    mut selected: Vec<ProviderCandidate>,
    local_fallback: Option<ProviderCandidate>,
    target: usize,
) -> Vec<ProviderCandidate> {
    let Some(local_fallback) = local_fallback else {
        return selected;
    };
    if selected
        .iter()
        .any(|candidate| candidate.candidate_id() == local_fallback.candidate_id())
    {
        return selected;
    }
    if selected.len() < target {
        selected.push(local_fallback);
    } else if target >= 2 {
        selected.pop();
        selected.push(local_fallback);
    }
    selected
}

fn ranked_candidate_is_in_quota_backoff(item: &RankedCandidate) -> bool {
    !item.health_ok
        && item
            .health_mode
            .as_deref()
            .is_some_and(|mode| mode.eq_ignore_ascii_case("quota"))
}

fn ranked_candidate_is_in_provider_backoff(item: &RankedCandidate) -> bool {
    ranked_candidate_is_in_quota_backoff(item)
        || (!item.health_ok
            && item.health_mode.as_deref().is_some_and(|mode| {
                [
                    "upstream",
                    "timeout",
                    "ollama_saturated",
                    "resource_saturated",
                    "nmc_constrained",
                    "provider_backoff",
                    "unconfigured",
                    "missing_endpoint",
                ]
                .iter()
                .any(|item| mode.eq_ignore_ascii_case(item))
            }))
}

fn ranked_candidate_is_transient_backoff(item: &RankedCandidate) -> bool {
    !item.health_ok
        && item.health_mode.as_deref().is_some_and(|mode| {
            [
                "upstream",
                "timeout",
                "ollama_saturated",
                "resource_saturated",
                "provider_backoff",
                "nmc_constrained",
            ]
            .iter()
            .any(|item| mode.eq_ignore_ascii_case(item))
        })
}

fn should_probe_transient_backoff_candidates(
    workflow: &str,
    role: &str,
    expected_json: bool,
    task_tags: &HashSet<String>,
    prompt_text: &str,
) -> bool {
    if env_bool_any(
        &[
            "GAIL_DISABLE_TRANSIENT_BACKOFF_PROBE",
            "REFINER_AI_DISABLE_TRANSIENT_BACKOFF_PROBE",
        ],
        false,
    ) {
        return false;
    }
    if env_bool_any(
        &[
            "GAIL_ALWAYS_TRANSIENT_BACKOFF_PROBE",
            "REFINER_AI_ALWAYS_TRANSIENT_BACKOFF_PROBE",
        ],
        false,
    ) {
        return true;
    }
    if is_interactive_workflow(workflow, role) {
        return false;
    }
    expected_json || text_or_tags_indicate_automation(workflow, role, task_tags, prompt_text)
}

fn message_indicates_provider_backoff(message: &str) -> bool {
    message_indicates_quota(message)
        || message_indicates_ollama_saturation(message)
        || message_indicates_resource_saturation(message)
        || message_indicates_nmc_constrained(message)
        || message_indicates_provider_auth_failure(message)
        || message_indicates_transient_provider_failure(message)
}

fn error_should_backoff_provider_family(candidate: &ProviderCandidate, message: &str) -> bool {
    if is_ollama_candidate(candidate) {
        return message_indicates_quota(message)
            || message_indicates_provider_auth_failure(message);
    }
    message_indicates_provider_backoff(message)
}

fn message_indicates_ollama_saturation(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("local ollama request queue is saturated")
        || lowered.contains("local model service is saturated")
}

fn message_indicates_resource_saturation(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("configured concurrency/resource budget is exhausted")
        || lowered.contains("resource budget exhausted")
        || lowered.contains("workload pool is saturated")
}

fn message_indicates_nmc_constrained(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("nmc/tracey telemetry reports constrained capacity")
        || lowered.contains("nmc telemetry reports constrained capacity")
        || lowered.contains("nmc_constrained")
}

fn message_indicates_transient_provider_failure(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    if message_indicates_provider_auth_failure(message)
        || message_indicates_permanent_model_failure(message)
    {
        return false;
    }
    lowered.contains("upstream error")
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
        || lowered.contains(" 502 ")
        || lowered.contains(" 503 ")
        || lowered.contains(" 504 ")
}

fn message_indicates_provider_auth_failure(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("authentication failed")
        || lowered.contains("unauthorized")
        || lowered.contains("forbidden")
        || lowered.contains("invalid api key")
        || lowered.contains("status\":401")
        || lowered.contains("status\":403")
        || lowered.contains("status 401")
        || lowered.contains("status 403")
        || lowered.contains("http 401")
        || lowered.contains("http 403")
}

fn message_indicates_permanent_model_failure(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("end of life")
        || lowered.contains("no longer available")
        || lowered.contains("not found for account")
        || lowered.contains("status\":404")
        || lowered.contains("status\":410")
        || lowered.contains("status 404")
        || lowered.contains("status 410")
        || lowered.contains("http 404")
        || lowered.contains("http 410")
        || lowered.contains("\"title\":\"gone\"")
        || lowered.contains("\"title\":\"not found\"")
}

fn runtime_failure_health_bucket(error: Option<&str>, latency_ms: Option<u64>) -> HealthBucket {
    let lowered = error.unwrap_or_default().to_ascii_lowercase();
    let mode = if message_indicates_ollama_saturation(error.unwrap_or_default()) {
        "ollama_saturated"
    } else if message_indicates_resource_saturation(error.unwrap_or_default()) {
        "resource_saturated"
    } else if message_indicates_nmc_constrained(error.unwrap_or_default()) {
        "nmc_constrained"
    } else if lowered.contains("timeout") || lowered.contains("timed out") {
        "timeout"
    } else if message_indicates_quota(error.unwrap_or_default()) {
        "quota"
    } else if message_indicates_provider_auth_failure(error.unwrap_or_default())
        || lowered.contains("not configured")
        || lowered.contains("unsupported")
    {
        "unconfigured"
    } else if message_indicates_permanent_model_failure(error.unwrap_or_default()) {
        "missing_endpoint"
    } else if message_indicates_transient_provider_failure(error.unwrap_or_default()) {
        "upstream"
    } else {
        "runtime_error"
    };
    HealthBucket {
        ok: Some(false),
        mode: Some(mode.to_string()),
        checked_at: None,
        latency_ms,
        message: error.map(ToOwned::to_owned),
    }
}

fn severity_for_issue_category(category: &str) -> &'static str {
    match category {
        "quota" | "upstream" | "timeout" => "warning",
        "unconfigured" | "missing_endpoint" => "critical",
        _ => "warning",
    }
}

fn orchestration_failure_status(message: &str) -> Option<StatusCode> {
    let mode = runtime_failure_health_bucket(Some(message), None)
        .mode
        .unwrap_or_default();
    match mode.as_str() {
        "quota" => Some(StatusCode::TOO_MANY_REQUESTS),
        "timeout" => Some(StatusCode::GATEWAY_TIMEOUT),
        "resource_saturated" | "ollama_saturated" | "nmc_constrained" => {
            Some(StatusCode::SERVICE_UNAVAILABLE)
        }
        "upstream" => Some(StatusCode::BAD_GATEWAY),
        "unconfigured" | "missing_endpoint" => Some(StatusCode::BAD_GATEWAY),
        _ => {
            let lowered = message.to_ascii_lowercase();
            if lowered.contains("adaptive backoff") || lowered.contains("retry after") {
                Some(StatusCode::SERVICE_UNAVAILABLE)
            } else {
                None
            }
        }
    }
}

fn infer_specialties(
    provider_type: &str,
    model: &str,
    source: Option<&str>,
    configured: &[String],
) -> HashSet<String> {
    let mut specialties = default_routing_profiles().base_provider_specialties(provider_type);
    let lowered_model = model.to_ascii_lowercase();
    if lowered_model.contains("codex") {
        specialties.extend(
            ["code", "planning", "review"]
                .into_iter()
                .map(ToOwned::to_owned),
        );
    }
    if lowered_model.contains("flash")
        || lowered_model.contains("mini")
        || lowered_model.contains("small")
    {
        specialties.insert("fast".to_string());
    }
    if lowered_model.contains("pro") || lowered_model.contains("o3") || lowered_model.contains("o4")
    {
        specialties.insert("reasoning".to_string());
    }
    if lowered_model.contains("embed") {
        specialties.insert("retrieval".to_string());
    }
    if source
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("local")
    {
        specialties.insert("local".to_string());
    }
    specialties.extend(
        configured
            .iter()
            .map(|item| normalize_key(item, "general"))
            .filter(|item| !item.is_empty()),
    );
    specialties
}

fn workflow_tags(workflow: &str, role: &str, text: &str) -> HashSet<String> {
    default_routing_profiles().workflow_tags(workflow, role, text)
}

fn expected_json(messages: &[crate::models::ChatMessage], system: Option<&str>) -> bool {
    let text = flatten_prompt_text(messages, system).to_ascii_lowercase();
    [
        "return only valid json",
        "respond with json only",
        "valid json",
        "json with keys",
        "output only json",
        "schema",
    ]
    .iter()
    .any(|hint| text.contains(hint))
}

fn task_tags_expect_json(task_tags: &HashSet<String>) -> bool {
    task_tags
        .iter()
        .any(|tag| matches!(tag.as_str(), "json" | "structured_data"))
}

fn try_parse_json(text: &str) -> Option<Value> {
    let payload = text.trim();
    if payload.is_empty() {
        return None;
    }
    serde_json::from_str(payload).ok().or_else(|| {
        if payload.starts_with("```") && payload.ends_with("```") {
            let inner = payload
                .trim_matches('`')
                .trim()
                .strip_prefix("json")
                .unwrap_or(payload.trim_matches('`').trim())
                .trim();
            serde_json::from_str(inner).ok()
        } else {
            None
        }
    })
}

fn quality_score(text: &str, expected_json: bool) -> f64 {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return -3.0;
    }
    let mut score = 0.6;
    if cleaned.len() >= 40 {
        score += 0.35;
    }
    if expected_json {
        match try_parse_json(cleaned) {
            Some(Value::Object(_)) => score += 2.45,
            Some(Value::Array(_)) => score += 2.35,
            Some(_) => score += 2.2,
            None => score -= 2.0,
        }
    }
    if expected_json && cleaned.contains("```") {
        score -= 0.4;
    }
    score
}

fn local_usage_telemetry(response: &ProviderInvocationResponse) -> LocalUsageTelemetry {
    let mut telemetry = LocalUsageTelemetry::default();
    if let Some(raw) = response.raw.as_ref()
        && let Some(local_usage) = raw.get("gail_local_usage")
    {
        telemetry.queue_wait_ms = local_usage
            .get("queue_wait_ms")
            .and_then(Value::as_u64)
            .or_else(|| raw.get("gail_ollama_queue_wait_ms").and_then(Value::as_u64));
        telemetry.inference_ms = local_usage
            .get("inference_ms")
            .and_then(Value::as_u64)
            .or_else(|| raw.get("gail_ollama_inference_ms").and_then(Value::as_u64));
        telemetry.total_tokens_estimate = local_usage
            .get("total_tokens_estimate")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .or_else(|| {
                raw.get("gail_ollama_total_tokens_estimate")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32)
            });
    }
    if telemetry.total_tokens_estimate.is_none() {
        telemetry.total_tokens_estimate = response.usage.as_ref().and_then(|usage| {
            usage.total.or_else(|| {
                usage
                    .prompt
                    .zip(usage.completion)
                    .map(|(prompt, completion)| prompt.saturating_add(completion))
            })
        });
    }
    telemetry
}

fn parse_model_size_billions(model: &str) -> Option<f64> {
    let lowered = model.trim().to_ascii_lowercase();
    for (index, ch) in lowered.char_indices() {
        if ch != 'b' {
            continue;
        }
        let mut start = index;
        for (scan_index, scan) in lowered[..index].char_indices().rev() {
            if scan.is_ascii_digit() || scan == '.' {
                start = scan_index;
            } else {
                break;
            }
        }
        if start < index {
            let candidate = &lowered[start..index];
            if candidate.chars().any(|ch| ch.is_ascii_digit())
                && let Ok(parsed) = candidate.parse::<f64>()
            {
                return Some(parsed);
            }
        }
    }
    None
}

fn candidate_meets_model_floor(candidate: &ProviderCandidate, min_model_size_b: f64) -> bool {
    if min_model_size_b <= 0.0 {
        return true;
    }
    parse_model_size_billions(candidate.configured_model.as_str())
        .map(|size| size + 0.000_1 >= min_model_size_b)
        .unwrap_or(true)
}

fn violates_strict_model_policy(
    strict_no_downgrade: bool,
    min_model_size_b: Option<f64>,
    configured_model: &str,
    resolved_model: &str,
) -> bool {
    if !strict_no_downgrade {
        return false;
    }
    let configured_size = parse_model_size_billions(configured_model);
    let resolved_size = parse_model_size_billions(resolved_model);
    if let (Some(configured), Some(resolved)) = (configured_size, resolved_size)
        && resolved + 0.000_1 < configured
    {
        return true;
    }
    if let (Some(minimum), Some(resolved)) =
        (min_model_size_b.filter(|value| *value > 0.0), resolved_size)
        && resolved + 0.000_1 < minimum
    {
        return true;
    }
    false
}

fn flatten_prompt_text(messages: &[crate::models::ChatMessage], system: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(system) = system {
        let system = system.trim();
        if !system.is_empty() {
            parts.push(system.to_string());
        }
    }
    for message in messages {
        let text = message.flattened_text();
        if !text.trim().is_empty() {
            parts.push(text);
        }
    }
    parts.join("\n")
}

fn candidate_scope_from_base_url(base_url: &str) -> Option<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = reqwest::Url::parse(with_scheme.as_str()).ok()?;
    let host = parsed.host_str()?;
    let mut scope = host.to_ascii_lowercase();
    if let Some(port) = parsed.port_or_known_default() {
        scope.push('_');
        scope.push_str(port.to_string().as_str());
    }
    let path = parsed.path().trim_matches('/');
    if !path.is_empty() {
        scope.push('_');
        scope.push_str(path);
    }
    Some(sanitize_candidate_scope(scope.as_str(), "endpoint"))
}

fn sanitize_candidate_scope(value: &str, fallback: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    let collapsed = out
        .split('_')
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if collapsed.is_empty() {
        fallback.to_string()
    } else {
        collapsed
    }
}

fn normalize_key(value: &str, fallback: &str) -> String {
    let cleaned = value.trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        fallback.to_string()
    } else {
        cleaned
    }
}

fn sorted_strings<T>(values: T) -> Vec<String>
where
    T: IntoIterator<Item = String>,
{
    let mut items = values.into_iter().collect::<Vec<_>>();
    items.sort();
    items
}

fn should_include_configured_candidates(
    include_configured: bool,
    request: &CompletionRequest,
    has_request_candidates: bool,
) -> bool {
    if include_configured {
        return true;
    }
    if !has_request_candidates {
        return true;
    }
    request.preferred_provider.is_some()
}

fn allow_unconfigured_ollama_request_models() -> bool {
    env_bool_any(
        &[
            "GAIL_ALLOW_UNCONFIGURED_OLLAMA_REQUEST_MODELS",
            "GAIL_ALLOW_UNCONFIGURED_OLLAMA_REQUEST_MODEL",
            "REFINER_AI_ALLOW_UNCONFIGURED_OLLAMA_REQUEST_MODELS",
        ],
        false,
    )
}

fn request_candidate_model_allowed(
    config: &GailConfig,
    provider: &str,
    model: Option<&str>,
) -> bool {
    request_candidate_model_allowed_with_policy(
        config,
        provider,
        model,
        allow_unconfigured_ollama_request_models(),
    )
}

fn request_candidate_model_allowed_with_policy(
    config: &GailConfig,
    provider: &str,
    model: Option<&str>,
    allow_unconfigured_ollama_models: bool,
) -> bool {
    if allow_unconfigured_ollama_models {
        return true;
    }
    if normalize_provider_type(provider) != "ollama" {
        return true;
    }

    let Some(requested_model) = model
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("default"))
    else {
        return true;
    };
    let requested_model = requested_model
        .strip_prefix("ollama/")
        .unwrap_or(requested_model)
        .to_ascii_lowercase();

    let configured_ollama_models = config
        .providers
        .iter()
        .filter(|profile| normalize_provider_type(profile.provider_type.as_str()) == "ollama")
        .filter_map(|profile| profile.model.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .collect::<HashSet<_>>();

    if configured_ollama_models.is_empty() {
        return true;
    }

    configured_ollama_models.contains(&requested_model)
}

fn should_return_degraded_fallback(
    request: &CompletionRequest,
    include_configured: bool,
    workflow: &str,
    role: &str,
    expected_json: bool,
    task_tags: &HashSet<String>,
    prompt_text: &str,
) -> bool {
    if env_bool_any(
        &[
            "GAIL_DISABLE_DEGRADED_FALLBACK",
            "REFINER_AI_DISABLE_DEGRADED_FALLBACK",
        ],
        false,
    ) {
        return false;
    }
    if env_bool_any(
        &[
            "GAIL_ALWAYS_DEGRADED_FALLBACK",
            "REFINER_AI_ALWAYS_DEGRADED_FALLBACK",
        ],
        false,
    ) {
        return true;
    }
    if request.preferred_provider.is_some() && !include_configured {
        return false;
    }
    if expected_json {
        return true;
    }
    if is_interactive_workflow(workflow, role) {
        return false;
    }
    if request.preferred_provider.is_none() {
        return true;
    }
    text_or_tags_indicate_automation(workflow, role, task_tags, prompt_text)
}

fn text_or_tags_indicate_automation(
    workflow: &str,
    role: &str,
    task_tags: &HashSet<String>,
    prompt_text: &str,
) -> bool {
    let haystack = format!(
        "{} {} {} {}",
        workflow,
        role,
        task_tags.iter().cloned().collect::<Vec<_>>().join(" "),
        prompt_text
    )
    .to_ascii_lowercase();
    [
        "agent",
        "aiindex",
        "automation",
        "code",
        "crypto",
        "evaluator",
        "json",
        "manager",
        "octobot",
        "planner",
        "planning",
        "portfolio",
        "rebalance",
        "refiner",
        "research",
        "researcher",
        "review",
        "reviewer",
        "signal",
        "structured_data",
        "strategy",
        "technicalanalysis",
        "tool",
        "trade",
        "trading",
    ]
    .iter()
    .any(|term| haystack.contains(term))
}

fn degraded_candidate_summary(role: &str) -> CandidateSummary {
    CandidateSummary {
        candidate_id: "gail/degraded_safety".to_string(),
        provider: "gail".to_string(),
        model: "degraded_safety".to_string(),
        configured_model: "degraded_safety".to_string(),
        resolved_model: "degraded_safety".to_string(),
        source: "internal_degraded_policy".to_string(),
        specialties: vec!["fallback".to_string(), "safety".to_string()],
        roles: vec![role.to_string()],
    }
}

fn invocation_summaries_from_results(
    results: &[InvocationResult],
) -> Vec<CandidateInvocationSummary> {
    results
        .iter()
        .map(|result| CandidateInvocationSummary {
            summary: result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str())),
            latency_ms: result.latency_ms,
            quality: result.quality,
            score: result.score,
            status: if result.response.is_some() {
                "ok".to_string()
            } else {
                "error".to_string()
            },
            error: result.error.clone(),
        })
        .collect()
}

fn ranked_candidate_summaries(candidates: &[RankedCandidate]) -> Vec<CandidateInvocationSummary> {
    candidates
        .iter()
        .map(|candidate| CandidateInvocationSummary {
            summary: candidate.candidate.summary(None),
            latency_ms: None,
            quality: -1.0,
            score: candidate.score,
            status: "skipped_backoff".to_string(),
            error: candidate
                .health_mode
                .as_ref()
                .map(|mode| format!("provider health backoff: {mode}")),
        })
        .collect()
}

fn degraded_fallback_text(
    expected_json: bool,
    workflow: &str,
    role: &str,
    prompt_text: &str,
    failures: &[String],
) -> String {
    let reason = failures
        .last()
        .map(|value| value.as_str())
        .unwrap_or("all provider candidates failed");
    if expected_json {
        let payload = if prompt_requests_manager_tool_call(prompt_text) {
            json!({
                "tool_name": "finish",
                "arguments": {
                    "status": "degraded",
                    "decision": "hold",
                    "action": "hold",
                    "should_trade": false,
                    "reason": reason,
                }
            })
        } else if prompt_requests_execution_plan(prompt_text) {
            json!({
                "steps": []
            })
        } else {
            json!({
                "status": "degraded",
                "decision": "hold",
                "action": "hold",
                "signal": "neutral",
                "confidence": 0.0,
                "should_trade": false,
                "orders": [],
                "trades": [],
                "risk": "provider_unavailable",
                "reason": reason,
            })
        };
        return payload.to_string();
    }
    if text_or_tags_indicate_automation(workflow, role, &HashSet::new(), prompt_text) {
        return format!(
            "HOLD / NO_TRADE: Gail detected that every configured AI provider is unavailable or in adaptive backoff. Reason: {reason}. Do not open new positions until provider health recovers."
        );
    }
    format!(
        "Gail degraded fallback: every configured AI provider is unavailable or in adaptive backoff. Reason: {reason}."
    )
}

fn prompt_requests_execution_plan(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    lowered.contains("executionplan")
        && lowered.contains("steps")
        && lowered.contains("additionalproperties")
}

fn prompt_requests_manager_tool_call(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    lowered.contains("tool_name")
        && lowered.contains("arguments")
        && (lowered.contains("manager") || lowered.contains("tool"))
}

fn classify_workload(workflow: &str, role: &str) -> WorkloadClass {
    if is_interactive_workflow(workflow, role) {
        return WorkloadClass::Interactive;
    }
    let workflow = workflow.to_ascii_lowercase();
    let role = role.to_ascii_lowercase();
    if workflow.contains("solver")
        || workflow.contains("refiner")
        || workflow.contains("conductor")
        || workflow.contains("automation")
        || workflow.contains("batch")
        || matches!(role.as_str(), "planner" | "reviewer" | "researcher")
    {
        WorkloadClass::Solver
    } else {
        WorkloadClass::Interactive
    }
}

fn is_interactive_workflow(workflow: &str, role: &str) -> bool {
    let workflow = workflow.to_ascii_lowercase();
    let role = role.to_ascii_lowercase();
    role == "assistant"
        || workflow.starts_with("assistant_")
        || workflow.starts_with("direct")
        || workflow.starts_with("ui_")
        || workflow.starts_with("playground")
        || workflow.contains("chat")
}

fn parse_bool(value: &str, default: bool) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

fn env_bool_any(names: &[&str], default: bool) -> bool {
    for name in names {
        if let Ok(value) = env::var(name) {
            return parse_bool(&value, default);
        }
    }
    default
}

fn env_string_any(names: &[&str]) -> Option<String> {
    for name in names {
        if let Ok(value) = env::var(name) {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            if matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "none" | "null" | "nil" | "undefined"
            ) {
                continue;
            }
            return Some(trimmed.trim_end_matches('/').to_string());
        }
    }
    None
}

fn env_int_any(names: &[&str], default: u64) -> u64 {
    for name in names {
        if let Ok(value) = env::var(name)
            && let Ok(parsed) = value.trim().parse::<u64>()
        {
            return parsed;
        }
    }
    default
}

fn env_float_any(names: &[&str], default: f64) -> f64 {
    for name in names {
        if let Ok(value) = env::var(name)
            && let Ok(parsed) = value.trim().parse::<f64>()
        {
            return parsed;
        }
    }
    default
}

fn preview_labels(mut labels: Vec<String>, limit: usize) -> String {
    labels.retain(|item| !item.trim().is_empty());
    if labels.is_empty() {
        return "none".to_string();
    }
    let preview = labels.into_iter().take(limit.max(1)).collect::<Vec<_>>();
    preview.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::ProviderProfile,
        models::{ChatMessage, MessageContent},
    };

    #[test]
    fn quality_score_prefers_valid_json() {
        assert!(quality_score("{\"ok\":true}", true) > quality_score("not json", true));
    }

    #[test]
    fn workflow_tags_include_keyword_and_profile_tags() {
        let tags = workflow_tags(
            "assistant_requirements",
            "assistant",
            "Need JSON schema for a reading quiz",
        );
        assert!(tags.contains("json"));
        assert!(tags.contains("requirements"));
    }

    #[test]
    fn expected_json_detects_schema_prompt() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("Return only valid JSON with keys: summary".to_string()),
        }];
        assert!(expected_json(&messages, None));
    }

    #[test]
    fn degraded_fallback_matches_execution_plan_schema() {
        let prompt = r#"{"name":"ExecutionPlan","schema":{"type":"object","properties":{"steps":{"type":"array"}},"required":["steps"],"additionalProperties":false}}"#;
        let text = degraded_fallback_text(
            true,
            "trading",
            "planner",
            prompt,
            &["provider unavailable".to_string()],
        );
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        assert_eq!(value, serde_json::json!({ "steps": [] }));
    }

    #[test]
    fn request_timeout_with_cap_respects_lower_bound() {
        assert_eq!(request_timeout_with_cap(Some(180), Some(45)), Some(45));
        assert_eq!(request_timeout_with_cap(Some(30), Some(45)), Some(30));
        assert_eq!(request_timeout_with_cap(None, Some(45)), Some(45));
    }

    #[test]
    fn provider_profile_is_usable_rejects_none_markers() {
        let nvidia = ProviderProfile {
            provider_type: "nvidia".to_string(),
            api_key: Some("nvapi-test".to_string()),
            ..ProviderProfile::default()
        };
        assert!(!has_usable_value(Some("None")));
        assert!(!has_usable_value(Some("null")));
        assert!(provider_profile_is_usable(&nvidia));
    }

    #[test]
    fn select_ranked_candidates_prefers_healthy_diverse_providers() {
        fn ranked(
            provider_type: &str,
            model: &str,
            score: f64,
            health_ok: bool,
        ) -> RankedCandidate {
            RankedCandidate {
                score,
                health_ok,
                health_mode: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: format!("{provider_type}-{model}"),
                    provider_type: provider_type.to_string(),
                    model: Some(model.to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some("http://example.internal".to_string()),
                    ..ProviderProfile::default()
                }),
            }
        }

        let selected = select_ranked_candidates(
            vec![
                ranked("nvidia", "moonshotai/kimi-k2-instruct-0905", 5.0, true),
                ranked("nvidia", "minimaxai/minimax-m2.7", 4.9, true),
                ranked("ollama", "llama3.2", 4.0, true),
                ranked("openai", "gpt-4o-mini", 6.0, false),
            ],
            3,
        );

        let labels = selected
            .iter()
            .map(|candidate| candidate.candidate_id())
            .collect::<Vec<_>>();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "nvidia/moonshotai/kimi-k2-instruct-0905");
        assert!(labels[1].starts_with("ollama/llama3.2"));
        assert_eq!(labels[2], "openai/gpt-4o-mini");
    }

    #[test]
    fn select_ranked_candidates_uses_fallback_family_before_duplicate_provider() {
        fn ranked(
            provider_type: &str,
            model: &str,
            score: f64,
            health_ok: bool,
        ) -> RankedCandidate {
            RankedCandidate {
                score,
                health_ok,
                health_mode: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: format!("{provider_type}-{model}"),
                    provider_type: provider_type.to_string(),
                    model: Some(model.to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some("http://example.internal".to_string()),
                    ..ProviderProfile::default()
                }),
            }
        }

        let selected = select_ranked_candidates(
            vec![
                ranked("nvidia", "moonshotai/kimi-k2-instruct-0905", 5.0, true),
                ranked("nvidia", "minimaxai/minimax-m2.7", 4.9, true),
                ranked("ollama", "llama3.2", 2.0, false),
            ],
            2,
        );

        let labels = selected
            .iter()
            .map(|candidate| candidate.candidate_id())
            .collect::<Vec<_>>();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], "nvidia/moonshotai/kimi-k2-instruct-0905");
        assert!(labels[1].starts_with("ollama/llama3.2"));
    }

    #[test]
    fn select_ranked_candidates_reserves_ollama_fallback_slot() {
        fn ranked(
            provider_type: &str,
            model: &str,
            score: f64,
            health_ok: bool,
        ) -> RankedCandidate {
            RankedCandidate {
                score,
                health_ok,
                health_mode: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: format!("{provider_type}-{model}"),
                    provider_type: provider_type.to_string(),
                    model: Some(model.to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some("http://example.internal".to_string()),
                    specialties: if provider_type == "ollama" {
                        vec!["local".to_string()]
                    } else {
                        Vec::new()
                    },
                    ..ProviderProfile::default()
                }),
            }
        }

        let selected = select_ranked_candidates(
            vec![
                ranked("openai", "gpt-4o-mini", 9.0, true),
                ranked("nvidia", "moonshotai/kimi-k2-instruct-0905", 8.0, true),
                ranked("gemini", "gemini-2.5-flash", 7.0, true),
                ranked("ollama", "llama3.2", 1.0, false),
            ],
            3,
        );

        let labels = selected
            .iter()
            .map(|candidate| candidate.candidate_id())
            .collect::<Vec<_>>();
        assert!(
            labels
                .iter()
                .any(|label| label.starts_with("ollama/llama3.2"))
        );
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn candidate_id_scopes_ollama_endpoints() {
        let first = ProviderCandidate::from_profile(ProviderProfile {
            name: "ollama-openai-compat".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("qwen2.5-coder:1.5b".to_string()),
            base_url: Some(
                "http://ollama-openai-compat.ollama.svc.cluster.local:11434".to_string(),
            ),
            ..ProviderProfile::default()
        });
        let second = ProviderCandidate::from_profile(ProviderProfile {
            name: "ollama-native".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("qwen2.5-coder:1.5b".to_string()),
            base_url: Some("http://ollama.ollama.svc.cluster.local:11434".to_string()),
            ..ProviderProfile::default()
        });
        let nvidia = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "nvidia".to_string(),
            model: Some("minimaxai/minimax-m2.7".to_string()),
            ..ProviderProfile::default()
        });

        assert_ne!(first.candidate_id(), second.candidate_id());
        assert!(
            first
                .candidate_id()
                .starts_with("ollama/qwen2.5-coder:1.5b@")
        );
        assert!(
            second
                .candidate_id()
                .starts_with("ollama/qwen2.5-coder:1.5b@")
        );
        assert_eq!(nvidia.candidate_id(), "nvidia/minimaxai/minimax-m2.7");
    }

    #[test]
    fn ranked_candidate_quota_backoff_detects_cached_quota_health() {
        let candidate = RankedCandidate {
            score: 1.0,
            health_ok: false,
            health_mode: Some("quota".to_string()),
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                provider_type: "nvidia".to_string(),
                model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
                api_key: Some("token".to_string()),
                ..ProviderProfile::default()
            }),
        };
        assert!(ranked_candidate_is_in_quota_backoff(&candidate));
        assert!(ranked_candidate_is_in_provider_backoff(&candidate));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_nested_429_as_quota() {
        let bucket = runtime_failure_health_bucket(
            Some(r#"nvidia upstream error: {"status":429,"title":"Too Many Requests"}"#),
            Some(12),
        );
        assert_eq!(bucket.mode.as_deref(), Some("quota"));
        assert_eq!(bucket.ok, Some(false));
        assert_eq!(bucket.latency_ms, Some(12));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_502_as_upstream_backoff() {
        let message = "nvidia upstream error: error sending request for url (https://integrate.api.nvidia.com/v1/chat/completions)";
        let bucket = runtime_failure_health_bucket(Some(message), Some(34));
        assert_eq!(bucket.mode.as_deref(), Some("upstream"));
        assert!(message_indicates_provider_backoff(message));
    }

    #[test]
    fn orchestration_failure_status_maps_adaptive_backoff_to_503() {
        let message = "all suitable providers are currently in adaptive backoff; retry after the recorded mitigation window";
        assert_eq!(
            orchestration_failure_status(message),
            Some(StatusCode::SERVICE_UNAVAILABLE)
        );
    }

    #[test]
    fn orchestration_failure_status_keeps_transient_upstream_as_502() {
        let message = "nvidia upstream error: error sending request for url (https://integrate.api.nvidia.com/v1/chat/completions)";
        assert_eq!(
            orchestration_failure_status(message),
            Some(StatusCode::BAD_GATEWAY)
        );
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_ollama_saturation_as_local_backoff() {
        let message = "ollama upstream error: local Ollama request queue is saturated; backing off before retrying in 120s";
        let bucket = runtime_failure_health_bucket(Some(message), Some(2000));
        assert_eq!(bucket.mode.as_deref(), Some("ollama_saturated"));
        assert!(message_indicates_provider_backoff(message));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_model_retirement_as_missing_endpoint() {
        let message = r#"nvidia upstream error: {"detail":"The model 'deepseek-ai/deepseek-v3.2' has reached its end of life on 2026-05-04T00:00:00Z and is no longer available.","status":410,"title":"Gone"}"#;
        let bucket = runtime_failure_health_bucket(Some(message), Some(19));
        assert_eq!(bucket.mode.as_deref(), Some("missing_endpoint"));
        assert!(!message_indicates_transient_provider_failure(message));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_auth_failure_as_unconfigured() {
        let message = r#"nvidia upstream error: {"detail":"Authentication failed","status":401,"title":"Unauthorized"}"#;
        let bucket = runtime_failure_health_bucket(Some(message), Some(21));
        assert_eq!(bucket.mode.as_deref(), Some("unconfigured"));
        assert!(message_indicates_provider_backoff(message));
    }

    #[test]
    fn host_budget_ratio_and_overload_detection_work_for_shared_host() {
        let candidate = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            host_group: Some("host-a".to_string()),
            host_cpu_budget: Some(16.0),
            host_ram_budget_mb: Some(65_536),
            host_vram_budget_mb: Some(24_576),
            ..ProviderProfile::default()
        });
        let safe = HostLoad {
            requests: 2,
            cpu: 10.0,
            ram_mb: 32_768,
            vram_mb: 12_000,
        };
        assert!(!host_budget_exceeded(&candidate, &safe));
        let overloaded = HostLoad {
            requests: 4,
            cpu: 18.0,
            ram_mb: 72_000,
            vram_mb: 20_000,
        };
        assert!(host_budget_ratio(&candidate, &overloaded) > 1.0);
        assert!(host_budget_exceeded(&candidate, &overloaded));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_resource_saturation() {
        let message =
            "candidate skipped because configured concurrency/resource budget is exhausted";
        let bucket = runtime_failure_health_bucket(Some(message), Some(5));
        assert_eq!(bucket.mode.as_deref(), Some("resource_saturated"));
        assert!(message_indicates_provider_backoff(message));
    }

    #[test]
    fn runtime_failure_health_bucket_classifies_nmc_constrained() {
        let message = "candidate skipped because NMC/Tracey telemetry reports constrained capacity (agent=tracey-1, host=node-a, status=healthy, mode=constrained, optimize_status=avoid, pressure_ratio=1.25)";
        let bucket = runtime_failure_health_bucket(Some(message), Some(7));
        assert_eq!(bucket.mode.as_deref(), Some("nmc_constrained"));
        assert!(message_indicates_provider_backoff(message));
    }

    #[test]
    fn cached_health_ttl_keeps_ollama_transient_failures_short_lived() {
        let default_ttl = 1800.0;
        let timeout_ttl = cached_health_ttl_seconds(true, Some("timeout"), default_ttl);
        let upstream_ttl = cached_health_ttl_seconds(true, Some("upstream"), default_ttl);
        let saturation_ttl = cached_health_ttl_seconds(true, Some("ollama_saturated"), default_ttl);

        assert!(timeout_ttl >= 1.0 && timeout_ttl <= 120.0);
        assert!(upstream_ttl >= 1.0 && upstream_ttl <= 120.0);
        assert!(saturation_ttl >= 1.0 && saturation_ttl <= 120.0);
        assert_eq!(
            cached_health_ttl_seconds(false, Some("timeout"), default_ttl),
            default_ttl
        );
    }

    #[test]
    fn provider_family_backoff_does_not_throttle_all_ollama_endpoints_on_saturation() {
        let ollama = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "ollama".to_string(),
            model: Some("qwen2.5-coder:1.5b".to_string()),
            base_url: Some(
                "http://ollama-openai-compat.ollama.svc.cluster.local:11434".to_string(),
            ),
            ..ProviderProfile::default()
        });
        let nvidia = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "nvidia".to_string(),
            model: Some("minimaxai/minimax-m2.7".to_string()),
            ..ProviderProfile::default()
        });
        let saturation = "ollama upstream error: local Ollama request queue is saturated; backing off before retrying in 90s";
        let upstream = "nvidia upstream error: error sending request for url (https://integrate.api.nvidia.com/v1/chat/completions)";

        assert!(!error_should_backoff_provider_family(&ollama, saturation));
        assert!(error_should_backoff_provider_family(&nvidia, upstream));
    }

    #[test]
    fn classify_workload_prefers_solver_for_project_solver_workflows() {
        assert_eq!(
            classify_workload("project_solver", "planner"),
            WorkloadClass::Solver
        );
        assert_eq!(
            classify_workload("direct", "assistant"),
            WorkloadClass::Interactive
        );
    }

    #[test]
    fn configured_candidates_are_included_for_preferred_provider_fallback() {
        let request = CompletionRequest {
            workflow: Some("project_solver".to_string()),
            role: Some("planner".to_string()),
            preferred_provider: Some("openai".to_string()),
            preferred_model: Some("gpt-4o-mini".to_string()),
            preferred_api_key: None,
            preferred_access_token: None,
            fallback_provider: None,
            fallback_model: None,
            fallback_api_key: None,
            fallback_access_token: None,
            base_url: None,
            include_configured: Some(false),
            selection_mode: None,
            max_candidates: None,
            messages: Vec::new(),
            system: None,
            max_tokens: None,
            temperature: None,
            timeout_seconds: None,
            reasoning_effort: None,
            request_category: None,
        };
        assert!(should_include_configured_candidates(false, &request, true));
    }

    #[test]
    fn configured_candidates_respect_explicit_non_preferred_request_mode() {
        let request = CompletionRequest {
            workflow: Some("direct".to_string()),
            role: Some("assistant".to_string()),
            preferred_provider: None,
            preferred_model: None,
            preferred_api_key: None,
            preferred_access_token: None,
            fallback_provider: Some("ollama".to_string()),
            fallback_model: Some("llama3.2:3b".to_string()),
            fallback_api_key: None,
            fallback_access_token: None,
            base_url: None,
            include_configured: Some(false),
            selection_mode: None,
            max_candidates: Some(1),
            messages: Vec::new(),
            system: None,
            max_tokens: None,
            temperature: None,
            timeout_seconds: None,
            reasoning_effort: None,
            request_category: None,
        };
        assert!(!should_include_configured_candidates(false, &request, true));
        assert!(should_include_configured_candidates(false, &request, false));
    }

    #[test]
    fn request_candidate_model_allowed_rejects_unconfigured_ollama_model_by_default() {
        let config = GailConfig {
            providers: vec![ProviderProfile {
                name: "ollama-native".to_string(),
                provider_type: "ollama".to_string(),
                model: Some("llama3.2:3b".to_string()),
                ..ProviderProfile::default()
            }],
            ..GailConfig::default()
        };
        assert!(!request_candidate_model_allowed_with_policy(
            &config,
            "ollama",
            Some("qwen2.5-coder:1.5b"),
            false,
        ));
        assert!(request_candidate_model_allowed_with_policy(
            &config,
            "ollama",
            Some("ollama/llama3.2:3b"),
            false,
        ));
    }

    #[test]
    fn request_candidate_model_allowed_can_permit_unconfigured_ollama_model() {
        let config = GailConfig {
            providers: vec![ProviderProfile {
                name: "ollama-native".to_string(),
                provider_type: "ollama".to_string(),
                model: Some("llama3.2:3b".to_string()),
                ..ProviderProfile::default()
            }],
            ..GailConfig::default()
        };
        assert!(request_candidate_model_allowed_with_policy(
            &config,
            "ollama",
            Some("qwen2.5-coder:1.5b"),
            true,
        ));
    }

    #[test]
    fn request_candidate_model_allowed_keeps_non_ollama_requests() {
        let config = GailConfig::default();
        assert!(request_candidate_model_allowed_with_policy(
            &config,
            "openai",
            Some("gpt-4o-mini"),
            false,
        ));
    }

    #[test]
    fn strict_model_policy_rejects_downgrade_and_floor_violations() {
        assert!(violates_strict_model_policy(
            true,
            Some(1.5),
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:0.5b"
        ));
        assert!(violates_strict_model_policy(
            true,
            Some(7.0),
            "llama3.2:3b",
            "llama3.2:3b"
        ));
        assert!(!violates_strict_model_policy(
            true,
            Some(1.5),
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b"
        ));
    }
}
