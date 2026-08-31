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
use sha2::{Digest, Sha256};
use tokio::{
    sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, oneshot},
    task::JoinSet,
    time::{Instant, sleep},
};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    aarnn_bridge::{AarnnMirrorClient, AarnnMirrorExchange},
    adaptive_schema, aer, api_issues,
    config::{
        ApiTokenConfig, AuditLoggingConfig, GailConfig, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS,
        ProviderProfile,
    },
    errors::{GailError, Result, message_indicates_quota},
    hardware::{detect_hardware, log_hardware_profile},
    llm_ledger::{LlmLedger, LlmLedgerRecord},
    metrics::{CandidateMetricsSummary, HealthBucket, LocalUsageTelemetry, MetricsStore},
    models::{
        AarnnMirrorDirection, AerDecodeRequest, AerDecodeResponse, AerEncodeRequest,
        AerEncodeResponse, AuthContext, CandidateInvocationSummary, CandidateSummary,
        CompletionRequest, CompletionResponse, CompletionTrace, HealthResponse,
        NeuromorphicAnalyzeRequest, NeuromorphicPredictRequest, NeuromorphicPredictResponse,
        ProviderCompletionRequest, ReadinessResponse, SelectionMode, SpecialistAnalysisResponse,
        TranscriptionResponse,
    },
    nmc_telemetry::{NmcAgentSignal, NmcTelemetryClient},
    prompt_budget::{PromptCompactionReport, compact_provider_request},
    provider_admission::{
        admission_endpoint_matches, admission_model_matches, admitted_for_kind, admitted_for_model,
    },
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

const DEFAULT_PROVIDER_HEALTH_TIMEOUT_SECONDS: u64 = 8;
const DEFAULT_READINESS_CACHE_TTL_SECONDS: u64 = 15;

/// Native llama.cpp readiness includes a real completion request.  A fixed
/// four-second deadline is too short for the CPU-only 4B endpoint while it is
/// loading or draining a previous request, so make the deadline configurable
/// and keep a bounded safe default for deployments that do not set it.
fn provider_health_timeout_seconds() -> u64 {
    env_int_any(
        &[
            "GAIL_PROVIDER_HEALTH_TIMEOUT_SECONDS",
            "REFINER_AI_PROVIDER_HEALTH_TIMEOUT_SECONDS",
        ],
        DEFAULT_PROVIDER_HEALTH_TIMEOUT_SECONDS,
    )
    .clamp(4, 60)
}

/// `/readyz` is called by both Kubernetes and external monitors.  Provider
/// readiness includes a real completion probe for native llama.cpp endpoints,
/// so it must not run inline for every kubelet request.  Keep the cache short
/// enough to notice a reboot or a failed model, while allowing the probe to
/// complete without making the Gail Service flap between ready and not-ready.
fn readiness_cache_ttl() -> Duration {
    Duration::from_secs(
        env_int_any(
            &[
                "GAIL_READINESS_CACHE_TTL_SECONDS",
                "REFINER_AI_READINESS_CACHE_TTL_SECONDS",
            ],
            DEFAULT_READINESS_CACHE_TTL_SECONDS,
        )
        .clamp(1, 120),
    )
}

fn completion_metric_source(response: &CompletionResponse) -> &'static str {
    if response
        .trace
        .as_ref()
        .is_some_and(|trace| trace.final_source.eq_ignore_ascii_case("aarnn"))
    {
        "snn"
    } else {
        "llm"
    }
}

fn completion_metric_success(response: &CompletionResponse) -> bool {
    response
        .trace
        .as_ref()
        .is_none_or(|trace| !trace.final_source.eq_ignore_ascii_case("degraded_policy"))
}

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
    load_released: Arc<Notify>,
    round_robin_cursors: Arc<Mutex<HashMap<String, usize>>>,
    interactive_pool: Arc<Semaphore>,
    solver_pool: Arc<Semaphore>,
    trading_pool: Arc<Semaphore>,
    postgres_dsn: Option<String>,
    readiness_cache: ReadinessCache,
}

#[derive(Default)]
struct ReadinessCache {
    state: Mutex<ReadinessCacheState>,
    refresh_finished: Notify,
}

#[derive(Default)]
struct ReadinessCacheState {
    value: Option<CachedReadiness>,
    refresh_in_progress: bool,
}

struct CachedReadiness {
    response: ReadinessResponse,
    refreshed_at: Instant,
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
    queue_wait_ms: Option<u64>,
    quality: f64,
    score: f64,
}

#[derive(Clone, Debug, Serialize)]
struct EndpointTelemetryRow {
    candidate_id: String,
    provider: Option<String>,
    configured_model: Option<String>,
    resolved_model: Option<String>,
    endpoint_scope: String,
    endpoint_host: Option<String>,
    endpoint_port: Option<u16>,
    endpoint_suffix: Option<String>,
    successes: u64,
    failures: u64,
    total: u64,
    success_rate: Option<f64>,
    ewma_latency_ms: Option<f64>,
    ewma_queue_wait_ms: Option<f64>,
    ewma_inference_ms: Option<f64>,
    last_status: Option<String>,
    last_error: Option<String>,
    updated_at: Option<f64>,
}

#[derive(Clone, Debug)]
struct RankedCandidate {
    candidate: ProviderCandidate,
    score: f64,
    health_ok: bool,
    health_mode: Option<String>,
    generation_tokens_per_second: Option<f64>,
}

#[derive(Default)]
struct LoadTracker {
    candidate_in_flight: HashMap<String, usize>,
    candidate_waiting: HashMap<String, usize>,
    host_usage: HashMap<String, HostLoad>,
}

struct CandidateWaitingGuard {
    service: GailService,
    candidate_id: Option<String>,
}

impl CandidateWaitingGuard {
    fn new(service: GailService, candidate_id: String) -> Self {
        Self {
            service,
            candidate_id: Some(candidate_id),
        }
    }

    async fn release(mut self) {
        if let Some(candidate_id) = self.candidate_id.take() {
            self.service.release_candidate_waiting(candidate_id).await;
        }
    }
}

impl Drop for CandidateWaitingGuard {
    fn drop(&mut self) {
        let Some(candidate_id) = self.candidate_id.take() else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let service = self.service.clone();
            handle.spawn(async move {
                service.release_candidate_waiting(candidate_id).await;
            });
        }
    }
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
    candidate_in_flight: usize,
    candidate_waiting: usize,
    candidate_limit_ratio: f64,
    candidate_limit_reached: bool,
    host_budget_ratio: f64,
    host_budget_reached: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateDispatchEstimate {
    samples: u64,
    useful_rate: f64,
    service_time_ms: f64,
    queue_depth: usize,
    candidate_parallelism: usize,
    estimated_completion_ms: f64,
    estimated_useful_completion_ms: f64,
}

#[derive(Clone, Debug)]
struct LoadReservation {
    candidate_id: String,
    host_group: Option<String>,
    resource_cost_cpu: f64,
    resource_cost_ram_mb: u64,
    resource_cost_vram_mb: u64,
}

struct LoadReservationGuard {
    service: GailService,
    reservation: Option<LoadReservation>,
}

impl LoadReservationGuard {
    fn new(service: GailService, reservation: LoadReservation) -> Self {
        Self {
            service,
            reservation: Some(reservation),
        }
    }

    async fn release(mut self) {
        if let Some(reservation) = self.reservation.take() {
            self.service.release_candidate_load(reservation).await;
        }
    }
}

impl Drop for LoadReservationGuard {
    fn drop(&mut self) {
        let Some(reservation) = self.reservation.take() else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let service = self.service.clone();
            handle.spawn(async move {
                service.release_candidate_load(reservation).await;
            });
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkloadClass {
    Interactive,
    Solver,
    Trading,
}

impl WorkloadClass {
    fn label(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Solver => "solver",
            Self::Trading => "trading",
        }
    }
}

#[derive(Clone, Debug)]
struct RoundRobinContext {
    provider_key: String,
    model_key: String,
    key: String,
    group_size: usize,
}

impl GailService {
    pub fn metrics(&self) -> MetricsStore {
        self.inner.metrics.clone()
    }
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
        // LlmLedger::from_config performs the shared schema migration when
        // the ledger is enabled.  Do not run the same DDL a second time here:
        // on a fresh rollout that duplicated migration can deadlock against
        // the mirror/trainer workers.  Comparative validation uses the same
        // tables, so it does not need a separate migration.
        let metrics = MetricsStore::new(config.storage.metrics_path.clone()).await?;
        let specialists = build_specialist_engines(&config, client.clone());
        let aarnn_bridge = AarnnMirrorClient::from_config(&config, client.clone(), &specialists);
        let nmc_telemetry = NmcTelemetryClient::from_config(&config, client.clone());
        let load_tracker = Arc::new(Mutex::new(LoadTracker::default()));
        let load_released = Arc::new(Notify::new());
        let round_robin_cursors = Arc::new(Mutex::new(HashMap::new()));
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
        let postgres_dsn = config.storage.postgres_dsn.clone();
        let trading_pool = Arc::new(Semaphore::new(
            env_int_any(
                &["GAIL_TRADING_POOL_MAX_IN_FLIGHT"],
                config.orchestration.trading_pool_max_in_flight as u64,
            )
            .max(1) as usize,
        ));
        tracing::info!(
            interactive_pool_size = interactive_pool.available_permits(),
            solver_pool_size = solver_pool.available_permits(),
            trading_pool_size = trading_pool.available_permits(),
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
                load_released: load_released.clone(),
                round_robin_cursors: round_robin_cursors.clone(),
                interactive_pool: interactive_pool.clone(),
                solver_pool: solver_pool.clone(),
                trading_pool: trading_pool.clone(),
                postgres_dsn: postgres_dsn.clone(),
                readiness_cache: ReadinessCache::default(),
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
                load_released,
                round_robin_cursors,
                interactive_pool,
                solver_pool,
                trading_pool,
                postgres_dsn,
                readiness_cache: ReadinessCache::default(),
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

    /// Resolve a direct request back to its configured profile when possible.
    /// This reuses host budgets, endpoint identity, context limits and telemetry
    /// metadata instead of silently discarding them at the direct API boundary.
    fn direct_provider_profile(&self, request: &ProviderCompletionRequest) -> ProviderProfile {
        let requested_provider = normalize_provider_type(&request.provider);
        let requested_model = request.model.as_deref().unwrap_or_default().trim();
        let requested_base = request.base_url.as_deref().unwrap_or_default().trim();
        let mut profile = self
            .inner
            .config
            .providers
            .iter()
            .find(|profile| {
                normalize_provider_type(&profile.provider_type) == requested_provider
                    && profile.model.as_deref().unwrap_or_default().trim() == requested_model
                    && profile.base_url.as_deref().unwrap_or_default().trim() == requested_base
            })
            .cloned()
            .unwrap_or_default();
        profile.name = if profile.name.trim().is_empty() {
            request.provider.clone()
        } else {
            profile.name
        };
        profile.provider_type = request.provider.clone();
        profile.model = request.model.clone().or(profile.model);
        profile.api_key = request.api_key.clone().or(profile.api_key);
        profile.access_token = request.access_token.clone().or(profile.access_token);
        profile.base_url = request.base_url.clone().or(profile.base_url);
        // Keep the configured source when the request names a configured
        // endpoint.  Direct callers (notably trading) still need the trained
        // model admission/readiness policy; replacing this with
        // `request_direct` made a stale replica indistinguishable from a
        // normal provider at this boundary.
        if profile.source.is_none() {
            profile.source = Some("request_direct".to_string());
        }
        profile
    }

    async fn trained_candidate_is_admitted(&self, profile: &ProviderProfile) -> bool {
        if !is_trained_llamacpp_profile(profile) {
            return true;
        }
        if !self.inner.config.comparative_validation.enabled {
            return true;
        }
        let Some(dsn) = self.inner.postgres_dsn.as_deref() else {
            return false;
        };
        let Some(model_version) = active_snapshot_id_for_routing(&self.inner.config) else {
            return false;
        };
        let Some(endpoint) = profile.base_url.as_deref() else {
            return false;
        };
        let Some(model) = profile.model.as_deref() else {
            return false;
        };
        match admitted_for_model(
            dsn,
            model_version.as_str(),
            self.inner
                .config
                .comparative_validation
                .admission_ttl_seconds,
        )
        .await
        {
            Ok(admissions) => admissions.iter().any(|admission| {
                admission.kind == "trained"
                    && admission_model_matches(admission.model.as_str(), model)
                    && admission_endpoint_matches(admission.endpoint.as_str(), endpoint)
            }),
            Err(error) => {
                tracing::debug!(error = %error, "direct trained-provider admission lookup failed");
                false
            }
        }
    }

    /// Apply the provider-specific prompt budget before queueing network work.
    fn prepare_provider_request(
        &self,
        profile: &ProviderProfile,
        request: &mut ProviderCompletionRequest,
    ) -> Option<PromptCompactionReport> {
        if !env_bool_any(
            &["GAIL_PROMPT_COMPACTION_ENABLED"],
            self.inner.config.orchestration.prompt_compaction_enabled,
        ) {
            return None;
        }
        let context_window_tokens = profile.context_window_tokens.or_else(|| {
            profile_uses_local_context_default(profile).then(|| {
                let configured = self
                    .inner
                    .config
                    .orchestration
                    .default_local_context_window_tokens as u64;
                let explicit = env_int_any(
                    &[
                        "GAIL_LOCAL_CONTEXT_WINDOW_TOKENS",
                        "GAIL_OLLAMA_CONTEXT_WINDOW_TOKENS",
                        "GAIL_OLLAMA_NUM_CTX",
                    ],
                    configured,
                );
                if explicit == 0 {
                    configured as usize
                } else {
                    explicit as usize
                }
            })
        })?;
        let chars_per_token = env_int_any(
            &["GAIL_PROMPT_CHARS_PER_TOKEN"],
            self.inner.config.orchestration.prompt_chars_per_token as u64,
        ) as usize;
        let safety_margin_tokens = env_int_any(
            &["GAIL_PROMPT_SAFETY_MARGIN_TOKENS"],
            self.inner.config.orchestration.prompt_safety_margin_tokens as u64,
        ) as usize;
        let report = compact_provider_request(
            request,
            context_window_tokens,
            chars_per_token,
            safety_margin_tokens,
        );
        if let Some(report) = report.as_ref() {
            tracing::warn!(
                provider = %profile.provider_type,
                model = %profile.model.as_deref().unwrap_or("default"),
                context_window_tokens = report.context_window_tokens,
                input_budget_tokens = report.input_budget_tokens,
                estimated_tokens_before = report.estimated_tokens_before,
                estimated_tokens_after = report.estimated_tokens_after,
                omitted_messages = report.omitted_messages,
                omitted_chars = report.omitted_chars,
                "compacted oversized provider prompt before dispatch"
            );
        }
        report
    }

    /// Reuse Gail's live health/load/latency ranker for direct consumers such
    /// as trading, which otherwise select a configured endpoint statically.
    pub(crate) async fn provider_runtime_routing_score(
        &self,
        profile: ProviderProfile,
        workflow: &str,
        role: &str,
    ) -> (ProviderProfile, f64, bool) {
        let candidate = ProviderCandidate::from_profile(profile.clone());
        let mut tags = workflow_tags(workflow, role, "");
        tags.insert(normalize_key(workflow, "general"));
        let ranked = self
            .rank_candidate(candidate, "unknown", workflow, workflow, role, None, &tags)
            .await;
        (profile, ranked.score, ranked.health_ok)
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
        let prompt_text = if audit.log_llm_prompts && audit.store_llm_content {
            self.optional_audit_text(Some(record.prompt_text.as_str()))
        } else {
            None
        };
        let response_text = if audit.log_llm_responses && audit.store_llm_content {
            self.optional_audit_text(record.response_text.as_deref())
        } else {
            None
        };
        let system_prompt = if audit.log_llm_prompts && audit.store_llm_content {
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
            build: crate::build_info::current(),
        }
    }

    /// Return application readiness without allowing slow provider probes to
    /// make the Kubernetes Service flap. The first request after startup
    /// performs a real provider probe. Once a result exists, callers receive
    /// the bounded snapshot immediately; expiry starts one background refresh
    /// and callers continue to receive the last result until that refresh
    /// completes. Concurrent cold-start callers coalesce behind one probe.
    pub async fn readiness(&self) -> ReadinessResponse {
        let ttl = readiness_cache_ttl();
        loop {
            let mut state = self.inner.readiness_cache.state.lock().await;
            if let Some(cached) = state.value.as_ref() {
                let response = cached.response.clone();
                if cached.refreshed_at.elapsed() < ttl {
                    return response;
                }

                if !state.refresh_in_progress {
                    state.refresh_in_progress = true;
                    let service = self.clone();
                    tokio::spawn(async move {
                        let _ = service.refresh_readiness().await;
                    });
                }
                return response;
            }

            if state.refresh_in_progress {
                // Register before releasing the mutex so a completion cannot
                // notify between the state check and waiter registration.
                let notified = self.inner.readiness_cache.refresh_finished.notified();
                drop(state);
                notified.await;
                continue;
            }

            state.refresh_in_progress = true;
            drop(state);
            return self.refresh_readiness().await;
        }
    }

    async fn refresh_readiness(&self) -> ReadinessResponse {
        let providers = self.provider_summaries(true).await;
        let providers_checked = providers.len();
        let providers_ready = providers
            .iter()
            .filter(|provider| provider["health"]["ok"].as_bool() == Some(true))
            .count();
        let ready =
            self.inner.config.orchestration.enabled && providers_checked > 0 && providers_ready > 0;
        let reason = if !self.inner.config.orchestration.enabled {
            "orchestration_disabled".to_string()
        } else if providers_checked == 0 {
            "no_configured_providers".to_string()
        } else if providers_ready == 0 {
            "no_application_ready_providers".to_string()
        } else {
            "application_ready".to_string()
        };
        let response = ReadinessResponse {
            ready,
            service: "gail".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            build: crate::build_info::current(),
            providers_checked,
            providers_ready,
            reason,
        };

        let mut state = self.inner.readiness_cache.state.lock().await;
        state.value = Some(CachedReadiness {
            response: response.clone(),
            refreshed_at: Instant::now(),
        });
        state.refresh_in_progress = false;
        drop(state);
        self.inner.readiness_cache.refresh_finished.notify_waiters();
        response
    }

    pub async fn provider_prometheus_metrics(&self) -> String {
        let mut rendered = self.inner.metrics.prometheus_metrics().await;
        rendered.push_str(&crate::aarnn_bridge::AarnnMirrorClient::evaluation_prometheus_metrics());
        rendered
    }

    pub async fn direct_complete(
        &self,
        request: ProviderCompletionRequest,
    ) -> Result<CompletionResponse> {
        let _ = self.inner.metrics.record_request_received().await;
        let api_source = request
            .source
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let prompt_tokens_estimate =
            estimate_request_prompt_tokens(&request.messages, request.system.as_deref());
        let processing_time_estimate_ms =
            self.inner.metrics.ai_response_time_estimate_ms("all").await;
        let started = Instant::now();
        let result = self.direct_complete_inner(request).await;
        let terminal_ok = result.as_ref().is_ok_and(completion_metric_success);
        let timed_out = result
            .as_ref()
            .err()
            .map(|error| error.to_string().to_ascii_lowercase().contains("timeout"))
            .unwrap_or(false);
        let _ = self
            .inner
            .metrics
            .record_request_terminal(terminal_ok, timed_out)
            .await;
        let source = result
            .as_ref()
            .map(completion_metric_source)
            .unwrap_or("llm");
        let _ = self
            .inner
            .metrics
            .record_ai_response_time_with_prompt(
                source,
                started.elapsed().as_millis() as u64,
                terminal_ok,
                Some(prompt_tokens_estimate),
            )
            .await;
        let _ = self
            .inner
            .metrics
            .record_api_source_response_time(
                api_source.as_str(),
                started.elapsed().as_millis() as u64,
                terminal_ok,
                Some(prompt_tokens_estimate),
            )
            .await;
        result.map(|mut response| {
            response.processing_time_estimate_ms = processing_time_estimate_ms;
            response
        })
    }

    async fn direct_complete_inner(
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
        // Classify from the actual prompt before taking a workload permit. A
        // number of OpenAI-compatible callers use the generic direct/assistant
        // route even for long research or solver work; classifying only from
        // workflow/role would incorrectly consume interactive capacity.
        let prompt_text = flatten_prompt_text(
            &effective_request.messages,
            effective_request.system.as_deref(),
        );
        let workload_class = classify_workload_with_context(
            effective_request.workflow.as_deref().unwrap_or("direct"),
            effective_request.role.as_deref().unwrap_or("assistant"),
            effective_request.request_category.as_deref(),
            effective_request.request_profile.as_deref(),
            effective_request.source.as_deref(),
            Some(prompt_text.as_str()),
        );
        if effective_request.min_model_size_b.is_none() {
            effective_request.min_model_size_b = self.model_floor_b(workload_class);
        }
        if effective_request.strict_no_downgrade.is_none() {
            effective_request.strict_no_downgrade = Some(self.strict_no_downgrade());
        }
        let request_id = Uuid::new_v4().to_string();
        info!(request_id = %request_id, workflow = ?effective_request.workflow, role = ?effective_request.role, request_category = ?effective_request.request_category, source = ?effective_request.source, lifecycle = "received", "GAIL_ORCHESTRATION_LIFECYCLE");
        info!(request_id = %request_id, lifecycle = "queued", "GAIL_ORCHESTRATION_LIFECYCLE");
        let profile = self.direct_provider_profile(&effective_request);
        if !self.trained_candidate_is_admitted(&profile).await {
            return Err(GailError::upstream(
                profile.provider_type.as_str(),
                Some(StatusCode::SERVICE_UNAVAILABLE),
                format!(
                    "trained provider {}/{} is not currently admitted for the active snapshot",
                    profile.model.as_deref().unwrap_or("unknown"),
                    profile.base_url.as_deref().unwrap_or("unknown endpoint")
                ),
            ));
        }
        self.prepare_provider_request(&profile, &mut effective_request);
        if !effective_request.messages.iter().any(|message| {
            message.role.eq_ignore_ascii_case("user") && !message.flattened_text().trim().is_empty()
        }) {
            return Err(GailError::bad_request(
                "direct provider requests require at least one non-empty user message",
            ));
        }
        let expected_json = expected_json(
            &effective_request.messages,
            effective_request.system.as_deref(),
        );
        let candidate = ProviderCandidate::from_profile(profile.clone());
        let workload_permit = match self.acquire_workload_permit(workload_class).await {
            Some(permit) => permit,
            None => {
                return Err(GailError::upstream(
                    "gail",
                    Some(StatusCode::SERVICE_UNAVAILABLE),
                    format!(
                        "{} workload pool is saturated; retry after {}ms",
                        workload_class.label(),
                        self.workload_pool_wait_timeout_ms_for(workload_class)
                    ),
                ));
            }
        };
        let Some(load_reservation) = self
            .reserve_candidate_load_with_backpressure(&candidate)
            .await
        else {
            drop(workload_permit);
            return Err(GailError::upstream(
                "gail",
                Some(StatusCode::SERVICE_UNAVAILABLE),
                format!(
                    "provider/host capacity remained saturated after {}ms",
                    self.candidate_queue_wait_timeout_ms()
                ),
            ));
        };
        let load_reservation_guard = LoadReservationGuard::new(self.clone(), load_reservation);
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
        let response_result = adapter.complete(&effective_request).await;
        drop(workload_permit);
        load_reservation_guard.release().await;
        let response = match response_result {
            Ok(response) => response,
            Err(error) => {
                if error.is_timeout() {
                    let _ = self
                        .inner
                        .metrics
                        .record_orchestration_event("timeout", None)
                        .await;
                    warn!(request_id = %request_id, lifecycle = "timed_out", "GAIL_ORCHESTRATION_LIFECYCLE");
                }
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
        if response.text.trim().is_empty() {
            let _ = self
                .inner
                .metrics
                .record_orchestration_event("empty_plan", None)
                .await;
            let error = GailError::upstream(
                response.provider.as_str(),
                Some(StatusCode::BAD_GATEWAY),
                format!(
                    "empty response text from {}/{}",
                    response.provider, response.model
                ),
            );
            api_issues::observe_provider_failure(
                candidate.provider_type.as_str(),
                candidate.configured_model.as_str(),
                "direct",
                "assistant",
                "empty_response",
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
                provider_resolved: Some(response.provider.clone()),
                model_resolved: Some(response.model.clone()),
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
                latency_ms: Some(response.latency_ms),
                usage: response
                    .usage
                    .as_ref()
                    .and_then(|value| serde_json::to_value(value).ok()),
                raw: response.raw.clone(),
                metadata: Some(json!({
                    "source": "direct_complete",
                    "reason": "empty_response",
                })),
                created_ts: current_ts(),
            })
            .await;
            return Err(error);
        }
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
        let aarnn_evaluation =
            self.aarnn_bridge()
                .zip(mirror_output.as_ref())
                .map(|(bridge, trace)| {
                    serde_json::to_value(bridge.evaluate_candidate(
                        trace,
                        response.text.as_str(),
                        prompt_text.as_str(),
                    ))
                    .unwrap_or(Value::Null)
                });
        let mut text = response.text.clone();
        let mut provider = response.provider.clone();
        let mut model = response.model.clone();
        let mut latency_ms = response.latency_ms;
        let mut usage = response.usage.clone();
        let mut raw = response.raw.clone();
        let mut final_source = "llm".to_string();
        let aarnn_admitted = if let Some(trace) = mirror_output.as_ref() {
            self.aarnn_candidate_admitted(trace).await
        } else {
            false
        };
        // Optionally promote an AARNN candidate reply over the LLM response when
        // the bridge confidence gates are explicitly configured to allow it.
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref())
            && ((bridge.should_promote_candidate(
                output_trace,
                response.text.as_str(),
                prompt_text.as_str(),
            )) || (aarnn_admitted
                && bridge.should_promote_admitted_candidate(
                    output_trace,
                    response.text.as_str(),
                    prompt_text.as_str(),
                )))
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
            processing_time_estimate_ms: None,
            usage,
            trace,
            raw,
        };
        info!(request_id = %completion_response.request_id, lifecycle = "provider_completed", parsed_valid = true, "GAIL_ORCHESTRATION_LIFECYCLE");
        info!(request_id = %completion_response.request_id, lifecycle = "selected", "GAIL_ORCHESTRATION_LIFECYCLE");
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
                "request_max_tokens": effective_request.max_tokens,
                "request_temperature": effective_request.temperature,
                "final_source": completion_response
                    .trace
                    .as_ref()
                    .map(|trace| trace.final_source.clone()),
                "aarnn_evaluation": aarnn_evaluation,
            })),
            created_ts: current_ts(),
        })
        .await;
        Ok(completion_response)
    }

    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let _ = self.inner.metrics.record_request_received().await;
        let api_source = request
            .source
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let prompt_tokens_estimate =
            estimate_request_prompt_tokens(&request.messages, request.system.as_deref());
        let processing_time_estimate_ms =
            self.inner.metrics.ai_response_time_estimate_ms("all").await;
        let started = Instant::now();
        let result = self.complete_inner(request).await;
        let terminal_ok = result.as_ref().is_ok_and(completion_metric_success);
        let timed_out = result
            .as_ref()
            .err()
            .map(|error| error.to_string().to_ascii_lowercase().contains("timeout"))
            .unwrap_or(false);
        let _ = self
            .inner
            .metrics
            .record_request_terminal(terminal_ok, timed_out)
            .await;
        let source = result
            .as_ref()
            .map(completion_metric_source)
            .unwrap_or("llm");
        let _ = self
            .inner
            .metrics
            .record_ai_response_time_with_prompt(
                source,
                started.elapsed().as_millis() as u64,
                terminal_ok,
                Some(prompt_tokens_estimate),
            )
            .await;
        let _ = self
            .inner
            .metrics
            .record_api_source_response_time(
                api_source.as_str(),
                started.elapsed().as_millis() as u64,
                terminal_ok,
                Some(prompt_tokens_estimate),
            )
            .await;
        result.map(|mut response| {
            response.processing_time_estimate_ms = processing_time_estimate_ms;
            response
        })
    }

    async fn complete_inner(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let request_id = request
            .request_id
            .as_deref()
            .map(str::trim)
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 128
                    && value.chars().all(|ch| ch.is_ascii_graphic())
            })
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let api_source = normalize_key(request.source.as_deref().unwrap_or("unknown"), "unknown");
        let workflow = normalize_key(request.workflow.as_deref().unwrap_or("general"), "general");
        let role = normalize_key(request.role.as_deref().unwrap_or("general"), "general");
        let prompt_text = flatten_prompt_text(&request.messages, request.system.as_deref());
        let workload_class = classify_workload_with_context(
            workflow.as_str(),
            role.as_str(),
            request.request_category.as_deref(),
            request.request_profile.as_deref(),
            request.source.as_deref(),
            Some(prompt_text.as_str()),
        );
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
        // `fastest` intentionally races endpoint replicas; other modes avoid
        // paying twice for equivalent model work in the same wave.
        let deduplicate_wave_models =
            self.deduplicate_model_candidates() && selection_mode != SelectionMode::Fastest;

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
            source: request.source.clone(),
            request_profile: request.request_profile.clone(),
        };

        let prompt_tokens_estimate = estimate_prompt_tokens(&prompt_text);
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
        let request_profile = derive_request_profile(
            request.request_profile.as_deref(),
            &workflow,
            &role,
            request.request_category.as_deref(),
            &task_tags,
        );
        let requested_output_tokens = request.max_tokens.unwrap_or(512).max(1);
        let mut specialist_meta = None;
        if !self.inner.specialists.is_empty()
            && (task_tags.contains("neuromorphic") || self.always_route_specialists())
        {
            let analyze_request = NeuromorphicAnalyzeRequest {
                text: prompt_text.clone(),
                workflow: Some(workflow.clone()),
                role: Some(role.clone()),
            };
            let specialist_started = Instant::now();
            let analysis =
                analyze_specialist_engines(&self.inner.specialists, &analyze_request).await;
            let _ = self
                .inner
                .metrics
                .record_ai_response_time(
                    "snn",
                    specialist_started.elapsed().as_millis() as u64,
                    true,
                )
                .await;
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
        self.retain_admitted_dynamic_candidates(&request, &mut candidates)
            .await;
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
        let before_role_filter = candidates.len();
        candidates.retain(|candidate| candidate_supports_role(candidate, role.as_str()));
        let role_filtered = before_role_filter.saturating_sub(candidates.len());
        if role_filtered > 0 {
            info!(
                request_id = %request_id,
                workflow = %workflow,
                role = %role,
                removed = role_filtered,
                "filtered candidates that do not declare the requested workflow role"
            );
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
            let api_source_clone = api_source.clone();
            let request_profile_clone = request_profile.clone();
            let workflow_clone = workflow.clone();
            let role_clone = role.clone();
            let request_category_clone = request.request_category.clone();
            let task_tags_clone = task_tags.clone();
            rank_join_set.spawn(async move {
                service
                    .rank_candidate(
                        candidate,
                        &api_source_clone,
                        &request_profile_clone,
                        &workflow_clone,
                        &role_clone,
                        request_category_clone.as_deref(),
                        &task_tags_clone,
                    )
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
        let dispatch_estimates = self
            .dispatch_estimates(
                &ranked,
                &api_source,
                &request_profile,
                &workflow,
                &role,
                request.request_category.as_deref(),
                requested_output_tokens,
            )
            .await;
        // Routing tags describe provider specialities; they are not a response
        // contract.  In particular, the assistant_requirements profile carries
        // a `json` speciality so structured requirements prompts can find the
        // right providers, but Refiner's normal assistant request is still
        // conversational.  An explicit prompt/system instruction, or an
        // explicit `json` request category, activates the JSON quality gate.
        let expected_json = expected_json(
            &provider_request.messages,
            provider_request.system.as_deref(),
        ) || request_category_expects_json(request.request_category.as_deref());
        let timeout_cap = self.candidate_timeout_cap(
            workload_class,
            &workflow,
            &role,
            expected_json,
            &task_tags,
            &prompt_text,
        );
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
                    !attempted_candidate_ids.contains(&candidate_attempt_key(&item.candidate))
                        && !throttled_provider_types.contains(&item.candidate.provider_type)
                })
                .cloned()
                .collect::<Vec<_>>();
            // Do not spend a request's entire timeout budget on a candidate
            // whose queue-adjusted useful ETA is already beyond that budget.
            // Keep the old fallback behaviour when every candidate is over
            // budget, since a degraded response is preferable to rejecting a
            // request with no available provider.  This is especially
            // important for solver JSON requests: a single saturated native
            // llama.cpp host must not block all subsequent fallback waves.
            let budgeted_unattempted = timeout_cap
                .map(|seconds| {
                    let budget_ms = seconds.saturating_mul(1_000) as f64;
                    unattempted
                        .iter()
                        .filter(|item| {
                            dispatch_estimates
                                .get(&item.candidate.candidate_id())
                                .map(|estimate| {
                                    estimate.estimated_useful_completion_ms <= budget_ms
                                })
                                .unwrap_or(true)
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| unattempted.clone());
            let selection_pool = if budgeted_unattempted.is_empty() {
                unattempted.clone()
            } else {
                if budgeted_unattempted.len() < unattempted.len() {
                    info!(
                        request_id = %request_id,
                        workflow = %workflow,
                        role = %role,
                        timeout_cap_seconds = ?timeout_cap,
                        skipped_candidates = unattempted.len() - budgeted_unattempted.len(),
                        "excluding candidates whose useful dispatch ETA exceeds the request timeout"
                    );
                }
                budgeted_unattempted
            };
            let remaining = selection_pool
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
                        let probe_target =
                            transient_backoff_probe_target(wave_size, transient_backoff.len());
                        info!(
                            workflow = %workflow,
                            role = %role,
                            probe_candidates = probe_target,
                            backoff_candidates = %preview_labels(
                                transient_backoff
                                    .iter()
                                    .map(|item| item.candidate.candidate_id())
                                    .collect::<Vec<_>>(),
                                6
                            ),
                            "all providers are in transient adaptive backoff; forcing a probe attempt"
                        );
                        forced_selected = Some(select_ranked_candidates(
                            transient_backoff,
                            probe_target,
                            deduplicate_wave_models,
                        ));
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
            let selected = if let Some(forced) = forced_selected {
                forced
            } else if selection_mode == SelectionMode::RoundRobin {
                self.select_round_robin_candidates(
                    remaining,
                    wave_size,
                    &workflow,
                    &role,
                    &dispatch_estimates,
                )
                .await
            } else {
                select_adaptive_candidates(
                    remaining,
                    wave_size,
                    deduplicate_wave_models,
                    &dispatch_estimates,
                )
            };
            if selected.is_empty() {
                if results.is_empty() {
                    return Err(GailError::bad_request(
                        "no provider candidates were selected",
                    ));
                }
                break;
            }
            let _ = self
                .inner
                .metrics
                .record_orchestration_event("candidate_selection", None)
                .await;
            wave_index += 1;
            for candidate in &selected {
                attempted_candidate_ids.insert(candidate_attempt_key(candidate));
            }

            let capacity_forecasts = selected
                .iter()
                .filter_map(|candidate| {
                    let estimate = dispatch_estimates.get(&candidate.candidate_id())?;
                    Some(format!(
                        "{}:queue={} lanes={} eta_ms={:.0} useful_eta_ms={:.0} service_ms={:.0} useful_rate={:.2}",
                        candidate.candidate_id(),
                        estimate.queue_depth,
                        estimate.candidate_parallelism,
                        estimate.estimated_completion_ms,
                        estimate.estimated_useful_completion_ms,
                        estimate.service_time_ms,
                        estimate.useful_rate,
                    ))
                })
                .collect::<Vec<_>>();

            info!(
                request_id = %request_id,
                workflow = %workflow,
                role = %role,
                fallback_wave = wave_index,
                timeout_cap_seconds = ?timeout_cap,
                candidates = %preview_labels(selected.iter().map(|item| item.label(None)).collect::<Vec<_>>(), 6),
                candidate_ids = %preview_labels(selected.iter().map(ProviderCandidate::candidate_id).collect::<Vec<_>>(), 6),
                candidate_hosts = %preview_labels(selected.iter().map(candidate_host_label).collect::<Vec<_>>(), 6),
                throttled_providers = %preview_labels(sorted_strings(throttled_provider_types.iter().cloned()), 6),
                tags = %preview_labels(task_tags.iter().cloned().collect::<Vec<_>>(), 8),
                capacity_forecasts = %preview_labels(capacity_forecasts, 6),
                "dispatching Gail orchestration"
            );
            info!(request_id = %request_id, workflow = %workflow, role = %role, lifecycle = "dispatched", "GAIL_ORCHESTRATION_LIFECYCLE");
            if wave_index > 1 {
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event("fallback", None)
                    .await;
            }

            let mut wave_results = if selected.len() == 1 {
                let wait_for_capacity =
                    !ranked_candidate_is_capacity_available(&ranked, &selected[0]);
                vec![
                    self.invoke_candidate(
                        selected[0].clone(),
                        provider_request.clone(),
                        expected_json,
                        timeout_cap,
                        workload_class,
                        wait_for_capacity,
                    )
                    .await,
                ]
            } else {
                let capacity_available_ids = ranked
                    .iter()
                    .filter(|item| item.health_ok)
                    .map(|item| item.candidate.candidate_id())
                    .collect::<HashSet<_>>();
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
                    capacity_available_ids,
                )
                .await?
            };

            returned_early |= wave_results.len() < selected.len() && selected.len() > 1;
            let wave_has_success = wave_results.iter().any(|result| {
                result.response.is_some() && result.quality >= early_success_min_quality
            });
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
                request_id = %request_id,
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
            if result.response.is_some() && result.quality < early_success_min_quality {
                result.error = Some(format!(
                    "response quality {:.3} was below the useful-response threshold {:.3}",
                    result.quality, early_success_min_quality
                ));
                result.response = None;
            }
            let candidate_summary = result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str()));
            if let Some(response) = result.response.as_ref() {
                let latency_penalty = response.latency_ms as f64 / 5000.0;
                let metrics_bonus = self
                    .inner
                    .metrics
                    .score_bonus_for_context(
                        candidate_summary.candidate_id.as_str(),
                        &api_source,
                        &request_profile,
                        &workflow,
                        &role,
                        request.request_category.as_deref(),
                    )
                    .await;
                result.score = result.quality - latency_penalty.min(1.25) + metrics_bonus;
                let mut telemetry = local_usage_telemetry(response);
                telemetry.prompt_tokens_estimate = Some(prompt_tokens_estimate);
                if let Some(queue_wait_ms) = result.queue_wait_ms {
                    telemetry.queue_wait_ms = Some(
                        telemetry
                            .queue_wait_ms
                            .unwrap_or(0)
                            .saturating_add(queue_wait_ms),
                    );
                }
                self.inner
                    .metrics
                    .record_result_with_context(
                        &candidate_summary,
                        &api_source,
                        &request_profile,
                        &workflow,
                        &role,
                        request.request_category.as_deref(),
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
                // A dispatch-time reservation race means another request won
                // the slot between ranking and invocation.  It is normal
                // scheduler contention, not evidence that the endpoint is
                // unhealthy.  Keep it in orchestration telemetry so capacity
                // tuning can see it, but do not count it as a provider
                // failure or overwrite a healthy endpoint's health bucket.
                if is_dispatch_capacity_race(result.error.as_deref()) {
                    let _ = self
                        .inner
                        .metrics
                        .record_orchestration_event("capacity_race", None)
                        .await;
                    continue;
                }
                let health_bucket =
                    runtime_failure_health_bucket(result.error.as_deref(), result.latency_ms);
                if result
                    .error
                    .as_deref()
                    .is_some_and(|error| error.to_ascii_lowercase().contains("timeout"))
                {
                    let _ = self
                        .inner
                        .metrics
                        .record_orchestration_event("timeout", None)
                        .await;
                    warn!(request_id = %request_id, lifecycle = "timed_out", "GAIL_ORCHESTRATION_LIFECYCLE");
                }
                let category = health_bucket
                    .mode
                    .clone()
                    .unwrap_or_else(|| "runtime_error".to_string());
                self.inner
                    .metrics
                    .record_result_with_context(
                        &candidate_summary,
                        &api_source,
                        &request_profile,
                        &workflow,
                        &role,
                        request.request_category.as_deref(),
                        false,
                        result.latency_ms,
                        Some(LocalUsageTelemetry {
                            prompt_tokens_estimate: Some(prompt_tokens_estimate),
                            queue_wait_ms: result.queue_wait_ms,
                            ..LocalUsageTelemetry::default()
                        }),
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
            request_id = %request_id,
            workflow = %workflow,
            role = %role,
            provider = %chosen_response.provider,
            model = %chosen_response.model,
            candidate_id = %chosen.candidate.candidate_id(),
            candidate_host = %candidate_host_label(&chosen.candidate),
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
        let aarnn_admitted = if let Some(trace) = mirror_output.as_ref() {
            self.aarnn_candidate_admitted(trace).await
        } else {
            false
        };
        // Optionally promote an AARNN candidate reply over the selected LLM
        // candidate when confidence/quality gates pass.
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref())
            && ((bridge.should_promote_candidate(
                output_trace,
                chosen_response.text.as_str(),
                mirrored_prompt_text.as_str(),
            )) || (aarnn_admitted
                && bridge.should_promote_admitted_candidate(
                    output_trace,
                    chosen_response.text.as_str(),
                    mirrored_prompt_text.as_str(),
                )))
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
            processing_time_estimate_ms: None,
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
            processing_time_estimate_ms: None,
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
        let processing_time_estimate_ms =
            self.inner.metrics.ai_response_time_estimate_ms("snn").await;
        let started = Instant::now();
        let mut response = analyze_specialist_engines(&self.inner.specialists, &request).await;
        let latency_ms = started.elapsed().as_millis() as u64;
        let _ = self
            .inner
            .metrics
            .record_ai_response_time("snn", latency_ms, true)
            .await;
        response.processing_time_estimate_ms = processing_time_estimate_ms;
        Ok(response)
    }

    pub async fn predict_neuromorphic(
        &self,
        request: NeuromorphicPredictRequest,
    ) -> Result<NeuromorphicPredictResponse> {
        let processing_time_estimate_ms =
            self.inner.metrics.ai_response_time_estimate_ms("snn").await;
        let started = Instant::now();
        let result = match self.select_specialist(request.engine_name.as_deref()) {
            Ok(engine) => engine.predict_request(&request).await,
            Err(error) => Err(error),
        };
        let latency_ms = started.elapsed().as_millis() as u64;
        let _ = self
            .inner
            .metrics
            .record_ai_response_time("snn", latency_ms, result.is_ok())
            .await;
        result.map(|mut response| {
            response.processing_time_estimate_ms = processing_time_estimate_ms;
            response
        })
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
        let processing_time_estimate_ms =
            self.inner.metrics.ai_response_time_estimate_ms("all").await;
        let endpoint_telemetry =
            summarize_endpoint_telemetry(&self.inner.metrics.summary(256).await.candidates);
        let dispatch_load = self.dispatch_load_snapshot().await;
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
            "adaptive_race_window_seed_ms": adaptive_race_window_seed_ms(),
            "adaptive_max_raced_candidates_seed":
                adaptive_race_candidate_seed(self.max_parallel_candidates()),
            "default_provider_concurrency_seed": default_provider_concurrency_seed(),
            "interactive_pool_max_in_flight": self.inner.config.orchestration.interactive_pool_max_in_flight,
            "solver_pool_max_in_flight": self.inner.config.orchestration.solver_pool_max_in_flight,
            "trading_pool_max_in_flight": self.inner.config.orchestration.trading_pool_max_in_flight,
            "workload_pool_wait_timeout_ms": self.workload_pool_wait_timeout_ms(),
            "interactive_pool_wait_timeout_ms": self.workload_pool_wait_timeout_ms_for(WorkloadClass::Interactive),
            "solver_pool_wait_timeout_ms": self.workload_pool_wait_timeout_ms_for(WorkloadClass::Solver),
            "trading_pool_wait_timeout_ms": self.workload_pool_wait_timeout_ms_for(WorkloadClass::Trading),
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
            "ai_response_times": self.inner.metrics.ai_response_time_summary().await,
            "processing_time_estimate_ms": processing_time_estimate_ms,
            "dispatch_load": dispatch_load,
            "endpoint_telemetry": {
                "count": endpoint_telemetry.len(),
                "candidates": endpoint_telemetry,
            },
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
                let mut probed_health = None;
                let health = if probe_health {
                    match build_adapter(client, &profile) {
                        Ok(adapter) => {
                            match adapter
                                .health(Some(provider_health_timeout_seconds()))
                                .await
                            {
                                Ok(status) => {
                                    probed_health = Some(status.clone());
                                    json!(status)
                                }
                                Err(error) => {
                                    let status = ProviderHealth {
                                        ok: false,
                                        message: Some(error.to_string()),
                                        ..ProviderHealth::default()
                                    };
                                    probed_health = Some(status.clone());
                                    json!(status)
                                }
                            }
                        }
                        Err(error) => {
                            let status = ProviderHealth {
                                ok: false,
                                message: Some(error.to_string()),
                                ..ProviderHealth::default()
                            };
                            probed_health = Some(status.clone());
                            json!(status)
                        }
                    }
                } else {
                    Value::Null
                };
                let metrics_probe = probed_health.map(|status| (profile.clone(), status));
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
                    metrics_probe,
                )
            });
        }
        let mut ordered: Vec<(usize, Value, Option<(ProviderProfile, ProviderHealth)>)> =
            Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(item) => ordered.push(item),
                Err(error) => {
                    tracing::warn!(error = %error, "provider summary task failed");
                }
            }
        }
        ordered.sort_by_key(|(index, _, _)| *index);
        // Network probes run concurrently, but metrics writes must be ordered:
        // each persistence operation snapshots the store, so concurrent writes
        // can otherwise overwrite a just-recorded health recovery.
        for (_, _, probe) in &ordered {
            let Some((profile, status)) = probe else {
                continue;
            };
            let candidate = ProviderCandidate::from_profile(profile.clone());
            let _ = self
                .inner
                .metrics
                .record_health(
                    &candidate.summary(None),
                    HealthBucket {
                        ok: Some(status.ok),
                        mode: status.mode.clone(),
                        checked_at: None,
                        latency_ms: status.latency_ms,
                        message: status.message.clone(),
                    },
                )
                .await;
        }
        ordered.into_iter().map(|(_, value, _)| value).collect()
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
        // Respect per-direction toggles so input/output mirroring can be tuned
        // independently without changing orchestration call sites.
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
        // Bounded wait keeps completion latency predictable; if the mirror path
        // is slow, Gail returns without blocking the main LLM response path.
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

    async fn aarnn_candidate_admitted(
        &self,
        trace: &crate::models::AarnnMirrorInvocationTrace,
    ) -> bool {
        if !self.inner.config.comparative_validation.enabled {
            return false;
        }
        let Some(dsn) = self.inner.postgres_dsn.as_deref() else {
            return false;
        };
        let Some(bridge) = self.aarnn_bridge() else {
            return false;
        };
        admitted_for_kind(
            dsn,
            "aarnn",
            trace.endpoint.as_str(),
            bridge.response_model(),
            self.inner
                .config
                .comparative_validation
                .admission_ttl_seconds,
        )
        .await
        .unwrap_or_else(|error| {
            tracing::debug!(error = %error, "AARNN comparative admission lookup failed");
            false
        })
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
        // Output mirroring reuses the same exchange builder used by input
        // mirroring so both directions carry identical metadata contracts.
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
        // Canonical map from orchestration context into bridge exchange fields.
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

    async fn record_llm_interaction(&self, mut record: LlmLedgerRecord) {
        if !self.audit_logging().store_llm_content
            && !self
                .inner
                .config
                .comparative_validation
                .retain_content_for_validation
        {
            record.prompt_text = summarize_llm_content(&record.prompt_text);
            record.system_prompt = record.system_prompt.as_deref().map(summarize_llm_content);
            record.response_text = record.response_text.as_deref().map(summarize_llm_content);
        }
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
        let aarnn_evaluation = response
            .trace
            .as_ref()
            .and_then(|trace| trace.aarnn_mirroring.as_ref())
            .and_then(|mirror| mirror.output.as_ref())
            .and_then(|output| {
                self.aarnn_bridge().map(|bridge| {
                    serde_json::to_value(bridge.evaluate_candidate(
                        output,
                        response.text.as_str(),
                        prompt_text,
                    ))
                    .unwrap_or(Value::Null)
                })
            });
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
                "request_max_tokens": provider_request.max_tokens,
                "request_temperature": provider_request.temperature,
                "selection_mode": response.trace.as_ref().map(|trace| trace.selection_mode.clone()),
                "final_source": final_source,
                "aarnn_evaluation": aarnn_evaluation,
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
        let mut skip_configured_ollama_profiles = false;
        if let Some(provider) = request.preferred_provider.as_ref() {
            let normalized_provider = normalize_provider_type(provider);
            let has_request_endpoint_override =
                has_usable_value(request.preferred_api_key.as_deref())
                    || has_usable_value(request.preferred_access_token.as_deref())
                    || has_usable_value(request.base_url.as_deref());
            let configured_pool_requested =
                should_include_configured_candidates(include_configured, request, true);
            let skip_implicit_configured_request_candidate = configured_pool_requested
                // Credentials supplied by an OpenAI-compatible client are
                // often generic gateway credentials (Refiner does this for
                // local models).  An explicit base URL is the actual
                // endpoint override; keep the configured model contract when
                // only those credentials are present.
                && !has_usable_value(request.base_url.as_deref())
                && configured_model_matches_request(
                    &self.inner.config,
                    provider,
                    request.preferred_model.as_deref(),
                );
            let skip_implicit_ollama_request_candidate = include_configured
                && normalized_provider == "ollama"
                && !has_request_endpoint_override;
            let request_model_allowed = request_candidate_model_allowed(
                &self.inner.config,
                provider,
                request.preferred_model.as_deref(),
            );
            if request_model_allowed
                && !skip_implicit_ollama_request_candidate
                && !skip_implicit_configured_request_candidate
            {
                candidates.push(self.request_candidate(
                    provider,
                    request.preferred_model.clone(),
                    request.preferred_api_key.clone(),
                    request.preferred_access_token.clone(),
                    request.base_url.clone(),
                    true,
                    "request_primary",
                ));
            } else if !request_model_allowed {
                tracing::warn!(
                    provider = %provider,
                    requested_model = ?request.preferred_model,
                    "ignoring unconfigured Ollama request model; using configured provider profiles"
                );
                skip_configured_ollama_profiles = include_configured
                    && normalized_provider == "ollama"
                    && !has_request_endpoint_override;
            } else {
                tracing::debug!(
                    provider = %provider,
                    requested_model = ?request.preferred_model,
                    configured_match = skip_implicit_configured_request_candidate,
                    "skipping implicit request model candidate in configured-pool mode"
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
            let requested_provider = request
                .preferred_provider
                .as_deref()
                .map(normalize_provider_type);
            let requested_model = request
                .preferred_model
                .as_deref()
                .map(str::trim)
                .filter(|model| !model.is_empty() && !model.eq_ignore_ascii_case("default"));
            let exact_model_routing = requested_provider
                .as_deref()
                .is_some_and(|provider| provider == "openai");
            candidates.extend(
                self.inner
                    .config
                    .providers
                    .iter()
                    .filter(|profile| {
                        // An explicit provider/model route names a concrete
                        // model contract. Do not let an unrelated configured
                        // model win merely because it is faster; in
                        // particular, gail-inhouse must never silently fall
                        // back to a Qwen profile. Matching endpoint replicas
                        // remain eligible and are ranked by live health and
                        // historical throughput.
                        let provider_matches = !exact_model_routing
                            || requested_provider.as_deref().is_none_or(|provider| {
                                normalize_provider_type(profile.provider_type.as_str()) == provider
                            });
                        let model_matches = !exact_model_routing
                            || requested_model.is_none_or(|model| {
                                profile.model.as_deref().is_some_and(|configured| {
                                    configured.eq_ignore_ascii_case(model)
                                })
                            });
                        !(skip_configured_ollama_profiles
                            && normalize_provider_type(profile.provider_type.as_str()) == "ollama")
                            && provider_matches
                            && model_matches
                    })
                    .cloned()
                    .map(ProviderCandidate::from_profile),
            );
            if skip_configured_ollama_profiles {
                tracing::info!(
                    requested_provider = ?request.preferred_provider,
                    requested_model = ?request.preferred_model,
                    "skipping configured Ollama profiles for unconfigured explicit Ollama request"
                );
            }
            let prefer_ollama_family = request
                .preferred_provider
                .as_deref()
                .map(normalize_provider_type)
                .is_some_and(|provider| provider == "ollama");
            let append_ollama_fallback = request.preferred_provider.is_none()
                || (prefer_ollama_family && candidates.is_empty());
            if append_ollama_fallback {
                append_local_ollama_fallback_candidate(&mut candidates);
            }
        }
        dedupe_candidates(candidates)
            .into_iter()
            .filter(provider_candidate_is_usable)
            .collect()
    }

    async fn retain_admitted_dynamic_candidates(
        &self,
        request: &CompletionRequest,
        candidates: &mut Vec<ProviderCandidate>,
    ) {
        // Explicit model requests still need the same comparative admission
        // and readiness gate as implicit routing when they name the trained
        // model.  Otherwise the configured replica list can send the request
        // to a stale/unavailable endpoint (for example SM00:18081) before it
        // ever reaches the currently promoted target.  Keep the historical
        // test/dev behaviour for requests without a durable active snapshot:
        // there is no admission record to consult in that case.
        let explicit_trained_request = request.preferred_model.as_deref().is_some_and(|model| {
            candidates.iter().any(|candidate| {
                is_trained_llamacpp_profile(&candidate.profile)
                    && candidate
                        .profile
                        .model
                        .as_deref()
                        .is_some_and(|configured| configured.eq_ignore_ascii_case(model))
            })
        });
        if !self.inner.config.comparative_validation.enabled
            || (request.preferred_model.is_some() && !explicit_trained_request)
        {
            return;
        }
        let Some(dsn) = self.inner.postgres_dsn.as_deref() else {
            if explicit_trained_request {
                return;
            }
            candidates.retain(|candidate| !is_trained_llamacpp_profile(&candidate.profile));
            return;
        };
        let Some(model_version) = active_snapshot_id_for_routing(&self.inner.config) else {
            if explicit_trained_request {
                return;
            }
            candidates.retain(|candidate| !is_trained_llamacpp_profile(&candidate.profile));
            return;
        };
        let admissions = match admitted_for_model(
            dsn,
            model_version.as_str(),
            self.inner
                .config
                .comparative_validation
                .admission_ttl_seconds,
        )
        .await
        {
            Ok(admissions) => admissions,
            Err(error) => {
                tracing::warn!(error = %error, "comparative admission lookup failed; keeping dynamic providers out of generic routing");
                candidates.retain(|candidate| !is_trained_llamacpp_profile(&candidate.profile));
                return;
            }
        };
        let before = candidates.len();
        let dynamic_before = candidates
            .iter()
            .filter(|candidate| is_trained_llamacpp_profile(&candidate.profile))
            .count();
        candidates.retain(|candidate| {
            if !is_trained_llamacpp_profile(&candidate.profile) {
                return true;
            }
            admissions.iter().any(|admission| {
                admission.kind == "trained"
                    && admission_model_matches(
                        admission.model.as_str(),
                        candidate.profile.model.as_deref().unwrap_or_default(),
                    )
                    && candidate
                        .profile
                        .base_url
                        .as_deref()
                        .is_some_and(|endpoint| {
                            admission_endpoint_matches(admission.endpoint.as_str(), endpoint)
                        })
            })
        });
        let admitted = candidates
            .iter()
            .filter(|candidate| is_trained_llamacpp_profile(&candidate.profile))
            .count()
            .min(dynamic_before);
        tracing::info!(
            model_version,
            admitted_dynamic_candidates = admitted,
            removed_dynamic_candidates = before.saturating_sub(candidates.len()),
            "applied comparative provider admission"
        );
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
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
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
        } else if candidate_uses_provider_family_backoff(&candidate)
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
            .score_bonus_for_context(
                candidate_id.as_str(),
                source,
                request_profile,
                workflow,
                role,
                request_category,
            )
            .await;
        let generation_tokens_per_second = self
            .inner
            .metrics
            .generation_tokens_per_second_for_context(
                candidate_id.as_str(),
                source,
                request_profile,
                workflow,
                role,
                request_category,
            )
            .await;
        // Prefer the largest configured model when candidates are otherwise
        // comparable. Persistent source/profile latency and queue metrics are
        // deliberately allowed to outweigh this bonus when the large model
        // is busy or too slow for this workload.
        let model_size_bonus = parse_model_size_billions(candidate.configured_model.as_str())
            .map(|size| (size.ln_1p() / 3.6).clamp(0.0, 1.0) * 0.65)
            .unwrap_or(0.0);
        RankedCandidate {
            health_ok,
            health_mode,
            generation_tokens_per_second,
            score: candidate.weight
                + candidate.priority_bias
                + (overlap * 0.85)
                + role_score
                + health_score
                + preferred_score
                + metrics_bonus
                + model_size_bonus
                - usage_penalty
                - resource_penalty
                - hard_limit_penalty
                - nmc_pressure_penalty
                - nmc_hard_limit_penalty,
            candidate,
        }
    }

    async fn dispatch_estimates(
        &self,
        ranked: &[RankedCandidate],
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
        requested_output_tokens: u32,
    ) -> HashMap<String, CandidateDispatchEstimate> {
        let mut estimates = HashMap::with_capacity(ranked.len());
        for item in ranked {
            let capacity = self
                .inner
                .metrics
                .candidate_capacity_estimate_for_context(
                    item.candidate.candidate_id().as_str(),
                    source,
                    request_profile,
                    workflow,
                    role,
                    request_category,
                    requested_output_tokens,
                )
                .await;
            let load = self.load_snapshot(&item.candidate).await;
            let service_time_ms = capacity
                .service_time_ms
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(5_000.0)
                .clamp(1.0, 3_600_000.0);
            let candidate_parallelism = item
                .candidate
                .max_concurrent_requests
                .unwrap_or_else(|| adaptive_provider_parallelism(&capacity))
                .max(1);
            let queue_depth = load
                .candidate_in_flight
                .saturating_add(load.candidate_waiting);
            let occupied_lanes = (queue_depth as f64 / candidate_parallelism as f64).ceil();
            let historical_queue_wait_ms = capacity
                .queue_wait_ms
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(0.0)
                .min(3_600_000.0);
            let estimated_completion_ms =
                historical_queue_wait_ms + (occupied_lanes + 1.0) * service_time_ms;
            let useful_rate = capacity.useful_rate.clamp(0.1, 1.0);
            let estimated_useful_completion_ms = estimated_completion_ms / useful_rate;
            estimates.insert(
                item.candidate.candidate_id(),
                CandidateDispatchEstimate {
                    samples: capacity.samples,
                    useful_rate,
                    service_time_ms,
                    queue_depth,
                    candidate_parallelism,
                    estimated_completion_ms,
                    estimated_useful_completion_ms,
                },
            );
        }
        estimates
    }

    async fn probe_health(&self, candidate: &ProviderCandidate) -> ProviderHealth {
        let cached = self
            .inner
            .metrics
            .health_snapshot(candidate.candidate_id().as_str())
            .await;
        let health_ttl_seconds = cached_health_ttl_seconds(
            is_ollama_candidate(candidate),
            is_local_llamacpp_candidate(candidate),
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
                .health(Some(provider_health_timeout_seconds()))
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

    async fn dispatch_load_snapshot(&self) -> Value {
        let tracker = self.inner.load_tracker.lock().await;
        let candidates = tracker
            .candidate_in_flight
            .keys()
            .chain(tracker.candidate_waiting.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|candidate_id| {
                (
                    candidate_id.clone(),
                    json!({
                        "in_flight": tracker.candidate_in_flight.get(candidate_id).copied().unwrap_or(0),
                        "waiting": tracker.candidate_waiting.get(candidate_id).copied().unwrap_or(0),
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let hosts = tracker
            .host_usage
            .iter()
            .map(|(host, usage)| {
                (
                    host.clone(),
                    json!({
                        "requests": usage.requests,
                        "cpu": usage.cpu,
                        "ram_mb": usage.ram_mb,
                        "vram_mb": usage.vram_mb,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        json!({
            "candidates": candidates,
            "hosts": hosts,
        })
    }

    async fn load_snapshot(&self, candidate: &ProviderCandidate) -> CandidateLoadSnapshot {
        let candidate_id = candidate.candidate_id();
        let tracker = self.inner.load_tracker.lock().await;
        let candidate_in_flight = tracker
            .candidate_in_flight
            .get(&candidate_id)
            .copied()
            .unwrap_or(0);
        let candidate_waiting = tracker
            .candidate_waiting
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
            candidate_in_flight,
            candidate_waiting,
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

    /// Wait for a candidate/host reservation without polling or blocking a
    /// Tokio worker. Capacity releases wake queued requests through `Notify`.
    async fn reserve_candidate_load_with_backpressure(
        &self,
        candidate: &ProviderCandidate,
    ) -> Option<LoadReservation> {
        let candidate_id = candidate.candidate_id();
        {
            let mut tracker = self.inner.load_tracker.lock().await;
            tracker
                .candidate_waiting
                .entry(candidate_id.clone())
                .and_modify(|value| *value += 1)
                .or_insert(1);
        }
        let waiting_guard = CandidateWaitingGuard::new(self.clone(), candidate_id);
        let started = Instant::now();
        let deadline =
            Instant::now() + Duration::from_millis(self.candidate_queue_wait_timeout_ms());
        loop {
            // `Notify::notified()` is lazy: merely constructing the future does
            // not register it for `notify_waiters()`. Pin and enable it before
            // checking capacity so a release between the check and await cannot
            // be lost and turn a short queue wait into a full timeout.
            let notified = self.inner.load_released.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(reservation) = self.reserve_candidate_load(candidate).await {
                waiting_guard.release().await;
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event(
                        "queue_wait",
                        Some(started.elapsed().as_millis() as u64),
                    )
                    .await;
                return Some(reservation);
            }
            if Instant::now() >= deadline {
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event("queue_wait_timeout", None)
                    .await;
                return None;
            }
            if tokio::time::timeout_at(deadline, notified.as_mut())
                .await
                .is_err()
            {
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event("queue_wait_timeout", None)
                    .await;
                return None;
            }
        }
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
        drop(tracker);
        self.inner.load_released.notify_waiters();
    }

    async fn release_candidate_waiting(&self, candidate_id: String) {
        let mut tracker = self.inner.load_tracker.lock().await;
        if let Some(current) = tracker
            .candidate_waiting
            .get(candidate_id.as_str())
            .copied()
        {
            if current <= 1 {
                tracker.candidate_waiting.remove(candidate_id.as_str());
            } else {
                tracker.candidate_waiting.insert(candidate_id, current - 1);
            }
        }
        drop(tracker);
        self.inner.load_released.notify_waiters();
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
        capacity_available_ids: HashSet<String>,
    ) -> Result<Vec<InvocationResult>> {
        let mut join_set = JoinSet::new();
        for candidate in selected.iter().cloned() {
            let service = self.clone();
            let request = provider_request.clone();
            let wait_for_capacity = !capacity_available_ids.contains(&candidate.candidate_id());
            join_set.spawn(async move {
                service
                    .invoke_candidate(
                        candidate,
                        request,
                        expected_json,
                        timeout_cap,
                        workload_class,
                        wait_for_capacity,
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
                                    queue_wait_ms: None,
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
                    queue_wait_ms: None,
                    quality: -1.0,
                    score: f64::NEG_INFINITY,
                },
            };
            info!(
                candidate = %result.candidate.label(result.response.as_ref().map(|value| value.model.as_str())),
                candidate_id = %result.candidate.candidate_id(),
                candidate_host = %candidate_host_label(&result.candidate),
                status = if result.response.is_some() { "ok" } else { "error" },
                latency_ms = ?result.latency_ms,
                quality = result.quality,
                error = ?result.error,
                "Gail candidate completed"
            );
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
        mut provider_request: ProviderCompletionRequest,
        expected_json: bool,
        timeout_cap: Option<u64>,
        workload_class: WorkloadClass,
        wait_for_capacity: bool,
    ) -> InvocationResult {
        self.prepare_provider_request(&candidate.profile, &mut provider_request);
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
                queue_wait_ms: None,
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        }
        let queue_wait_started = Instant::now();
        let Some(_workload_permit) = self.acquire_workload_permit(workload_class).await else {
            return InvocationResult {
                candidate,
                response: None,
                error: Some(format!(
                    "{} workload pool is saturated; retry after {}ms",
                    workload_class.label(),
                    self.workload_pool_wait_timeout_ms_for(workload_class),
                )),
                latency_ms: None,
                queue_wait_ms: Some(queue_wait_started.elapsed().as_millis() as u64),
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        };
        let load_reservation = if wait_for_capacity {
            self.reserve_candidate_load_with_backpressure(&candidate)
                .await
        } else {
            // The ranker observed capacity for this candidate, but another
            // request may have claimed it between ranking and dispatch. Do
            // not turn that normal race into a long queue wait: the caller's
            // fallback wave can immediately try the next model tier.
            self.reserve_candidate_load(&candidate).await
        };
        let Some(load_reservation) = load_reservation else {
            let event = if wait_for_capacity {
                "queue_wait_timeout"
            } else {
                "capacity_race"
            };
            let _ = self
                .inner
                .metrics
                .record_orchestration_event(event, None)
                .await;
            return InvocationResult {
                candidate,
                response: None,
                error: Some(format!(
                    "candidate capacity was unavailable at dispatch{}",
                    if wait_for_capacity {
                        format!(
                            " after {}ms of queue waiting",
                            self.candidate_queue_wait_timeout_ms()
                        )
                    } else {
                        " (reservation race; trying fallback candidates)".to_string()
                    }
                )),
                latency_ms: None,
                queue_wait_ms: Some(queue_wait_started.elapsed().as_millis() as u64),
                quality: -1.0,
                score: f64::NEG_INFINITY,
            };
        };
        let queue_wait_ms = queue_wait_started.elapsed().as_millis() as u64;
        let load_reservation_guard = LoadReservationGuard::new(self.clone(), load_reservation);
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
        let provider_request_for_invocation = provider_request;
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
                            queue_wait_ms: Some(queue_wait_ms),
                            quality: -1.0,
                            score: f64::NEG_INFINITY,
                        };
                    }
                };
                match adapter.complete(&effective).await {
                    Ok(response) => {
                        let latency_ms = started.elapsed().as_millis() as u64;
                        if response.text.trim().is_empty() {
                            if retry_empty && attempts < 2 {
                                continue;
                            }
                            let _ = self
                                .inner
                                .metrics
                                .record_orchestration_event("empty_plan", None)
                                .await;
                            return InvocationResult {
                                candidate: candidate_for_invocation.clone(),
                                response: None,
                                error: Some(format!(
                                    "empty response text from {}/{}",
                                    candidate_for_invocation.profile.provider_type, response.model
                                )),
                                latency_ms: Some(latency_ms),
                                queue_wait_ms: Some(queue_wait_ms),
                                quality: -1.0,
                                score: f64::NEG_INFINITY,
                            };
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
                                queue_wait_ms: Some(queue_wait_ms),
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
                            queue_wait_ms: Some(queue_wait_ms),
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
                            queue_wait_ms: Some(queue_wait_ms),
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
                    queue_wait_ms: Some(queue_wait_ms),
                    quality: -1.0,
                    score: f64::NEG_INFINITY,
                },
            }
        } else {
            invocation.await
        };
        load_reservation_guard.release().await;
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

    async fn select_round_robin_candidates(
        &self,
        ranked: Vec<RankedCandidate>,
        max_candidates: usize,
        workflow: &str,
        role: &str,
        estimates: &HashMap<String, CandidateDispatchEstimate>,
    ) -> Vec<ProviderCandidate> {
        let Some(context) = round_robin_context(&ranked, workflow, role) else {
            return select_adaptive_candidates(
                ranked,
                max_candidates,
                self.deduplicate_model_candidates(),
                estimates,
            );
        };
        let offset = self
            .next_round_robin_offset(context.key.as_str(), context.group_size)
            .await;
        let reordered = reorder_ranked_candidates_for_round_robin(ranked, &context, offset);
        select_adaptive_candidates(
            reordered,
            max_candidates,
            self.deduplicate_model_candidates(),
            estimates,
        )
    }

    async fn next_round_robin_offset(&self, key: &str, group_size: usize) -> usize {
        if group_size <= 1 {
            return 0;
        }
        let mut cursors = self.inner.round_robin_cursors.lock().await;
        let cursor = cursors.entry(key.to_string()).or_insert(0);
        let offset = *cursor % group_size;
        *cursor = cursor.wrapping_add(1);
        offset
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
        .clamp(1, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS)
    }

    fn workload_pool_wait_timeout_ms_for(&self, class: WorkloadClass) -> u64 {
        let (names, configured) = match class {
            WorkloadClass::Interactive => (
                [
                    "GAIL_INTERACTIVE_POOL_WAIT_TIMEOUT_MS",
                    "REFINER_AI_INTERACTIVE_POOL_WAIT_TIMEOUT_MS",
                ],
                self.inner
                    .config
                    .orchestration
                    .interactive_pool_wait_timeout_ms,
            ),
            WorkloadClass::Solver => (
                [
                    "GAIL_SOLVER_POOL_WAIT_TIMEOUT_MS",
                    "REFINER_AI_SOLVER_POOL_WAIT_TIMEOUT_MS",
                ],
                self.inner.config.orchestration.solver_pool_wait_timeout_ms,
            ),
            WorkloadClass::Trading => (
                [
                    "GAIL_TRADING_POOL_WAIT_TIMEOUT_MS",
                    "GAIL_ADVISORY_QUEUE_WAIT_TIMEOUT_MS",
                ],
                self.inner.config.orchestration.trading_pool_wait_timeout_ms,
            ),
        };
        env_int_any(&names, configured).clamp(1, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS)
    }

    fn candidate_queue_wait_timeout_ms(&self) -> u64 {
        env_int_any(
            &["GAIL_CANDIDATE_QUEUE_WAIT_TIMEOUT_MS"],
            self.inner
                .config
                .orchestration
                .candidate_queue_wait_timeout_ms,
        )
        .clamp(1, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS)
    }

    fn deduplicate_model_candidates(&self) -> bool {
        env_bool_any(
            &["GAIL_DEDUPLICATE_MODEL_CANDIDATES"],
            self.inner.config.orchestration.deduplicate_model_candidates,
        )
    }

    fn model_floor_b(&self, workload_class: WorkloadClass) -> Option<f64> {
        let configured = match workload_class {
            WorkloadClass::Interactive => self.inner.config.orchestration.interactive_model_floor_b,
            WorkloadClass::Solver => self.inner.config.orchestration.solver_model_floor_b,
            WorkloadClass::Trading => self.inner.config.orchestration.interactive_model_floor_b,
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
            WorkloadClass::Trading => env_float_any(&["GAIL_TRADING_MODEL_FLOOR_B"], configured),
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
        let started = Instant::now();
        let wait_timeout = Duration::from_millis(self.workload_pool_wait_timeout_ms_for(class));
        let semaphore = match class {
            WorkloadClass::Interactive => self.inner.interactive_pool.clone(),
            WorkloadClass::Solver => self.inner.solver_pool.clone(),
            WorkloadClass::Trading => self.inner.trading_pool.clone(),
        };
        match tokio::time::timeout(wait_timeout, semaphore.acquire_owned()).await {
            Ok(Ok(permit)) => {
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event(
                        "queue_wait",
                        Some(started.elapsed().as_millis() as u64),
                    )
                    .await;
                Some(permit)
            }
            _ => {
                let _ = self
                    .inner
                    .metrics
                    .record_orchestration_event("queue_wait_timeout", None)
                    .await;
                None
            }
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
            "round_robin" | "roundrobin" | "rr" => SelectionMode::RoundRobin,
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
        workload_class: WorkloadClass,
        workflow: &str,
        role: &str,
        expected_json: bool,
        task_tags: &HashSet<String>,
        prompt_text: &str,
    ) -> Option<u64> {
        let default = if workload_class == WorkloadClass::Interactive {
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
        if expected_json
            && (prompt_requests_execution_plan(prompt_text)
                || prompt_requests_manager_tool_call(prompt_text)
                || prompt_requests_signal_synthesis_output(prompt_text))
        {
            // Multi-agent planning and synthesis payloads (ExecutionPlan,
            // ManagerToolCall, SignalSynthesisOutput) need a full request
            // budget; forcing automation caps here collapses the response into
            // degraded no-op envelopes.
            return base;
        }
        let automation_request = expected_json
            || text_or_tags_indicate_automation(workflow, role, task_tags, prompt_text);
        if workload_class == WorkloadClass::Interactive && !automation_request {
            return base;
        }
        if !automation_request {
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

fn summarize_llm_content(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    format!(
        "<redacted chars={} sha256={}>",
        value.chars().count(),
        hex::encode(digest)
    )
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
        if is_ollama_candidate(self) {
            let explicit_name = self.profile.name.trim();
            if !explicit_name.is_empty()
                && !explicit_name.eq_ignore_ascii_case(self.provider_type.as_str())
            {
                return Some(sanitize_candidate_scope(explicit_name, "endpoint"));
            }
            return self
                .profile
                .base_url
                .as_deref()
                .and_then(candidate_scope_from_base_url);
        }
        if !self.provider_type.eq_ignore_ascii_case("openai") {
            return None;
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
            host_id: Some(candidate_host_label(self)),
            specialties: sorted_strings(self.specialties.clone()),
            roles: sorted_strings(self.roles.clone()),
        }
    }
}

fn dedupe_candidates(candidates: Vec<ProviderCandidate>) -> Vec<ProviderCandidate> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for candidate in candidates {
        let key = candidate_attempt_key(&candidate);
        if seen.insert(key) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn candidate_attempt_key(candidate: &ProviderCandidate) -> String {
    format!(
        "{}::{}::{}",
        candidate.provider_type,
        candidate.configured_model.trim(),
        candidate.profile.base_url.as_deref().unwrap_or("").trim()
    )
}

/// Identity for equivalent inference work. Endpoint is intentionally excluded:
/// endpoint diversity belongs in fallback waves, not duplicate concurrent work.
fn candidate_model_key(candidate: &ProviderCandidate) -> String {
    format!(
        "{}::{}",
        candidate.provider_type.trim().to_ascii_lowercase(),
        candidate.configured_model.trim().to_ascii_lowercase(),
    )
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

fn profile_uses_local_context_default(profile: &ProviderProfile) -> bool {
    if normalize_provider_type(&profile.provider_type) == "ollama" {
        return true;
    }
    let Some(base_url) = profile.base_url.as_deref() else {
        return false;
    };
    let with_scheme = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let Some(host) = reqwest::Url::parse(&with_scheme)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    if host == "localhost"
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".svc")
        || host.ends_with(".svc.cluster.local")
    {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|address| match address {
            std::net::IpAddr::V4(address) => {
                address.is_private() || address.is_loopback() || address.is_link_local()
            }
            std::net::IpAddr::V6(address) => address.is_loopback() || address.is_unique_local(),
        })
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

fn candidate_supports_role(candidate: &ProviderCandidate, role: &str) -> bool {
    let requested = normalize_key(role, "general");
    candidate.roles.is_empty()
        || candidate.roles.contains(&requested)
        // Older OpenAI-compatible callers omit `role`; Gail normalises that
        // to `general`, while legacy profiles commonly declare `assistant`.
        || (requested == "general" && candidate.roles.contains("assistant"))
}

fn has_usable_value(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| !looks_like_placeholder_secret(value))
        .unwrap_or(false)
}

fn env_has_usable_value(names: &[&str]) -> bool {
    names
        .iter()
        .any(|name| has_usable_value(env::var(name).ok().as_deref()))
}

fn looks_like_placeholder_secret(value: &str) -> bool {
    let trimmed = value.trim();
    let lowered = trimmed.to_ascii_lowercase();
    if matches!(
        lowered.as_str(),
        "none"
            | "null"
            | "nil"
            | "undefined"
            | "changeme"
            | "replace_me"
            | "redacted"
            | "<redacted>"
            | "***"
    ) {
        return true;
    }
    trimmed.starts_with("${")
        || (trimmed.starts_with('$') && trimmed.len() > 1)
        || (trimmed.starts_with("{{") && trimmed.ends_with("}}"))
}

fn candidate_host_label(candidate: &ProviderCandidate) -> String {
    candidate
        .host_group
        .as_deref()
        .or(candidate.nmc_host.as_deref())
        .or_else(|| candidate.profile.base_url.as_deref())
        .unwrap_or("unknown")
        .to_string()
}

fn select_ranked_candidates(
    ranked: Vec<RankedCandidate>,
    max_candidates: usize,
    deduplicate_models: bool,
) -> Vec<ProviderCandidate> {
    let target = max_candidates.max(1);
    let mut selected = Vec::new();
    let mut selected_ids = HashSet::new();
    let mut selected_models = HashSet::new();
    let mut selected_provider_types = HashSet::new();
    // `rank_candidate` folds the current candidate/host capacity snapshot
    // into `health_ok`. Prefer healthy endpoints with the highest observed
    // generation throughput. Model size and the full routing score remain
    // deterministic tie-breakers for endpoints without throughput history.
    let mut ordered = ranked.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .health_ok
            .cmp(&left.health_ok)
            .then_with(|| {
                match (
                    right.generation_tokens_per_second,
                    left.generation_tokens_per_second,
                ) {
                    (Some(right), Some(left)) => right
                        .partial_cmp(&left)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                }
            })
            .then_with(|| {
                model_size_tier(&right.candidate.configured_model)
                    .cmp(&model_size_tier(&left.candidate.configured_model))
            })
    });
    let local_fallback = if target >= 2 {
        best_local_fallback_candidate(&ranked)
    } else {
        None
    };

    // First pass preserves the existing preference for distinct provider
    // families. `ordered` already puts healthy/high-tier candidates first.
    for item in &ordered {
        if selected_provider_types.contains(&item.candidate.provider_type) {
            continue;
        }
        let candidate_key = candidate_attempt_key(&item.candidate);
        let model_key = candidate_model_key(&item.candidate);
        if deduplicate_models && selected_models.contains(&model_key) {
            continue;
        }
        if selected_ids.insert(candidate_key) {
            selected_provider_types.insert(item.candidate.provider_type.clone());
            selected_models.insert(model_key);
            selected.push(item.candidate.clone());
            if selected.len() == target {
                return ensure_local_fallback_selected(
                    selected,
                    local_fallback,
                    target,
                    deduplicate_models,
                );
            }
        }
    }

    for item in &ordered {
        let candidate_key = candidate_attempt_key(&item.candidate);
        let model_key = candidate_model_key(&item.candidate);
        if deduplicate_models && selected_models.contains(&model_key) {
            continue;
        }
        if selected_ids.insert(candidate_key) {
            selected_models.insert(model_key);
            selected.push(item.candidate.clone());
            if selected.len() == target {
                return ensure_local_fallback_selected(
                    selected,
                    local_fallback,
                    target,
                    deduplicate_models,
                );
            }
        }
    }

    ensure_local_fallback_selected(selected, local_fallback, target, deduplicate_models)
}

/// Select the shortest expected useful completion rather than blindly racing
/// the highest-ranked providers. A second provider is raced only when its
/// quality-adjusted ETA is within the configured window of the best candidate;
/// the rest remain ordered fallbacks. This lets a fast but busy provider lose
/// the next request to a slower idle provider when the latter will finish
/// sooner, while preserving redundancy for near-ties.
fn select_adaptive_candidates(
    ranked: Vec<RankedCandidate>,
    max_candidates: usize,
    deduplicate_models: bool,
    estimates: &HashMap<String, CandidateDispatchEstimate>,
) -> Vec<ProviderCandidate> {
    if ranked.is_empty() {
        return Vec::new();
    }
    let mut ordered = ranked;
    ordered.sort_by(|left, right| {
        let left_eta = estimates
            .get(&left.candidate.candidate_id())
            .map(|estimate| estimate.estimated_useful_completion_ms)
            .unwrap_or(f64::INFINITY);
        let right_eta = estimates
            .get(&right.candidate.candidate_id())
            .map(|estimate| estimate.estimated_useful_completion_ms)
            .unwrap_or(f64::INFINITY);
        left_eta
            .total_cmp(&right_eta)
            .then_with(|| right.health_ok.cmp(&left.health_ok))
            .then_with(|| right.score.total_cmp(&left.score))
    });
    let best_eta = estimates
        .get(&ordered[0].candidate.candidate_id())
        .map(|estimate| estimate.estimated_useful_completion_ms)
        .unwrap_or(5_000.0);
    let policy = adaptive_dispatch_policy(estimates, max_candidates);
    let frontier = best_eta + policy.race_window_ms;
    let race_limit = policy.max_raced_candidates;
    let fallback = ordered.first().cloned();
    let competitive = ordered
        .into_iter()
        .filter(|item| {
            estimates
                .get(&item.candidate.candidate_id())
                .map(|estimate| estimate.estimated_useful_completion_ms <= frontier)
                .unwrap_or(false)
        })
        .take(race_limit)
        .collect::<Vec<_>>();
    let competitive = if competitive.is_empty() {
        vec![fallback.expect("adaptive candidates are non-empty")]
    } else {
        competitive
    };
    let target = competitive.len().min(max_candidates.max(1));
    let mut selected = Vec::with_capacity(target);
    let mut ids = HashSet::new();
    let mut models = HashSet::new();
    let mut providers = HashSet::new();
    for item in competitive.iter() {
        let candidate_id = item.candidate.candidate_id();
        let model_key = candidate_model_key(&item.candidate);
        if providers.contains(&item.candidate.provider_type)
            || (deduplicate_models && models.contains(&model_key))
        {
            continue;
        }
        if ids.insert(candidate_id) {
            providers.insert(item.candidate.provider_type.clone());
            models.insert(model_key);
            selected.push(item.candidate.clone());
            if selected.len() == target {
                return selected;
            }
        }
    }
    for item in competitive {
        let candidate_id = item.candidate.candidate_id();
        let model_key = candidate_model_key(&item.candidate);
        if deduplicate_models && models.contains(&model_key) {
            continue;
        }
        if ids.insert(candidate_id) {
            models.insert(model_key);
            selected.push(item.candidate);
            if selected.len() == target {
                break;
            }
        }
    }
    selected
}

fn ranked_candidate_is_capacity_available(
    ranked: &[RankedCandidate],
    candidate: &ProviderCandidate,
) -> bool {
    ranked
        .iter()
        .find(|item| item.candidate.candidate_id() == candidate.candidate_id())
        .is_some_and(|item| item.health_ok)
}

fn round_robin_context(
    ranked: &[RankedCandidate],
    workflow: &str,
    role: &str,
) -> Option<RoundRobinContext> {
    let anchor = ranked
        .iter()
        .find(|item| item.health_ok)
        .or_else(|| ranked.first())?;
    let provider_key = normalize_key(anchor.candidate.provider_type.as_str(), "openai");
    let model_key = normalize_key(anchor.candidate.configured_model.as_str(), "");
    if provider_key.is_empty() || model_key.is_empty() {
        return None;
    }
    let group_size = ranked
        .iter()
        .filter(|item| candidate_matches_round_robin_group(item, &provider_key, &model_key))
        .count();
    if group_size < 2 {
        return None;
    }
    Some(RoundRobinContext {
        provider_key: provider_key.clone(),
        model_key: model_key.clone(),
        key: format!("{workflow}:{role}:{provider_key}:{model_key}"),
        group_size,
    })
}

fn reorder_ranked_candidates_for_round_robin(
    ranked: Vec<RankedCandidate>,
    context: &RoundRobinContext,
    offset: usize,
) -> Vec<RankedCandidate> {
    if ranked.len() < 2 || context.group_size < 2 {
        return ranked;
    }
    let mut group = Vec::with_capacity(context.group_size);
    let mut rest = Vec::with_capacity(ranked.len().saturating_sub(context.group_size));
    for item in ranked {
        if candidate_matches_round_robin_group(&item, &context.provider_key, &context.model_key) {
            group.push(item);
        } else {
            rest.push(item);
        }
    }
    if group.len() < 2 {
        group.extend(rest);
        return group;
    }
    let group_len = group.len();
    group.rotate_left(offset % group_len);
    let mut healthy = Vec::with_capacity(group.len());
    let mut unhealthy = Vec::new();
    for item in group {
        if item.health_ok {
            healthy.push(item);
        } else {
            unhealthy.push(item);
        }
    }
    healthy.extend(unhealthy);
    healthy.extend(rest);
    healthy
}

fn candidate_matches_round_robin_group(
    item: &RankedCandidate,
    provider_key: &str,
    model_key: &str,
) -> bool {
    normalize_key(item.candidate.provider_type.as_str(), "openai") == provider_key
        && normalize_key(item.candidate.configured_model.as_str(), "") == model_key
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

/// Cloud/API profiles do not always declare a concurrency limit. This is a
/// cold-start seed for the virtual lane count used by ETA calculation; once
/// useful service and queue observations exist it is adjusted per candidate.
fn default_provider_concurrency_seed() -> usize {
    env_int_any(&["GAIL_DEFAULT_PROVIDER_CONCURRENCY"], 4).clamp(1, 64) as usize
}

fn adaptive_race_window_seed_ms() -> f64 {
    env_float_any(&["GAIL_ADAPTIVE_RACE_WINDOW_MS"], 750.0).clamp(0.0, 30_000.0)
}

fn adaptive_race_candidate_seed(target: usize) -> usize {
    env_int_any(&["GAIL_ADAPTIVE_MAX_RACED_CANDIDATES"], 2).clamp(1, target.max(1) as u64) as usize
}

fn adaptive_provider_parallelism(capacity: &crate::metrics::CandidateCapacityEstimate) -> usize {
    let seed = default_provider_concurrency_seed();
    if capacity.samples < 4 {
        return seed;
    }
    let Some(service_time_ms) = capacity.service_time_ms.filter(|value| *value > 0.0) else {
        return seed;
    };
    let queue_ratio = capacity.queue_wait_ms.unwrap_or_default().max(0.0) / service_time_ms;
    let mut lanes = seed as isize;
    if capacity.useful_rate < 0.5 || queue_ratio > 0.75 {
        lanes -= 1;
    } else if capacity.useful_rate > 0.85 && queue_ratio < 0.15 {
        lanes += 1;
        if capacity.samples >= 20 && queue_ratio < 0.05 {
            lanes += 1;
        }
    }
    lanes.clamp(1, 64) as usize
}

#[derive(Clone, Copy, Debug)]
struct AdaptiveDispatchPolicy {
    race_window_ms: f64,
    max_raced_candidates: usize,
}

fn adaptive_dispatch_policy(
    estimates: &HashMap<String, CandidateDispatchEstimate>,
    max_candidates: usize,
) -> AdaptiveDispatchPolicy {
    let seed_window = adaptive_race_window_seed_ms();
    let seed_raced = adaptive_race_candidate_seed(max_candidates);
    let confidence = estimates
        .values()
        .map(|estimate| estimate.samples)
        .max()
        .unwrap_or_default() as f64
        / 20.0;
    let confidence = confidence.clamp(0.0, 1.0);
    let best_service_ms = estimates
        .values()
        .map(|estimate| estimate.service_time_ms)
        .filter(|value| value.is_finite() && *value > 0.0)
        .min_by(f64::total_cmp)
        .unwrap_or(5_000.0);
    // A fixed millisecond window is too wide for short requests and too
    // narrow for long generations. Use the seed during cold start, then
    // converge towards a fraction of the observed service time.
    let learned_window_ms = (best_service_ms * 0.20).clamp(100.0, 3_000.0);
    let race_window_ms =
        (seed_window * (1.0 - confidence) + learned_window_ms * confidence).clamp(50.0, 3_000.0);
    let best_eta = estimates
        .values()
        .map(|estimate| estimate.estimated_useful_completion_ms)
        .filter(|value| value.is_finite() && *value > 0.0)
        .min_by(f64::total_cmp)
        .unwrap_or(5_000.0);
    let near_tie_count = estimates
        .values()
        .filter(|estimate| estimate.estimated_useful_completion_ms <= best_eta + race_window_ms)
        .count();
    let learned_raced = if near_tie_count <= 1 {
        1
    } else {
        (near_tie_count as f64).sqrt().ceil() as usize
    };
    let max_raced_candidates = ((seed_raced as f64 * (1.0 - confidence))
        + (learned_raced as f64 * confidence))
        .round()
        .clamp(1.0, max_candidates.max(1) as f64) as usize;
    AdaptiveDispatchPolicy {
        race_window_ms,
        max_raced_candidates,
    }
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

/// Native llama.cpp endpoints disappear during host reboots while Gail keeps
/// running. Keep both positive and failed endpoint probes short-lived so a
/// persisted health snapshot cannot hide a recovered node (or keep routing
/// to a node that has just gone down) for the general provider TTL.
fn local_llamacpp_health_ttl_seconds() -> f64 {
    env_float_any(&["GAIL_LOCAL_HEALTH_TTL_SECONDS"], 30.0).max(5.0)
}

fn is_local_llamacpp_candidate(candidate: &ProviderCandidate) -> bool {
    // Both the primary and optional trained llama.cpp profiles are managed
    // by Ansible.  Treat the trained source as local too; otherwise it falls
    // back to the general provider TTL and a stale positive snapshot can
    // keep routing requests to an endpoint that is no longer listening.
    candidate
        .source
        .to_ascii_lowercase()
        .starts_with("ansible_llamacpp")
}

fn cached_health_ttl_seconds(
    is_ollama: bool,
    is_local_llamacpp: bool,
    mode: Option<&str>,
    default_ttl: f64,
) -> f64 {
    if is_local_llamacpp {
        return local_llamacpp_health_ttl_seconds();
    }
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
    deduplicate_models: bool,
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
    if deduplicate_models
        && selected
            .iter()
            .any(|candidate| candidate_model_key(candidate) == candidate_model_key(&local_fallback))
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
                    "error",
                    "runtime_error",
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

fn transient_backoff_probe_target(wave_size: usize, candidate_count: usize) -> usize {
    let configured = env_int_any(
        &[
            "GAIL_TRANSIENT_BACKOFF_PROBE_CANDIDATES",
            "REFINER_AI_TRANSIENT_BACKOFF_PROBE_CANDIDATES",
        ],
        2,
    ) as usize;
    transient_backoff_probe_target_with_config(wave_size, candidate_count, configured)
}

fn transient_backoff_probe_target_with_config(
    wave_size: usize,
    candidate_count: usize,
    configured: usize,
) -> usize {
    configured
        .max(wave_size.max(1))
        .max(1)
        .min(candidate_count.max(1))
}

fn message_indicates_provider_backoff(message: &str) -> bool {
    message_indicates_quota(message)
        || message_indicates_ollama_saturation(message)
        || message_indicates_resource_saturation(message)
        || message_indicates_nmc_constrained(message)
        || message_indicates_provider_auth_failure(message)
        || message_indicates_transient_provider_failure(message)
}

fn candidate_uses_provider_family_backoff(candidate: &ProviderCandidate) -> bool {
    if is_ollama_candidate(candidate) {
        return false;
    }
    !candidate_has_explicit_endpoint(candidate)
}

fn candidate_has_explicit_endpoint(candidate: &ProviderCandidate) -> bool {
    candidate
        .profile
        .base_url
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
}

fn error_should_backoff_provider_family(candidate: &ProviderCandidate, message: &str) -> bool {
    if !candidate_uses_provider_family_backoff(candidate) {
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

fn is_dispatch_capacity_race(error: Option<&str>) -> bool {
    error.is_some_and(|message| {
        message
            .to_ascii_lowercase()
            .contains("capacity was unavailable at dispatch (reservation race;")
    })
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

fn request_category_expects_json(request_category: Option<&str>) -> bool {
    request_category
        .unwrap_or_default()
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .map(|part| part.trim().to_ascii_lowercase())
        .any(|part| matches!(part.as_str(), "json" | "structured_data"))
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
    let lowered = cleaned.to_ascii_lowercase();
    let obvious_non_answers = [
        "i can't help",
        "i cannot help",
        "i'm unable",
        "i am unable",
        "cannot answer",
        "no answer",
        "service unavailable",
        "model not found",
        "upstream error",
        "as an ai language model",
    ];
    if obvious_non_answers
        .iter()
        .any(|marker| lowered.contains(marker))
        || matches!(lowered.as_str(), "n/a" | "null" | "none" | "..." | "-")
    {
        return -1.5;
    }
    let mut score = 0.6;
    if cleaned.len() >= 40 {
        score += 0.35;
    }
    if expected_json {
        match try_parse_json(cleaned) {
            Some(Value::Object(value)) if !value.is_empty() => score += 2.45,
            Some(Value::Array(value)) if !value.is_empty() => score += 2.35,
            Some(Value::Object(_)) | Some(Value::Array(_)) | Some(Value::Null) => {
                return -1.5;
            }
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
        telemetry.completion_tokens_estimate = local_usage
            .get("completion_tokens_estimate")
            .and_then(Value::as_u64)
            .map(|value| value as u32)
            .or_else(|| {
                local_usage
                    .get("eval_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32)
            });
    }
    // Native llama.cpp exposes decode-only timing in its OpenAI-compatible
    // `timings` object. Prefer it over end-to-end latency when present so the
    // routing metric describes sustained generation throughput rather than
    // queue wait, mirroring, and HTTP overhead. This is especially important
    // for short readiness probes, which otherwise make a healthy GPU appear
    // to produce only a few tokens per second.
    if let Some(timings) = response.raw.as_ref().and_then(|raw| raw.get("timings")) {
        if telemetry.inference_ms.is_none() {
            telemetry.inference_ms = timings
                .get("predicted_ms")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| value.round() as u64);
        }
        if telemetry.completion_tokens_estimate.is_none() {
            telemetry.completion_tokens_estimate = timings
                .get("predicted_n")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
        }
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
    if telemetry.completion_tokens_estimate.is_none() {
        telemetry.completion_tokens_estimate =
            response.usage.as_ref().and_then(|usage| usage.completion);
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

/// Coarse routing tiers for the configured local pool. The boundaries are
/// intentionally broad so model aliases such as `qwen3.6:35b`, `qwen3.5:9b`
/// and `qwen3.5:4b` retain the expected 35B > 9B > 4B ordering without
/// hard-coding a particular vendor or model family.
fn model_size_tier(model: &str) -> u8 {
    match parse_model_size_billions(model) {
        Some(size) if size >= 20.0 => 3,
        Some(size) if size >= 7.0 => 2,
        Some(size) if size > 0.0 => 1,
        _ => 0,
    }
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

fn estimate_prompt_tokens(text: &str) -> u32 {
    ((text.chars().count() as u32).saturating_add(3) / 4).max(1)
}

fn estimate_request_prompt_tokens(
    messages: &[crate::models::ChatMessage],
    system: Option<&str>,
) -> u32 {
    estimate_prompt_tokens(flatten_prompt_text(messages, system).as_str())
}

fn derive_request_profile(
    explicit: Option<&str>,
    workflow: &str,
    role: &str,
    request_category: Option<&str>,
    task_tags: &HashSet<String>,
) -> String {
    if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return normalize_key(value, "general");
    }
    if let Some(value) = request_category
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return normalize_key(value, "general");
    }
    for candidate in [
        "project_solver",
        "coding",
        "code",
        "research",
        "trading",
        "market",
        "json",
        "planning",
        "review",
    ] {
        if task_tags.contains(candidate) {
            return candidate.to_string();
        }
    }
    if !workflow.eq_ignore_ascii_case("general") {
        normalize_key(workflow, "general")
    } else if !role.eq_ignore_ascii_case("general") {
        normalize_key(role, "general")
    } else {
        "general".to_string()
    }
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

fn summarize_endpoint_telemetry(
    candidates: &[CandidateMetricsSummary],
) -> Vec<EndpointTelemetryRow> {
    let mut rows = candidates
        .iter()
        .filter_map(|candidate| {
            let endpoint_scope = candidate_endpoint_scope(candidate.candidate_id.as_str())?;
            let (endpoint_host, endpoint_port) = endpoint_host_port_from_scope(endpoint_scope);
            let endpoint_suffix = endpoint_host.as_deref().and_then(endpoint_host_suffix);
            Some(EndpointTelemetryRow {
                candidate_id: candidate.candidate_id.clone(),
                provider: candidate.provider.clone(),
                configured_model: candidate.configured_model.clone(),
                resolved_model: candidate
                    .resolved_model
                    .clone()
                    .or_else(|| candidate.model.clone()),
                endpoint_scope: endpoint_scope.to_string(),
                endpoint_host,
                endpoint_port,
                endpoint_suffix,
                successes: candidate.successes,
                failures: candidate.failures,
                total: candidate.total,
                success_rate: candidate.success_rate,
                ewma_latency_ms: candidate.ewma_latency_ms,
                ewma_queue_wait_ms: candidate.ewma_queue_wait_ms,
                ewma_inference_ms: candidate.ewma_inference_ms,
                last_status: candidate.last_status.clone(),
                last_error: candidate.last_error.clone(),
                updated_at: candidate.updated_at,
            })
        })
        .collect::<Vec<_>>();

    rows.sort_by(|left, right| {
        left.provider
            .cmp(&right.provider)
            .then_with(|| left.configured_model.cmp(&right.configured_model))
            .then_with(|| left.endpoint_scope.cmp(&right.endpoint_scope))
    });
    rows
}

fn candidate_endpoint_scope(candidate_id: &str) -> Option<&str> {
    candidate_id
        .rsplit_once('@')
        .map(|(_, scope)| scope.trim())
        .filter(|scope| !scope.is_empty())
}

fn endpoint_host_port_from_scope(scope: &str) -> (Option<String>, Option<u16>) {
    let segments = scope.split('_').collect::<Vec<_>>();
    if segments.len() >= 4
        && segments[..4]
            .iter()
            .all(|segment| segment.chars().all(|char| char.is_ascii_digit()))
    {
        let host = format!(
            "{}.{}.{}.{}",
            segments[0], segments[1], segments[2], segments[3]
        );
        let port = segments
            .get(4)
            .filter(|segment| segment.chars().all(|char| char.is_ascii_digit()))
            .and_then(|segment| segment.parse::<u16>().ok());
        return (Some(host), port);
    }
    (None, None)
}

fn endpoint_host_suffix(host: &str) -> Option<String> {
    host.rsplit('.')
        .next()
        .filter(|octet| !octet.is_empty())
        .map(|octet| format!(".{octet}"))
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

fn configured_model_matches_request(
    config: &GailConfig,
    provider: &str,
    model: Option<&str>,
) -> bool {
    let Some(requested_model) = model
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("default"))
    else {
        return false;
    };
    let normalized_provider = normalize_provider_type(provider);
    config.providers.iter().any(|profile| {
        normalize_provider_type(profile.provider_type.as_str()) == normalized_provider
            && profile
                .model
                .as_deref()
                .is_some_and(|configured| configured.trim().eq_ignore_ascii_case(requested_model))
    })
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

fn is_trained_llamacpp_profile(profile: &ProviderProfile) -> bool {
    profile
        .source
        .as_deref()
        .is_some_and(|source| source.eq_ignore_ascii_case("ansible_llamacpp_trained"))
}

fn active_snapshot_id_for_routing(config: &GailConfig) -> Option<String> {
    let path = std::path::PathBuf::from(&config.trainer.output_root).join("active_snapshot.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    value
        .get("snapshot_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
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
        host_id: Some("internal".to_string()),
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
        // Prefer schema-specific fallbacks first to avoid returning manager-tool
        // envelopes for structured payloads like SignalSynthesisOutput.
        let payload = if prompt_requests_execution_plan(prompt_text) {
            degraded_execution_plan_payload(prompt_text)
        } else if prompt_requests_signal_synthesis_output(prompt_text) {
            json!({
                "synthesized_signals": [
                    {
                        "asset": "MARKET",
                        "direction": "neutral",
                        "strength": 0.0,
                        "consensus_level": "weak",
                        "trading_instruction": "Hold / no trade while provider health recovers."
                    }
                ],
                "market_outlook": "neutral",
                "summary": format!(
                    "Degraded fallback: providers unavailable or in adaptive backoff ({reason})."
                )
            })
        } else if prompt_requests_manager_tool_call(prompt_text) {
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

#[derive(Clone, Debug)]
struct PromptAgentDescriptor {
    name: String,
    channel: Option<String>,
}

fn degraded_execution_plan_payload(prompt_text: &str) -> Value {
    let steps = build_degraded_execution_plan_steps(prompt_text);
    if execution_plan_prompt_needs_loop_fields(prompt_text) {
        json!({
            "steps": steps,
            "loop": false,
            "loop_condition": null,
            "max_iterations": null,
        })
    } else {
        json!({
            "steps": steps,
        })
    }
}

fn execution_plan_prompt_needs_loop_fields(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    lowered.contains("loop_condition")
        || lowered.contains("max_iterations")
        || lowered.contains("\"loop\"")
}

fn build_degraded_execution_plan_steps(prompt_text: &str) -> Vec<Value> {
    let agents = parse_prompt_agents(prompt_text);
    if agents.is_empty() {
        return Vec::new();
    }

    let channel_to_agent = agents
        .iter()
        .filter_map(|agent| {
            agent
                .channel
                .as_ref()
                .map(|channel| (channel.trim().to_ascii_lowercase(), agent.name.clone()))
        })
        .collect::<HashMap<_, _>>();

    let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
    for (source_channel, target_channel) in parse_prompt_relations(prompt_text) {
        let Some(source_agent) =
            channel_to_agent.get(source_channel.trim().to_ascii_lowercase().as_str())
        else {
            continue;
        };
        let Some(target_agent) =
            channel_to_agent.get(target_channel.trim().to_ascii_lowercase().as_str())
        else {
            continue;
        };
        let wait_for = dependencies.entry(target_agent.clone()).or_default();
        if !wait_for.iter().any(|existing| existing == source_agent) {
            wait_for.push(source_agent.clone());
        }
    }

    let mut fallback_steps = Vec::with_capacity(agents.len());
    let mut previous_agent: Option<String> = None;
    for agent in agents {
        let agent_name = agent.name;
        let mut wait_for = dependencies.remove(agent_name.as_str()).unwrap_or_default();
        if wait_for.is_empty()
            && let Some(previous) = previous_agent.as_ref()
        {
            wait_for.push(previous.clone());
        }
        let current_agent = agent_name.clone();
        fallback_steps.push(json!({
            "agent_name": agent_name,
            "instructions": [],
            "wait_for": wait_for,
            "skip": false,
            "step_type": "agent",
            "debate_config": null,
        }));
        previous_agent = Some(current_agent);
    }

    fallback_steps
}

fn parse_prompt_agents(prompt_text: &str) -> Vec<PromptAgentDescriptor> {
    let Some(section) = extract_labeled_section(
        prompt_text,
        "Agents:",
        &["Relations:", "Initial Data:", "Instructions:"],
    ) else {
        return Vec::new();
    };

    let mut agents = Vec::new();
    let mut seen = HashSet::new();

    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(section) {
        for item in items {
            let Some(name) = item.get("name").and_then(Value::as_str) else {
                continue;
            };
            let normalized = name.trim();
            if normalized.is_empty() {
                continue;
            }
            let dedupe_key = normalized.to_ascii_lowercase();
            if !seen.insert(dedupe_key) {
                continue;
            }
            let channel = item
                .get("channel")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            agents.push(PromptAgentDescriptor {
                name: normalized.to_string(),
                channel,
            });
        }
        if !agents.is_empty() {
            return agents;
        }
    }

    let names = extract_named_values_from_section(section, "name");
    let channels = extract_named_values_from_section(section, "channel");
    for (index, name) in names.into_iter().enumerate() {
        let normalized = name.trim();
        if normalized.is_empty() {
            continue;
        }
        let dedupe_key = normalized.to_ascii_lowercase();
        if !seen.insert(dedupe_key) {
            continue;
        }
        let channel = channels
            .get(index)
            .map(String::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        agents.push(PromptAgentDescriptor {
            name: normalized.to_string(),
            channel,
        });
    }

    agents
}

fn parse_prompt_relations(prompt_text: &str) -> Vec<(String, String)> {
    let Some(section) = extract_labeled_section(
        prompt_text,
        "Relations:",
        &["Initial Data:", "Instructions:"],
    ) else {
        return Vec::new();
    };

    if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(section) {
        let parsed = items
            .iter()
            .filter_map(|item| {
                let source = item.get("source")?.as_str()?.trim();
                let target = item.get("target")?.as_str()?.trim();
                if source.is_empty() || target.is_empty() {
                    return None;
                }
                Some((source.to_string(), target.to_string()))
            })
            .collect::<Vec<_>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }

    let sources = extract_named_values_from_section(section, "source");
    let targets = extract_named_values_from_section(section, "target");
    sources.into_iter().zip(targets).collect()
}

fn extract_labeled_section<'a>(
    text: &'a str,
    label: &str,
    end_markers: &[&str],
) -> Option<&'a str> {
    let start = text.find(label)?;
    let after_label = &text[start + label.len()..];
    let mut end = after_label.len();
    for marker in end_markers {
        if let Some(position) = after_label.find(marker) {
            end = end.min(position);
        }
    }
    Some(after_label[..end].trim())
}

fn extract_named_values_from_section(section: &str, field_name: &str) -> Vec<String> {
    let needle = format!("\"{field_name}\"");
    section
        .split(needle.as_str())
        .skip(1)
        .filter_map(|tail| {
            let (_, after_colon) = tail.split_once(':')?;
            let (_, after_start_quote) = after_colon.split_once('"')?;
            let (value, _) = after_start_quote.split_once('"')?;
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        })
        .collect()
}

fn prompt_requests_execution_plan(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    lowered.contains("executionplan")
        && lowered.contains("steps")
        && lowered.contains("additionalproperties")
}

fn prompt_requests_manager_tool_call(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    let has_shape = lowered.contains("tool_name") && lowered.contains("arguments");
    let manager_markers = [
        "managertoolcall",
        "run_agent",
        "run_debate",
        "finish",
        "agent_name",
        "debator_agent_names",
        "judge_agent_name",
        "team execution manager",
    ];
    lowered.contains("managertoolcall")
        || (has_shape
            && manager_markers
                .iter()
                .any(|marker| lowered.contains(marker)))
}

fn prompt_requests_signal_synthesis_output(prompt_text: &str) -> bool {
    let lowered = prompt_text.to_ascii_lowercase();
    lowered.contains("signalsynthesisoutput")
        || (lowered.contains("synthesized_signals")
            && lowered.contains("market_outlook")
            && lowered.contains("summary"))
}

fn classify_workload(workflow: &str, role: &str) -> WorkloadClass {
    classify_workload_with_context(workflow, role, None, None, None, None)
}

/// Classify a request before acquiring a global permit.  OctoBot's
/// OpenAI-compatible path can use a generic `direct`/`assistant` route while
/// still carrying a trading category or trading-shaped prompt.  Looking only
/// at workflow and role incorrectly puts those calls in the interactive pool,
/// allowing long planner/research work to starve trading advisories.
fn classify_workload_with_context(
    workflow: &str,
    role: &str,
    request_category: Option<&str>,
    request_profile: Option<&str>,
    source: Option<&str>,
    prompt_text: Option<&str>,
) -> WorkloadClass {
    let workflow_lower = workflow.to_ascii_lowercase();
    let role_lower = role.to_ascii_lowercase();
    let category_lower = request_category.unwrap_or_default().to_ascii_lowercase();
    let profile_lower = request_profile.unwrap_or_default().to_ascii_lowercase();
    let source_lower = source.unwrap_or_default().to_ascii_lowercase();
    let prompt_lower = prompt_text.unwrap_or_default().to_ascii_lowercase();
    let trading_markers = [
        "trading",
        "advisory",
        "buy/hold/sell",
        "buy_hold_sell",
        "target_symbol",
        "market data",
        "octobot",
    ];
    if trading_markers.iter().any(|marker| {
        workflow_lower.contains(marker)
            || role_lower.contains(marker)
            || category_lower.contains(marker)
            || profile_lower.contains(marker)
            || source_lower.contains(marker)
            || prompt_lower.contains(marker)
    }) {
        return WorkloadClass::Trading;
    }
    // Long-running solver/research traffic must not consume the interactive
    // pool merely because its caller uses a generic direct/assistant route.
    // The prompt is intentionally a secondary signal: explicit trading
    // markers above always win, while these markers route planning work to the
    // solver pool and leave interactive capacity available for short calls.
    let solver_markers = [
        "project_solver",
        "solver",
        "research",
        "planner",
        "researcher",
        "execution plan",
        "technical analysis",
        "code solution",
        "coding task",
        "refiner",
        "conductor",
    ];
    if solver_markers.iter().any(|marker| {
        workflow_lower.contains(marker)
            || role_lower.contains(marker)
            || profile_lower.contains(marker)
            || source_lower.contains(marker)
            || prompt_lower.contains(marker)
    }) {
        return WorkloadClass::Solver;
    }
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
    fn quality_score_rejects_obvious_non_answers_and_empty_structures() {
        assert!(quality_score("I cannot help with that.", false) < 0.5);
        assert!(quality_score("{}", true) < 0.5);
        assert!(quality_score("null", true) < 0.5);
        assert!(quality_score("A useful answer with enough detail.", false) >= 0.5);
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

    #[tokio::test]
    async fn readiness_performs_initial_probe_and_caches_all_unavailable_result() {
        let service = GailService::new(GailConfig::default())
            .await
            .expect("service should initialise");

        let first = service.readiness().await;
        assert!(!first.ready);
        assert_eq!(first.reason, "no_configured_providers");
        {
            let state = service.inner.readiness_cache.state.lock().await;
            assert!(state.value.is_some());
            assert!(!state.refresh_in_progress);
        }

        // A second call is served from the snapshot and does not invalidate
        // the result merely because the endpoint is polled again.
        assert_eq!(service.readiness().await.reason, first.reason);
    }

    #[tokio::test]
    async fn readiness_coalesces_concurrent_cold_start_callers() {
        let service = GailService::new(GailConfig::default())
            .await
            .expect("service should initialise");
        let cache = &service.inner.readiness_cache;
        {
            let mut state = cache.state.lock().await;
            state.refresh_in_progress = true;
        }

        let notify_service = service.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            let notify_cache = &notify_service.inner.readiness_cache;
            let mut state = notify_cache.state.lock().await;
            state.value = Some(CachedReadiness {
                response: ReadinessResponse {
                    ready: false,
                    service: "gail".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    build: crate::build_info::current(),
                    providers_checked: 1,
                    providers_ready: 0,
                    reason: "no_application_ready_providers".to_string(),
                },
                refreshed_at: Instant::now(),
            });
            state.refresh_in_progress = false;
            drop(state);
            notify_cache.refresh_finished.notify_waiters();
        });

        let response = service.readiness().await;
        assert_eq!(response.reason, "no_application_ready_providers");
    }

    #[tokio::test]
    async fn readiness_expiry_returns_snapshot_while_refreshing() {
        let service = GailService::new(GailConfig::default())
            .await
            .expect("service should initialise");
        let initial = service.readiness().await;
        {
            let mut state = service.inner.readiness_cache.state.lock().await;
            state
                .value
                .as_mut()
                .expect("initial readiness should be cached")
                .refreshed_at = Instant::now() - Duration::from_secs(121);
        }

        // The expired value remains available while one refresh runs. This
        // is what keeps kubelet from treating a slow probe as process failure.
        let response = service.readiness().await;
        assert_eq!(response.reason, initial.reason);
        for _ in 0..20 {
            if !service
                .inner
                .readiness_cache
                .state
                .lock()
                .await
                .refresh_in_progress
            {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("readiness refresh did not finish");
    }

    #[test]
    fn explicit_json_request_category_keeps_json_contract() {
        assert!(request_category_expects_json(Some("json")));
        assert!(request_category_expects_json(Some("structured_data")));
        assert!(!request_category_expects_json(Some(
            "assistant_requirements"
        )));
    }

    #[test]
    fn conversational_assistant_output_is_not_json_gated_by_routing_profile() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(
                "Please explain the requirements in plain language and suggest next steps."
                    .to_string(),
            ),
        }];
        let tags = workflow_tags("assistant_requirements", "assistant", "requirements");

        assert!(tags.contains("json"));
        assert!(!expected_json(&messages, None));
        assert!(
            quality_score(
                "The requirements are clear. Start by validating the input, then run the review.",
                false,
            ) >= 0.5
        );
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
    fn degraded_fallback_execution_plan_uses_prompt_agents_when_available() {
        let prompt = r#"Analyze the following team structure and create an execution plan:
Team: SimpleAIEvaluatorAgentsTeam
Agents: [
  {"name":"SentimentAnalysisAIAgentProducer","channel":"SentimentAnalysisAIAgentChannel"},
  {"name":"SummarizationAIAgentProducer","channel":"SummarizationAIAgentChannel"}
]
Relations: [
  {"source":"SentimentAnalysisAIAgentChannel","target":"SummarizationAIAgentChannel"}
]
Initial Data: {}
Instructions: None
Return only a JSON data instance that satisfies this schema:
{"name":"ExecutionPlan","schema":{"type":"object","properties":{"steps":{"type":"array"},"loop":{"type":"boolean"},"loop_condition":{"anyOf":[{"type":"string"},{"type":"null"}]},"max_iterations":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["steps","loop","loop_condition","max_iterations"],"additionalProperties":false}}"#;
        let text = degraded_fallback_text(
            true,
            "general",
            "general",
            prompt,
            &["provider unavailable".to_string()],
        );
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        let steps = value["steps"].as_array().expect("steps array");
        assert_eq!(steps.len(), 2);
        assert_eq!(
            steps[0]["agent_name"],
            serde_json::Value::String("SentimentAnalysisAIAgentProducer".to_string())
        );
        assert_eq!(
            steps[1]["agent_name"],
            serde_json::Value::String("SummarizationAIAgentProducer".to_string())
        );
        assert_eq!(
            steps[1]["wait_for"],
            serde_json::json!(["SentimentAnalysisAIAgentProducer"])
        );
        assert_eq!(value["loop"], serde_json::Value::Bool(false));
        assert_eq!(value["loop_condition"], serde_json::Value::Null);
        assert_eq!(value["max_iterations"], serde_json::Value::Null);
    }

    #[test]
    fn degraded_fallback_matches_signal_synthesis_schema() {
        let prompt = "Return only valid JSON for SignalSynthesisOutput with synthesized_signals, market_outlook, and summary.";
        let text = degraded_fallback_text(
            true,
            "trading",
            "assistant",
            prompt,
            &["provider unavailable".to_string()],
        );
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        let object = value.as_object().expect("object");
        assert!(object.contains_key("synthesized_signals"));
        assert_eq!(value["market_outlook"], "neutral");
        assert!(
            value["summary"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        );
        assert_eq!(
            value["synthesized_signals"]
                .as_array()
                .expect("synthesized_signals array")
                .len(),
            1
        );
        assert!(
            !object.contains_key("action"),
            "signal synthesis fallback should avoid unrelated hold/action keys"
        );
    }

    #[test]
    fn degraded_fallback_prefers_signal_synthesis_over_manager_tool_call() {
        let prompt = "Return only valid JSON for SignalSynthesisOutput with synthesized_signals, market_outlook, and summary. Do not return ManagerToolCall fields like tool_name and arguments.";
        let text = degraded_fallback_text(
            true,
            "trading",
            "assistant",
            prompt,
            &["provider unavailable".to_string()],
        );
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        let object = value.as_object().expect("object");
        assert!(object.contains_key("synthesized_signals"));
        assert!(!object.contains_key("tool_name"));
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
        assert!(!has_usable_value(Some("${GEMINI_API_KEY}")));
        assert!(!has_usable_value(Some("{{ vault_gemini_api_key }}")));
        assert!(!has_usable_value(Some("changeme")));
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
                generation_tokens_per_second: None,
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
                ranked("openai", "gpt-5.3-codex", 6.0, false),
            ],
            3,
            true,
        );

        let labels = selected
            .iter()
            .map(|candidate| candidate.candidate_id())
            .collect::<Vec<_>>();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "nvidia/moonshotai/kimi-k2-instruct-0905");
        assert!(labels[1].starts_with("ollama/llama3.2"));
        assert!(labels[2].starts_with("openai/gpt-5.3-codex@"));
    }

    #[test]
    fn select_ranked_candidates_prefers_available_35b_over_smaller_models() {
        let ranked = |provider: &str, model: &str, health_ok: bool| RankedCandidate {
            score: 1.0,
            health_ok,
            health_mode: (!health_ok).then(|| "resource_saturated".to_string()),
            generation_tokens_per_second: None,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                name: format!("{provider}-{model}"),
                provider_type: provider.to_string(),
                model: Some(model.to_string()),
                api_key: Some("token".to_string()),
                base_url: Some(format!("http://{provider}.internal")),
                ..ProviderProfile::default()
            }),
        };

        let selected = select_ranked_candidates(
            vec![
                ranked("openai", "qwen3.6:35b", true),
                ranked("ollama", "qwen3.5:9b", true),
                ranked("ollama", "qwen3.5:4b", true),
            ],
            1,
            true,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].configured_model, "qwen3.6:35b");
    }

    #[test]
    fn select_ranked_candidates_falls_from_busy_35b_to_available_9b() {
        let ranked = |provider: &str, model: &str, health_ok: bool| RankedCandidate {
            score: 1.0,
            health_ok,
            health_mode: (!health_ok).then(|| "resource_saturated".to_string()),
            generation_tokens_per_second: None,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                name: format!("{provider}-{model}"),
                provider_type: provider.to_string(),
                model: Some(model.to_string()),
                api_key: Some("token".to_string()),
                base_url: Some(format!("http://{provider}.internal")),
                ..ProviderProfile::default()
            }),
        };

        let selected = select_ranked_candidates(
            vec![
                ranked("openai", "qwen3.6:35b", false),
                ranked("ollama9", "qwen3.5:9b", true),
                ranked("ollama4", "qwen3.5:4b", true),
            ],
            1,
            true,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].configured_model, "qwen3.5:9b");
    }

    #[test]
    fn select_ranked_candidates_falls_from_busy_35b_and_9b_to_4b() {
        let ranked = |provider: &str, model: &str, health_ok: bool| RankedCandidate {
            score: 1.0,
            health_ok,
            health_mode: (!health_ok).then(|| "resource_saturated".to_string()),
            generation_tokens_per_second: None,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                name: format!("{provider}-{model}"),
                provider_type: provider.to_string(),
                model: Some(model.to_string()),
                api_key: Some("token".to_string()),
                base_url: Some(format!("http://{provider}.internal")),
                ..ProviderProfile::default()
            }),
        };

        let selected = select_ranked_candidates(
            vec![
                ranked("openai", "qwen3.6:35b", false),
                ranked("ollama9", "qwen3.5:9b", false),
                ranked("ollama4", "qwen3.5:4b", true),
            ],
            1,
            true,
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].configured_model, "qwen3.5:4b");
    }

    #[test]
    fn model_size_tiers_match_local_pool_order() {
        assert_eq!(model_size_tier("qwen3.6:35b"), 3);
        assert_eq!(model_size_tier("qwen3.5:9b"), 2);
        assert_eq!(model_size_tier("qwen3.5:4b"), 1);
        assert_eq!(model_size_tier("unlabelled-model"), 0);
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
                generation_tokens_per_second: None,
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
            true,
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
    fn adaptive_selection_prefers_idle_candidate_over_busy_fast_candidate() {
        fn ranked(host: &str) -> RankedCandidate {
            RankedCandidate {
                score: 1.0,
                health_ok: true,
                health_mode: None,
                generation_tokens_per_second: Some(40.0),
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: host.to_string(),
                    provider_type: "openai".to_string(),
                    model: Some("qwen3.5:9b".to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some(format!("http://{host}:18080/v1")),
                    ..ProviderProfile::default()
                }),
            }
        }

        let busy_fast = ranked("busy-fast");
        let idle_slow = ranked("idle-slow");
        let mut estimates = HashMap::new();
        estimates.insert(
            busy_fast.candidate.candidate_id(),
            CandidateDispatchEstimate {
                samples: 20,
                useful_rate: 1.0,
                service_time_ms: 1_000.0,
                queue_depth: 6,
                candidate_parallelism: 4,
                estimated_completion_ms: 7_000.0,
                estimated_useful_completion_ms: 7_000.0,
            },
        );
        estimates.insert(
            idle_slow.candidate.candidate_id(),
            CandidateDispatchEstimate {
                samples: 20,
                useful_rate: 0.9,
                service_time_ms: 3_000.0,
                queue_depth: 0,
                candidate_parallelism: 4,
                estimated_completion_ms: 3_000.0,
                estimated_useful_completion_ms: 3_333.0,
            },
        );

        let selected = select_adaptive_candidates(vec![busy_fast, idle_slow], 1, false, &estimates);
        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].profile.base_url.as_deref(),
            Some("http://idle-slow:18080/v1")
        );
    }

    #[test]
    fn adaptive_selection_races_only_near_equal_useful_etas() {
        fn ranked(host: &str) -> RankedCandidate {
            RankedCandidate {
                score: 1.0,
                health_ok: true,
                health_mode: None,
                generation_tokens_per_second: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: host.to_string(),
                    provider_type: "openai".to_string(),
                    model: Some("qwen3.5:9b".to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some(format!("http://{host}:18080/v1")),
                    ..ProviderProfile::default()
                }),
            }
        }

        let first = ranked("first");
        let near = ranked("near");
        let far = ranked("far");
        let mut estimates = HashMap::new();
        for (candidate, eta) in [(&first, 1_000.0), (&near, 1_500.0), (&far, 8_000.0)] {
            estimates.insert(
                candidate.candidate.candidate_id(),
                CandidateDispatchEstimate {
                    samples: 0,
                    useful_rate: 1.0,
                    service_time_ms: eta,
                    queue_depth: 0,
                    candidate_parallelism: 4,
                    estimated_completion_ms: eta,
                    estimated_useful_completion_ms: eta,
                },
            );
        }

        let selected = select_adaptive_candidates(vec![first, near, far], 8, false, &estimates);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|candidate| {
            candidate.candidate_id().contains("first") || candidate.candidate_id().contains("near")
        }));
    }

    #[test]
    fn adaptive_policy_converges_from_seed_to_observed_service_time() {
        let estimates = HashMap::from([
            (
                "fast".to_string(),
                CandidateDispatchEstimate {
                    samples: 20,
                    useful_rate: 1.0,
                    service_time_ms: 200.0,
                    queue_depth: 0,
                    candidate_parallelism: 6,
                    estimated_completion_ms: 200.0,
                    estimated_useful_completion_ms: 200.0,
                },
            ),
            (
                "near".to_string(),
                CandidateDispatchEstimate {
                    samples: 20,
                    useful_rate: 1.0,
                    service_time_ms: 220.0,
                    queue_depth: 0,
                    candidate_parallelism: 6,
                    estimated_completion_ms: 220.0,
                    estimated_useful_completion_ms: 220.0,
                },
            ),
            (
                "far".to_string(),
                CandidateDispatchEstimate {
                    samples: 20,
                    useful_rate: 1.0,
                    service_time_ms: 1_000.0,
                    queue_depth: 0,
                    candidate_parallelism: 6,
                    estimated_completion_ms: 1_000.0,
                    estimated_useful_completion_ms: 1_000.0,
                },
            ),
        ]);
        let policy = adaptive_dispatch_policy(&estimates, 8);
        assert!(policy.race_window_ms < 750.0);
        assert!(policy.race_window_ms >= 100.0);
        assert_eq!(policy.max_raced_candidates, 2);
    }

    #[test]
    fn provider_concurrency_seed_adjusts_with_useful_queue_capacity() {
        let seed = default_provider_concurrency_seed();
        let healthy = crate::metrics::CandidateCapacityEstimate {
            samples: 20,
            useful_rate: 0.95,
            service_time_ms: Some(1_000.0),
            queue_wait_ms: Some(20.0),
            ..crate::metrics::CandidateCapacityEstimate::default()
        };
        let constrained = crate::metrics::CandidateCapacityEstimate {
            samples: 20,
            useful_rate: 0.4,
            service_time_ms: Some(1_000.0),
            queue_wait_ms: Some(900.0),
            ..crate::metrics::CandidateCapacityEstimate::default()
        };
        assert_eq!(adaptive_provider_parallelism(&healthy), (seed + 2).min(64));
        assert_eq!(
            adaptive_provider_parallelism(&constrained),
            seed.saturating_sub(1).max(1)
        );
    }

    #[test]
    fn select_ranked_candidates_can_include_multiple_endpoints_when_deduplication_is_disabled() {
        fn ranked(base_url: &str, score: f64) -> RankedCandidate {
            RankedCandidate {
                score,
                health_ok: true,
                health_mode: None,
                generation_tokens_per_second: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: format!("llamacpp-{}", base_url.replace(':', "-")),
                    provider_type: "openai".to_string(),
                    model: Some("qwen3-32b-centriq2400".to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some(base_url.to_string()),
                    ..ProviderProfile::default()
                }),
            }
        }

        let selected = select_ranked_candidates(
            vec![
                ranked("http://192.168.1.60:18080/v1", 5.0),
                ranked("http://192.168.1.62:18080/v1", 4.9),
            ],
            2,
            false,
        );

        assert_eq!(selected.len(), 2);
        let selected_base_urls = selected
            .iter()
            .map(|candidate| candidate.profile.base_url.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        assert!(
            selected_base_urls
                .iter()
                .any(|url| url == "http://192.168.1.60:18080/v1")
        );
        assert!(
            selected_base_urls
                .iter()
                .any(|url| url == "http://192.168.1.62:18080/v1")
        );
    }

    #[test]
    fn select_ranked_candidates_orders_healthy_endpoints_by_generation_throughput() {
        let ranked = |host: &str, throughput: Option<f64>, health_ok: bool| RankedCandidate {
            score: 1.0,
            health_ok,
            health_mode: (!health_ok).then(|| "error".to_string()),
            generation_tokens_per_second: throughput,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                name: host.to_string(),
                provider_type: "openai".to_string(),
                model: Some("qwen3.5:9b".to_string()),
                api_key: Some("token".to_string()),
                base_url: Some(format!("http://{host}:18080/v1")),
                ..ProviderProfile::default()
            }),
        };

        let selected = select_ranked_candidates(
            vec![
                ranked("slow", Some(3.5), true),
                ranked("fast", Some(45.6), true),
                ranked("middle", Some(31.7), true),
                ranked("offline", Some(120.0), false),
            ],
            4,
            false,
        );
        let hosts = selected
            .iter()
            .map(|candidate| candidate.profile.base_url.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(
            hosts,
            vec![
                "http://fast:18080/v1",
                "http://middle:18080/v1",
                "http://slow:18080/v1",
                "http://offline:18080/v1",
            ]
        );
    }

    #[test]
    fn select_ranked_candidates_deduplicates_same_model_across_endpoints() {
        let ranked = |base_url: &str, score: f64| RankedCandidate {
            score,
            health_ok: true,
            health_mode: None,
            generation_tokens_per_second: None,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                name: base_url.to_string(),
                provider_type: "openai".to_string(),
                model: Some("qwen3.6:35b".to_string()),
                api_key: Some("token".to_string()),
                base_url: Some(base_url.to_string()),
                ..ProviderProfile::default()
            }),
        };
        let selected = select_ranked_candidates(
            vec![
                ranked("http://qc02:18080/v1", 5.0),
                ranked("http://qc03:18080/v1", 4.0),
            ],
            2,
            true,
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].profile.base_url.as_deref(),
            Some("http://qc02:18080/v1")
        );
    }

    #[test]
    fn round_robin_reorder_rotates_equivalent_endpoints() {
        fn ranked(base_url: &str, score: f64) -> RankedCandidate {
            RankedCandidate {
                score,
                health_ok: true,
                health_mode: None,
                generation_tokens_per_second: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    name: format!("llamacpp-{}", base_url.replace(':', "-")),
                    provider_type: "openai".to_string(),
                    model: Some("qwen3.6:35b".to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some(base_url.to_string()),
                    ..ProviderProfile::default()
                }),
            }
        }

        let ranked_candidates = vec![
            ranked("http://192.168.1.60:18080/v1", 5.0),
            ranked("http://192.168.1.62:18080/v1", 4.9),
            ranked("http://192.168.1.63:18080/v1", 4.8),
            RankedCandidate {
                score: 1.0,
                health_ok: true,
                health_mode: None,
                generation_tokens_per_second: None,
                candidate: ProviderCandidate::from_profile(ProviderProfile {
                    provider_type: "ollama".to_string(),
                    model: Some("qwen3.5:4b".to_string()),
                    api_key: Some("token".to_string()),
                    base_url: Some("http://ollama.internal:11434".to_string()),
                    ..ProviderProfile::default()
                }),
            },
        ];

        let context = round_robin_context(&ranked_candidates, "general", "assistant")
            .expect("round-robin context");
        let rotated = reorder_ranked_candidates_for_round_robin(ranked_candidates, &context, 1);
        let selected = select_ranked_candidates(rotated, 1, true);
        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].profile.base_url.as_deref(),
            Some("http://192.168.1.62:18080/v1")
        );
    }

    #[test]
    fn endpoint_scoped_candidates_do_not_trigger_provider_family_backoff() {
        let local_openai = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "openai".to_string(),
            model: Some("qwen3.6:35b".to_string()),
            api_key: Some("token".to_string()),
            base_url: Some("http://192.168.1.60:18080/v1".to_string()),
            ..ProviderProfile::default()
        });
        assert!(!candidate_uses_provider_family_backoff(&local_openai));
        assert!(!error_should_backoff_provider_family(
            &local_openai,
            "upstream error: error sending request",
        ));
        assert!(error_should_backoff_provider_family(
            &local_openai,
            "status 401 unauthorized",
        ));

        let shared_openai = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "openai".to_string(),
            model: Some("gpt-5".to_string()),
            api_key: Some("token".to_string()),
            base_url: None,
            ..ProviderProfile::default()
        });
        assert!(candidate_uses_provider_family_backoff(&shared_openai));
        assert!(error_should_backoff_provider_family(
            &shared_openai,
            "upstream error: bad gateway",
        ));
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
                generation_tokens_per_second: None,
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
                ranked("openai", "gpt-5.3-codex", 9.0, true),
                ranked("nvidia", "moonshotai/kimi-k2-instruct-0905", 8.0, true),
                ranked("gemini", "gemini-2.5-flash", 7.0, true),
                ranked("ollama", "llama3.2", 1.0, false),
            ],
            3,
            true,
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
    fn candidate_id_scopes_openai_endpoints() {
        let first = ProviderCandidate::from_profile(ProviderProfile {
            name: "LlamaCppQC00".to_string(),
            provider_type: "openai".to_string(),
            model: Some("qwen3-32b-centriq2400".to_string()),
            api_key: Some("token".to_string()),
            base_url: Some("http://192.168.1.60:18080/v1".to_string()),
            ..ProviderProfile::default()
        });
        let second = ProviderCandidate::from_profile(ProviderProfile {
            name: "LlamaCppQC02".to_string(),
            provider_type: "openai".to_string(),
            model: Some("qwen3-32b-centriq2400".to_string()),
            api_key: Some("token".to_string()),
            base_url: Some("http://192.168.1.62:18080/v1".to_string()),
            ..ProviderProfile::default()
        });
        let openai_cloud = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "openai".to_string(),
            model: Some("gpt-5.3-codex".to_string()),
            api_key: Some("token".to_string()),
            ..ProviderProfile::default()
        });

        assert_ne!(first.candidate_id(), second.candidate_id());
        assert!(
            first
                .candidate_id()
                .starts_with("openai/qwen3-32b-centriq2400@")
        );
        assert!(
            second
                .candidate_id()
                .starts_with("openai/qwen3-32b-centriq2400@")
        );
        assert_eq!(openai_cloud.candidate_id(), "openai/gpt-5.3-codex");
    }

    #[test]
    fn endpoint_telemetry_summarizes_scoped_success_and_failure_counts() {
        fn candidate_metric(
            candidate_id: &str,
            provider: &str,
            model: &str,
            successes: u64,
            failures: u64,
        ) -> CandidateMetricsSummary {
            let total = successes + failures;
            CandidateMetricsSummary {
                candidate_id: candidate_id.to_string(),
                host_id: Some("test-host".to_string()),
                provider: Some(provider.to_string()),
                model: Some(model.to_string()),
                configured_model: Some(model.to_string()),
                resolved_model: Some(model.to_string()),
                specialties: Vec::new(),
                successes,
                failures,
                total,
                average_latency_ms: Some(420.0),
                min_latency_ms: Some(100),
                max_latency_ms: Some(900),
                success_rate: if total > 0 {
                    Some(successes as f64 / total as f64)
                } else {
                    None
                },
                ewma_latency_ms: Some(420.0),
                successful_latency_samples: 3,
                successful_latency_average_ms: Some(420.0),
                successful_latency_min_ms: Some(100),
                successful_latency_max_ms: Some(900),
                successful_latency_ewma_ms: Some(420.0),
                failed_latency_samples: 0,
                failed_latency_average_ms: None,
                failed_latency_min_ms: None,
                failed_latency_max_ms: None,
                failed_latency_ewma_ms: None,
                stale_failures: 0,
                queue_wait_average_ms: Some(80.0),
                queue_wait_min_ms: Some(10),
                queue_wait_max_ms: Some(200),
                ewma_queue_wait_ms: Some(80.0),
                ewma_inference_ms: Some(340.0),
                ewma_prompt_tokens: None,
                ewma_tokens_estimate: None,
                ewma_completion_tokens: None,
                generation_tokens_per_second_average: Some(20.0),
                generation_tokens_per_second_min: Some(10.0),
                generation_tokens_per_second_max: Some(30.0),
                generation_tokens_per_second_ewma: Some(20.0),
                ewma_quality: 0.5,
                last_status: Some("success".to_string()),
                last_error: None,
                updated_at: Some(1_789_999_999.0),
                health_ok: Some(true),
                health_mode: Some("runtime_completion".to_string()),
                health_checked_at: Some(1_789_999_999.0),
                routing_profiles: std::collections::HashMap::new(),
            }
        }

        let rows = summarize_endpoint_telemetry(&[
            candidate_metric(
                "openai/qwen3.6:35b@192_168_1_60_18080_v1",
                "openai",
                "qwen3.6:35b",
                21,
                63,
            ),
            candidate_metric(
                "openai/qwen3.6:35b@192_168_1_62_18080_v1",
                "openai",
                "qwen3.6:35b",
                18,
                49,
            ),
            candidate_metric("openai/gpt-5.3-codex", "openai", "gpt-5.3-codex", 5, 1),
        ]);

        assert_eq!(rows.len(), 2, "unscoped candidates should be excluded");
        assert_eq!(rows[0].endpoint_scope, "192_168_1_60_18080_v1");
        assert_eq!(rows[0].endpoint_host.as_deref(), Some("192.168.1.60"));
        assert_eq!(rows[0].endpoint_suffix.as_deref(), Some(".60"));
        assert_eq!(rows[0].successes, 21);
        assert_eq!(rows[0].failures, 63);
        assert_eq!(rows[1].endpoint_scope, "192_168_1_62_18080_v1");
        assert_eq!(rows[1].endpoint_host.as_deref(), Some("192.168.1.62"));
        assert_eq!(rows[1].endpoint_suffix.as_deref(), Some(".62"));
    }

    #[tokio::test]
    async fn load_reservation_guard_drop_releases_candidate_slot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut config = GailConfig::default();
        config.storage.metrics_path = temp.path().join("metrics.json").display().to_string();
        config.storage.adaptive_schema_path =
            temp.path().join("adaptive.json").display().to_string();
        config.storage.api_issues_path = temp.path().join("api_issues.json").display().to_string();
        config.storage.llm_ledger_path = temp.path().join("llm_ledger.jsonl").display().to_string();
        config.storage.trainer_output_path = temp.path().join("training").display().to_string();
        config.providers = vec![ProviderProfile {
            name: "LlamaCppQC00".to_string(),
            provider_type: "openai".to_string(),
            model: Some("qwen3-32b-centriq2400".to_string()),
            api_key: Some("token".to_string()),
            base_url: Some("http://192.168.1.60:18080/v1".to_string()),
            max_concurrent_requests: Some(1),
            ..ProviderProfile::default()
        }];
        let service = GailService::new(config).await.expect("service");
        let candidate = ProviderCandidate::from_profile(service.config().providers[0].clone());

        let reservation = service
            .reserve_candidate_load(&candidate)
            .await
            .expect("reserve load");
        let guard = LoadReservationGuard::new(service.clone(), reservation);
        drop(guard);
        tokio::time::sleep(Duration::from_millis(20)).await;

        let second = service
            .reserve_candidate_load(&candidate)
            .await
            .expect("candidate reservation should be released after guard drop");

        // A dispatch-time race must be handled by the caller's fallback
        // wave, not by entering the long queue wait intended for a candidate
        // that was already known to be busy during ranking.
        let race_started = Instant::now();
        assert!(
            service.reserve_candidate_load(&candidate).await.is_none(),
            "the single-slot candidate must reject the competing reservation"
        );
        assert!(
            race_started.elapsed() < Duration::from_secs(1),
            "a reservation race must return without queueing"
        );

        // The next request must remain pending without polling, then wake as
        // soon as the active reservation releases capacity.
        let waiting_service = service.clone();
        let waiting_candidate = candidate.clone();
        let waiter = tokio::spawn(async move {
            waiting_service
                .reserve_candidate_load_with_backpressure(&waiting_candidate)
                .await
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "waiter should queue at the limit");

        service.release_candidate_load(second).await;
        let queued_reservation = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("capacity release should wake the waiter")
            .expect("waiter task should complete")
            .expect("waiter should acquire released capacity");
        service.release_candidate_load(queued_reservation).await;
        let tracker = service.inner.load_tracker.lock().await;
        assert_eq!(
            tracker.candidate_waiting.get(&candidate.candidate_id()),
            None
        );
    }

    #[test]
    fn ranked_candidate_quota_backoff_detects_cached_quota_health() {
        let candidate = RankedCandidate {
            score: 1.0,
            health_ok: false,
            health_mode: Some("quota".to_string()),
            generation_tokens_per_second: None,
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
    fn ranked_candidate_health_error_enters_provider_backoff() {
        let candidate = RankedCandidate {
            score: 1.0,
            health_ok: false,
            health_mode: Some("error".to_string()),
            generation_tokens_per_second: None,
            candidate: ProviderCandidate::from_profile(ProviderProfile {
                provider_type: "openai".to_string(),
                model: Some("gail-inhouse:latest".to_string()),
                base_url: Some("http://192.168.1.66:18081/v1".to_string()),
                api_key: Some("token".to_string()),
                source: Some("ansible_llamacpp_trained".to_string()),
                ..ProviderProfile::default()
            }),
        };
        assert!(ranked_candidate_is_in_provider_backoff(&candidate));
    }

    #[test]
    fn trained_llamacpp_profiles_are_secondary_to_generic_routing() {
        let trained = ProviderProfile {
            provider_type: "openai".to_string(),
            model: Some("gail-inhouse:latest".to_string()),
            source: Some("ansible_llamacpp_trained".to_string()),
            ..ProviderProfile::default()
        };
        let primary = ProviderProfile {
            provider_type: "openai".to_string(),
            model: Some("qwen3.5:9b".to_string()),
            source: Some("ansible_llamacpp".to_string()),
            ..ProviderProfile::default()
        };

        assert!(is_trained_llamacpp_profile(&trained));
        assert!(!is_trained_llamacpp_profile(&primary));
        assert!(!is_trained_llamacpp_profile(&ProviderProfile::default()));
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
    fn dispatch_capacity_race_is_not_provider_failure() {
        assert!(is_dispatch_capacity_race(Some(
            "candidate capacity was unavailable at dispatch (reservation race; trying fallback candidates)"
        )));
        assert!(!is_dispatch_capacity_race(Some(
            "candidate capacity was unavailable at dispatch after 250ms of queue waiting"
        )));
        assert!(!is_dispatch_capacity_race(Some("connection refused")));
    }

    #[test]
    fn cached_health_ttl_keeps_ollama_transient_failures_short_lived() {
        let default_ttl = 1800.0;
        let timeout_ttl = cached_health_ttl_seconds(true, false, Some("timeout"), default_ttl);
        let upstream_ttl = cached_health_ttl_seconds(true, false, Some("upstream"), default_ttl);
        let saturation_ttl =
            cached_health_ttl_seconds(true, false, Some("ollama_saturated"), default_ttl);

        assert!(timeout_ttl >= 1.0 && timeout_ttl <= 120.0);
        assert!(upstream_ttl >= 1.0 && upstream_ttl <= 120.0);
        assert!(saturation_ttl >= 1.0 && saturation_ttl <= 120.0);
        assert_eq!(
            cached_health_ttl_seconds(false, false, Some("timeout"), default_ttl),
            default_ttl
        );
    }

    #[test]
    fn cached_health_ttl_keeps_ansible_llamacpp_restarts_short_lived() {
        let default_ttl = 1800.0;
        let local_ttl = cached_health_ttl_seconds(true, true, None, default_ttl);
        assert!(local_ttl >= 5.0 && local_ttl <= 120.0);
        assert_eq!(
            cached_health_ttl_seconds(false, true, Some("upstream"), default_ttl),
            local_ttl
        );
    }

    #[test]
    fn trained_ansible_llamacpp_profiles_use_short_health_ttl() {
        let candidate = ProviderCandidate::from_profile(ProviderProfile {
            source: Some("ansible_llamacpp_trained".to_string()),
            provider_type: "openai".to_string(),
            model: Some("gail-inhouse:latest".to_string()),
            base_url: Some("http://127.0.0.1:18081/v1".to_string()),
            ..ProviderProfile::default()
        });

        assert!(is_local_llamacpp_candidate(&candidate));
        assert!(
            cached_health_ttl_seconds(false, is_local_llamacpp_candidate(&candidate), None, 1800.0,)
                <= 120.0
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
        assert_eq!(
            classify_workload("trading", "assistant"),
            WorkloadClass::Trading
        );
        assert_eq!(
            classify_workload("direct", "trading"),
            WorkloadClass::Trading
        );
    }

    #[test]
    fn classify_workload_routes_generic_octobot_advisories_to_reserved_pool() {
        assert_eq!(
            classify_workload_with_context(
                "direct",
                "assistant",
                Some("trading_advisory"),
                None,
                None,
                None,
            ),
            WorkloadClass::Trading
        );
        assert_eq!(
            classify_workload_with_context(
                "direct",
                "assistant",
                None,
                Some("octobot_trading"),
                None,
                None,
            ),
            WorkloadClass::Trading
        );
    }

    #[test]
    fn classify_workload_routes_generic_solver_prompts_away_from_interactive_pool() {
        assert_eq!(
            classify_workload_with_context(
                "direct",
                "assistant",
                None,
                None,
                Some("refiner"),
                Some("Research the problem and propose a coding solution."),
            ),
            WorkloadClass::Solver
        );
        assert_eq!(
            classify_workload_with_context(
                "general",
                "general",
                None,
                None,
                Some("continuum"),
                Some("Create an execution plan for the planner."),
            ),
            WorkloadClass::Solver
        );
    }

    #[test]
    fn configured_candidates_are_filtered_to_declared_workflow_roles() {
        let planner = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "ollama".to_string(),
            model: Some("planner".to_string()),
            roles: vec!["planner".to_string()],
            ..ProviderProfile::default()
        });
        let generalist = ProviderCandidate::from_profile(ProviderProfile {
            provider_type: "ollama".to_string(),
            model: Some("generalist".to_string()),
            roles: vec![
                "general".to_string(),
                "planner".to_string(),
                "reviewer".to_string(),
                "researcher".to_string(),
            ],
            ..ProviderProfile::default()
        });

        assert!(candidate_supports_role(&planner, "planner"));
        assert!(!candidate_supports_role(&planner, "researcher"));
        assert!(candidate_supports_role(&generalist, "researcher"));
    }

    #[test]
    fn configured_candidates_are_included_for_preferred_provider_fallback() {
        let request = CompletionRequest {
            request_id: None,
            workflow: Some("project_solver".to_string()),
            role: Some("planner".to_string()),
            preferred_provider: Some("openai".to_string()),
            preferred_model: Some("gpt-5.3-codex".to_string()),
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
            source: None,
            request_profile: None,
        };
        assert!(should_include_configured_candidates(false, &request, true));
    }

    #[test]
    fn configured_candidates_respect_explicit_non_preferred_request_mode() {
        let request = CompletionRequest {
            request_id: None,
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
            source: None,
            request_profile: None,
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
            Some("gpt-5.3-codex"),
            false,
        ));
    }

    #[test]
    fn configured_model_matches_request_for_exact_provider_and_model() {
        let config = GailConfig {
            providers: vec![ProviderProfile {
                name: "trained".to_string(),
                provider_type: "openai".to_string(),
                model: Some("gail-inhouse:latest".to_string()),
                base_url: Some("http://sm00:18081/v1".to_string()),
                ..ProviderProfile::default()
            }],
            ..GailConfig::default()
        };

        assert!(configured_model_matches_request(
            &config,
            "OPENAI",
            Some("GAIL-INHOUSE:LATEST"),
        ));
        assert!(!configured_model_matches_request(
            &config,
            "openai",
            Some("other-model"),
        ));
        assert!(!configured_model_matches_request(
            &config,
            "ollama",
            Some("gail-inhouse:latest"),
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

    #[test]
    fn transient_backoff_probe_target_keeps_minimum_two_candidates_for_small_waves() {
        assert_eq!(transient_backoff_probe_target_with_config(1, 5, 2), 2,);
    }

    #[test]
    fn transient_backoff_probe_target_never_exceeds_remaining_candidates() {
        assert_eq!(transient_backoff_probe_target_with_config(4, 3, 8), 3,);
    }
}
