use std::{fs, path::Path};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    errors::{GailError, Result},
    models::{AarnnResponsePreference, SelectionMode},
    trading::config::TradingConfig,
};

pub const MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS: u64 = 1_200_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct GailConfig {
    pub server: ServerConfig,
    pub security: SecurityConfig,
    pub orchestration: OrchestrationConfig,
    pub llm_ledger: LlmLedgerConfig,
    pub mirror_worker: MirrorWorkerConfig,
    pub trainer: TrainerConfig,
    pub aarnn_bridge: AarnnBridgeConfig,
    pub audit_logging: AuditLoggingConfig,
    pub nmc_telemetry: NmcTelemetryConfig,
    pub providers: Vec<ProviderProfile>,
    pub specialists: Vec<SpecialistProfile>,
    pub storage: StorageConfig,
    pub trading: TradingConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub public_base_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub api_tokens: Vec<ApiTokenConfig>,
    pub allow_unauthenticated_health: bool,
    pub allow_unauthenticated_metrics: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ApiTokenConfig {
    pub client_id: String,
    pub token: String,
    pub scopes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OrchestrationConfig {
    pub enabled: bool,
    pub selection_mode: SelectionMode,
    pub max_parallel_candidates: usize,
    pub interactive_pool_max_in_flight: usize,
    pub solver_pool_max_in_flight: usize,
    pub workload_pool_wait_timeout_ms: u64,
    /// Bounded backpressure wait for a provider/host resource reservation.
    pub candidate_queue_wait_timeout_ms: u64,
    /// Prevent equivalent provider/model work from running twice in one wave.
    pub deduplicate_model_candidates: bool,
    /// Compact oversized prompt histories before they reach a provider.
    pub prompt_compaction_enabled: bool,
    /// Context window assumed for local Ollama/OpenAI-compatible profiles
    /// without an explicit provider-level value.
    #[serde(alias = "default_ollama_context_window_tokens")]
    pub default_local_context_window_tokens: usize,
    /// Conservative tokenizer-independent estimate used for prompt budgeting.
    pub prompt_chars_per_token: usize,
    /// Space reserved for provider chat templates and token-estimation variance.
    pub prompt_safety_margin_tokens: usize,
    pub include_configured_candidates: bool,
    pub health_ttl_seconds: f64,
    pub early_success_enabled: bool,
    pub early_success_settle_seconds: f64,
    pub early_success_min_quality: f64,
    pub candidate_timeout_cap_seconds: Option<u64>,
    pub automation_candidate_timeout_cap_seconds: Option<u64>,
    pub interactive_model_floor_b: f64,
    pub solver_model_floor_b: f64,
    pub strict_no_downgrade: bool,
    pub always_route_specialists: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AarnnBridgeConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub access_token: Option<String>,
    pub timeout_seconds: f64,
    pub queue_capacity: usize,
    pub worker_count: usize,
    pub enqueue_timeout_ms: u64,
    pub candidate_wait_timeout_ms: u64,
    pub mirror_input: bool,
    pub mirror_output: bool,
    pub request_candidate_reply: bool,
    pub response_preference: AarnnResponsePreference,
    pub candidate_confidence_threshold: f64,
    pub candidate_min_reply_chars: usize,
    pub network_id: Option<String>,
    pub node_id: Option<String>,
    pub max_text_chars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NmcTelemetryConfig {
    pub enabled: bool,
    pub base_url: Option<String>,
    pub access_token: Option<String>,
    pub timeout_seconds: f64,
    pub cache_ttl_seconds: f64,
    pub stale_after_seconds: u64,
    pub adaptive_policy: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditLoggingConfig {
    pub enabled: bool,
    pub log_llm_prompts: bool,
    pub log_llm_responses: bool,
    pub log_aer_payloads: bool,
    pub max_chars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmLedgerConfig {
    pub enabled: bool,
    pub queue_capacity: usize,
    pub enqueue_timeout_ms: u64,
    pub max_prompt_chars: usize,
    pub max_response_chars: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MirrorWorkerConfig {
    pub enabled: bool,
    pub poll_interval_ms: u64,
    pub batch_size: usize,
    pub max_attempts: u32,
    pub retry_backoff_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainerConfig {
    pub enabled: bool,
    pub poll_interval_seconds: u64,
    pub min_samples: usize,
    pub max_samples_per_snapshot: usize,
    pub include_degraded: bool,
    pub max_attempts: u32,
    pub retry_backoff_seconds: u64,
    pub algorithm: String,
    pub command_template: Option<String>,
    pub command_timeout_seconds: u64,
    pub model_prefix: String,
    pub model_alias: String,
    pub ollama_base_model: String,
    pub rotate_keep: usize,
    pub register_with_ollama: bool,
    pub ollama_cli: String,
    pub ollama_host: Option<String>,
    pub output_root: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub metrics_path: String,
    pub adaptive_schema_path: String,
    pub api_issues_path: String,
    pub llm_ledger_path: String,
    pub trainer_output_path: String,
    pub postgres_dsn: Option<String>,
    pub ollama_model_store_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderProfile {
    pub name: String,
    #[serde(rename = "provider", alias = "type")]
    pub provider_type: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub access_token: Option<String>,
    pub base_url: Option<String>,
    /// Provider/model context window. `None` means provider-managed except for
    /// Ollama, which uses the orchestration default.
    pub context_window_tokens: Option<usize>,
    pub roles: Vec<String>,
    pub specialties: Vec<String>,
    pub weight: f64,
    pub preferred: bool,
    pub source: Option<String>,
    pub host_group: Option<String>,
    pub priority_bias: f64,
    pub usage_penalty_decay_seconds: f64,
    pub max_concurrent_requests: Option<usize>,
    pub resource_cost_cpu: f64,
    pub resource_cost_ram_mb: u64,
    pub resource_cost_vram_mb: u64,
    pub host_cpu_budget: Option<f64>,
    pub host_ram_budget_mb: Option<u64>,
    pub host_vram_budget_mb: Option<u64>,
    pub nmc_agent_id: Option<String>,
    pub nmc_host: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpecialistProfile {
    pub name: String,
    #[serde(rename = "type", alias = "engine")]
    pub engine_type: String,
    pub endpoint: Option<String>,
    pub socket_path: Option<String>,
    pub repo_root: Option<String>,
    pub allow_offline_heuristic: bool,
    pub sensory_size: usize,
    pub output_size: usize,
    pub aer_sensory_base: u32,
    pub aer_output_base: u32,
    pub timeout_seconds: f64,
    pub health_ttl_seconds: f64,
    pub spike_threshold: f64,
    pub roles: Vec<String>,
    pub specialties: Vec<String>,
    pub keyword_hints: Vec<String>,
    pub guidance_lines: Vec<String>,
    pub description: Option<String>,
    pub weight: f64,
    pub prefer_aarnn_designs: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:8080".to_string(),
            public_base_url: None,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_tokens: Vec::new(),
            allow_unauthenticated_health: true,
            allow_unauthenticated_metrics: true,
        }
    }
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            selection_mode: SelectionMode::Best,
            max_parallel_candidates: 3,
            interactive_pool_max_in_flight: 8,
            solver_pool_max_in_flight: 4,
            workload_pool_wait_timeout_ms: 30_000,
            candidate_queue_wait_timeout_ms: 30_000,
            deduplicate_model_candidates: true,
            prompt_compaction_enabled: true,
            default_local_context_window_tokens: 16_384,
            prompt_chars_per_token: 4,
            prompt_safety_margin_tokens: 1_024,
            include_configured_candidates: true,
            health_ttl_seconds: 1800.0,
            early_success_enabled: true,
            early_success_settle_seconds: 0.75,
            early_success_min_quality: 0.5,
            candidate_timeout_cap_seconds: Some(45),
            automation_candidate_timeout_cap_seconds: Some(12),
            interactive_model_floor_b: 0.5,
            solver_model_floor_b: 1.5,
            strict_no_downgrade: true,
            always_route_specialists: false,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            metrics_path: "data/provider_metrics.json".to_string(),
            adaptive_schema_path: "data/adaptive_api_schema.json".to_string(),
            api_issues_path: "data/api_issues.json".to_string(),
            llm_ledger_path: "data/llm_interactions.jsonl".to_string(),
            trainer_output_path: "data/training".to_string(),
            postgres_dsn: None,
            ollama_model_store_path: None,
        }
    }
}

impl Default for LlmLedgerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue_capacity: 4096,
            enqueue_timeout_ms: 25,
            max_prompt_chars: 65_536,
            max_response_chars: 65_536,
        }
    }
}

impl Default for MirrorWorkerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: 2_000,
            batch_size: 64,
            max_attempts: 6,
            retry_backoff_seconds: 60,
        }
    }
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_seconds: 60,
            min_samples: 64,
            max_samples_per_snapshot: 512,
            include_degraded: false,
            max_attempts: 6,
            retry_backoff_seconds: 300,
            algorithm: "qlora_sft".to_string(),
            command_template: None,
            command_timeout_seconds: 86_400,
            model_prefix: "gail-inhouse".to_string(),
            model_alias: "gail-inhouse:latest".to_string(),
            ollama_base_model: "qwen2.5-coder:1.5b".to_string(),
            rotate_keep: 6,
            register_with_ollama: true,
            ollama_cli: "ollama".to_string(),
            ollama_host: None,
            output_root: "data/training".to_string(),
        }
    }
}

impl Default for AarnnBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: None,
            access_token: None,
            timeout_seconds: 4.0,
            queue_capacity: 256,
            worker_count: 2,
            enqueue_timeout_ms: 35,
            candidate_wait_timeout_ms: 300,
            mirror_input: true,
            mirror_output: true,
            request_candidate_reply: true,
            response_preference: AarnnResponsePreference::LlmPreferred,
            candidate_confidence_threshold: 0.98,
            candidate_min_reply_chars: 24,
            network_id: None,
            node_id: None,
            max_text_chars: 8192,
        }
    }
}

impl Default for NmcTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: None,
            access_token: None,
            timeout_seconds: 2.0,
            cache_ttl_seconds: 5.0,
            stale_after_seconds: 300,
            adaptive_policy: Some("balanced".to_string()),
        }
    }
}

impl Default for AuditLoggingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_llm_prompts: true,
            log_llm_responses: true,
            log_aer_payloads: true,
            max_chars: 65_536,
        }
    }
}

impl Default for ProviderProfile {
    fn default() -> Self {
        Self {
            name: String::new(),
            provider_type: String::new(),
            model: None,
            api_key: None,
            access_token: None,
            base_url: None,
            context_window_tokens: None,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: 0.0,
            preferred: false,
            source: None,
            host_group: None,
            priority_bias: 0.0,
            usage_penalty_decay_seconds: 600.0,
            max_concurrent_requests: None,
            resource_cost_cpu: 0.0,
            resource_cost_ram_mb: 0,
            resource_cost_vram_mb: 0,
            host_cpu_budget: None,
            host_ram_budget_mb: None,
            host_vram_budget_mb: None,
            nmc_agent_id: None,
            nmc_host: None,
        }
    }
}

impl Default for SpecialistProfile {
    fn default() -> Self {
        Self {
            name: "AARNN".to_string(),
            engine_type: "aarnn".to_string(),
            endpoint: None,
            socket_path: None,
            repo_root: None,
            allow_offline_heuristic: true,
            sensory_size: 32,
            output_size: 16,
            aer_sensory_base: 4096,
            aer_output_base: 16384,
            timeout_seconds: 2.0,
            health_ttl_seconds: 300.0,
            spike_threshold: 0.5,
            roles: vec![
                "planner".to_string(),
                "reviewer".to_string(),
                "researcher".to_string(),
                "assistant".to_string(),
            ],
            specialties: vec![
                "aarnn".to_string(),
                "snn".to_string(),
                "neuromorphic".to_string(),
                "aer".to_string(),
            ],
            keyword_hints: Vec::new(),
            guidance_lines: Vec::new(),
            description: None,
            weight: 0.0,
            prefer_aarnn_designs: true,
        }
    }
}

impl GailConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let raw = fs::read_to_string(path)?;
            let rendered = interpolate_env(&raw);
            let mut config: GailConfig = serde_yaml::from_str(&rendered)?;
            config.normalize()?;
            return Ok(config);
        }
        let mut config = GailConfig::default();
        config.normalize()?;
        Ok(config)
    }

    fn normalize(&mut self) -> Result<()> {
        let trainer_command_env = std::env::var("GAIL_TRAINER_COMMAND")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let ollama_host_env = std::env::var("OLLAMA_HOST")
            .ok()
            .filter(|value| !value.trim().is_empty());
        if self.server.bind_addr.trim().is_empty() {
            return Err(GailError::invalid_config(
                "server.bind_addr must not be empty",
            ));
        }
        self.server.public_base_url =
            normalize_optional_url(self.server.public_base_url.as_deref());
        if self.orchestration.max_parallel_candidates == 0 {
            self.orchestration.max_parallel_candidates = 1;
        }
        self.orchestration.interactive_pool_max_in_flight = self
            .orchestration
            .interactive_pool_max_in_flight
            .clamp(1, 4096);
        self.orchestration.solver_pool_max_in_flight =
            self.orchestration.solver_pool_max_in_flight.clamp(1, 4096);
        self.orchestration.workload_pool_wait_timeout_ms = self
            .orchestration
            .workload_pool_wait_timeout_ms
            .clamp(1, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS);
        self.orchestration.candidate_queue_wait_timeout_ms = self
            .orchestration
            .candidate_queue_wait_timeout_ms
            .clamp(1, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS);
        self.orchestration.default_local_context_window_tokens = self
            .orchestration
            .default_local_context_window_tokens
            .clamp(1_024, 4_194_304);
        self.orchestration.prompt_chars_per_token =
            self.orchestration.prompt_chars_per_token.clamp(1, 16);
        self.orchestration.prompt_safety_margin_tokens = self
            .orchestration
            .prompt_safety_margin_tokens
            .clamp(0, 262_144);
        if self.orchestration.health_ttl_seconds < 30.0 {
            self.orchestration.health_ttl_seconds = 30.0;
        }
        self.orchestration.interactive_model_floor_b =
            self.orchestration.interactive_model_floor_b.max(0.0);
        self.orchestration.solver_model_floor_b = self.orchestration.solver_model_floor_b.max(0.0);
        if self.orchestration.solver_model_floor_b < self.orchestration.interactive_model_floor_b {
            self.orchestration.solver_model_floor_b = self.orchestration.interactive_model_floor_b;
        }
        self.aarnn_bridge.endpoint = normalize_optional_url(self.aarnn_bridge.endpoint.as_deref());
        self.aarnn_bridge.access_token =
            normalize_optional_string(self.aarnn_bridge.access_token.as_deref());
        self.aarnn_bridge.network_id =
            normalize_optional_string(self.aarnn_bridge.network_id.as_deref());
        self.aarnn_bridge.node_id = normalize_optional_string(self.aarnn_bridge.node_id.as_deref());
        self.aarnn_bridge.timeout_seconds = self.aarnn_bridge.timeout_seconds.max(0.2);
        self.aarnn_bridge.queue_capacity = self.aarnn_bridge.queue_capacity.clamp(8, 32_768);
        self.aarnn_bridge.worker_count = self.aarnn_bridge.worker_count.clamp(1, 128);
        self.aarnn_bridge.enqueue_timeout_ms =
            self.aarnn_bridge.enqueue_timeout_ms.clamp(1, 10_000);
        self.aarnn_bridge.candidate_wait_timeout_ms =
            self.aarnn_bridge.candidate_wait_timeout_ms.min(30_000);
        self.aarnn_bridge.candidate_confidence_threshold = self
            .aarnn_bridge
            .candidate_confidence_threshold
            .clamp(0.0, 1.0);
        self.aarnn_bridge.candidate_min_reply_chars =
            self.aarnn_bridge.candidate_min_reply_chars.max(1);
        self.aarnn_bridge.max_text_chars = self.aarnn_bridge.max_text_chars.clamp(128, 65_536);
        self.audit_logging.max_chars = self.audit_logging.max_chars.clamp(256, 262_144);
        self.nmc_telemetry.base_url =
            normalize_optional_url(self.nmc_telemetry.base_url.as_deref());
        self.nmc_telemetry.access_token =
            normalize_optional_string(self.nmc_telemetry.access_token.as_deref());
        self.nmc_telemetry.timeout_seconds = self.nmc_telemetry.timeout_seconds.max(0.2);
        self.nmc_telemetry.cache_ttl_seconds = self.nmc_telemetry.cache_ttl_seconds.max(0.2);
        self.nmc_telemetry.stale_after_seconds =
            self.nmc_telemetry.stale_after_seconds.clamp(5, 86_400);
        self.nmc_telemetry.adaptive_policy =
            normalize_nmc_policy(self.nmc_telemetry.adaptive_policy.as_deref());
        if self.storage.metrics_path.trim().is_empty() {
            self.storage.metrics_path = "data/provider_metrics.json".to_string();
        }
        if self.storage.adaptive_schema_path.trim().is_empty() {
            self.storage.adaptive_schema_path = "data/adaptive_api_schema.json".to_string();
        }
        if self.storage.api_issues_path.trim().is_empty() {
            self.storage.api_issues_path = "data/api_issues.json".to_string();
        }
        if self.storage.llm_ledger_path.trim().is_empty() {
            self.storage.llm_ledger_path = "data/llm_interactions.jsonl".to_string();
        }
        if self.storage.trainer_output_path.trim().is_empty() {
            self.storage.trainer_output_path = "data/training".to_string();
        }
        if self
            .storage
            .postgres_dsn
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            self.storage.postgres_dsn = std::env::var("GAIL_POSTGRES_DSN")
                .ok()
                .or_else(|| std::env::var("DATABASE_URL").ok())
                .filter(|value| !value.trim().is_empty());
        }
        self.llm_ledger.queue_capacity = self.llm_ledger.queue_capacity.clamp(128, 131_072);
        self.llm_ledger.enqueue_timeout_ms = self.llm_ledger.enqueue_timeout_ms.clamp(1, 60_000);
        self.llm_ledger.max_prompt_chars = self.llm_ledger.max_prompt_chars.clamp(256, 262_144);
        self.llm_ledger.max_response_chars = self.llm_ledger.max_response_chars.clamp(256, 262_144);
        self.mirror_worker.poll_interval_ms =
            self.mirror_worker.poll_interval_ms.clamp(100, 60_000);
        self.mirror_worker.batch_size = self.mirror_worker.batch_size.clamp(1, 4096);
        self.mirror_worker.max_attempts = self.mirror_worker.max_attempts.clamp(1, 1_000);
        self.mirror_worker.retry_backoff_seconds =
            self.mirror_worker.retry_backoff_seconds.clamp(1, 86_400);
        self.trainer.poll_interval_seconds = self.trainer.poll_interval_seconds.clamp(5, 86_400);
        self.trainer.min_samples = self.trainer.min_samples.clamp(1, 1_000_000);
        self.trainer.max_samples_per_snapshot =
            self.trainer.max_samples_per_snapshot.clamp(1, 1_000_000);
        self.trainer.max_attempts = self.trainer.max_attempts.clamp(1, 1_000);
        self.trainer.retry_backoff_seconds = self.trainer.retry_backoff_seconds.clamp(1, 604_800);
        self.trainer.algorithm = normalize_optional_string(Some(self.trainer.algorithm.as_str()))
            .unwrap_or_else(|| "qlora_sft".to_string());
        self.trainer.command_template = normalize_optional_string(
            self.trainer
                .command_template
                .as_deref()
                .or(trainer_command_env.as_deref()),
        );
        self.trainer.command_timeout_seconds =
            self.trainer.command_timeout_seconds.clamp(30, 604_800);
        self.trainer.model_prefix =
            normalize_optional_string(Some(self.trainer.model_prefix.as_str()))
                .unwrap_or_else(|| "gail-inhouse".to_string());
        self.trainer.model_alias =
            normalize_optional_string(Some(self.trainer.model_alias.as_str()))
                .unwrap_or_else(|| "gail-inhouse:latest".to_string());
        self.trainer.ollama_base_model =
            normalize_optional_string(Some(self.trainer.ollama_base_model.as_str()))
                .unwrap_or_else(|| "qwen2.5-coder:1.5b".to_string());
        self.trainer.rotate_keep = self.trainer.rotate_keep.clamp(1, 128);
        self.trainer.ollama_cli = normalize_optional_string(Some(self.trainer.ollama_cli.as_str()))
            .unwrap_or_else(|| "ollama".to_string());
        self.trainer.ollama_host = normalize_optional_url(
            self.trainer
                .ollama_host
                .as_deref()
                .or(ollama_host_env.as_deref()),
        );
        self.trainer.output_root =
            normalize_optional_string(Some(self.trainer.output_root.as_str()))
                .unwrap_or_else(|| self.storage.trainer_output_path.clone());
        for provider in &mut self.providers {
            provider.provider_type =
                normalize_optional_string(Some(provider.provider_type.as_str()))
                    .unwrap_or_default();
            provider.model = normalize_optional_string(provider.model.as_deref());
            provider.api_key = normalize_optional_string(provider.api_key.as_deref());
            provider.access_token = normalize_optional_string(provider.access_token.as_deref());
            provider.base_url = normalize_optional_url(provider.base_url.as_deref());
            provider.context_window_tokens = provider
                .context_window_tokens
                .map(|value| value.clamp(1_024, 4_194_304));
            provider.roles = provider
                .roles
                .iter()
                .filter_map(|item| normalize_optional_string(Some(item.as_str())))
                .collect();
            provider.specialties = provider
                .specialties
                .iter()
                .filter_map(|item| normalize_optional_string(Some(item.as_str())))
                .collect();
            provider.source = normalize_optional_string(provider.source.as_deref());
            provider.host_group = normalize_optional_string(provider.host_group.as_deref());
            provider.priority_bias = provider.priority_bias.clamp(-10.0, 10.0);
            provider.usage_penalty_decay_seconds = provider.usage_penalty_decay_seconds.max(30.0);
            provider.max_concurrent_requests = provider.max_concurrent_requests.and_then(|value| {
                if value == 0 {
                    None
                } else {
                    Some(value.min(4096))
                }
            });
            provider.resource_cost_cpu = provider.resource_cost_cpu.max(0.0);
            provider.resource_cost_ram_mb = provider.resource_cost_ram_mb.min(16_777_216);
            provider.resource_cost_vram_mb = provider.resource_cost_vram_mb.min(16_777_216);
            provider.host_cpu_budget = provider.host_cpu_budget.and_then(|value| {
                if value <= 0.0 {
                    None
                } else {
                    Some(value.min(4096.0))
                }
            });
            provider.host_ram_budget_mb = provider.host_ram_budget_mb.and_then(|value| {
                if value == 0 {
                    None
                } else {
                    Some(value.min(16_777_216))
                }
            });
            provider.host_vram_budget_mb = provider.host_vram_budget_mb.and_then(|value| {
                if value == 0 {
                    None
                } else {
                    Some(value.min(16_777_216))
                }
            });
            provider.nmc_agent_id = normalize_optional_string(provider.nmc_agent_id.as_deref());
            provider.nmc_host = normalize_optional_string(provider.nmc_host.as_deref());
            if provider.source.is_none() {
                provider.source = Some("config".to_string());
            }
            provider.name = normalize_optional_string(Some(provider.name.as_str()))
                .unwrap_or_else(|| provider.provider_type.clone());
            if provider.name.trim().is_empty() {
                provider.name = provider.provider_type.clone();
            }
        }
        self.providers
            .retain(|provider| !provider.provider_type.trim().is_empty());
        for specialist in &mut self.specialists {
            if specialist.name.trim().is_empty() {
                specialist.name = "AARNN".to_string();
            }
            if specialist.keyword_hints.is_empty() {
                specialist.keyword_hints = specialist.specialties.clone();
            }
        }
        self.trading.normalize();
        Ok(())
    }
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    let cleaned = value.map(str::trim).filter(|value| !value.is_empty())?;
    let lowered = cleaned.to_ascii_lowercase();
    if matches!(lowered.as_str(), "none" | "null" | "nil" | "undefined") {
        return None;
    }
    Some(cleaned.to_owned())
}

fn normalize_optional_url(value: Option<&str>) -> Option<String> {
    let value = normalize_optional_string(value)?;
    if value.contains("://") {
        Some(value.trim_end_matches('/').to_string())
    } else {
        Some(format!("http://{}", value.trim_end_matches('/')))
    }
}

fn normalize_nmc_policy(value: Option<&str>) -> Option<String> {
    let value = normalize_optional_string(value)?;
    let normalized = value.to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        "balanced" | "throughput" | "risk" | "energy"
    ) {
        Some(normalized)
    } else {
        None
    }
}

fn interpolate_env(raw: &str) -> String {
    let regex = Regex::new(r"\$\{([A-Z0-9_]+)\}").expect("env interpolation regex");
    regex
        .replace_all(raw, |captures: &regex::Captures<'_>| {
            std::env::var(&captures[1]).unwrap_or_default()
        })
        .to_string()
}

#[cfg(test)]
mod tests {
    use reqwest::Client;

    use super::{GailConfig, MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS};
    use crate::{aarnn_bridge::AarnnMirrorClient, nmc_telemetry::NmcTelemetryClient};

    #[test]
    fn optional_workflow_components_default_to_enabled() {
        let config = GailConfig::default();
        assert!(config.llm_ledger.enabled);
        assert!(config.aarnn_bridge.enabled);
        assert!(config.nmc_telemetry.enabled);
        assert!(config.trainer.register_with_ollama);
        assert!(config.trading.backtesting_enabled);
    }

    #[test]
    fn default_enabled_optional_clients_remain_safe_without_endpoints() {
        let config = GailConfig::default();
        let client = Client::builder().build().expect("http client");
        let aarnn_bridge = AarnnMirrorClient::from_config(&config, client.clone(), &[]);
        let nmc_client = NmcTelemetryClient::from_config(&config, client);
        assert!(
            aarnn_bridge.is_none(),
            "bridge should stay disabled at runtime when no endpoint is configured"
        );
        assert!(
            nmc_client.is_none(),
            "NMC telemetry should stay disabled at runtime when base_url is unset"
        );
    }

    #[test]
    fn workload_pool_wait_timeout_allows_twenty_minutes() {
        let mut config = GailConfig::default();
        config.orchestration.workload_pool_wait_timeout_ms = 1_200_000;
        config.normalize().expect("config normalize");
        assert_eq!(
            config.orchestration.workload_pool_wait_timeout_ms,
            1_200_000
        );
    }

    #[test]
    fn workload_pool_wait_timeout_clamps_to_maximum() {
        let mut config = GailConfig::default();
        config.orchestration.workload_pool_wait_timeout_ms = 9_999_999;
        config.normalize().expect("config normalize");
        assert_eq!(
            config.orchestration.workload_pool_wait_timeout_ms,
            MAX_WORKLOAD_POOL_WAIT_TIMEOUT_MS
        );
    }
}
