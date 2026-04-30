use std::{fs, path::Path};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    errors::{GailError, Result},
    models::{AarnnResponsePreference, SelectionMode},
    trading::config::TradingConfig,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GailConfig {
    pub server: ServerConfig,
    pub security: SecurityConfig,
    pub orchestration: OrchestrationConfig,
    pub aarnn_bridge: AarnnBridgeConfig,
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
    pub include_configured_candidates: bool,
    pub health_ttl_seconds: f64,
    pub early_success_enabled: bool,
    pub early_success_settle_seconds: f64,
    pub early_success_min_quality: f64,
    pub candidate_timeout_cap_seconds: Option<u64>,
    pub always_route_specialists: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AarnnBridgeConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub access_token: Option<String>,
    pub timeout_seconds: f64,
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
pub struct StorageConfig {
    pub metrics_path: String,
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
    pub roles: Vec<String>,
    pub specialties: Vec<String>,
    pub weight: f64,
    pub preferred: bool,
    pub source: Option<String>,
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

impl Default for GailConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            security: SecurityConfig::default(),
            orchestration: OrchestrationConfig::default(),
            aarnn_bridge: AarnnBridgeConfig::default(),
            providers: Vec::new(),
            specialists: Vec::new(),
            storage: StorageConfig::default(),
            trading: TradingConfig::default(),
        }
    }
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
        }
    }
}

impl Default for ApiTokenConfig {
    fn default() -> Self {
        Self {
            client_id: String::new(),
            token: String::new(),
            scopes: Vec::new(),
        }
    }
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            selection_mode: SelectionMode::Best,
            max_parallel_candidates: 3,
            include_configured_candidates: true,
            health_ttl_seconds: 1800.0,
            early_success_enabled: true,
            early_success_settle_seconds: 0.75,
            early_success_min_quality: 0.5,
            candidate_timeout_cap_seconds: Some(45),
            always_route_specialists: false,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            metrics_path: "data/provider_metrics.json".to_string(),
            ollama_model_store_path: None,
        }
    }
}

impl Default for AarnnBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            access_token: None,
            timeout_seconds: 4.0,
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

impl Default for ProviderProfile {
    fn default() -> Self {
        Self {
            name: String::new(),
            provider_type: String::new(),
            model: None,
            api_key: None,
            access_token: None,
            base_url: None,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: 0.0,
            preferred: false,
            source: None,
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
        if self.server.bind_addr.trim().is_empty() {
            return Err(GailError::invalid_config(
                "server.bind_addr must not be empty",
            ));
        }
        if self.orchestration.max_parallel_candidates == 0 {
            self.orchestration.max_parallel_candidates = 1;
        }
        if self.orchestration.health_ttl_seconds < 30.0 {
            self.orchestration.health_ttl_seconds = 30.0;
        }
        self.aarnn_bridge.endpoint = normalize_optional_url(self.aarnn_bridge.endpoint.as_deref());
        self.aarnn_bridge.access_token =
            normalize_optional_string(self.aarnn_bridge.access_token.as_deref());
        self.aarnn_bridge.network_id =
            normalize_optional_string(self.aarnn_bridge.network_id.as_deref());
        self.aarnn_bridge.node_id = normalize_optional_string(self.aarnn_bridge.node_id.as_deref());
        self.aarnn_bridge.timeout_seconds = self.aarnn_bridge.timeout_seconds.max(0.2);
        self.aarnn_bridge.candidate_confidence_threshold = self
            .aarnn_bridge
            .candidate_confidence_threshold
            .clamp(0.0, 1.0);
        self.aarnn_bridge.candidate_min_reply_chars =
            self.aarnn_bridge.candidate_min_reply_chars.max(1);
        self.aarnn_bridge.max_text_chars = self.aarnn_bridge.max_text_chars.clamp(128, 65_536);
        if self.storage.metrics_path.trim().is_empty() {
            self.storage.metrics_path = "data/provider_metrics.json".to_string();
        }
        for provider in &mut self.providers {
            if provider.source.is_none() {
                provider.source = Some("config".to_string());
            }
            if provider.name.trim().is_empty() {
                provider.name = provider.provider_type.clone();
            }
        }
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
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_optional_url(value: Option<&str>) -> Option<String> {
    let value = normalize_optional_string(value)?;
    if value.contains("://") {
        Some(value.trim_end_matches('/').to_string())
    } else {
        Some(format!("http://{}", value.trim_end_matches('/')))
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
