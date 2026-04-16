use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::aer::AerEvent;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    Fastest,
    #[default]
    Best,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageUrlValue {
    pub url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlValue },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn flattened_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => text.clone(),
                    ContentPart::ImageUrl { image_url } => image_url.url.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn image_data_urls(&self) -> Vec<String> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    ContentPart::Text { .. } => None,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

impl ChatMessage {
    pub fn flattened_text(&self) -> String {
        self.content.flattened_text()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CostInfo {
    pub amount: f64,
    pub currency: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt: Option<u32>,
    pub completion: Option<u32>,
    pub total: Option<u32>,
    pub cached: Option<u32>,
    pub cost: Option<CostInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderCompletionRequest {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub access_token: Option<String>,
    pub base_url: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub reasoning_effort: Option<String>,
    pub request_category: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub workflow: Option<String>,
    pub role: Option<String>,
    pub preferred_provider: Option<String>,
    pub preferred_model: Option<String>,
    pub preferred_api_key: Option<String>,
    pub preferred_access_token: Option<String>,
    pub fallback_provider: Option<String>,
    pub fallback_model: Option<String>,
    pub fallback_api_key: Option<String>,
    pub fallback_access_token: Option<String>,
    pub base_url: Option<String>,
    pub include_configured: Option<bool>,
    pub selection_mode: Option<SelectionMode>,
    pub max_candidates: Option<usize>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub timeout_seconds: Option<u64>,
    pub reasoning_effort: Option<String>,
    pub request_category: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateSummary {
    pub candidate_id: String,
    pub provider: String,
    pub model: String,
    pub configured_model: String,
    pub resolved_model: String,
    pub source: String,
    pub specialties: Vec<String>,
    pub roles: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateInvocationSummary {
    #[serde(flatten)]
    pub summary: CandidateSummary,
    pub latency_ms: Option<u64>,
    pub quality: f64,
    pub score: f64,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionTrace {
    pub workflow: String,
    pub role: String,
    pub task_tags: Vec<String>,
    pub selection_mode: SelectionMode,
    pub returned_early: bool,
    pub early_success_enabled: bool,
    pub early_success_settle_seconds: f64,
    pub selected: CandidateSummary,
    pub candidates: Vec<CandidateInvocationSummary>,
    pub metrics_store_path: String,
    pub specialist_engines: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub request_id: String,
    pub text: String,
    pub provider: String,
    pub model: String,
    pub latency_ms: u64,
    pub usage: Option<TokenUsage>,
    pub trace: Option<CompletionTrace>,
    pub raw: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub request_id: String,
    pub text: String,
    pub provider: String,
    pub model: String,
    pub latency_ms: u64,
    pub usage: Option<TokenUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromorphicPredictRequest {
    pub engine_name: Option<String>,
    #[serde(default)]
    pub inputs: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromorphicPredictResponse {
    pub score: f64,
    pub fired: bool,
    pub mode: String,
    pub threshold: f64,
    pub input_spikes: Vec<u8>,
    pub output_spikes: Vec<u8>,
    pub aer_payload_hex: String,
    pub raw: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuromorphicAnalyzeRequest {
    pub text: String,
    pub workflow: Option<String>,
    pub role: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpecialistAnalysisResponse {
    pub relevant: bool,
    pub engine_count: usize,
    pub engines: Vec<Value>,
    pub selected: Option<Value>,
    pub combined_specialties: Vec<String>,
    pub context_blocks: Vec<String>,
    pub context: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AerEncodeRequest {
    pub ts_us: Option<u64>,
    pub base_addr: u32,
    pub spikes: Option<Vec<u8>>,
    pub events: Option<Vec<AerEvent>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AerEncodeResponse {
    pub payload_hex: String,
    pub event_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AerDecodeRequest {
    pub payload_hex: String,
    pub base_addr: Option<u32>,
    pub length: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AerDecodeResponse {
    pub events: Vec<AerEvent>,
    pub spikes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub service: String,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthContext {
    pub client_id: Option<String>,
}
