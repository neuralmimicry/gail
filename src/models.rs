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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AarnnResponsePreference {
    #[default]
    LlmPreferred,
    PreferAarnnWhenConfident,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AarnnMirrorDirection {
    Input,
    Output,
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
pub struct OpenAIReasoningConfig {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub format_type: Option<String>,
    pub json_schema: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAITextConfig {
    pub format: Option<OpenAIResponseFormat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    pub instructions: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub response_format: Option<OpenAIResponseFormat>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: Option<OpenAIReasoningConfig>,
    pub workflow: Option<String>,
    pub role: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub access_token: Option<String>,
    pub base_url: Option<String>,
    pub include_configured: Option<bool>,
    pub selection_mode: Option<SelectionMode>,
    pub max_candidates: Option<usize>,
    pub request_category: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAIResponseRequest {
    pub model: String,
    pub input: Value,
    pub instructions: Option<String>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub text: Option<OpenAITextConfig>,
    pub reasoning: Option<OpenAIReasoningConfig>,
    pub workflow: Option<String>,
    pub role: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub access_token: Option<String>,
    pub base_url: Option<String>,
    pub include_configured: Option<bool>,
    pub selection_mode: Option<SelectionMode>,
    pub max_candidates: Option<usize>,
    pub request_category: Option<String>,
    pub metadata: Option<Value>,
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
    pub final_source: String,
    pub final_provider: String,
    pub final_model: String,
    pub aarnn_mirroring: Option<AarnnMirrorTrace>,
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
pub struct AarnnMirrorRequest {
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
    #[serde(default)]
    pub message_roles: Vec<String>,
    pub aer_base: u32,
    pub output_base: u32,
    pub aer_payload_hex: String,
    #[serde(default)]
    pub sensory_spikes: Vec<u8>,
    pub network_id: Option<String>,
    pub node_id: Option<String>,
    pub request_candidate_reply: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AarnnMirrorCandidate {
    pub reply_text: Option<String>,
    pub confidence: Option<f64>,
    #[serde(default)]
    pub usable: bool,
    pub source: Option<String>,
    #[serde(default)]
    pub output_spike_indices: Vec<u32>,
    pub output_aer_payload_hex: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AarnnMirrorStimulus {
    #[serde(default)]
    pub attempted: bool,
    #[serde(default)]
    pub accepted_batches: usize,
    pub target: Option<String>,
    pub network_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AarnnMirrorResponse {
    #[serde(default)]
    pub accepted: bool,
    pub request_id: Option<String>,
    pub conversation_id: Option<String>,
    pub direction: Option<AarnnMirrorDirection>,
    #[serde(default)]
    pub text_chars: usize,
    #[serde(default)]
    pub spike_count: usize,
    pub aer_payload_hex: Option<String>,
    pub candidate: Option<AarnnMirrorCandidate>,
    pub stimulation: Option<AarnnMirrorStimulus>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AarnnMirrorInvocationTrace {
    pub direction: AarnnMirrorDirection,
    pub accepted: bool,
    pub endpoint: String,
    pub latency_ms: u64,
    pub text_chars: usize,
    pub spike_count: usize,
    pub candidate: Option<AarnnMirrorCandidate>,
    pub stimulation: Option<AarnnMirrorStimulus>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AarnnMirrorTrace {
    pub enabled: bool,
    pub endpoint: String,
    pub mirror_input: bool,
    pub mirror_output: bool,
    pub response_preference: AarnnResponsePreference,
    pub candidate_confidence_threshold: f64,
    pub candidate_min_reply_chars: usize,
    pub input: Option<AarnnMirrorInvocationTrace>,
    pub output: Option<AarnnMirrorInvocationTrace>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AarnnBridgeStatus {
    pub enabled: bool,
    pub available: bool,
    pub endpoint: Option<String>,
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
    pub sensory_size: usize,
    pub output_size: usize,
    pub aer_sensory_base: u32,
    pub aer_output_base: u32,
    pub max_text_chars: usize,
    pub reason: Option<String>,
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
