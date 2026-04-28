pub mod gemini;
pub mod ollama;
pub mod openai;

use std::{collections::HashSet, env, time::Duration};

use http::{
    HeaderMap, HeaderValue, StatusCode,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use rand::Rng;
use reqwest::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::time::sleep;

use crate::{
    config::{GailConfig, ProviderProfile},
    errors::{GailError, Result},
    models::{CostInfo, ProviderCompletionRequest, TokenUsage},
};

pub use gemini::GeminiProvider;
pub use ollama::{OllamaInventoryStatus, OllamaProvider};
pub use openai::OpenAIProvider;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct ProviderHealth {
    pub ok: bool,
    pub status_code: Option<u16>,
    pub latency_ms: Option<u64>,
    pub message: Option<String>,
    pub mode: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProviderInvocationResponse {
    pub text: String,
    pub raw: Option<Value>,
    pub latency_ms: u64,
    pub provider: String,
    pub model: String,
    pub usage: Option<TokenUsage>,
}

#[derive(Clone, Debug)]
pub struct TranscriptionInput {
    pub filename: String,
    pub mime_type: Option<String>,
    pub bytes: Vec<u8>,
    pub timeout_seconds: Option<u64>,
}

#[derive(Clone)]
pub enum ProviderAdapter {
    OpenAI(OpenAIProvider),
    Nvidia(OpenAIProvider),
    Gemini(GeminiProvider),
    Ollama(OllamaProvider),
}

impl ProviderAdapter {
    pub fn provider_type(&self) -> &'static str {
        match self {
            Self::OpenAI(_) => "openai",
            Self::Nvidia(_) => "nvidia",
            Self::Gemini(_) => "gemini",
            Self::Ollama(_) => "ollama",
        }
    }

    pub fn model(&self) -> &str {
        match self {
            Self::OpenAI(provider) => provider.model(),
            Self::Nvidia(provider) => provider.model(),
            Self::Gemini(provider) => provider.model(),
            Self::Ollama(provider) => provider.model(),
        }
    }

    pub async fn complete(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<ProviderInvocationResponse> {
        match self {
            Self::OpenAI(provider) => provider.complete(request).await,
            Self::Nvidia(provider) => provider.complete(request).await,
            Self::Gemini(provider) => provider.complete(request).await,
            Self::Ollama(provider) => provider.complete(request).await,
        }
    }

    pub async fn transcribe(
        &self,
        input: &TranscriptionInput,
    ) -> Result<ProviderInvocationResponse> {
        match self {
            Self::OpenAI(provider) => provider.transcribe(input).await,
            Self::Nvidia(provider) => provider.transcribe(input).await,
            Self::Gemini(provider) => provider.transcribe(input).await,
            Self::Ollama(provider) => provider.transcribe(input).await,
        }
    }

    pub async fn health(&self, timeout_seconds: Option<u64>) -> Result<ProviderHealth> {
        match self {
            Self::OpenAI(provider) => provider.health(timeout_seconds).await,
            Self::Nvidia(provider) => provider.health(timeout_seconds).await,
            Self::Gemini(provider) => provider.health(timeout_seconds).await,
            Self::Ollama(provider) => provider.health(timeout_seconds).await,
        }
    }

    pub async fn ollama_inventory(&self, config: &GailConfig) -> Option<OllamaInventoryStatus> {
        match self {
            Self::Ollama(provider) => provider.inventory_status(config).await.ok(),
            _ => None,
        }
    }
}

pub fn build_adapter(client: Client, profile: &ProviderProfile) -> Result<ProviderAdapter> {
    match normalize_provider_type(profile.provider_type.as_str()).as_str() {
        "openai" => Ok(ProviderAdapter::OpenAI(OpenAIProvider::new(
            client, profile,
        )?)),
        "nvidia" => Ok(ProviderAdapter::Nvidia(OpenAIProvider::new_nvidia(
            client, profile,
        )?)),
        "gemini" => Ok(ProviderAdapter::Gemini(GeminiProvider::new(
            client, profile,
        )?)),
        "ollama" => Ok(ProviderAdapter::Ollama(OllamaProvider::new(
            client, profile,
        ))),
        other => Err(GailError::bad_request(format!(
            "unsupported provider type: {other}"
        ))),
    }
}

pub fn normalize_provider_type(raw: &str) -> String {
    let cleaned = raw.trim().to_ascii_lowercase();
    match cleaned.as_str() {
        "chatgpt" | "gpt" => "openai".to_string(),
        "google" => "gemini".to_string(),
        "nim" | "nvidia_nim" => "nvidia".to_string(),
        other => other.to_string(),
    }
}

pub fn provider_request_from_profile(
    profile: &ProviderProfile,
    request: &ProviderCompletionRequest,
) -> ProviderCompletionRequest {
    ProviderCompletionRequest {
        provider: normalize_provider_type(profile.provider_type.as_str()),
        model: profile.model.clone().or_else(|| request.model.clone()),
        api_key: request.api_key.clone().or_else(|| profile.api_key.clone()),
        access_token: request
            .access_token
            .clone()
            .or_else(|| profile.access_token.clone()),
        base_url: request
            .base_url
            .clone()
            .or_else(|| profile.base_url.clone()),
        messages: request.messages.clone(),
        system: request.system.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        timeout_seconds: request.timeout_seconds,
        reasoning_effort: request.reasoning_effort.clone(),
        request_category: request.request_category.clone(),
    }
}

pub fn total_input_chars(messages: &[crate::models::ChatMessage], system: Option<&str>) -> usize {
    let system_chars = system.unwrap_or_default().len();
    system_chars
        + messages
            .iter()
            .map(|message| message.flattened_text().len())
            .sum::<usize>()
}

pub fn estimate_tokens_from_chars(chars: usize) -> u32 {
    ((chars.max(1) / 4) as u32).max(1)
}

pub fn flatten_prompt_text(
    messages: &[crate::models::ChatMessage],
    system: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    if let Some(system) = system {
        if !system.trim().is_empty() {
            parts.push(system.trim().to_string());
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

pub fn looks_like_json_request(
    messages: &[crate::models::ChatMessage],
    system: Option<&str>,
) -> bool {
    let combined = flatten_prompt_text(messages, system).to_ascii_lowercase();
    [
        "return only valid json",
        "respond with json only",
        "output only json",
        "valid json",
        "schema",
    ]
    .iter()
    .any(|hint| combined.contains(hint))
}

pub fn prompt_cache_key(system: Option<&str>, model: Option<&str>, kind: &str) -> Option<String> {
    if env_bool(
        "OPENAI_PROMPT_CACHE_KEY_AUTO",
        env_bool("PROMPT_CACHE_KEY_AUTO", true),
    ) == false
    {
        return None;
    }
    let mode = env::var("OPENAI_PROMPT_CACHE_KEY_MODE").unwrap_or_else(|_| "system".to_string());
    let basis = match mode.trim().to_ascii_lowercase().as_str() {
        "model" => model.unwrap_or_default().to_string(),
        _ => system.unwrap_or_default().to_string(),
    };
    if basis.trim().is_empty() {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(format!("{kind}:{basis}"));
    Some(format!(
        "pcache:{}",
        hex::encode(digest.finalize())[..16].to_string()
    ))
}

pub fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        })
        .unwrap_or(default)
}

pub fn env_int(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

pub fn env_float(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

pub fn data_url_parts(url: &str) -> Option<(String, String)> {
    if !url.starts_with("data:") {
        return None;
    }
    let (prefix, data) = url.split_once(",")?;
    let mime = prefix
        .trim_start_matches("data:")
        .trim_end_matches(";base64")
        .to_string();
    Some((mime, data.to_string()))
}

pub fn should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retry_after_seconds(headers: &HeaderMap) -> Option<f64> {
    headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok())
}

pub async fn post_json_with_retries(
    provider: &str,
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    payload: &Value,
    timeout: Duration,
    max_retries: usize,
) -> Result<reqwest::Response> {
    let base = env_float("LLM_BACKOFF_BASE", 0.5);
    let max_backoff = env_float("LLM_BACKOFF_MAX", 8.0);
    let mut attempt = 0usize;
    loop {
        let response = client
            .post(url)
            .headers(headers.clone())
            .timeout(timeout)
            .json(payload)
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) if !should_retry_status(response.status()) || attempt >= max_retries => {
                return Ok(response);
            }
            Ok(response) => {
                let delay = retry_after_seconds(response.headers()).unwrap_or_else(|| {
                    let jitter = rand::rng().random_range(0.0..0.2);
                    (base * 2_f64.powi(attempt as i32)).min(max_backoff) + jitter
                });
                attempt += 1;
                sleep(Duration::from_secs_f64(delay)).await;
            }
            Err(error) if attempt >= max_retries => {
                return Err(GailError::upstream(provider, None, error.to_string()));
            }
            Err(error) => {
                let delay = (base * 2_f64.powi(attempt as i32)).min(max_backoff);
                tracing::debug!(provider, attempt, error = %error, "retrying failed POST");
                attempt += 1;
                sleep(Duration::from_secs_f64(delay)).await;
            }
        }
    }
}

pub async fn get_with_retries(
    provider: &str,
    client: &Client,
    url: &str,
    headers: &HeaderMap,
    timeout: Duration,
    max_retries: usize,
) -> Result<reqwest::Response> {
    let base = env_float("LLM_BACKOFF_BASE", 0.5);
    let max_backoff = env_float("LLM_BACKOFF_MAX", 8.0);
    let mut attempt = 0usize;
    loop {
        let response = client
            .get(url)
            .headers(headers.clone())
            .timeout(timeout)
            .send()
            .await;
        match response {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) if !should_retry_status(response.status()) || attempt >= max_retries => {
                return Ok(response);
            }
            Ok(response) => {
                let delay = retry_after_seconds(response.headers()).unwrap_or_else(|| {
                    let jitter = rand::rng().random_range(0.0..0.2);
                    (base * 2_f64.powi(attempt as i32)).min(max_backoff) + jitter
                });
                attempt += 1;
                sleep(Duration::from_secs_f64(delay)).await;
            }
            Err(error) if attempt >= max_retries => {
                return Err(GailError::upstream(provider, None, error.to_string()));
            }
            Err(error) => {
                let delay = (base * 2_f64.powi(attempt as i32)).min(max_backoff);
                tracing::debug!(provider, attempt, error = %error, "retrying failed GET");
                attempt += 1;
                sleep(Duration::from_secs_f64(delay)).await;
            }
        }
    }
}

pub fn error_message(payload: &Value) -> String {
    payload
        .get("error")
        .and_then(|error| {
            if error.is_string() {
                error.as_str().map(ToOwned::to_owned)
            } else {
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| {
                        error
                            .get("code")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
            }
        })
        .or_else(|| {
            payload
                .get("message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| payload.to_string())
}

pub fn is_model_not_found(status: StatusCode, message: &str) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::UNPROCESSABLE_ENTITY
    ) && {
        let lowered = message.to_ascii_lowercase();
        lowered.contains("model_not_found")
            || (lowered.contains("model")
                && (lowered.contains("not found")
                    || lowered.contains("does not exist")
                    || lowered.contains("unknown")))
    }
}

pub fn auth_headers(api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let value = format!("Bearer {api_key}");
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&value).expect("authorization header"),
    );
    headers
}

pub fn extract_cost(usage: &TokenUsage, provider: &str, model: &str) -> Option<CostInfo> {
    let raw = env::var("REFINER_LLM_PRICING").unwrap_or_default();
    let payload: Value = serde_json::from_str(&raw).ok()?;
    let entry = payload
        .get(format!("{provider}/{model}"))
        .or_else(|| payload.get(format!("{provider}:{model}")))
        .or_else(|| payload.get(model))
        .or_else(|| payload.get("default"))?;
    let unit = entry
        .get("unit")
        .and_then(Value::as_str)
        .unwrap_or("per_1m_tokens");
    let denominator = if matches!(unit, "per_1k_tokens" | "per_1k" | "1k") {
        1_000.0
    } else {
        1_000_000.0
    };
    let prompt_rate = entry.get("prompt").and_then(Value::as_f64).unwrap_or(0.0);
    let completion_rate = entry
        .get("completion")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let cached_rate = entry
        .get("cached")
        .and_then(Value::as_f64)
        .or_else(|| entry.get("prompt_cached").and_then(Value::as_f64))
        .unwrap_or(prompt_rate);
    let prompt = usage
        .prompt
        .unwrap_or(0)
        .saturating_sub(usage.cached.unwrap_or(0));
    let completion = usage.completion.unwrap_or(0);
    let cached = usage.cached.unwrap_or(0);
    let amount = ((prompt as f64 * prompt_rate)
        + (completion as f64 * completion_rate)
        + (cached as f64 * cached_rate))
        / denominator;
    Some(CostInfo {
        amount: (amount * 100_000_000.0).round() / 100_000_000.0,
        currency: entry
            .get("currency")
            .and_then(Value::as_str)
            .unwrap_or("USD")
            .to_string(),
    })
}

pub fn infer_capabilities_from_text(prompt: &str) -> HashSet<String> {
    let lowered = prompt.to_ascii_lowercase();
    let mut capabilities = HashSet::new();
    if lowered.contains("image") || lowered.contains("vision") || lowered.contains("diagram") {
        capabilities.insert("vision".to_string());
    }
    if lowered.contains("code")
        || lowered.contains("refactor")
        || lowered.contains("python")
        || lowered.contains("rust")
    {
        capabilities.insert("code".to_string());
    }
    if lowered.contains("json") || lowered.contains("schema") {
        capabilities.insert("json".to_string());
    }
    if lowered.contains("research") || lowered.contains("citation") || lowered.contains("evidence")
    {
        capabilities.insert("research".to_string());
    }
    if capabilities.is_empty() {
        capabilities.insert("general".to_string());
    }
    capabilities
}

pub fn infer_capabilities_from_model(model: &str) -> HashSet<String> {
    let lowered = model.to_ascii_lowercase();
    let mut capabilities = HashSet::new();
    if lowered.contains("vision") || lowered.contains("vl") {
        capabilities.insert("vision".to_string());
    }
    if lowered.contains("code") || lowered.contains("coder") || lowered.contains("codex") {
        capabilities.insert("code".to_string());
    }
    if lowered.contains("embedding") || lowered.contains("embed") {
        capabilities.insert("retrieval".to_string());
    }
    if lowered.contains("flash") || lowered.contains("mini") {
        capabilities.insert("fast".to_string());
    }
    if lowered.contains("o3") || lowered.contains("o4") || lowered.contains("pro") {
        capabilities.insert("reasoning".to_string());
    }
    if capabilities.is_empty() {
        capabilities.insert("general".to_string());
    }
    capabilities
}

pub fn build_internal_request(
    provider: &str,
    model: &str,
    api_key: Option<String>,
    access_token: Option<String>,
    base_url: Option<String>,
    messages: Vec<crate::models::ChatMessage>,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    timeout_seconds: Option<u64>,
    reasoning_effort: Option<String>,
    request_category: Option<String>,
) -> ProviderCompletionRequest {
    ProviderCompletionRequest {
        provider: provider.to_string(),
        model: Some(model.to_string()),
        api_key,
        access_token,
        base_url,
        messages,
        system,
        max_tokens,
        temperature,
        timeout_seconds,
        reasoning_effort,
        request_category,
    }
}

pub fn response_with_usage(
    text: String,
    raw: Value,
    latency_ms: u64,
    provider: &str,
    model: &str,
    mut usage: Option<TokenUsage>,
) -> ProviderInvocationResponse {
    if let Some(ref mut usage) = usage {
        if usage.cost.is_none() {
            usage.cost = extract_cost(usage, provider, model);
        }
    }
    ProviderInvocationResponse {
        text,
        raw: Some(raw),
        latency_ms,
        provider: provider.to_string(),
        model: model.to_string(),
        usage,
    }
}
