use std::{
    env,
    time::{Duration, Instant},
};

use http::{HeaderMap, StatusCode};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::{
    adaptive_schema,
    config::ProviderProfile,
    errors::{GailError, Result},
    models::{MessageContent, ProviderCompletionRequest, TokenUsage},
};

use super::{
    ProviderHealth, ProviderInvocationResponse, TranscriptionInput, auth_headers, env_bool,
    env_int, error_message, get_with_retries, is_model_not_found, post_json_with_retries,
    prompt_cache_key, response_with_usage, total_input_chars,
};

#[derive(Clone)]
pub struct OpenAIProvider {
    client: Client,
    provider_name: String,
    model: String,
    default_model: String,
    api_key: String,
    base_url: String,
    supports_responses: bool,
    supports_transcriptions: bool,
    supports_prompt_cache: bool,
}

impl OpenAIProvider {
    pub fn new(client: Client, profile: &ProviderProfile) -> Result<Self> {
        Self::new_with_settings(
            client,
            profile,
            "openai",
            "OPENAI_API_KEY",
            "OPENAI_DEFAULT_MODEL",
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            true,
            true,
            true,
        )
    }

    pub fn new_nvidia(client: Client, profile: &ProviderProfile) -> Result<Self> {
        Self::new_with_settings(
            client,
            profile,
            "nvidia",
            "NVIDIA_API_KEY",
            "NVIDIA_DEFAULT_MODEL",
            "https://integrate.api.nvidia.com/v1",
            "moonshotai/kimi-k2-instruct-0905",
            false,
            false,
            false,
        )
    }

    fn new_with_settings(
        client: Client,
        profile: &ProviderProfile,
        provider_name: &str,
        api_key_env: &str,
        default_model_env: &str,
        default_base_url: &str,
        default_model_fallback: &str,
        supports_responses: bool,
        supports_transcriptions: bool,
        supports_prompt_cache: bool,
    ) -> Result<Self> {
        let api_key = profile
            .api_key
            .clone()
            .or_else(|| env::var(api_key_env).ok())
            .ok_or_else(|| GailError::bad_request(format!("{api_key_env} not configured")))?;
        let default_model =
            env::var(default_model_env).unwrap_or_else(|_| default_model_fallback.to_string());
        let model = profile
            .model
            .clone()
            .unwrap_or_else(|| default_model.clone());
        let base_url = normalize_base_url(
            profile
                .base_url
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(default_base_url),
        );
        Ok(Self {
            client,
            provider_name: provider_name.to_string(),
            model,
            default_model,
            api_key,
            base_url,
            supports_responses,
            supports_transcriptions,
            supports_prompt_cache,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn complete(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<ProviderInvocationResponse> {
        self.complete_once(request).await
    }

    async fn complete_once(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<ProviderInvocationResponse> {
        let api_key = request
            .api_key
            .clone()
            .unwrap_or_else(|| self.api_key.clone());
        let mut model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let base_url = self.resolve_base_url(request.base_url.as_deref());
        let headers = auth_headers(&api_key);
        let effective_effort = request.reasoning_effort.clone().filter(|effort| {
            openai_model_supports_reasoning_effort(&model) && !effort.trim().is_empty()
        });
        let use_responses = self.supports_responses
            && (env_bool("OPENAI_USE_RESPONSES", false) || effective_effort.is_some());
        let temperature = request.temperature.unwrap_or(0.2);
        let base_timeout = request.timeout_seconds.unwrap_or(env_int(
            "OPENAI_TIMEOUT_SECONDS",
            env_int("LLM_TIMEOUT_SECONDS", 180),
        ));
        let timeout = Duration::from_secs(base_timeout.max(1));
        let max_retries = env_int("OPENAI_MAX_RETRIES", env_int("LLM_MAX_RETRIES", 2)) as usize;

        if use_responses {
            let total_chars = total_input_chars(&request.messages, request.system.as_deref());
            let mut max_tokens = request.max_tokens;
            let mut effort = effective_effort.clone();
            let mut prompt_cache_enabled = true;
            let mut replay_response_id: Option<String> = None;
            let mut replay_prompt: Option<String> = None;
            for attempt in 0..3 {
                let url = endpoint(&base_url, "responses");
                let payload = build_responses_payload(
                    &model,
                    request,
                    max_tokens,
                    temperature,
                    effort.clone(),
                    prompt_cache_enabled && self.supports_prompt_cache,
                    replay_response_id.clone(),
                    replay_prompt.clone(),
                );
                let started = Instant::now();
                let response = post_json_with_retries(
                    self.provider_name.as_str(),
                    &self.client,
                    &url,
                    &headers,
                    &payload,
                    timeout,
                    max_retries,
                )
                .await?;
                let latency_ms = started.elapsed().as_millis() as u64;
                let status = response.status();
                let body = response.text().await?;
                let data: Value =
                    serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
                if !status.is_success() {
                    let message = error_message(&data);
                    if effort.is_some() && unsupported_reasoning(status, &message) {
                        effort = None;
                        continue;
                    }
                    if prompt_cache_enabled && prompt_cache_rejected(status, &message) {
                        prompt_cache_enabled = false;
                        continue;
                    }
                    if attempt == 0
                        && is_model_not_found(status, &message)
                        && model != self.default_model
                    {
                        model = self.default_model.clone();
                        continue;
                    }
                    return Err(GailError::upstream(
                        self.provider_name.as_str(),
                        Some(status),
                        message,
                    ));
                }
                let mut data = data;
                if status == StatusCode::ACCEPTED {
                    let request_id = extract_async_request_id(&data).ok_or_else(|| {
                        GailError::upstream(
                            self.provider_name.as_str(),
                            Some(status),
                            "accepted response missing async request id",
                        )
                    })?;
                    data = poll_async_status(
                        self.provider_name.as_str(),
                        &self.client,
                        &headers,
                        &base_url,
                        request_id.as_str(),
                        timeout,
                    )
                    .await?;
                } else if background_enabled(
                    total_chars,
                    max_tokens,
                    effort.as_deref(),
                    base_timeout,
                ) && matches!(
                    data.get("status").and_then(Value::as_str),
                    Some("queued") | Some("in_progress")
                ) {
                    let response_id = data.get("id").and_then(Value::as_str).ok_or_else(|| {
                        GailError::upstream(
                            self.provider_name.as_str(),
                            None,
                            "background response id missing",
                        )
                    })?;
                    data = poll_background_response(
                        self.provider_name.as_str(),
                        &self.client,
                        &headers,
                        &base_url,
                        response_id,
                        timeout,
                    )
                    .await?;
                }
                if let Some(async_model) = data.get("model").and_then(Value::as_str) {
                    model = async_model.to_string();
                }
                if let Some(reason) = data
                    .get("incomplete_details")
                    .and_then(|value| value.get("reason"))
                    .and_then(Value::as_str)
                {
                    if reason == "max_output_tokens" && attempt < 2 {
                        max_tokens = Some(max_tokens.unwrap_or(1024).saturating_mul(2).max(1024));
                        replay_response_id = data
                            .get("id")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned);
                        replay_prompt = Some(
                            env::var("OPENAI_RESPONSES_REPLAY_PROMPT").unwrap_or_else(|_| {
                                "Replay your previous answer in full, preserving the exact format. Do not omit steps or add commentary.".to_string()
                            }),
                        );
                        continue;
                    }
                }
                let text = extract_openai_response_text(&data);
                let usage = extract_openai_usage(&data);
                if text.trim().is_empty() {
                    return Err(GailError::upstream(
                        self.provider_name.as_str(),
                        None,
                        "empty OpenAI response text",
                    ));
                }
                return Ok(response_with_usage(
                    text,
                    data,
                    latency_ms,
                    self.provider_name.as_str(),
                    &model,
                    usage,
                ));
            }
            return Err(GailError::upstream(
                self.provider_name.as_str(),
                None,
                "OpenAI responses retries exhausted",
            ));
        }

        let url = endpoint(&base_url, "chat/completions");
        for attempt in 0..2 {
            let mut messages = Vec::new();
            if let Some(system) = request.system.as_ref() {
                messages.push(json!({"role": "system", "content": system}));
            }
            for message in &request.messages {
                messages.push(json!({
                    "role": message.role,
                    "content": message_content_to_openai_chat(&message.content),
                }));
            }
            let mut payload = json!({
                "model": model,
                "messages": messages,
                "temperature": temperature,
            });
            if let Some(max_tokens) = request.max_tokens {
                payload["max_tokens"] = json!(max_tokens);
            }
            if self.supports_prompt_cache {
                if let Some(cache_key) =
                    prompt_cache_key(request.system.as_deref(), Some(&model), "chat")
                {
                    payload["prompt_cache_key"] = json!(cache_key);
                }
            }
            let started = Instant::now();
            let response = post_json_with_retries(
                self.provider_name.as_str(),
                &self.client,
                &url,
                &headers,
                &payload,
                timeout,
                max_retries,
            )
            .await?;
            let latency_ms = started.elapsed().as_millis() as u64;
            let status = response.status();
            let body = response.text().await?;
            let mut data: Value =
                serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
            if status == StatusCode::ACCEPTED {
                let request_id = extract_async_request_id(&data).ok_or_else(|| {
                    GailError::upstream(
                        self.provider_name.as_str(),
                        Some(status),
                        "accepted response missing async request id",
                    )
                })?;
                data = poll_async_status(
                    self.provider_name.as_str(),
                    &self.client,
                    &headers,
                    &base_url,
                    request_id.as_str(),
                    timeout,
                )
                .await?;
                if let Some(async_model) = data.get("model").and_then(Value::as_str) {
                    model = async_model.to_string();
                }
            } else if !status.is_success() {
                let message = error_message(&data);
                if attempt == 0
                    && is_model_not_found(status, &message)
                    && model != self.default_model
                {
                    model = self.default_model.clone();
                    continue;
                }
                return Err(GailError::upstream(
                    self.provider_name.as_str(),
                    Some(status),
                    message,
                ));
            }
            let text = data
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let usage = extract_openai_usage(&data);
            return Ok(response_with_usage(
                text,
                data,
                latency_ms,
                self.provider_name.as_str(),
                &model,
                usage,
            ));
        }
        Err(GailError::upstream(
            self.provider_name.as_str(),
            None,
            "OpenAI chat retries exhausted",
        ))
    }

    pub async fn transcribe(
        &self,
        input: &TranscriptionInput,
    ) -> Result<ProviderInvocationResponse> {
        if !self.supports_transcriptions {
            return Err(GailError::bad_request(format!(
                "{} does not support audio transcription via Gail",
                self.provider_name
            )));
        }
        let form = reqwest::multipart::Form::new()
            .text("model", "whisper-1")
            .part(
                "file",
                reqwest::multipart::Part::bytes(input.bytes.clone())
                    .file_name(input.filename.clone())
                    .mime_str(
                        input
                            .mime_type
                            .as_deref()
                            .unwrap_or("application/octet-stream"),
                    )
                    .map_err(|error| GailError::Multipart(error.to_string()))?,
            );
        let started = Instant::now();
        let url = endpoint(&self.base_url, "audio/transcriptions");
        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .timeout(Duration::from_secs(
                input.timeout_seconds.unwrap_or(60).max(1),
            ))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                adaptive_schema::observe_failure(
                    self.provider_name.as_str(),
                    "POST",
                    &url,
                    "audio transcription",
                    None,
                    &error.to_string(),
                )
                .await;
                return Err(error.into());
            }
        };
        let latency_ms = started.elapsed().as_millis() as u64;
        let status = response.status();
        let body = response.text().await?;
        let data: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        if !status.is_success() {
            adaptive_schema::observe_failure(
                self.provider_name.as_str(),
                "POST",
                &url,
                "audio transcription",
                Some(status.as_u16()),
                &error_message(&data),
            )
            .await;
            return Err(GailError::upstream(
                self.provider_name.as_str(),
                Some(status),
                error_message(&data),
            ));
        }
        adaptive_schema::observe_success(
            self.provider_name.as_str(),
            "POST",
            &url,
            "audio transcription",
            &data,
        )
        .await;
        let text = data
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Ok(response_with_usage(
            text,
            data,
            latency_ms,
            self.provider_name.as_str(),
            "whisper-1",
            None,
        ))
    }

    pub async fn health(&self, timeout_seconds: Option<u64>) -> Result<ProviderHealth> {
        let headers = auth_headers(&self.api_key);
        let timeout = Duration::from_secs(timeout_seconds.unwrap_or(60).max(1));
        let started = Instant::now();
        let models_url = endpoint(&self.base_url, "models");
        let response = get_with_retries(
            self.provider_name.as_str(),
            &self.client,
            &models_url,
            &headers,
            timeout,
            env_int("OPENAI_MAX_RETRIES", 1) as usize,
        )
        .await?;
        let latency_ms = started.elapsed().as_millis() as u64;
        if response.status().is_success() {
            return Ok(ProviderHealth {
                ok: true,
                status_code: Some(response.status().as_u16()),
                latency_ms: Some(latency_ms),
                message: Some("ok".to_string()),
                mode: Some("http".to_string()),
            });
        }
        if self.provider_name != "openai" {
            let chat_url = endpoint(&self.base_url, "chat/completions");
            let payload = json!({
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0,
            });
            let started = Instant::now();
            let response = post_json_with_retries(
                self.provider_name.as_str(),
                &self.client,
                &chat_url,
                &headers,
                &payload,
                timeout,
                env_int("OPENAI_MAX_RETRIES", 1) as usize,
            )
            .await?;
            let latency_ms = started.elapsed().as_millis() as u64;
            let status = response.status();
            let body = response.text().await?;
            let mut data: Value =
                serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
            if status == StatusCode::ACCEPTED {
                if let Some(request_id) = extract_async_request_id(&data) {
                    data = poll_async_status(
                        self.provider_name.as_str(),
                        &self.client,
                        &headers,
                        &self.base_url,
                        request_id.as_str(),
                        timeout,
                    )
                    .await?;
                    let final_status = data
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| "ok".to_string());
                    return Ok(ProviderHealth {
                        ok: true,
                        status_code: Some(StatusCode::OK.as_u16()),
                        latency_ms: Some(latency_ms),
                        message: Some(final_status),
                        mode: Some("http".to_string()),
                    });
                }
            }
            return Ok(ProviderHealth {
                ok: status.is_success(),
                status_code: Some(status.as_u16()),
                latency_ms: Some(latency_ms),
                message: Some(if status.is_success() {
                    "ok".to_string()
                } else {
                    error_message(&data)
                }),
                mode: Some("http".to_string()),
            });
        }
        Ok(ProviderHealth {
            ok: response.status().is_success(),
            status_code: Some(response.status().as_u16()),
            latency_ms: Some(latency_ms),
            message: Some(if response.status().is_success() {
                "ok".to_string()
            } else {
                response.status().to_string()
            }),
            mode: Some("http".to_string()),
        })
    }

    fn resolve_base_url(&self, request_base_url: Option<&str>) -> String {
        normalize_base_url(request_base_url.unwrap_or(self.base_url.as_str()))
    }
}

fn openai_model_supports_reasoning_effort(model: &str) -> bool {
    let normalized = model
        .trim()
        .trim_start_matches("openai/")
        .trim_start_matches("openai:")
        .to_ascii_lowercase();
    normalized.starts_with("o1")
        || normalized.starts_with("o3")
        || normalized.starts_with("o4")
        || normalized.starts_with("codex")
}

fn background_enabled(
    total_chars: usize,
    max_tokens: Option<u32>,
    effort: Option<&str>,
    timeout_seconds: u64,
) -> bool {
    if env_bool("OPENAI_RESPONSES_BACKGROUND", false) || env_bool("OPENAI_BACKGROUND", false) {
        return true;
    }
    if !env_bool("OPENAI_BACKGROUND_AUTO", true) {
        return false;
    }
    let min_chars = env_int("OPENAI_BACKGROUND_AUTO_MIN_INPUT_CHARS", 20_000) as usize;
    let min_output = env_int("OPENAI_BACKGROUND_AUTO_MIN_OUTPUT_TOKENS", 1_200) as u32;
    let min_timeout = env_int("OPENAI_BACKGROUND_AUTO_MIN_TIMEOUT_SECONDS", 120);
    total_chars >= min_chars
        || max_tokens.unwrap_or_default() >= min_output
        || (timeout_seconds <= min_timeout
            && matches!(effort, Some("medium") | Some("high") | Some("xhigh")))
}

fn build_responses_payload(
    model: &str,
    request: &ProviderCompletionRequest,
    max_tokens: Option<u32>,
    temperature: f32,
    effort: Option<String>,
    prompt_cache_enabled: bool,
    previous_response_id: Option<String>,
    replay_prompt: Option<String>,
) -> Value {
    let input = if let (Some(_previous_response_id), Some(replay_prompt)) =
        (previous_response_id.clone(), replay_prompt)
    {
        let mut replay_items = Vec::new();
        if let Some(system) = request.system.as_ref() {
            replay_items.push(json!({"role": "system", "content": system}));
        }
        replay_items.push(json!({"role": "user", "content": replay_prompt}));
        json!(replay_items)
    } else {
        json!(responses_input(request))
    };
    let mut payload = json!({
        "model": model,
        "input": input,
    });
    if let Some(previous_response_id) = previous_response_id {
        payload["previous_response_id"] = json!(previous_response_id);
        payload["store"] = json!(true);
    }
    if let Some(effort) = effort.as_ref() {
        payload["reasoning"] = json!({"effort": effort});
    } else {
        payload["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = max_tokens {
        payload["max_output_tokens"] = json!(max_tokens);
    }
    if prompt_cache_enabled {
        if let Some(cache_key) =
            prompt_cache_key(request.system.as_deref(), Some(model), "responses")
        {
            payload["prompt_cache_key"] = json!(cache_key);
        }
    }
    if background_enabled(
        total_input_chars(&request.messages, request.system.as_deref()),
        max_tokens,
        effort.as_deref(),
        request.timeout_seconds.unwrap_or(180),
    ) {
        payload["background"] = json!(true);
        payload["store"] = json!(true);
    }
    payload
}

fn responses_input(request: &ProviderCompletionRequest) -> Vec<Value> {
    let mut input = Vec::new();
    if let Some(system) = request.system.as_ref() {
        input.push(json!({"role": "system", "content": system}));
    }
    for message in &request.messages {
        input.push(json!({
            "role": message.role,
            "content": message.flattened_text(),
        }));
    }
    input
}

fn message_content_to_openai_chat(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => json!(text),
        MessageContent::Parts(parts) => {
            let content = parts
                .iter()
                .map(|part| match part {
                    crate::models::ContentPart::Text { text } => {
                        json!({"type": "text", "text": text})
                    }
                    crate::models::ContentPart::ImageUrl { image_url } => json!({
                        "type": "image_url",
                        "image_url": {"url": image_url.url},
                    }),
                })
                .collect::<Vec<_>>();
            json!(content)
        }
    }
}

fn unsupported_reasoning(status: StatusCode, message: &str) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::UNPROCESSABLE_ENTITY
    ) && message.to_ascii_lowercase().contains("reasoning.effort")
}

fn prompt_cache_rejected(status: StatusCode, message: &str) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST
            | StatusCode::FORBIDDEN
            | StatusCode::NOT_FOUND
            | StatusCode::UNPROCESSABLE_ENTITY
    ) && {
        let lowered = message.to_ascii_lowercase();
        lowered.contains("prompt_cache")
            || lowered.contains("prompt cache")
            || lowered.contains("cache_key")
    }
}

async fn poll_background_response(
    provider: &str,
    client: &Client,
    headers: &HeaderMap,
    base_url: &str,
    response_id: &str,
    timeout: Duration,
) -> Result<Value> {
    let poll_interval = Duration::from_secs_f64(
        env::var("OPENAI_BACKGROUND_POLL_INTERVAL_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(20.0),
    );
    let started = Instant::now();
    loop {
        if started.elapsed() > timeout.saturating_mul(2) {
            return Err(GailError::upstream(
                provider,
                Some(StatusCode::GATEWAY_TIMEOUT),
                "background response polling timed out",
            ));
        }
        let response = get_with_retries(
            provider,
            client,
            &endpoint(base_url, format!("responses/{response_id}")),
            headers,
            timeout,
            env_int("OPENAI_MAX_RETRIES", 2) as usize,
        )
        .await?;
        let status = response.status();
        let body = response.text().await?;
        let data: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        if !status.is_success() {
            return Err(GailError::upstream(
                provider,
                Some(status),
                error_message(&data),
            ));
        }
        match data.get("status").and_then(Value::as_str) {
            Some("queued") | Some("in_progress") => sleep(poll_interval).await,
            _ => return Ok(data),
        }
    }
}

async fn poll_async_status(
    provider: &str,
    client: &Client,
    headers: &HeaderMap,
    base_url: &str,
    request_id: &str,
    timeout: Duration,
) -> Result<Value> {
    let poll_interval = Duration::from_secs_f64(
        env::var("OPENAI_BACKGROUND_POLL_INTERVAL_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(10.0),
    );
    let started = Instant::now();
    loop {
        if started.elapsed() > timeout.saturating_mul(2) {
            return Err(GailError::upstream(
                provider,
                Some(StatusCode::GATEWAY_TIMEOUT),
                "async status polling timed out",
            ));
        }
        let response = get_with_retries(
            provider,
            client,
            &endpoint(base_url, format!("status/{request_id}")),
            headers,
            timeout,
            env_int("OPENAI_MAX_RETRIES", 2) as usize,
        )
        .await?;
        let status = response.status();
        let body = response.text().await?;
        let data: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        if status == StatusCode::ACCEPTED {
            sleep(poll_interval).await;
            continue;
        }
        if !status.is_success() {
            return Err(GailError::upstream(
                provider,
                Some(status),
                error_message(&data),
            ));
        }
        return Ok(data);
    }
}

fn extract_async_request_id(data: &Value) -> Option<String> {
    for key in ["requestId", "request_id", "id"] {
        if let Some(value) = data.get(key).and_then(Value::as_str) {
            let cleaned = value.trim();
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }
    None
}

fn normalize_base_url(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_string()
}

fn endpoint(base_url: &str, path: impl AsRef<str>) -> String {
    format!(
        "{}/{}",
        normalize_base_url(base_url),
        path.as_ref().trim_start_matches('/'),
    )
}

fn extract_openai_response_text(data: &Value) -> String {
    if let Some(output_text) = data.get("output_text").and_then(Value::as_str) {
        if !output_text.trim().is_empty() {
            return output_text.to_string();
        }
    }
    let mut chunks = Vec::new();
    fn extract(value: &Value, chunks: &mut Vec<String>) {
        match value {
            Value::String(text) if !text.trim().is_empty() => chunks.push(text.clone()),
            Value::Array(items) => items.iter().for_each(|item| extract(item, chunks)),
            Value::Object(map) => {
                for key in ["text", "output_text", "summary", "refusal"] {
                    if let Some(Value::String(text)) = map.get(key) {
                        if !text.trim().is_empty() {
                            chunks.push(text.clone());
                            return;
                        }
                    }
                }
                if let Some(content) = map.get("content") {
                    extract(content, chunks);
                }
            }
            _ => {}
        }
    }
    if let Some(output) = data.get("output") {
        extract(output, &mut chunks);
    }
    chunks.join("\n")
}

fn extract_openai_usage(data: &Value) -> Option<TokenUsage> {
    let usage = data.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    let completion = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .map(|value| value as u32)
        .or_else(|| Some(prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0))));
    let cached = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .map(|value| value as u32);
    Some(TokenUsage {
        prompt,
        completion,
        total,
        cached,
        cost: None,
    })
}
