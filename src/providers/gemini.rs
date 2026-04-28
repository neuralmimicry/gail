use std::{
    collections::HashMap,
    env,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use base64::Engine;
use http::StatusCode;
use reqwest::Client;
use serde_json::{Value, json};

use crate::{
    config::ProviderProfile,
    errors::{GailError, Result},
    models::{ContentPart, MessageContent, ProviderCompletionRequest, TokenUsage},
};

use super::{
    ProviderHealth, ProviderInvocationResponse, TranscriptionInput, data_url_parts, env_bool,
    env_int, error_message, get_with_retries, is_model_not_found, looks_like_json_request,
    post_json_with_retries, response_with_usage,
};

static GEMINI_CACHE: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

fn cache_store() -> &'static Mutex<HashMap<String, String>> {
    GEMINI_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub struct GeminiProvider {
    client: Client,
    model: String,
    default_model: String,
    api_key: Option<String>,
    access_token: Option<String>,
}

impl GeminiProvider {
    pub fn new(client: Client, profile: &ProviderProfile) -> Result<Self> {
        let api_key = profile
            .api_key
            .clone()
            .or_else(|| env::var("GEMINI_API_KEY").ok());
        let access_token = profile
            .access_token
            .clone()
            .or_else(|| env::var("GEMINI_ACCESS_TOKEN").ok())
            .or_else(|| env::var("GOOGLE_ACCESS_TOKEN").ok());
        if api_key.is_none() && access_token.is_none() {
            return Err(GailError::bad_request(
                "GEMINI_API_KEY or GEMINI_ACCESS_TOKEN/GOOGLE_ACCESS_TOKEN must be configured",
            ));
        }
        let default_model =
            env::var("GEMINI_DEFAULT_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
        let model = profile
            .model
            .clone()
            .unwrap_or_else(|| default_model.clone());
        Ok(Self {
            client,
            model,
            default_model,
            api_key,
            access_token,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn complete(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<ProviderInvocationResponse> {
        let mut model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let api_key = request.api_key.clone().or_else(|| self.api_key.clone());
        let access_token = request
            .access_token
            .clone()
            .or_else(|| self.access_token.clone());
        let headers = gemini_headers(api_key.as_deref(), access_token.as_deref())?;
        let timeout = Duration::from_secs(
            request
                .timeout_seconds
                .unwrap_or(env_int("LLM_TIMEOUT_SECONDS", 180))
                .max(1),
        );
        let max_retries = env_int("LLM_MAX_RETRIES", 2) as usize;
        let mut skip_cache = false;
        for attempt in 0..2 {
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            );
            let payload = build_payload(request, &model, skip_cache).await?;
            let started = Instant::now();
            let response = post_json_with_retries(
                "gemini",
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
                if attempt == 0 && cached_content_error(status, &message) {
                    skip_cache = true;
                    continue;
                }
                if attempt == 0
                    && is_model_not_found(status, &message)
                    && model != self.default_model
                {
                    model = self.default_model.clone();
                    continue;
                }
                return Err(GailError::upstream("gemini", Some(status), message));
            }
            let text = data
                .get("candidates")
                .and_then(Value::as_array)
                .and_then(|candidates| candidates.first())
                .and_then(|candidate| candidate.get("content"))
                .and_then(|content| content.get("parts"))
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            let usage = extract_gemini_usage(&data);
            return Ok(response_with_usage(
                text, data, latency_ms, "gemini", &model, usage,
            ));
        }
        Err(GailError::upstream(
            "gemini",
            None,
            "Gemini retries exhausted",
        ))
    }

    pub async fn transcribe(
        &self,
        input: &TranscriptionInput,
    ) -> Result<ProviderInvocationResponse> {
        let mime_type = input.mime_type.clone().unwrap_or_else(|| {
            mime_guess::from_path(&input.filename)
                .first_or_octet_stream()
                .to_string()
        });
        let request = ProviderCompletionRequest {
            provider: "gemini".to_string(),
            model: Some(self.model.clone()),
            api_key: self.api_key.clone(),
            access_token: self.access_token.clone(),
            base_url: None,
            messages: vec![crate::models::ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Please transcribe this audio or video file exactly.".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: crate::models::ImageUrlValue {
                            url: format!(
                                "data:{mime_type};base64,{}",
                                base64::engine::general_purpose::STANDARD.encode(&input.bytes)
                            ),
                        },
                    },
                ]),
            }],
            system: None,
            max_tokens: None,
            temperature: Some(0.1),
            timeout_seconds: input.timeout_seconds,
            reasoning_effort: None,
            request_category: None,
        };
        self.complete(&request).await
    }

    pub async fn health(&self, timeout_seconds: Option<u64>) -> Result<ProviderHealth> {
        let headers = gemini_headers(self.api_key.as_deref(), self.access_token.as_deref())?;
        let started = Instant::now();
        let response = get_with_retries(
            "gemini",
            &self.client,
            "https://generativelanguage.googleapis.com/v1beta/models",
            &headers,
            Duration::from_secs(
                timeout_seconds
                    .unwrap_or(env_int("LLM_TIMEOUT_SECONDS", 60))
                    .max(1),
            ),
            env_int("LLM_MAX_RETRIES", 1) as usize,
        )
        .await?;
        let latency_ms = started.elapsed().as_millis() as u64;
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
}

fn gemini_headers(api_key: Option<&str>, access_token: Option<&str>) -> Result<http::HeaderMap> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    if let Some(api_key) = api_key {
        headers.insert(
            "x-goog-api-key",
            http::HeaderValue::from_str(api_key)
                .map_err(|error| GailError::bad_request(error.to_string()))?,
        );
    } else if let Some(access_token) = access_token {
        headers.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_str(&format!("Bearer {access_token}"))
                .map_err(|error| GailError::bad_request(error.to_string()))?,
        );
    } else {
        return Err(GailError::bad_request("Gemini credentials not configured"));
    }
    Ok(headers)
}

async fn build_payload(
    request: &ProviderCompletionRequest,
    model: &str,
    skip_cache: bool,
) -> Result<Value> {
    let contents = request
        .messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role,
                "parts": message_parts(&message.content),
            })
        })
        .collect::<Vec<_>>();
    let mut generation_config = json!({
        "temperature": request.temperature.unwrap_or(0.2),
    });
    if looks_like_json_request(&request.messages, request.system.as_deref()) {
        generation_config["responseMimeType"] = json!("application/json");
    }
    if let Some(max_tokens) = request.max_tokens {
        generation_config["maxOutputTokens"] = json!(max_tokens);
    }
    let mut payload = json!({
        "contents": contents,
        "generationConfig": generation_config,
    });
    if !skip_cache && env_bool("GEMINI_EXPLICIT_CACHE", true) {
        if let Some(cached_content) = maybe_cached_content(model, request.system.as_deref()).await?
        {
            payload["cachedContent"] = json!(cached_content);
        } else if let Some(system) = request.system.as_ref() {
            payload["systemInstruction"] = json!({"parts": [{"text": system}]});
        }
    } else if let Some(system) = request.system.as_ref() {
        payload["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    Ok(payload)
}

fn message_parts(content: &MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(text) => vec![json!({"text": text})],
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(json!({"text": text})),
                ContentPart::ImageUrl { image_url } => data_url_parts(&image_url.url).map(|(mime_type, data)| {
                    json!({"inline_data": {"mime_type": mime_type, "data": data}})
                }),
            })
            .collect(),
    }
}

async fn maybe_cached_content(model: &str, system: Option<&str>) -> Result<Option<String>> {
    let cache_key = format!("{}::{}", model, system.unwrap_or("gemini-default-cache"));
    if let Some(existing) = cache_store()
        .lock()
        .expect("gemini cache lock")
        .get(&cache_key)
        .cloned()
    {
        return Ok(Some(existing));
    }
    if system.unwrap_or_default().trim().is_empty() {
        return Ok(None);
    }
    let seed = env::var("GEMINI_CACHE_SEED").unwrap_or_else(|_| ".".to_string());
    let payload = json!({
        "model": if model.starts_with("models/") { model.to_string() } else { format!("models/{model}") },
        "contents": [{"role": "user", "parts": [{"text": seed}]}],
        "systemInstruction": {"parts": [{"text": system.unwrap_or_default()}]},
    });
    let headers = gemini_headers(
        env::var("GEMINI_API_KEY").ok().as_deref(),
        env::var("GEMINI_ACCESS_TOKEN")
            .ok()
            .or_else(|| env::var("GOOGLE_ACCESS_TOKEN").ok())
            .as_deref(),
    )?;
    let client = Client::builder().build()?;
    let response = client
        .post("https://generativelanguage.googleapis.com/v1beta/cachedContents")
        .headers(headers)
        .timeout(Duration::from_secs(20))
        .json(&payload)
        .send()
        .await?;
    if !response.status().is_success() {
        return Ok(None);
    }
    let data: Value = response.json().await?;
    let name = data
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if let Some(name) = name.clone() {
        cache_store()
            .lock()
            .expect("gemini cache lock")
            .insert(cache_key, name);
    }
    Ok(name)
}

fn cached_content_error(status: StatusCode, message: &str) -> bool {
    matches!(
        status,
        StatusCode::BAD_REQUEST | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND
    ) && {
        let lowered = message.to_ascii_lowercase();
        lowered.contains("cachedcontent") || lowered.contains("cached content")
    }
}

fn extract_gemini_usage(data: &Value) -> Option<TokenUsage> {
    let usage = data.get("usageMetadata")?;
    Some(TokenUsage {
        prompt: usage
            .get("promptTokenCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        completion: usage
            .get("candidatesTokenCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        total: usage
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        cached: usage
            .get("cachedContentTokenCount")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        cost: None,
    })
}
