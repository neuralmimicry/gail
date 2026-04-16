use std::{collections::HashSet, env, time::{Duration, Instant}};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sysinfo::{Disks, System};

use crate::{
    config::{GailConfig, ProviderProfile},
    errors::{GailError, Result},
    models::{MessageContent, ProviderCompletionRequest, TokenUsage},
};

use super::{
    ProviderHealth, ProviderInvocationResponse, TranscriptionInput,
    data_url_parts, env_bool, env_int, error_message,
    infer_capabilities_from_model, infer_capabilities_from_text, is_model_not_found,
    post_json_with_retries, response_with_usage,
};

#[derive(Clone)]
pub struct OllamaProvider {
    client: Client,
    model: String,
    default_model: String,
    base_url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaInventoryStatus {
    pub provider: Value,
    pub resources: Value,
    pub counts: Value,
    pub models: Vec<Value>,
}

#[derive(Clone, Debug)]
struct ModelResolution {
    selected_model: Option<String>,
    requested_model: String,
    recommended_downloads: Vec<String>,
}

impl OllamaProvider {
    pub fn new(client: Client, profile: &ProviderProfile) -> Self {
        let default_model = env::var("OLLAMA_DEFAULT_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
        let model = profile.model.clone().unwrap_or_else(|| default_model.clone());
        let base_url = profile
            .base_url
            .clone()
            .or_else(|| env::var("OLLAMA_BASE_URL").ok())
            .unwrap_or_else(|| "http://localhost:11434".to_string())
            .trim_end_matches('/')
            .to_string();
        Self {
            client,
            model,
            default_model,
            base_url,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn complete(&self, request: &ProviderCompletionRequest) -> Result<ProviderInvocationResponse> {
        self.complete_once(request).await
    }

    async fn complete_once(&self, request: &ProviderCompletionRequest) -> Result<ProviderInvocationResponse> {
        let base_url = request.base_url.clone().unwrap_or_else(|| self.base_url.clone());
        let mut model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let prompt = collapse_messages(request);
        let images = request
            .messages
            .iter()
            .flat_map(|message| match &message.content {
                MessageContent::Text(_) => Vec::new(),
                MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|part| match part {
                        crate::models::ContentPart::ImageUrl { image_url } => {
                            data_url_parts(&image_url.url).map(|(_, data)| data)
                        }
                        crate::models::ContentPart::Text { .. } => None,
                    })
                    .collect::<Vec<_>>(),
            })
            .collect::<Vec<_>>();
        for attempt in 0..2 {
            let resolution = self.resolve_model_for_request(&base_url, &model, &prompt).await?;
            let selected_model = resolution
                .selected_model
                .clone()
                .ok_or_else(|| GailError::upstream("ollama", None, format!("Ollama model {} is not safely available locally. Recommended downloads: {}", resolution.requested_model, resolution.recommended_downloads.join(", "))))?;
            model = selected_model.clone();
            let mut payload = json!({
                "model": model,
                "prompt": prompt,
                "options": {"temperature": request.temperature.unwrap_or(0.2)},
                "stream": false,
            });
            if !images.is_empty() {
                payload["images"] = json!(images);
            }
            let started = Instant::now();
            let response = post_json_with_retries(
                "ollama",
                &self.client,
                &format!("{base_url}/api/generate"),
                &json_headers(),
                &payload,
                Duration::from_secs(request.timeout_seconds.unwrap_or(env_int("LLM_TIMEOUT_SECONDS", 180)).max(1)),
                env_int("LLM_MAX_RETRIES", 2) as usize,
            )
            .await?;
            let latency_ms = started.elapsed().as_millis() as u64;
            let status = response.status();
            let body = response.text().await?;
            let data: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
            if !status.is_success() {
                let message = error_message(&data);
                if attempt == 0 && is_model_not_found(status, &message) && model != self.default_model {
                    model = self.default_model.clone();
                    continue;
                }
                return Err(GailError::upstream("ollama", Some(status), message));
            }
            let text = data.get("response").and_then(Value::as_str).unwrap_or_default().to_string();
            let usage = Some(TokenUsage {
                prompt: data.get("prompt_eval_count").and_then(Value::as_u64).map(|value| value as u32),
                completion: data.get("eval_count").and_then(Value::as_u64).map(|value| value as u32),
                total: Some(
                    data.get("prompt_eval_count").and_then(Value::as_u64).unwrap_or(0) as u32
                        + data.get("eval_count").and_then(Value::as_u64).unwrap_or(0) as u32,
                ),
                cached: None,
                cost: None,
            });
            return Ok(response_with_usage(text, data, latency_ms, "ollama", &model, usage));
        }
        Err(GailError::upstream("ollama", None, "Ollama retries exhausted"))
    }

    pub async fn transcribe(&self, _input: &TranscriptionInput) -> Result<ProviderInvocationResponse> {
        Err(GailError::bad_request("Ollama transcription is not supported by Gail"))
    }

    pub async fn health(&self, timeout_seconds: Option<u64>) -> Result<ProviderHealth> {
        let started = Instant::now();
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(timeout_seconds.unwrap_or(env_int("LLM_TIMEOUT_SECONDS", 60)).max(1)))
            .send()
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

    pub async fn inventory_status(&self, config: &GailConfig) -> Result<OllamaInventoryStatus> {
        let started = Instant::now();
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(10))
            .send()
            .await?;
        let latency_ms = started.elapsed().as_millis() as u64;
        let status = response.status();
        let body = response.text().await?;
        let payload: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        let models = payload
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let available_memory = System::new_all().available_memory();
        let disks = Disks::new_with_refreshed_list();
        let model_store = config
            .storage
            .ollama_model_store_path
            .clone()
            .or_else(|| env::var("OLLAMA_MODELS").ok())
            .unwrap_or_else(|| format!("{}/.ollama/models", env::var("HOME").unwrap_or_else(|_| "/root".to_string())));
        let disk_available = disks
            .iter()
            .find(|disk| model_store.starts_with(disk.mount_point().to_string_lossy().as_ref()))
            .map(|disk| disk.available_space())
            .or_else(|| disks.iter().next().map(|disk| disk.available_space()))
            .unwrap_or(0);
        Ok(OllamaInventoryStatus {
            provider: json!({
                "name": "ollama",
                "base_url": self.base_url,
                "reachable": status.is_success(),
                "status_code": status.as_u16(),
                "latency_ms": latency_ms,
                "message": if status.is_success() { "ok".to_string() } else { status.to_string() },
                "auto_pull_guard": !env_bool("OLLAMA_ALLOW_AUTO_PULL", false),
            }),
            resources: json!({
                "memory_available_bytes": available_memory,
                "disk_available_bytes": disk_available,
                "model_store_path": model_store,
            }),
            counts: json!({
                "total_models": models.len(),
                "installed_models": models.len(),
                "ready_models": models.len(),
                "download_candidates": 0,
                "blocked_memory": 0,
                "blocked_disk": 0,
            }),
            models,
        })
    }

    async fn resolve_model_for_request(&self, base_url: &str, requested_model: &str, prompt_text: &str) -> Result<ModelResolution> {
        let tags = fetch_ollama_tags(&self.client, base_url).await?;
        let models = tags
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let requested_aliases = aliases_for_model(requested_model);
        let prompt_capabilities = infer_capabilities_from_text(prompt_text);
        let requested_capabilities = infer_capabilities_from_model(requested_model);
        if let Some(installed) = models.iter().find(|entry| {
            entry
                .get("name")
                .and_then(Value::as_str)
                .map(|name| aliases_for_model(name).iter().any(|alias| requested_aliases.contains(alias)))
                .unwrap_or(false)
        }) {
            let model = installed.get("name").and_then(Value::as_str).unwrap_or(requested_model);
            return Ok(ModelResolution {
                selected_model: Some(model.to_string()),
                requested_model: requested_model.to_string(),
                recommended_downloads: Vec::new(),
            });
        }
        let mut best: Option<(i32, String)> = None;
        for entry in &models {
            let Some(name) = entry.get("name").and_then(Value::as_str) else {
                continue;
            };
            let capabilities = infer_capabilities_from_model(name);
            let wanted = prompt_capabilities
                .union(&requested_capabilities)
                .cloned()
                .collect::<HashSet<_>>();
            let overlap = capabilities.intersection(&wanted).count() as i32;
            let bonus = if name.to_ascii_lowercase().contains("coder") && prompt_capabilities.contains("code") { 2 } else { 0 };
            let score = overlap + bonus;
            if best.as_ref().map(|(best_score, _)| score > *best_score).unwrap_or(true) {
                best = Some((score, name.to_string()));
            }
        }
        if let Some((_, selected_model)) = best {
            return Ok(ModelResolution {
                selected_model: Some(selected_model),
                requested_model: requested_model.to_string(),
                recommended_downloads: Vec::new(),
            });
        }
        let allow_pull = env_bool("OLLAMA_ALLOW_AUTO_PULL", false);
        Ok(ModelResolution {
            selected_model: allow_pull.then(|| requested_model.to_string()),
            requested_model: requested_model.to_string(),
            recommended_downloads: vec![requested_model.to_string()],
        })
    }
}

fn collapse_messages(request: &ProviderCompletionRequest) -> String {
    let mut prompt = String::new();
    if let Some(system) = request.system.as_ref() {
        prompt.push_str("System: ");
        prompt.push_str(system);
        prompt.push('\n');
    }
    for message in &request.messages {
        prompt.push_str(message.role.as_str());
        prompt.push_str(": ");
        prompt.push_str(message.flattened_text().as_str());
        prompt.push('\n');
    }
    prompt
}

fn aliases_for_model(model: &str) -> Vec<String> {
    let normalized = model.trim().to_ascii_lowercase();
    let bare = normalized.replace("latest", "").replace(':', "");
    vec![normalized.clone(), bare]
}

fn json_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("application/json"));
    headers
}

async fn fetch_ollama_tags(client: &Client, base_url: &str) -> Result<Value> {
    let response = client
        .get(format!("{base_url}/api/tags"))
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    let payload: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
    if !status.is_success() {
        return Err(GailError::upstream("ollama", Some(status), error_message(&payload)));
    }
    Ok(payload)
}
