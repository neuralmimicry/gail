use std::{
    collections::HashSet,
    env,
    time::{Duration, Instant},
};

use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sysinfo::{Disks, System};
use tokio::sync::Semaphore;

use crate::{
    adaptive_schema,
    config::{GailConfig, ProviderProfile},
    errors::{GailError, Result},
    models::{MessageContent, ProviderCompletionRequest, TokenUsage},
};

use super::{
    ProviderHealth, ProviderInvocationResponse, TranscriptionInput, data_url_parts, env_bool,
    env_int, error_message, infer_capabilities_from_model, infer_capabilities_from_text,
    is_model_not_found, post_json_with_retries, response_with_usage,
};

const PROVIDER_HEALTH_FALLBACK_TIMEOUT_SECONDS: u64 = 4;

static OLLAMA_REQUEST_SEMAPHORE: Lazy<Semaphore> =
    Lazy::new(|| Semaphore::new(env_int("GAIL_OLLAMA_MAX_CONCURRENT_REQUESTS", 1).max(1) as usize));

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
        let default_model = env::var("GAIL_OLLAMA_MODEL")
            .ok()
            .or_else(|| env::var("OLLAMA_MODEL").ok())
            .or_else(|| env::var("OLLAMA_DEFAULT_MODEL").ok())
            .unwrap_or_else(|| "llama3.2".to_string());
        let model = profile
            .model
            .clone()
            .unwrap_or_else(|| default_model.clone());
        let base_url = profile
            .base_url
            .clone()
            .or_else(|| env::var("GAIL_OLLAMA_BASE_URL").ok())
            .or_else(|| env::var("OLLAMA_BASE_URL").ok())
            .or_else(|| env::var("OLLAMA_HOST").ok())
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
        let queue_timeout =
            Duration::from_secs(env_int("GAIL_OLLAMA_QUEUE_TIMEOUT_SECONDS", 2).max(1));
        let _permit = match tokio::time::timeout(queue_timeout, OLLAMA_REQUEST_SEMAPHORE.acquire())
            .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                return Err(GailError::upstream(
                    "ollama",
                    None,
                    "Ollama request limiter is closed",
                ));
            }
            Err(_) => {
                return Err(GailError::upstream(
                    "ollama",
                    None,
                    "timeout waiting for local Ollama request slot; local model service is saturated",
                ));
            }
        };
        let base_url = request
            .base_url
            .clone()
            .unwrap_or_else(|| self.base_url.clone());
        let mut model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let prompt = collapse_messages(request);
        let max_predict = env_int("GAIL_OLLAMA_MAX_PREDICT", 512).max(1) as u32;
        let num_predict = request
            .max_tokens
            .map(|value| value.max(1).min(max_predict))
            .unwrap_or(max_predict);
        let default_timeout = env_int("GAIL_OLLAMA_TIMEOUT_SECONDS", 30).max(1);
        let timeout_seconds = request
            .timeout_seconds
            .map(|seconds| seconds.max(1).min(default_timeout))
            .unwrap_or(default_timeout);
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
            let resolution = self
                .resolve_model_for_request(&base_url, &model, &prompt)
                .await?;
            let selected_model = resolution
                .selected_model
                .clone()
                .ok_or_else(|| GailError::upstream("ollama", None, format!("Ollama model {} is not safely available locally. Recommended downloads: {}", resolution.requested_model, resolution.recommended_downloads.join(", "))))?;
            model = selected_model.clone();
            let mut payload = json!({
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": request.temperature.unwrap_or(0.2),
                    "num_predict": num_predict
                },
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
                Duration::from_secs(timeout_seconds),
                env_int("GAIL_OLLAMA_MAX_RETRIES", 0) as usize,
            )
            .await?;
            let latency_ms = started.elapsed().as_millis() as u64;
            let status = response.status();
            let body = response.text().await?;
            let data: Value =
                serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
            if !status.is_success() {
                let message = error_message(&data);
                if attempt == 0
                    && is_model_not_found(status, &message)
                    && model != self.default_model
                {
                    model = self.default_model.clone();
                    continue;
                }
                return Err(GailError::upstream("ollama", Some(status), message));
            }
            let text = data
                .get("response")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let usage = Some(TokenUsage {
                prompt: data
                    .get("prompt_eval_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32),
                completion: data
                    .get("eval_count")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32),
                total: Some(
                    data.get("prompt_eval_count")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32
                        + data.get("eval_count").and_then(Value::as_u64).unwrap_or(0) as u32,
                ),
                cached: None,
                cost: None,
            });
            return Ok(response_with_usage(
                text, data, latency_ms, "ollama", &model, usage,
            ));
        }
        Err(GailError::upstream(
            "ollama",
            None,
            "Ollama retries exhausted",
        ))
    }

    pub async fn transcribe(
        &self,
        _input: &TranscriptionInput,
    ) -> Result<ProviderInvocationResponse> {
        Err(GailError::bad_request(
            "Ollama transcription is not supported by Gail",
        ))
    }

    pub async fn health(&self, timeout_seconds: Option<u64>) -> Result<ProviderHealth> {
        let started = Instant::now();
        let url = format!("{}/api/tags", self.base_url);
        let tags_timeout_seconds = timeout_seconds
            .unwrap_or(PROVIDER_HEALTH_FALLBACK_TIMEOUT_SECONDS)
            .max(1);
        let response = match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(tags_timeout_seconds))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                adaptive_schema::observe_failure(
                    "ollama",
                    "GET",
                    &url,
                    "tags health",
                    None,
                    &error.to_string(),
                )
                .await;
                return Err(error.into());
            }
        };
        let status = response.status();
        let body = response.text().await?;
        let payload: Value =
            serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        if status.is_success() {
            adaptive_schema::observe_success("ollama", "GET", &url, "tags health", &payload).await;
        } else {
            adaptive_schema::observe_failure(
                "ollama",
                "GET",
                &url,
                "tags health",
                Some(status.as_u16()),
                &payload.to_string(),
            )
            .await;
        }
        let tags_latency_ms = started.elapsed().as_millis() as u64;
        if !status.is_success() {
            return Ok(ProviderHealth {
                ok: false,
                status_code: Some(status.as_u16()),
                latency_ms: Some(tags_latency_ms),
                message: Some(status.to_string()),
                mode: Some("http".to_string()),
            });
        }

        if !env_bool("GAIL_OLLAMA_HEALTH_GENERATE_PROBE", true) {
            return Ok(ProviderHealth {
                ok: true,
                status_code: Some(status.as_u16()),
                latency_ms: Some(tags_latency_ms),
                message: Some("ok".to_string()),
                mode: Some("tags".to_string()),
            });
        }

        let Some(model) = select_matching_installed_model(&payload, &self.model)
            .or_else(|| select_matching_installed_model(&payload, &self.default_model))
        else {
            return Ok(ProviderHealth {
                ok: false,
                status_code: Some(status.as_u16()),
                latency_ms: Some(tags_latency_ms),
                message: Some(format!(
                    "Ollama model {} is not installed and auto-pull is disabled",
                    self.model
                )),
                mode: Some("missing_endpoint".to_string()),
            });
        };
        self.generate_probe_health(&model, timeout_seconds).await
    }

    async fn generate_probe_health(
        &self,
        model: &str,
        timeout_seconds: Option<u64>,
    ) -> Result<ProviderHealth> {
        let generate_url = format!("{}/api/generate", self.base_url);
        let probe_timeout_seconds = env_int(
            "GAIL_OLLAMA_HEALTH_GENERATE_TIMEOUT_SECONDS",
            timeout_seconds.unwrap_or(PROVIDER_HEALTH_FALLBACK_TIMEOUT_SECONDS),
        )
        .max(1);
        let probe_payload = json!({
            "model": model,
            "prompt": "Reply OK.",
            "options": {
                "temperature": 0,
                "num_predict": 1
            },
            "stream": false,
        });
        let started = Instant::now();
        let response = self
            .client
            .post(&generate_url)
            .timeout(Duration::from_secs(probe_timeout_seconds))
            .json(&probe_payload)
            .send()
            .await;
        let latency_ms = started.elapsed().as_millis() as u64;
        match response {
            Ok(response) => {
                let status = response.status();
                let body = response.text().await?;
                let payload: Value =
                    serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
                if status.is_success() {
                    adaptive_schema::observe_success(
                        "ollama",
                        "POST",
                        &generate_url,
                        "generate health",
                        &payload,
                    )
                    .await;
                    Ok(ProviderHealth {
                        ok: true,
                        status_code: Some(status.as_u16()),
                        latency_ms: Some(latency_ms),
                        message: Some("ok".to_string()),
                        mode: Some("generate_probe".to_string()),
                    })
                } else {
                    adaptive_schema::observe_failure(
                        "ollama",
                        "POST",
                        &generate_url,
                        "generate health",
                        Some(status.as_u16()),
                        &payload.to_string(),
                    )
                    .await;
                    Ok(ProviderHealth {
                        ok: false,
                        status_code: Some(status.as_u16()),
                        latency_ms: Some(latency_ms),
                        message: Some(error_message(&payload)),
                        mode: Some("upstream".to_string()),
                    })
                }
            }
            Err(error) => {
                adaptive_schema::observe_failure(
                    "ollama",
                    "POST",
                    &generate_url,
                    "generate health",
                    None,
                    &error.to_string(),
                )
                .await;
                Ok(ProviderHealth {
                    ok: false,
                    status_code: None,
                    latency_ms: Some(latency_ms),
                    message: Some(format!("generate probe failed: {error}")),
                    mode: Some(if error.is_timeout() {
                        "timeout".to_string()
                    } else {
                        "upstream".to_string()
                    }),
                })
            }
        }
    }

    pub async fn inventory_status(&self, config: &GailConfig) -> Result<OllamaInventoryStatus> {
        let started = Instant::now();
        let url = format!("{}/api/tags", self.base_url);
        let response = match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                adaptive_schema::observe_failure(
                    "ollama",
                    "GET",
                    &url,
                    "tags inventory",
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
        let payload: Value =
            serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
        if status.is_success() {
            adaptive_schema::observe_success("ollama", "GET", &url, "tags inventory", &payload)
                .await;
        } else {
            adaptive_schema::observe_failure(
                "ollama",
                "GET",
                &url,
                "tags inventory",
                Some(status.as_u16()),
                &payload.to_string(),
            )
            .await;
        }
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
            .unwrap_or_else(|| {
                format!(
                    "{}/.ollama/models",
                    env::var("HOME").unwrap_or_else(|_| "/root".to_string())
                )
            });
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

    async fn resolve_model_for_request(
        &self,
        base_url: &str,
        requested_model: &str,
        prompt_text: &str,
    ) -> Result<ModelResolution> {
        let tags = fetch_ollama_tags(&self.client, base_url).await?;
        let models = tags
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if let Some(model) = select_matching_model_from_entries(&models, requested_model) {
            return Ok(ModelResolution {
                selected_model: Some(model),
                requested_model: requested_model.to_string(),
                recommended_downloads: Vec::new(),
            });
        }
        let prompt_capabilities = infer_capabilities_from_text(prompt_text);
        let requested_capabilities = infer_capabilities_from_model(requested_model);
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
            let bonus = if name.to_ascii_lowercase().contains("coder")
                && prompt_capabilities.contains("code")
            {
                2
            } else {
                0
            };
            let score = overlap + bonus;
            if best
                .as_ref()
                .map(|(best_score, _)| score > *best_score)
                .unwrap_or(true)
            {
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
    let base = normalized
        .split_once(':')
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| normalized.clone());
    let bare = normalized.replace("latest", "").replace(':', "");
    vec![normalized.clone(), base, bare]
}

fn select_matching_installed_model(tags: &Value, requested_model: &str) -> Option<String> {
    let models = tags.get("models").and_then(Value::as_array)?;
    select_matching_model_from_entries(models, requested_model)
}

fn select_matching_model_from_entries(models: &[Value], requested_model: &str) -> Option<String> {
    let requested_normalized = requested_model.trim().to_ascii_lowercase();
    if let Some(exact) = models.iter().find_map(|entry| {
        entry
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| name.trim().eq_ignore_ascii_case(&requested_normalized))
            .map(ToOwned::to_owned)
    }) {
        return Some(exact);
    }
    let requested_aliases = aliases_for_model(requested_model);
    models.iter().find_map(|entry| {
        entry
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| {
                aliases_for_model(name)
                    .iter()
                    .any(|alias| requested_aliases.contains(alias))
            })
            .map(ToOwned::to_owned)
    })
}

fn json_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    headers
}

async fn fetch_ollama_tags(client: &Client, base_url: &str) -> Result<Value> {
    let url = format!("{base_url}/api/tags");
    let response = match client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            adaptive_schema::observe_failure(
                "ollama",
                "GET",
                &url,
                "tags",
                None,
                &error.to_string(),
            )
            .await;
            return Err(error.into());
        }
    };
    let status = response.status();
    let body = response.text().await?;
    let payload: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
    if !status.is_success() {
        adaptive_schema::observe_failure(
            "ollama",
            "GET",
            &url,
            "tags",
            Some(status.as_u16()),
            &payload.to_string(),
        )
        .await;
        return Err(GailError::upstream(
            "ollama",
            Some(status),
            error_message(&payload),
        ));
    }
    adaptive_schema::observe_success("ollama", "GET", &url, "tags", &payload).await;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_matching_accepts_base_name_for_tagged_model() {
        let tags = json!({
            "models": [
                {"name": "llama3.2:3b"}
            ]
        });

        assert_eq!(
            select_matching_installed_model(&tags, "llama3.2").as_deref(),
            Some("llama3.2:3b")
        );
    }

    #[test]
    fn model_matching_keeps_exact_tag_when_available() {
        let tags = json!({
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "llama3.2:3b"}
            ]
        });

        assert_eq!(
            select_matching_installed_model(&tags, "llama3.2:3b").as_deref(),
            Some("llama3.2:3b")
        );
    }
}
