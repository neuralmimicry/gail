use std::{
    collections::{HashMap, HashSet},
    env,
    time::{Duration, Instant},
};

use http::StatusCode;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sysinfo::{Disks, System};
use tokio::sync::{Mutex, Semaphore, SemaphorePermit};

use crate::{
    adaptive_schema,
    config::{GailConfig, ProviderProfile},
    errors::{GailError, Result},
    models::{MessageContent, ProviderCompletionRequest, TokenUsage},
};

use super::{
    ProviderHealth, ProviderInvocationResponse, TranscriptionInput, data_url_parts, env_bool,
    env_float, env_int, error_message, infer_capabilities_from_model, infer_capabilities_from_text,
    is_model_not_found, post_json_with_retries, response_with_usage,
};

async fn acquire_ollama_request_permit(
    total_timeout_seconds: u64,
) -> Result<SemaphorePermit<'static>> {
    let queue_timeout_seconds = resolved_ollama_queue_timeout_seconds(total_timeout_seconds);
    let queue_timeout = Duration::from_secs(queue_timeout_seconds);

    match tokio::time::timeout(queue_timeout, OLLAMA_REQUEST_SEMAPHORE.acquire()).await {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err(GailError::upstream(
            "ollama",
            None,
            "Ollama request limiter is closed",
        )),
        Err(_) => Err(GailError::upstream(
            "ollama",
            Some(StatusCode::TOO_MANY_REQUESTS),
            format!(
                "local Ollama request queue is saturated after waiting {queue_timeout_seconds}s"
            ),
        )),
    }
}
const PROVIDER_HEALTH_FALLBACK_TIMEOUT_SECONDS: u64 = 12;
const OLLAMA_SATURATION_BACKOFF_SECONDS: u64 = 20;
const OLLAMA_COOLDOWN_SKIP_LOG_INTERVAL_SECONDS: u64 = 5;

static OLLAMA_REQUEST_SEMAPHORE: Lazy<Semaphore> =
    Lazy::new(|| Semaphore::new(resolved_ollama_max_concurrent_requests() as usize));
static OLLAMA_SATURATED_UNTIL: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));
static OLLAMA_ENDPOINT_SATURATED_UNTIL: Lazy<Mutex<HashMap<String, Instant>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static OLLAMA_ENDPOINT_SKIP_LOGGED_AT: Lazy<Mutex<HashMap<String, Instant>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn resolved_ollama_max_concurrent_requests() -> u64 {
    let configured = env_int("GAIL_OLLAMA_MAX_CONCURRENT_REQUESTS", 1).max(1);
    if cfg!(test) {
        configured.max(8)
    } else {
        configured
    }
}

fn resolved_ollama_total_timeout_seconds(request_timeout_seconds: Option<u64>) -> u64 {
    let default_timeout = env_int("GAIL_OLLAMA_TIMEOUT_SECONDS", 90).max(1);
    let min_request_timeout = env_int("GAIL_OLLAMA_MIN_REQUEST_TIMEOUT_SECONDS", 1)
        .max(1)
        .min(default_timeout);

    request_timeout_seconds
        .unwrap_or(default_timeout)
        .max(1)
        .max(min_request_timeout)
        .min(default_timeout)
}

fn resolved_ollama_queue_timeout_seconds(total_timeout_seconds: u64) -> u64 {
    let configured_queue_timeout = env_int("GAIL_OLLAMA_QUEUE_TIMEOUT_SECONDS", 20).max(1);
    let min_request_timeout = env_int("GAIL_OLLAMA_MIN_REQUEST_TIMEOUT_SECONDS", 1)
        .max(1)
        .min(total_timeout_seconds.max(1));
    let queue_budget = total_timeout_seconds
        .max(1)
        .saturating_sub(min_request_timeout)
        .max(1);
    configured_queue_timeout.min(queue_budget)
}

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

#[derive(Clone, Copy, Debug)]
struct LocalResourceSnapshot {
    memory_available_mb: f64,
    disk_available_mb: f64,
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

    fn base_url_candidates(&self, request_base_url: Option<&str>) -> Vec<String> {
        let mut candidates = Vec::new();
        let request_base = request_base_url.and_then(normalize_base_url);
        let has_explicit_request_base = request_base.is_some();
        let request_base_is_cluster = request_base
            .as_deref()
            .is_some_and(base_url_uses_cluster_service_host);
        let self_base_is_cluster = base_url_uses_cluster_service_host(self.base_url.as_str());
        let include_cluster_fallback_defaults = request_base_is_cluster || self_base_is_cluster;
        let internal_base_url = env::var("GAIL_OLLAMA_INTERNAL_BASE_URL").ok();
        let native_cluster_base_url = "http://ollama.ollama.svc.cluster.local:11434";
        let openai_compat_cluster_base_url =
            "http://ollama-openai-compat.ollama.svc.cluster.local:11434";
        push_base_url_candidate(&mut candidates, request_base.as_deref());
        push_base_url_candidate(&mut candidates, Some(self.base_url.as_str()));
        let allow_cross_endpoint_fallback = env_bool(
            "GAIL_OLLAMA_ALLOW_CROSS_ENDPOINT_FALLBACK",
            !has_explicit_request_base || include_cluster_fallback_defaults,
        );
        if allow_cross_endpoint_fallback && include_cluster_fallback_defaults {
            push_base_url_candidate(&mut candidates, internal_base_url.as_deref());
            push_base_url_candidate(&mut candidates, Some(native_cluster_base_url));
            push_base_url_candidate(&mut candidates, Some(openai_compat_cluster_base_url));
        }
        for value in env_base_url_list("GAIL_OLLAMA_FALLBACK_BASE_URLS")
            .into_iter()
            .chain(env_base_url_list("OLLAMA_FALLBACK_BASE_URLS"))
        {
            push_base_url_candidate(&mut candidates, Some(value.as_str()));
        }

        let has_non_local_candidate = candidates
            .iter()
            .any(|candidate| !base_url_uses_local_ollama_host(candidate));
        if candidates.is_empty() || (allow_cross_endpoint_fallback && has_non_local_candidate) {
            push_base_url_candidate(&mut candidates, internal_base_url.as_deref());
            push_base_url_candidate(&mut candidates, Some(native_cluster_base_url));
            push_base_url_candidate(&mut candidates, Some(openai_compat_cluster_base_url));
        }

        let mut with_variants = Vec::new();
        for candidate in candidates {
            push_base_url_candidate(&mut with_variants, Some(candidate.as_str()));
            for variant in endpoint_scheme_variants(candidate.as_str()) {
                push_base_url_candidate(&mut with_variants, Some(variant.as_str()));
            }
        }

        let max_endpoints = env_int("GAIL_OLLAMA_MAX_ENDPOINTS", 4).max(1) as usize;
        with_variants.truncate(max_endpoints);
        with_variants
    }

    pub async fn complete(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<ProviderInvocationResponse> {
        if ollama_family_saturation_enabled()
            && let Some(remaining) = ollama_saturation_remaining().await
        {
            return Err(ollama_saturated_error(remaining));
        }
        let total_timeout_seconds = resolved_ollama_total_timeout_seconds(request.timeout_seconds);
        let _permit = acquire_ollama_request_permit(total_timeout_seconds).await?;
        let base_urls = self.base_url_candidates(request.base_url.as_deref());
        let queue_wait_ms = 0;
        let deadline = Instant::now() + Duration::from_secs(total_timeout_seconds);
        let mut last_error = None;
        for (index, base_url) in base_urls.iter().enumerate() {
            if let Some(remaining) = ollama_endpoint_saturation_remaining(base_url).await {
                if should_log_ollama_endpoint_cooldown_skip(base_url).await {
                    tracing::info!(
                        base_url = %base_url,
                        endpoint_index = index,
                        cooldown_seconds = remaining.as_secs().max(1),
                        "skipping Ollama endpoint because it is in saturation cooldown"
                    );
                }
                last_error = Some(ollama_saturated_error(remaining));
                continue;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                last_error = Some(GailError::upstream(
                    "ollama",
                    Some(StatusCode::GATEWAY_TIMEOUT),
                    format!(
                        "Ollama adaptive endpoint budget exhausted after {total_timeout_seconds}s"
                    ),
                ));
                break;
            }

            let endpoints_remaining = base_urls.len().saturating_sub(index).max(1);
            let endpoint_budget = compute_ollama_endpoint_budget(remaining, endpoints_remaining);

            let mut scoped_request = request.clone();
            scoped_request.base_url = Some(base_url.clone());
            scoped_request.timeout_seconds =
                Some(endpoint_budget.as_secs().max(1).min(total_timeout_seconds));

            let result = tokio::time::timeout(
                endpoint_budget,
                self.complete_once(&scoped_request, queue_wait_ms),
            )
            .await;
            match result {
                Err(_) => {
                    let message = format!(
                        "Ollama adaptive endpoint budget exhausted for endpoint {base_url} within {}s of total {}s budget",
                        endpoint_budget.as_secs().max(1),
                        total_timeout_seconds,
                    );

                    adaptive_schema::observe_failure(
                        "ollama",
                        "POST",
                        &format!("{base_url}/api/generate"),
                        "generate adaptive endpoint",
                        None,
                        &message,
                    )
                    .await;

                    last_error = Some(GailError::upstream(
                        "ollama",
                        Some(StatusCode::GATEWAY_TIMEOUT),
                        message,
                    ));

                    if index + 1 < base_urls.len()
                        && !deadline.saturating_duration_since(Instant::now()).is_zero()
                    {
                        continue;
                    }

                    break;
                }
                Ok(Ok(mut response)) => {
                    if let Some(raw) = response.raw.as_mut() {
                        raw["gail_ollama_base_url"] = json!(base_url);
                        raw["gail_ollama_endpoint_index"] = json!(index);
                        raw["gail_ollama_endpoint_failover"] = json!(index > 0);
                        raw["gail_ollama_total_timeout_seconds"] = json!(total_timeout_seconds);
                    }
                    if index > 0 {
                        let observed_raw = response.raw.clone().unwrap_or(Value::Null);
                        adaptive_schema::observe_success(
                            "ollama",
                            "POST",
                            &format!("{base_url}/api/generate"),
                            "generate adaptive endpoint",
                            &observed_raw,
                        )
                        .await;
                    }
                    return Ok(response);
                }
                Ok(Err(error)) => {
                    let message = error.to_string();
                    adaptive_schema::observe_failure(
                        "ollama",
                        "POST",
                        &format!("{base_url}/api/generate"),
                        "generate adaptive endpoint",
                        None,
                        &message,
                    )
                    .await;
                    if message_indicates_ollama_saturation(&message) {
                        let endpoint_backoff = mark_ollama_endpoint_saturated(base_url).await;
                        let family_backoff = if ollama_family_saturation_enabled() {
                            Some(mark_ollama_saturated().await)
                        } else {
                            None
                        };
                        tracing::warn!(
                            base_url = %base_url,
                            endpoint_index = index,
                            error = %message,
                            endpoint_backoff_seconds = endpoint_backoff.as_secs().max(1),
                            family_backoff_seconds = family_backoff.map(|value| value.as_secs().max(1)),
                            "Ollama local model service saturated; backing off before retrying"
                        );
                        last_error = Some(error);
                        continue;
                    }
                    if message_indicates_ollama_endpoint_transport_failure(&message) {
                        let endpoint_backoff = mark_ollama_endpoint_saturated(base_url).await;
                        tracing::warn!(
                            base_url = %base_url,
                            endpoint_index = index,
                            error = %message,
                            endpoint_backoff_seconds = endpoint_backoff.as_secs().max(1),
                            "Ollama endpoint transport failure detected; cooling down endpoint before adaptive fallback"
                        );
                        last_error = Some(error);
                        continue;
                    }
                    tracing::warn!(
                        base_url = %base_url,
                        endpoint_index = index,
                        error = %message,
                        "Ollama endpoint failed; trying adaptive fallback endpoint if budget remains"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            GailError::upstream("ollama", None, "no Ollama endpoints were configured")
        }))
    }

    async fn complete_once(
        &self,
        request: &ProviderCompletionRequest,
        queue_wait_ms: u64,
    ) -> Result<ProviderInvocationResponse> {
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
        let default_timeout = env_int("GAIL_OLLAMA_TIMEOUT_SECONDS", 90).max(1);
        let timeout_seconds = request
            .timeout_seconds
            .map(|seconds| seconds.max(1).min(default_timeout))
            .unwrap_or(default_timeout);
        let tags_timeout_seconds = env_int("GAIL_OLLAMA_TAGS_TIMEOUT_SECONDS", 4)
            .max(1)
            .min(timeout_seconds.max(1));
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
        if model_looks_non_generative(model.as_str()) {
            return Err(GailError::bad_request(format!(
                "Ollama model {model} does not support text generation via /api/generate"
            )));
        }
        for attempt in 0..2 {
            let resolution = self
                .resolve_model_for_request(
                    &base_url,
                    &model,
                    &prompt,
                    tags_timeout_seconds,
                    request.min_model_size_b,
                    request.strict_no_downgrade.unwrap_or(false),
                )
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
            let inference_ms = started.elapsed().as_millis() as u64;
            let latency_ms = queue_wait_ms.saturating_add(inference_ms);
            let status = response.status();
            let body = response.text().await?;
            let mut data: Value =
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
            let prompt_tokens_actual = data
                .get("prompt_eval_count")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
            let completion_tokens_actual = data
                .get("eval_count")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
            let prompt_tokens_estimate =
                prompt_tokens_actual.unwrap_or_else(|| ((prompt.len().max(1) / 4) as u32).max(1));
            let completion_tokens_estimate =
                completion_tokens_actual.unwrap_or_else(|| ((text.len().max(1) / 4) as u32).max(1));
            let total_tokens_estimate =
                prompt_tokens_estimate.saturating_add(completion_tokens_estimate);
            let usage = Some(TokenUsage {
                prompt: Some(prompt_tokens_actual.unwrap_or(prompt_tokens_estimate)),
                completion: Some(completion_tokens_actual.unwrap_or(completion_tokens_estimate)),
                total: Some(
                    prompt_tokens_actual
                        .unwrap_or(prompt_tokens_estimate)
                        .saturating_add(
                            completion_tokens_actual.unwrap_or(completion_tokens_estimate),
                        ),
                ),
                cached: None,
                cost: None,
            });
            data["gail_ollama_queue_wait_ms"] = json!(queue_wait_ms);
            data["gail_ollama_inference_ms"] = json!(inference_ms);
            data["gail_ollama_total_tokens_estimate"] = json!(total_tokens_estimate);
            data["gail_local_usage"] = json!({
                "provider": "ollama",
                "queue_wait_ms": queue_wait_ms,
                "inference_ms": inference_ms,
                "prompt_tokens_estimate": prompt_tokens_estimate,
                "completion_tokens_estimate": completion_tokens_estimate,
                "total_tokens_estimate": total_tokens_estimate,
                "prompt_eval_count": prompt_tokens_actual,
                "eval_count": completion_tokens_actual,
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
        if ollama_family_saturation_enabled()
            && let Some(remaining) = ollama_saturation_remaining().await
        {
            return Ok(ProviderHealth {
                ok: false,
                status_code: None,
                latency_ms: None,
                message: Some(ollama_saturated_message(remaining)),
                mode: Some("ollama_saturated".to_string()),
            });
        }
        let mut last_health = None;
        let mut last_error = None;
        for (index, base_url) in self.base_url_candidates(None).iter().enumerate() {
            match self.health_for_base_url(base_url, timeout_seconds).await {
                Ok(mut health) if health.ok => {
                    if index > 0 {
                        health.message = Some(format!("ok via adaptive endpoint {base_url}"));
                        health.mode = health.mode.map(|mode| format!("adaptive_{mode}"));
                    }
                    return Ok(health);
                }
                Ok(health) => {
                    last_health = Some(health);
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }
        if let Some(health) = last_health {
            return Ok(health);
        }
        Err(last_error.unwrap_or_else(|| {
            GailError::upstream("ollama", None, "no Ollama endpoints were configured")
        }))
    }

    async fn health_for_base_url(
        &self,
        base_url: &str,
        timeout_seconds: Option<u64>,
    ) -> Result<ProviderHealth> {
        let started = Instant::now();
        let url = format!("{base_url}/api/tags");
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

        if !env_bool("GAIL_OLLAMA_HEALTH_GENERATE_PROBE", false) {
            let configured = self.model.as_str();
            let default_model = self.default_model.as_str();
            let configured_installed = select_matching_installed_model(&payload, configured);
            let default_installed = select_matching_installed_model(&payload, default_model);
            let selected_for_health = configured_installed.clone().or(default_installed.clone());
            if let Some(model_name) = selected_for_health {
                if model_looks_non_generative(model_name.as_str()) {
                    return Ok(ProviderHealth {
                        ok: false,
                        status_code: Some(status.as_u16()),
                        latency_ms: Some(tags_latency_ms),
                        message: Some(format!(
                            "Ollama model {} is embedding-only and does not support generate",
                            model_name
                        )),
                        mode: Some("missing_endpoint".to_string()),
                    });
                }
            } else {
                return Ok(ProviderHealth {
                    ok: false,
                    status_code: Some(status.as_u16()),
                    latency_ms: Some(tags_latency_ms),
                    message: Some(format!(
                        "configured model {} (or default {}) is not installed on {}",
                        configured, default_model, base_url
                    )),
                    mode: Some("missing_endpoint".to_string()),
                });
            }
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
        self.generate_probe_health(base_url, &model, timeout_seconds)
            .await
    }

    async fn generate_probe_health(
        &self,
        base_url: &str,
        model: &str,
        timeout_seconds: Option<u64>,
    ) -> Result<ProviderHealth> {
        let generate_url = format!("{base_url}/api/generate");
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
        let mut last_error = None;
        for base_url in self.base_url_candidates(None) {
            match self
                .inventory_status_for_base_url(config, base_url.as_str())
                .await
            {
                Ok(status) => return Ok(status),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            GailError::upstream("ollama", None, "no Ollama endpoints were configured")
        }))
    }

    async fn inventory_status_for_base_url(
        &self,
        config: &GailConfig,
        base_url: &str,
    ) -> Result<OllamaInventoryStatus> {
        let started = Instant::now();
        let url = format!("{base_url}/api/tags");
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
                "base_url": base_url,
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
        timeout_seconds: u64,
        min_model_size_b: Option<f64>,
        strict_no_downgrade: bool,
    ) -> Result<ModelResolution> {
        let tags = fetch_ollama_tags(&self.client, base_url, timeout_seconds).await?;
        let models = tags
            .get("models")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let generative_models = models
            .iter()
            .filter(|entry| {
                entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| !model_looks_non_generative(name))
                    .unwrap_or(false)
            })
            .cloned()
            .collect::<Vec<_>>();
        let requested_size_b = model_size_billions(requested_model).unwrap_or(0.0);
        let required_min_size_b = min_model_size_b.unwrap_or(0.0).max(if strict_no_downgrade {
            requested_size_b
        } else {
            0.0
        });
        let resource_snapshot = local_resource_snapshot();
        if let Some(model) = select_matching_model_from_entries(&generative_models, requested_model)
            .filter(|model| {
                model_meets_size_floor(model.as_str(), required_min_size_b)
                    && model_fits_resources(model.as_str(), &resource_snapshot)
            })
        {
            return Ok(ModelResolution {
                selected_model: Some(model),
                requested_model: requested_model.to_string(),
                recommended_downloads: Vec::new(),
            });
        }
        if let Some(selected_model) = select_best_local_model(
            &generative_models,
            requested_model,
            prompt_text,
            required_min_size_b,
            &resource_snapshot,
        ) {
            return Ok(ModelResolution {
                selected_model: Some(selected_model),
                requested_model: requested_model.to_string(),
                recommended_downloads: Vec::new(),
            });
        }
        let allow_pull = env_bool("OLLAMA_ALLOW_AUTO_PULL", false);
        let mut recommended_downloads = Vec::new();
        if model_looks_non_generative(requested_model) {
            return Ok(ModelResolution {
                selected_model: None,
                requested_model: requested_model.to_string(),
                recommended_downloads: vec![requested_model.to_string()],
            });
        }
        if let Some(candidate) =
            next_size_up_pull_candidate(requested_model, required_min_size_b.max(requested_size_b))
        {
            recommended_downloads.push(candidate);
        }
        if !recommended_downloads
            .iter()
            .any(|model| model.eq_ignore_ascii_case(requested_model))
        {
            recommended_downloads.push(requested_model.to_string());
        }
        if allow_pull {
            let pull_timeout = env_int("GAIL_OLLAMA_PULL_TIMEOUT_SECONDS", 240).max(5);
            for candidate in &recommended_downloads {
                if !model_fits_resources(candidate.as_str(), &resource_snapshot) {
                    continue;
                }
                if pull_ollama_model(
                    &self.client,
                    base_url,
                    candidate.as_str(),
                    Duration::from_secs(pull_timeout),
                )
                .await
                .is_ok()
                {
                    return Ok(ModelResolution {
                        selected_model: Some(candidate.clone()),
                        requested_model: requested_model.to_string(),
                        recommended_downloads: Vec::new(),
                    });
                }
            }
        }
        if strict_no_downgrade && required_min_size_b > 0.0 {
            return Ok(ModelResolution {
                selected_model: None,
                requested_model: requested_model.to_string(),
                recommended_downloads,
            });
        }
        Ok(ModelResolution {
            selected_model: if allow_pull {
                Some(requested_model.to_string())
            } else {
                None
            },
            requested_model: requested_model.to_string(),
            recommended_downloads,
        })
    }
}

fn model_size_billions(model: &str) -> Option<f64> {
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

fn model_meets_size_floor(model: &str, min_size_b: f64) -> bool {
    if min_size_b <= 0.0 {
        return true;
    }
    model_size_billions(model)
        .map(|size| size + 0.000_1 >= min_size_b)
        .unwrap_or(true)
}

fn model_looks_non_generative(model: &str) -> bool {
    let lowered = model.trim().to_ascii_lowercase();
    lowered.contains("embed")
        || lowered.contains("embedding")
        || lowered.contains("rerank")
        || lowered.contains("re-rank")
        || lowered.contains("retrieval")
}

fn local_resource_snapshot() -> LocalResourceSnapshot {
    let system = System::new_all();
    let memory_available_mb = system.available_memory() as f64 / (1024.0 * 1024.0);
    let disks = Disks::new_with_refreshed_list();
    let disk_available_bytes = disks
        .iter()
        .map(|disk| disk.available_space())
        .max()
        .unwrap_or(0);
    let disk_available_mb = disk_available_bytes as f64 / (1024.0 * 1024.0);
    LocalResourceSnapshot {
        memory_available_mb,
        disk_available_mb,
    }
}

fn model_fits_resources(model: &str, resources: &LocalResourceSnapshot) -> bool {
    let Some(size_b) = model_size_billions(model) else {
        return true;
    };
    let ram_per_b_mb = env_float("GAIL_OLLAMA_RAM_MB_PER_B", 1400.0).max(256.0);
    let disk_per_b_mb = env_float("GAIL_OLLAMA_DISK_MB_PER_B", 900.0).max(128.0);
    let reserve_ratio = env_float("GAIL_OLLAMA_RESOURCE_RESERVE_RATIO", 1.15).clamp(1.0, 4.0);
    let required_ram_mb = size_b * ram_per_b_mb * reserve_ratio;
    let required_disk_mb = size_b * disk_per_b_mb * reserve_ratio;
    resources.memory_available_mb >= required_ram_mb
        && resources.disk_available_mb >= required_disk_mb
}

fn select_best_local_model(
    models: &[Value],
    requested_model: &str,
    prompt_text: &str,
    min_model_size_b: f64,
    resources: &LocalResourceSnapshot,
) -> Option<String> {
    let prompt_capabilities = infer_capabilities_from_text(prompt_text);
    let requested_capabilities = infer_capabilities_from_model(requested_model);
    let wanted = prompt_capabilities
        .union(&requested_capabilities)
        .cloned()
        .collect::<HashSet<_>>();
    let requested_family = requested_model
        .split_once(':')
        .map(|(family, _)| family)
        .unwrap_or(requested_model)
        .to_ascii_lowercase();
    let mut best: Option<(f64, String)> = None;
    for entry in models {
        let Some(name) = entry.get("name").and_then(Value::as_str) else {
            continue;
        };
        if !model_meets_size_floor(name, min_model_size_b) || !model_fits_resources(name, resources)
        {
            continue;
        }
        let capabilities = infer_capabilities_from_model(name);
        let overlap = capabilities.intersection(&wanted).count() as f64;
        let family = name
            .split_once(':')
            .map(|(value, _)| value)
            .unwrap_or(name)
            .to_ascii_lowercase();
        let family_bonus = if family == requested_family {
            1.25
        } else {
            0.0
        };
        let coder_bonus = if name.to_ascii_lowercase().contains("coder")
            && prompt_capabilities.contains("code")
        {
            0.9
        } else {
            0.0
        };
        let size_penalty = model_size_billions(name)
            .map(|size| {
                if min_model_size_b > 0.0 {
                    (size - min_model_size_b).abs() * 0.12
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let score = (overlap * 1.6) + family_bonus + coder_bonus - size_penalty;
        if best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true)
        {
            best = Some((score, name.to_string()));
        }
    }
    best.map(|(_, model)| model)
}

fn next_size_up_pull_candidate(requested_model: &str, min_size_b: f64) -> Option<String> {
    let base = requested_model
        .split_once(':')
        .map(|(value, _)| value.to_string())
        .unwrap_or_else(|| requested_model.to_string());
    let canonical_sizes = [0.5_f64, 1.0, 1.5, 3.0, 7.0, 8.0, 14.0, 32.0, 70.0];
    let threshold = min_size_b.max(model_size_billions(requested_model).unwrap_or(0.0));
    canonical_sizes
        .iter()
        .copied()
        .find(|size| *size + 0.000_1 >= threshold)
        .map(|size| {
            if (size.fract() - 0.0).abs() < f64::EPSILON {
                format!("{base}:{}b", size as u32)
            } else {
                format!("{base}:{size:.1}b")
            }
        })
}

async fn pull_ollama_model(
    client: &Client,
    base_url: &str,
    model: &str,
    timeout: Duration,
) -> Result<()> {
    let url = format!("{base_url}/api/pull");
    let payload = json!({
        "name": model,
        "stream": false,
    });
    let response = client
        .post(&url)
        .timeout(timeout)
        .json(&payload)
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let data: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({"message": body}));
    if status.is_success() {
        adaptive_schema::observe_success("ollama", "POST", &url, "pull model", &data).await;
        Ok(())
    } else {
        adaptive_schema::observe_failure(
            "ollama",
            "POST",
            &url,
            "pull model",
            Some(status.as_u16()),
            &data.to_string(),
        )
        .await;
        Err(GailError::upstream(
            "ollama",
            Some(status),
            error_message(&data),
        ))
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

fn env_base_url_list(name: &str) -> Vec<String> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
                .filter_map(normalize_base_url)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn ollama_saturation_backoff() -> Duration {
    Duration::from_secs(
        env_int(
            "GAIL_OLLAMA_SATURATION_BACKOFF_SECONDS",
            OLLAMA_SATURATION_BACKOFF_SECONDS,
        )
        .max(1),
    )
}

fn ollama_fallback_reserve_seconds() -> u64 {
    env_int("GAIL_OLLAMA_FALLBACK_RESERVE_SECONDS", 12)
        .max(1)
        .min(120)
}

fn compute_ollama_endpoint_budget(remaining: Duration, endpoints_remaining: usize) -> Duration {
    if endpoints_remaining <= 1 {
        return remaining.max(Duration::from_secs(1));
    }

    let endpoints_remaining = endpoints_remaining.max(1) as u32;
    let min_attempt_budget = Duration::from_secs(2).min(remaining.max(Duration::from_secs(1)));
    let split_budget = (remaining / endpoints_remaining)
        .max(min_attempt_budget)
        .min(remaining);

    // Keep a reserve for fallback endpoints, but cap that reserve to one-third of
    // the remaining request budget so short explicit timeouts are not fragmented
    // into tiny endpoint windows.
    let fallback_reserve = Duration::from_secs(ollama_fallback_reserve_seconds())
        .min(remaining / 3)
        .min(remaining);
    let preferred_budget = remaining.saturating_sub(fallback_reserve);
    if preferred_budget >= min_attempt_budget {
        preferred_budget.min(remaining)
    } else {
        split_budget
    }
}

fn ollama_cooldown_skip_log_interval() -> Duration {
    Duration::from_secs(
        env_int(
            "GAIL_OLLAMA_COOLDOWN_SKIP_LOG_INTERVAL_SECONDS",
            OLLAMA_COOLDOWN_SKIP_LOG_INTERVAL_SECONDS,
        )
        .min(600),
    )
}

fn ollama_family_saturation_enabled() -> bool {
    env_bool("GAIL_OLLAMA_FAMILY_SATURATION_BACKOFF", false)
}

async fn ollama_saturation_remaining() -> Option<Duration> {
    let mut saturated_until = OLLAMA_SATURATED_UNTIL.lock().await;
    let until = (*saturated_until)?;
    let now = Instant::now();
    if until <= now {
        *saturated_until = None;
        None
    } else {
        Some(until.saturating_duration_since(now))
    }
}

async fn mark_ollama_saturated() -> Duration {
    let backoff = ollama_saturation_backoff();
    let mut saturated_until = OLLAMA_SATURATED_UNTIL.lock().await;
    *saturated_until = Some(Instant::now() + backoff);
    backoff
}

async fn ollama_endpoint_saturation_remaining(base_url: &str) -> Option<Duration> {
    let key = normalize_base_url(base_url).unwrap_or_else(|| base_url.to_string());
    let mut guard = OLLAMA_ENDPOINT_SATURATED_UNTIL.lock().await;
    let until = guard.get(&key).copied()?;
    let now = Instant::now();
    if until <= now {
        guard.remove(&key);
        None
    } else {
        Some(until.saturating_duration_since(now))
    }
}

async fn mark_ollama_endpoint_saturated(base_url: &str) -> Duration {
    let backoff = ollama_saturation_backoff();
    let key = normalize_base_url(base_url).unwrap_or_else(|| base_url.to_string());
    let mut guard = OLLAMA_ENDPOINT_SATURATED_UNTIL.lock().await;
    guard.insert(key, Instant::now() + backoff);
    backoff
}

async fn should_log_ollama_endpoint_cooldown_skip(base_url: &str) -> bool {
    let interval = ollama_cooldown_skip_log_interval();
    if interval.is_zero() {
        return true;
    }
    let key = normalize_base_url(base_url).unwrap_or_else(|| base_url.to_string());
    let now = Instant::now();
    let mut guard = OLLAMA_ENDPOINT_SKIP_LOGGED_AT.lock().await;
    guard.retain(|_, logged_at| now.saturating_duration_since(*logged_at) <= interval);
    if let Some(last_logged_at) = guard.get(&key)
        && now.saturating_duration_since(*last_logged_at) < interval
    {
        return false;
    }
    guard.insert(key, now);
    true
}

fn ollama_saturated_message(remaining: Duration) -> String {
    format!(
        "local Ollama request queue is saturated; backing off before retrying in {}s",
        remaining.as_secs().max(1)
    )
}

fn ollama_saturated_error(remaining: Duration) -> GailError {
    GailError::upstream(
        "ollama",
        Some(StatusCode::SERVICE_UNAVAILABLE),
        ollama_saturated_message(remaining),
    )
}

fn message_indicates_ollama_saturation(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("local ollama request queue is saturated")
        || lowered.contains("local model service is saturated")
        || lowered.contains("adaptive endpoint budget exhausted")
}

fn message_indicates_ollama_endpoint_transport_failure(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("error sending request for url")
        || lowered.contains("connection refused")
        || lowered.contains("connection reset")
        || lowered.contains("connection closed")
        || lowered.contains("dns error")
        || lowered.contains("failed to lookup address information")
}

fn normalize_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "none" | "null" | "nil" | "undefined"
        )
    {
        return None;
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    Some(with_scheme.trim_end_matches('/').to_string())
}

fn push_base_url_candidate(candidates: &mut Vec<String>, value: Option<&str>) {
    let Some(normalized) = value.and_then(normalize_base_url) else {
        return;
    };
    if !candidates
        .iter()
        .any(|existing| existing.eq_ignore_ascii_case(&normalized))
    {
        candidates.push(normalized);
    }
}

fn endpoint_scheme_variants(base_url: &str) -> Vec<String> {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return Vec::new();
    };
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    if url.scheme() != "http" || is_local_ollama_host(host.as_str()) {
        return Vec::new();
    }
    let mut https_url = url;
    if https_url.set_scheme("https").is_ok() {
        vec![https_url.to_string().trim_end_matches('/').to_string()]
    } else {
        Vec::new()
    }
}

fn base_url_uses_local_ollama_host(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_ascii_lowercase()))
        .map(|host| is_local_ollama_host(host.as_str()))
        .unwrap_or(false)
}

fn base_url_uses_cluster_service_host(base_url: &str) -> bool {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_ascii_lowercase()))
        .map(|host| {
            host.ends_with(".svc") || host.ends_with(".svc.cluster.local") || host.contains(".svc.")
        })
        .unwrap_or(false)
}

fn is_local_ollama_host(host: &str) -> bool {
    host == "localhost"
        || host == "127.0.0.1"
        || host == "::1"
        || host.ends_with(".local")
        || host.ends_with(".svc")
        || host.ends_with(".svc.cluster.local")
        || host.contains(".svc.")
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

async fn fetch_ollama_tags(client: &Client, base_url: &str, timeout_seconds: u64) -> Result<Value> {
    let url = format!("{base_url}/api/tags");
    let response = match client
        .get(&url)
        .timeout(Duration::from_secs(timeout_seconds.max(1)))
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
    use crate::models::{ChatMessage, MessageContent};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    #[test]
    fn model_size_parser_extracts_billion_suffix() {
        assert_eq!(model_size_billions("qwen2.5-coder:0.5b"), Some(0.5));
        assert_eq!(model_size_billions("llama3.2:3b"), Some(3.0));
        assert_eq!(model_size_billions("mistral"), None);
    }

    #[test]
    fn model_floor_keeps_unknown_size_aliases_eligible() {
        assert!(model_meets_size_floor("llama3.2:3b", 1.5));
        assert!(!model_meets_size_floor("llama3.2:1b", 1.5));
        assert!(model_meets_size_floor("mistral", 1.5));
    }

    #[test]
    fn non_generative_model_detection_matches_embedding_families() {
        assert!(model_looks_non_generative("nomic-embed-text"));
        assert!(model_looks_non_generative("bge-rerank-v2"));
        assert!(!model_looks_non_generative("qwen2.5-coder:1.5b"));
    }

    #[test]
    fn next_size_up_pull_candidate_respects_floor() {
        assert_eq!(
            next_size_up_pull_candidate("qwen2.5-coder:0.5b", 1.5).as_deref(),
            Some("qwen2.5-coder:1.5b")
        );
    }

    #[test]
    fn public_http_ollama_endpoint_adds_https_variant() {
        assert_eq!(
            endpoint_scheme_variants("http://ollama.neuralmimicry.ai"),
            vec!["https://ollama.neuralmimicry.ai".to_string()]
        );
        assert!(
            endpoint_scheme_variants("http://ollama.ollama.svc.cluster.local:11434").is_empty()
        );
    }

    #[test]
    fn endpoint_budget_prefers_primary_for_short_overall_timeout() {
        let budget = compute_ollama_endpoint_budget(Duration::from_secs(18), 3);
        assert_eq!(budget, Duration::from_secs(12));
    }

    #[test]
    fn endpoint_budget_uses_remaining_when_only_one_endpoint_left() {
        let budget = compute_ollama_endpoint_budget(Duration::from_secs(11), 1);
        assert_eq!(budget, Duration::from_secs(11));
    }

    #[test]
    fn ollama_saturated_error_uses_service_unavailable_status() {
        let error = ollama_saturated_error(Duration::from_secs(7));
        match error {
            crate::errors::GailError::Upstream { status, .. } => {
                assert_eq!(status, Some(StatusCode::SERVICE_UNAVAILABLE))
            }
            other => panic!("expected upstream error, got: {other:?}"),
        }
    }

    #[test]
    fn timeout_messages_are_not_treated_as_ollama_saturation_or_transport_failure() {
        let timeout = "operation timed out while contacting Ollama";
        assert!(!message_indicates_ollama_saturation(timeout));
        assert!(!message_indicates_ollama_endpoint_transport_failure(
            timeout
        ));
    }

    #[test]
    fn connection_failures_still_trigger_endpoint_transport_cooldown() {
        let failure =
            "error sending request for url (http://ollama/api/generate): connection reset by peer";
        assert!(message_indicates_ollama_endpoint_transport_failure(failure));
    }

    #[test]
    fn select_best_local_model_excludes_embedding_only_candidates() {
        let models = vec![
            json!({"name": "nomic-embed-text:latest"}),
            json!({"name": "llama3.2:3b"}),
        ];
        let resources = LocalResourceSnapshot {
            memory_available_mb: 1_000_000.0,
            disk_available_mb: 1_000_000.0,
        };
        let selected = select_best_local_model(
            &models,
            "qwen2.5-coder:7b",
            "Reply with exactly: OK",
            0.0,
            &resources,
        );
        assert_eq!(selected.as_deref(), Some("llama3.2:3b"));
    }

    #[tokio::test]
    async fn completion_fails_over_from_request_endpoint_to_provider_endpoint() {
        let bad = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({
                "error": "bad gateway"
            })))
            .mount(&bad)
            .await;

        let good = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&good)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "local answer",
                "prompt_eval_count": 2,
                "eval_count": 3
            })))
            .mount(&good)
            .await;

        let provider = OllamaProvider {
            client: Client::new(),
            model: "llama3.2".to_string(),
            default_model: "llama3.2".to_string(),
            base_url: good.uri(),
        };
        let request = ProviderCompletionRequest {
            provider: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            api_key: None,
            access_token: None,
            base_url: Some(bad.uri()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            }],
            system: None,
            max_tokens: Some(8),
            temperature: Some(0.0),
            timeout_seconds: Some(4),
            reasoning_effort: None,
            request_category: None,
            workflow: None,
            role: None,
            min_model_size_b: None,
            strict_no_downgrade: None,
        };

        let response = provider
            .complete(&request)
            .await
            .expect("fallback response");
        assert_eq!(response.text, "local answer");
        let raw = response.raw.expect("raw metadata");
        assert_eq!(raw["gail_ollama_base_url"], good.uri());
        assert_eq!(raw["gail_ollama_endpoint_failover"], true);
    }

    #[tokio::test]
    async fn completion_skips_embedding_only_inventory_for_generate_requests() {
        let endpoint = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "nomic-embed-text:latest"}]
            })))
            .mount(&endpoint)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "should not be used"
            })))
            .expect(0)
            .mount(&endpoint)
            .await;

        let provider = OllamaProvider {
            client: Client::new(),
            model: "qwen2.5-coder:7b".to_string(),
            default_model: "qwen2.5-coder:7b".to_string(),
            base_url: endpoint.uri(),
        };
        let request = ProviderCompletionRequest {
            provider: "ollama".to_string(),
            model: Some("qwen2.5-coder:7b".to_string()),
            api_key: None,
            access_token: None,
            base_url: Some(endpoint.uri()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            }],
            system: None,
            max_tokens: Some(8),
            temperature: Some(0.0),
            timeout_seconds: Some(4),
            reasoning_effort: None,
            request_category: None,
            workflow: None,
            role: None,
            min_model_size_b: Some(1.5),
            strict_no_downgrade: Some(true),
        };

        let error = provider
            .complete_once(&request, 0)
            .await
            .expect_err("should fail");
        assert!(
            error
                .to_string()
                .contains("is not safely available locally")
        );
    }
}
