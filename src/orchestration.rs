use std::{collections::HashSet, env, sync::Arc, time::Duration};

use axum::http::{HeaderMap, header::AUTHORIZATION};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::{
    task::JoinSet,
    time::{Instant, sleep},
};
use tracing::info;
use uuid::Uuid;

use crate::{
    aarnn_bridge::{AarnnMirrorClient, AarnnMirrorExchange},
    aer,
    config::{ApiTokenConfig, GailConfig, ProviderProfile},
    errors::{GailError, Result},
    metrics::{HealthBucket, MetricsStore},
    models::{
        AarnnMirrorDirection, AerDecodeRequest, AerDecodeResponse, AerEncodeRequest,
        AerEncodeResponse, AuthContext, CandidateInvocationSummary, CandidateSummary,
        CompletionRequest, CompletionResponse, CompletionTrace, HealthResponse,
        NeuromorphicAnalyzeRequest, NeuromorphicPredictRequest, NeuromorphicPredictResponse,
        ProviderCompletionRequest, SelectionMode, SpecialistAnalysisResponse,
        TranscriptionResponse,
    },
    providers::{
        ProviderHealth, ProviderInvocationResponse, TranscriptionInput, build_adapter,
        normalize_provider_type, provider_request_from_profile,
    },
    routing::{default_routing_profiles, resolve_routing_profiles_path},
    specialists::{
        SpecialistEngine, analyze_specialist_engines, build_specialist_engines,
        specialist_engine_summaries,
    },
    trading::TradingBridge,
};

const PROVIDER_HEALTH_TIMEOUT_SECONDS: u64 = 4;

#[derive(Clone)]
pub struct GailService {
    inner: Arc<GailServiceInner>,
}

#[derive(Clone)]
struct GailServiceInner {
    config: GailConfig,
    client: Client,
    metrics: MetricsStore,
    specialists: Vec<SpecialistEngine>,
    aarnn_bridge: Option<AarnnMirrorClient>,
    trading_bridge: Option<TradingBridge>,
}

#[derive(Clone, Debug)]
struct ProviderCandidate {
    profile: ProviderProfile,
    source: String,
    provider_type: String,
    configured_model: String,
    preferred: bool,
    weight: f64,
    specialties: HashSet<String>,
    roles: HashSet<String>,
}

#[derive(Debug)]
struct InvocationResult {
    candidate: ProviderCandidate,
    response: Option<ProviderInvocationResponse>,
    error: Option<String>,
    latency_ms: Option<u64>,
    quality: f64,
    score: f64,
}

impl GailService {
    pub async fn new(config: GailConfig) -> Result<Self> {
        let client = Client::builder()
            .use_rustls_tls()
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Duration::from_secs(30))
            .user_agent(format!("gail/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        let metrics = MetricsStore::new(config.storage.metrics_path.clone()).await?;
        let specialists = build_specialist_engines(&config, client.clone());
        let aarnn_bridge = AarnnMirrorClient::from_config(&config, client.clone(), &specialists);

        // Construct a preliminary service (without trading) to pass into the trading bridge.
        let preliminary = Self {
            inner: Arc::new(GailServiceInner {
                config: config.clone(),
                client: client.clone(),
                metrics: metrics.clone(),
                specialists: specialists.clone(),
                aarnn_bridge: aarnn_bridge.clone(),
                trading_bridge: None,
            }),
        };

        // Start trading bridge if configured.
        let trading_bridge = if config.trading.is_viable() {
            tracing::info!("trading: bridge is enabled — starting background loop");
            let trading_cfg = config.trading.clone();
            let (bridge, _handle) = TradingBridge::start(trading_cfg, preliminary).await;
            // Note: _handle is dropped here; the background task keeps running because
            // tokio::spawn holds it. The bridge background task will only stop when
            // GailService is dropped (shutdown). For a clean shutdown we would store
            // _handle, but for now the task runs for the lifetime of the process.
            Some(bridge)
        } else {
            None
        };

        Ok(Self {
            inner: Arc::new(GailServiceInner {
                config,
                client,
                metrics,
                specialists,
                aarnn_bridge,
                trading_bridge,
            }),
        })
    }

    pub fn config(&self) -> &GailConfig {
        &self.inner.config
    }

    fn aarnn_bridge(&self) -> Option<&AarnnMirrorClient> {
        self.inner.aarnn_bridge.as_ref()
    }

    pub fn trading_bridge(&self) -> Option<&TradingBridge> {
        self.inner.trading_bridge.as_ref()
    }

    pub fn authorize(&self, headers: &HeaderMap, required_scope: &str) -> Result<AuthContext> {
        let Some(token_config) = self.matching_token(headers, required_scope) else {
            return Err(GailError::unauthorized());
        };
        Ok(AuthContext {
            client_id: Some(token_config.client_id.clone()),
        })
    }

    pub fn can_access_health_unauthenticated(&self) -> bool {
        self.inner.config.security.allow_unauthenticated_health
    }

    pub async fn health(&self) -> HealthResponse {
        HealthResponse {
            ok: true,
            service: "gail".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub async fn direct_complete(
        &self,
        request: ProviderCompletionRequest,
    ) -> Result<CompletionResponse> {
        let request_id = Uuid::new_v4().to_string();
        let prompt_text = flatten_prompt_text(&request.messages, request.system.as_deref());
        let expected_json = expected_json(&request.messages, request.system.as_deref());
        let mut profile = ProviderProfile::default();
        profile.name = request.provider.clone();
        profile.provider_type = request.provider.clone();
        profile.model = request.model.clone();
        profile.api_key = request.api_key.clone();
        profile.access_token = request.access_token.clone();
        profile.base_url = request.base_url.clone();
        profile.source = Some("request_direct".to_string());
        let candidate = ProviderCandidate::from_profile(profile.clone());
        let mirror_input = self.spawn_aarnn_mirror(self.build_aarnn_exchange(
            request_id.as_str(),
            request_id.as_str(),
            "direct",
            "assistant",
            AarnnMirrorDirection::Input,
            Some(request.provider.as_str()),
            request.model.as_deref(),
            request.request_category.as_deref(),
            request.system.as_deref(),
            None,
            prompt_text.as_str(),
            &request.messages,
        ));
        let adapter = build_adapter(self.inner.client.clone(), &profile)?;
        let response = adapter.complete(&request).await?;
        let quality = quality_score(response.text.as_str(), expected_json);
        let mirror_output = self
            .run_aarnn_output_mirror(
                request_id.as_str(),
                request_id.as_str(),
                "direct",
                "assistant",
                Some(response.provider.as_str()),
                Some(response.model.as_str()),
                request.request_category.as_deref(),
                request.system.as_deref(),
                Some(prompt_text.as_str()),
                response.text.as_str(),
                &request.messages,
            )
            .await;
        let mirror_input = self.await_aarnn_mirror_task(mirror_input).await;
        let mut text = response.text.clone();
        let mut provider = response.provider.clone();
        let mut model = response.model.clone();
        let mut latency_ms = response.latency_ms;
        let mut usage = response.usage.clone();
        let mut raw = response.raw.clone();
        let mut final_source = "llm".to_string();
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref()) {
            if bridge.should_promote_candidate(output_trace, response.text.as_str()) {
                if let Some(reply_text) = bridge.promoted_reply(output_trace) {
                    text = reply_text;
                    provider = "aarnn".to_string();
                    model = bridge.response_model().to_string();
                    latency_ms = latency_ms.saturating_add(output_trace.latency_ms);
                    usage = None;
                    raw = Some(json!({
                        "selected_source": "aarnn",
                        "aarnn_candidate": output_trace.candidate.clone(),
                        "llm_provider": response.provider,
                        "llm_model": response.model,
                        "llm_raw": response.raw,
                    }));
                    final_source = "aarnn".to_string();
                }
            }
        }
        let trace = if mirror_input.is_some() || mirror_output.is_some() {
            Some(CompletionTrace {
                workflow: "direct".to_string(),
                role: "assistant".to_string(),
                task_tags: vec!["direct".to_string()],
                selection_mode: SelectionMode::Fastest,
                returned_early: false,
                early_success_enabled: false,
                early_success_settle_seconds: 0.0,
                selected: candidate.summary(Some(response.model.as_str())),
                candidates: vec![CandidateInvocationSummary {
                    summary: candidate.summary(Some(response.model.as_str())),
                    latency_ms: Some(response.latency_ms),
                    quality,
                    score: quality,
                    status: "ok".to_string(),
                    error: None,
                }],
                metrics_store_path: self.inner.metrics.path(),
                specialist_engines: None,
                final_source,
                final_provider: provider.clone(),
                final_model: model.clone(),
                aarnn_mirroring: self
                    .aarnn_bridge()
                    .map(|bridge| bridge.build_trace(mirror_input.clone(), mirror_output.clone())),
            })
        } else {
            None
        };
        Ok(CompletionResponse {
            request_id,
            text,
            provider,
            model,
            latency_ms,
            usage,
            trace,
            raw,
        })
    }

    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let request_id = Uuid::new_v4().to_string();
        let workflow = normalize_key(request.workflow.as_deref().unwrap_or("general"), "general");
        let role = normalize_key(request.role.as_deref().unwrap_or("general"), "general");
        let selection_mode = request
            .selection_mode
            .clone()
            .unwrap_or_else(|| self.selection_mode());
        let include_configured = request
            .include_configured
            .unwrap_or_else(|| self.include_configured_candidates());
        let max_candidates = request
            .max_candidates
            .unwrap_or_else(|| self.max_parallel_candidates());
        let timeout_cap = self.candidate_timeout_cap(&workflow, &role);
        let early_success_enabled = self.early_success_enabled(&workflow, &role, &selection_mode);
        let early_success_settle_seconds =
            self.early_success_settle_seconds(&workflow, &role, &selection_mode);
        let early_success_min_quality = self.early_success_min_quality();

        let mut provider_request = ProviderCompletionRequest {
            provider: request
                .preferred_provider
                .clone()
                .unwrap_or_else(|| "openai".to_string()),
            model: request.preferred_model.clone(),
            api_key: request.preferred_api_key.clone(),
            access_token: request.preferred_access_token.clone(),
            base_url: request.base_url.clone(),
            messages: request.messages.clone(),
            system: request.system.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            timeout_seconds: request.timeout_seconds,
            reasoning_effort: request.reasoning_effort.clone(),
            request_category: request.request_category.clone(),
        };

        let prompt_text = flatten_prompt_text(
            &provider_request.messages,
            provider_request.system.as_deref(),
        );
        let mut task_tags = workflow_tags(&workflow, &role, &prompt_text);
        if let Some(category) = request
            .request_category
            .as_deref()
            .or(provider_request.request_category.as_deref())
        {
            for tag in category.split(|ch: char| !ch.is_ascii_alphanumeric()) {
                let normalized = normalize_key(tag, "");
                if !normalized.is_empty() {
                    task_tags.insert(normalized);
                }
            }
        }
        let mut specialist_meta = None;
        if !self.inner.specialists.is_empty()
            && (task_tags.contains("neuromorphic") || self.always_route_specialists())
        {
            let analyze_request = NeuromorphicAnalyzeRequest {
                text: prompt_text.clone(),
                workflow: Some(workflow.clone()),
                role: Some(role.clone()),
            };
            let analysis =
                analyze_specialist_engines(&self.inner.specialists, &analyze_request).await;
            if analysis.relevant {
                task_tags.insert("neuromorphic".to_string());
                task_tags.insert("aer".to_string());
                task_tags.extend(analysis.combined_specialties.iter().cloned());
                if !analysis.context.is_empty() {
                    provider_request.system = Some(match provider_request.system {
                        Some(system) if !system.trim().is_empty() => {
                            format!("{system}\n\n{}", analysis.context)
                        }
                        _ => analysis.context.clone(),
                    });
                }
                info!(
                    workflow = %workflow,
                    role = %role,
                    engine_count = analysis.engine_count,
                    "attached neuromorphic specialist context"
                );
            }
            specialist_meta = Some(analysis);
        }
        let mirrored_prompt_text = flatten_prompt_text(
            &provider_request.messages,
            provider_request.system.as_deref(),
        );
        let mirror_input = self.spawn_aarnn_mirror(self.build_aarnn_exchange(
            request_id.as_str(),
            request_id.as_str(),
            workflow.as_str(),
            role.as_str(),
            AarnnMirrorDirection::Input,
            Some(provider_request.provider.as_str()),
            provider_request.model.as_deref(),
            provider_request.request_category.as_deref(),
            provider_request.system.as_deref(),
            None,
            mirrored_prompt_text.as_str(),
            &provider_request.messages,
        ));

        let mut candidates = self.build_candidates(&request, include_configured);
        if candidates.is_empty() {
            return Err(GailError::bad_request(
                "no LLM providers are configured or supplied for orchestration",
            ));
        }
        let mut ranked = Vec::new();
        for candidate in candidates.drain(..) {
            let score = self
                .rank_candidate(&candidate, &workflow, &role, &task_tags)
                .await;
            ranked.push((score, candidate));
        }
        ranked.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected = ranked
            .into_iter()
            .take(max_candidates.max(1))
            .map(|(_, candidate)| candidate)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Err(GailError::bad_request(
                "no provider candidates were selected",
            ));
        }

        info!(
            workflow = %workflow,
            role = %role,
            candidates = %preview_labels(selected.iter().map(|item| item.label(None)).collect::<Vec<_>>(), 6),
            tags = %preview_labels(task_tags.iter().cloned().collect::<Vec<_>>(), 8),
            "dispatching Gail orchestration"
        );

        let expected_json = expected_json(
            &provider_request.messages,
            provider_request.system.as_deref(),
        );
        let mut results = if selected.len() == 1 {
            vec![
                self.invoke_candidate(
                    selected[0].clone(),
                    provider_request.clone(),
                    expected_json,
                    timeout_cap,
                )
                .await,
            ]
        } else {
            self.invoke_candidates(
                selected.clone(),
                provider_request.clone(),
                expected_json,
                selection_mode.clone(),
                early_success_enabled,
                early_success_settle_seconds,
                early_success_min_quality,
                timeout_cap,
            )
            .await?
        };

        let returned_early = results.len() < selected.len() && selected.len() > 1;
        let mut successful = Vec::new();
        let mut failures = Vec::new();
        for result in results.iter_mut() {
            let candidate_summary = result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str()));
            if let Some(response) = result.response.as_ref() {
                let latency_penalty = response.latency_ms as f64 / 5000.0;
                let metrics_bonus = self
                    .inner
                    .metrics
                    .score_bonus(candidate_summary.candidate_id.as_str(), &workflow, &role)
                    .await;
                result.score = result.quality - latency_penalty.min(1.25) + metrics_bonus;
                self.inner
                    .metrics
                    .record_result(
                        &candidate_summary,
                        &workflow,
                        &role,
                        true,
                        Some(response.latency_ms),
                        result.quality,
                        None,
                    )
                    .await?;
                successful.push(candidate_summary);
            } else {
                self.inner
                    .metrics
                    .record_result(
                        &candidate_summary,
                        &workflow,
                        &role,
                        false,
                        result.latency_ms,
                        -1.0,
                        result.error.as_deref(),
                    )
                    .await?;
                failures.push(
                    result
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown error".to_string()),
                );
            }
        }

        let Some(chosen_index) = results
            .iter()
            .enumerate()
            .filter(|(_, result)| result.response.is_some())
            .max_by(|(_, left), (_, right)| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
        else {
            let message = failures
                .last()
                .cloned()
                .unwrap_or_else(|| "LLM orchestration returned no responses".to_string());
            return Err(GailError::upstream("gail", None, message));
        };

        let chosen = results.swap_remove(chosen_index);
        let chosen_response = chosen.response.expect("chosen successful result");
        let selected_summary = chosen
            .candidate
            .summary(Some(chosen_response.model.as_str()));
        let mirror_output = self
            .run_aarnn_output_mirror(
                request_id.as_str(),
                request_id.as_str(),
                workflow.as_str(),
                role.as_str(),
                Some(chosen_response.provider.as_str()),
                Some(chosen_response.model.as_str()),
                provider_request.request_category.as_deref(),
                provider_request.system.as_deref(),
                Some(mirrored_prompt_text.as_str()),
                chosen_response.text.as_str(),
                &provider_request.messages,
            )
            .await;
        let mirror_input = self.await_aarnn_mirror_task(mirror_input).await;
        let candidate_summaries = std::iter::once((
            selected_summary.clone(),
            chosen.latency_ms,
            chosen.quality,
            chosen.score,
            chosen.error.clone(),
            true,
        ))
        .chain(results.into_iter().map(|result| {
            let summary = result
                .candidate
                .summary(result.response.as_ref().map(|value| value.model.as_str()));
            (
                summary,
                result.latency_ms,
                result.quality,
                result.score,
                result.error,
                result.response.is_some(),
            )
        }))
        .map(
            |(summary, latency_ms, quality, score, error, ok)| CandidateInvocationSummary {
                summary,
                latency_ms,
                quality,
                score,
                status: if ok { "ok" } else { "error" }.to_string(),
                error,
            },
        )
        .collect::<Vec<_>>();

        info!(
            workflow = %workflow,
            role = %role,
            provider = %chosen_response.provider,
            model = %chosen_response.model,
            returned_early,
            "selected Gail orchestration result"
        );

        let mut text = chosen_response.text.clone();
        let mut provider = chosen_response.provider.clone();
        let mut model = chosen_response.model.clone();
        let mut latency_ms = chosen_response.latency_ms;
        let mut usage = chosen_response.usage.clone();
        let mut raw = chosen_response.raw.clone();
        let mut final_source = "llm".to_string();
        if let (Some(bridge), Some(output_trace)) = (self.aarnn_bridge(), mirror_output.as_ref()) {
            if bridge.should_promote_candidate(output_trace, chosen_response.text.as_str()) {
                if let Some(reply_text) = bridge.promoted_reply(output_trace) {
                    text = reply_text;
                    provider = "aarnn".to_string();
                    model = bridge.response_model().to_string();
                    latency_ms = latency_ms.saturating_add(output_trace.latency_ms);
                    usage = None;
                    raw = Some(json!({
                        "selected_source": "aarnn",
                        "aarnn_candidate": output_trace.candidate.clone(),
                        "llm_provider": chosen_response.provider,
                        "llm_model": chosen_response.model,
                        "llm_raw": chosen_response.raw,
                    }));
                    final_source = "aarnn".to_string();
                }
            }
        }

        let trace = CompletionTrace {
            workflow: workflow.clone(),
            role: role.clone(),
            task_tags: sorted_strings(task_tags),
            selection_mode: selection_mode.clone(),
            returned_early,
            early_success_enabled,
            early_success_settle_seconds,
            selected: selected_summary,
            candidates: candidate_summaries,
            metrics_store_path: self.inner.metrics.path(),
            specialist_engines: specialist_meta
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            final_source,
            final_provider: provider.clone(),
            final_model: model.clone(),
            aarnn_mirroring: self
                .aarnn_bridge()
                .map(|bridge| bridge.build_trace(mirror_input.clone(), mirror_output.clone())),
        };

        Ok(CompletionResponse {
            request_id,
            text,
            provider,
            model,
            latency_ms,
            usage,
            trace: Some(trace),
            raw,
        })
    }

    pub async fn transcribe(
        &self,
        provider: String,
        model: Option<String>,
        api_key: Option<String>,
        access_token: Option<String>,
        base_url: Option<String>,
        input: TranscriptionInput,
    ) -> Result<TranscriptionResponse> {
        let profile = ProviderProfile {
            name: provider.clone(),
            provider_type: provider,
            model,
            api_key,
            access_token,
            base_url,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: 0.0,
            preferred: true,
            source: Some("request_transcribe".to_string()),
        };
        let adapter = build_adapter(self.inner.client.clone(), &profile)?;
        let response = adapter.transcribe(&input).await?;
        Ok(TranscriptionResponse {
            request_id: Uuid::new_v4().to_string(),
            text: response.text,
            provider: response.provider,
            model: response.model,
            latency_ms: response.latency_ms,
            usage: response.usage,
        })
    }

    pub async fn analyze_neuromorphic(
        &self,
        request: NeuromorphicAnalyzeRequest,
    ) -> Result<SpecialistAnalysisResponse> {
        Ok(analyze_specialist_engines(&self.inner.specialists, &request).await)
    }

    pub async fn predict_neuromorphic(
        &self,
        request: NeuromorphicPredictRequest,
    ) -> Result<NeuromorphicPredictResponse> {
        let engine = self.select_specialist(request.engine_name.as_deref())?;
        engine.predict_request(&request).await
    }

    pub fn encode_aer(&self, request: AerEncodeRequest) -> Result<AerEncodeResponse> {
        let ts_us = request.ts_us.unwrap_or(0);
        let events = if let Some(events) = request.events {
            events
        } else {
            aer::spikes_to_events(
                ts_us,
                request.base_addr,
                &request.spikes.unwrap_or_default(),
            )
        };
        let payload = aer::encode_events(&events);
        Ok(AerEncodeResponse {
            payload_hex: aer::payload_hex(&payload),
            event_count: events.len(),
        })
    }

    pub fn decode_aer(&self, request: AerDecodeRequest) -> Result<AerDecodeResponse> {
        let payload = hex::decode(request.payload_hex)
            .map_err(|error| GailError::bad_request(error.to_string()))?;
        let events = aer::decode_events(&payload)?;
        let spikes = match (request.base_addr, request.length) {
            (Some(base_addr), Some(length)) => aer::decode_spikes(&payload, base_addr, length)?,
            (Some(base_addr), None) => aer::decode_spikes_auto(&payload, base_addr)?,
            (None, Some(length)) => {
                let base_addr = events.first().map(|event| event.addr).unwrap_or_default();
                aer::decode_spikes(&payload, base_addr, length)?
            }
            (None, None) => {
                let base_addr = events.first().map(|event| event.addr).unwrap_or_default();
                aer::decode_spikes_auto(&payload, base_addr)?
            }
        };
        Ok(AerDecodeResponse { events, spikes })
    }

    pub async fn orchestration_status_value(
        &self,
        candidate_limit: usize,
        probe_engines: bool,
        probe_providers: bool,
    ) -> Value {
        let providers = self.provider_summaries(probe_providers).await;
        let engines = specialist_engine_summaries(
            &self.inner.config,
            self.inner.client.clone(),
            probe_engines,
        )
        .await;
        let metrics = self.inner.metrics.summary(candidate_limit.max(1)).await;
        let model_inventory = self.first_ollama_inventory().await;
        let routing_profiles_path = resolve_routing_profiles_path(None::<&std::path::Path>)
            .ok()
            .map(|path| path.display().to_string());
        let routing_profiles_version = default_routing_profiles().version;
        let aarnn_bridge = AarnnMirrorClient::status(&self.inner.config, &self.inner.specialists);
        json!({
            "enabled": self.inner.config.orchestration.enabled,
            "routing_profiles_path": routing_profiles_path,
            "routing_profiles_version": routing_profiles_version,
            "selection_mode": self.selection_mode(),
            "max_parallel_candidates": self.max_parallel_candidates(),
            "health_ttl_seconds": self.health_ttl_seconds(),
            "provider_count": providers.len(),
            "providers": providers,
            "engine_count": engines.len(),
            "engines": engines,
            "aarnn_bridge": aarnn_bridge,
            "metrics": metrics,
            "model_inventory": model_inventory,
        })
    }

    async fn provider_summaries(&self, probe_health: bool) -> Vec<Value> {
        let mut providers = Vec::new();
        for profile in &self.inner.config.providers {
            let provider_type = normalize_provider_type(profile.provider_type.as_str());
            let health = if probe_health {
                match build_adapter(self.inner.client.clone(), profile) {
                    Ok(adapter) => {
                        match adapter.health(Some(PROVIDER_HEALTH_TIMEOUT_SECONDS)).await {
                            Ok(status) => json!(status),
                            Err(error) => json!({"ok": false, "message": error.to_string()}),
                        }
                    }
                    Err(error) => json!({"ok": false, "message": error.to_string()}),
                }
            } else {
                Value::Null
            };
            providers.push(json!({
                "name": profile.name,
                "provider": provider_type,
                "model": profile.model,
                "source": profile.source,
                "roles": profile.roles,
                "specialties": profile.specialties,
                "weight": profile.weight,
                "preferred": profile.preferred,
                "base_url": profile.base_url,
                "health": health,
            }));
        }
        providers
    }

    async fn first_ollama_inventory(&self) -> Value {
        for profile in &self.inner.config.providers {
            if normalize_provider_type(profile.provider_type.as_str()) != "ollama" {
                continue;
            }
            if let Ok(adapter) = build_adapter(self.inner.client.clone(), profile) {
                if let Some(inventory) = adapter.ollama_inventory(&self.inner.config).await {
                    return serde_json::to_value(inventory).unwrap_or(Value::Null);
                }
            }
        }
        Value::Null
    }

    fn spawn_aarnn_mirror(
        &self,
        exchange: AarnnMirrorExchange,
    ) -> Option<tokio::task::JoinHandle<crate::models::AarnnMirrorInvocationTrace>> {
        let bridge = self.inner.aarnn_bridge.clone()?;
        let should_mirror = match exchange.direction {
            AarnnMirrorDirection::Input => bridge.should_mirror_input(),
            AarnnMirrorDirection::Output => bridge.should_mirror_output(),
        };
        if !should_mirror {
            return None;
        }
        Some(tokio::spawn(async move { bridge.mirror(exchange).await }))
    }

    async fn await_aarnn_mirror_task(
        &self,
        task: Option<tokio::task::JoinHandle<crate::models::AarnnMirrorInvocationTrace>>,
    ) -> Option<crate::models::AarnnMirrorInvocationTrace> {
        let task = task?;
        match task.await {
            Ok(trace) => Some(trace),
            Err(error) => {
                tracing::warn!(error = %error, "AARNN mirror task join failed");
                None
            }
        }
    }

    async fn run_aarnn_output_mirror(
        &self,
        request_id: &str,
        conversation_id: &str,
        workflow: &str,
        role: &str,
        provider: Option<&str>,
        model: Option<&str>,
        request_category: Option<&str>,
        system: Option<&str>,
        prompt_text: Option<&str>,
        text: &str,
        messages: &[crate::models::ChatMessage],
    ) -> Option<crate::models::AarnnMirrorInvocationTrace> {
        let bridge = self.aarnn_bridge()?;
        if !bridge.should_mirror_output() {
            return None;
        }
        Some(
            bridge
                .mirror(self.build_aarnn_exchange(
                    request_id,
                    conversation_id,
                    workflow,
                    role,
                    AarnnMirrorDirection::Output,
                    provider,
                    model,
                    request_category,
                    system,
                    prompt_text,
                    text,
                    messages,
                ))
                .await,
        )
    }

    fn build_aarnn_exchange(
        &self,
        request_id: &str,
        conversation_id: &str,
        workflow: &str,
        role: &str,
        direction: AarnnMirrorDirection,
        provider: Option<&str>,
        model: Option<&str>,
        request_category: Option<&str>,
        system: Option<&str>,
        prompt_text: Option<&str>,
        text: &str,
        messages: &[crate::models::ChatMessage],
    ) -> AarnnMirrorExchange {
        AarnnMirrorExchange {
            request_id: request_id.to_string(),
            conversation_id: conversation_id.to_string(),
            workflow: workflow.to_string(),
            role: role.to_string(),
            direction,
            provider: provider.map(ToOwned::to_owned),
            model: model.map(ToOwned::to_owned),
            request_category: request_category.map(ToOwned::to_owned),
            system: system.map(ToOwned::to_owned),
            prompt_text: prompt_text.map(ToOwned::to_owned),
            text: text.to_string(),
            message_roles: messages
                .iter()
                .map(|message| message.role.clone())
                .collect(),
        }
    }

    fn matching_token<'a>(
        &'a self,
        headers: &HeaderMap,
        required_scope: &str,
    ) -> Option<&'a ApiTokenConfig> {
        let header = headers.get(AUTHORIZATION)?.to_str().ok()?;
        let token = header.strip_prefix("Bearer ")?.trim();
        if token.is_empty() {
            return None;
        }
        self.inner.config.security.api_tokens.iter().find(|config| {
            config.token == token
                && (config.scopes.is_empty()
                    || config
                        .scopes
                        .iter()
                        .any(|scope| scope == "*" || scope.eq_ignore_ascii_case(required_scope)))
        })
    }

    fn build_candidates(
        &self,
        request: &CompletionRequest,
        include_configured: bool,
    ) -> Vec<ProviderCandidate> {
        let mut candidates = Vec::new();
        if let Some(provider) = request.preferred_provider.as_ref() {
            candidates.push(self.request_candidate(
                provider,
                request.preferred_model.clone(),
                request.preferred_api_key.clone(),
                request.preferred_access_token.clone(),
                request.base_url.clone(),
                true,
                "request_primary",
            ));
        }
        if let Some(provider) = request.fallback_provider.as_ref() {
            candidates.push(self.request_candidate(
                provider,
                request.fallback_model.clone(),
                request.fallback_api_key.clone(),
                request.fallback_access_token.clone(),
                request.base_url.clone(),
                false,
                "request_fallback",
            ));
        }
        if include_configured || candidates.is_empty() {
            candidates.extend(
                self.inner
                    .config
                    .providers
                    .iter()
                    .cloned()
                    .map(ProviderCandidate::from_profile),
            );
        }
        dedupe_candidates(candidates)
    }

    fn request_candidate(
        &self,
        provider: &str,
        model: Option<String>,
        api_key: Option<String>,
        access_token: Option<String>,
        base_url: Option<String>,
        preferred: bool,
        source: &str,
    ) -> ProviderCandidate {
        ProviderCandidate::from_profile(ProviderProfile {
            name: provider.trim().to_string(),
            provider_type: provider.trim().to_string(),
            model,
            api_key,
            access_token,
            base_url,
            roles: Vec::new(),
            specialties: Vec::new(),
            weight: if preferred { 0.4 } else { 0.0 },
            preferred,
            source: Some(source.to_string()),
        })
    }

    async fn rank_candidate(
        &self,
        candidate: &ProviderCandidate,
        workflow: &str,
        role: &str,
        task_tags: &HashSet<String>,
    ) -> f64 {
        let overlap = task_tags.intersection(&candidate.specialties).count() as f64;
        let role_score = if candidate.roles.is_empty() {
            0.0
        } else if candidate.roles.contains(role) {
            0.6
        } else {
            -0.9
        };
        let health = self.probe_health(candidate).await;
        let health_score = if health.ok { 0.4 } else { -1.4 };
        let preferred_score = if candidate.preferred { 0.7 } else { 0.0 };
        let metrics_bonus = self
            .inner
            .metrics
            .score_bonus(candidate.candidate_id().as_str(), workflow, role)
            .await;
        candidate.weight
            + (overlap * 0.85)
            + role_score
            + health_score
            + preferred_score
            + metrics_bonus
    }

    async fn probe_health(&self, candidate: &ProviderCandidate) -> ProviderHealth {
        if !self
            .inner
            .metrics
            .should_probe(candidate.candidate_id().as_str(), self.health_ttl_seconds())
            .await
        {
            let cached = self
                .inner
                .metrics
                .health_snapshot(candidate.candidate_id().as_str())
                .await;
            return ProviderHealth {
                ok: cached.ok.unwrap_or(false),
                status_code: None,
                latency_ms: cached.latency_ms,
                message: cached.message,
                mode: cached.mode,
            };
        }

        let health = match build_adapter(self.inner.client.clone(), &candidate.profile) {
            Ok(adapter) => adapter
                .health(Some(PROVIDER_HEALTH_TIMEOUT_SECONDS))
                .await
                .unwrap_or_else(|error| ProviderHealth {
                    ok: false,
                    status_code: None,
                    latency_ms: None,
                    message: Some(error.to_string()),
                    mode: Some("error".to_string()),
                }),
            Err(error) => ProviderHealth {
                ok: false,
                status_code: None,
                latency_ms: None,
                message: Some(error.to_string()),
                mode: Some("unconfigured".to_string()),
            },
        };
        let summary = candidate.summary(None);
        let _ = self
            .inner
            .metrics
            .record_health(
                &summary,
                HealthBucket {
                    ok: Some(health.ok),
                    mode: health.mode.clone(),
                    checked_at: None,
                    latency_ms: health.latency_ms,
                    message: health.message.clone(),
                },
            )
            .await;
        health
    }

    async fn invoke_candidates(
        &self,
        selected: Vec<ProviderCandidate>,
        provider_request: ProviderCompletionRequest,
        expected_json: bool,
        selection_mode: SelectionMode,
        early_success_enabled: bool,
        early_success_settle_seconds: f64,
        early_success_min_quality: f64,
        timeout_cap: Option<u64>,
    ) -> Result<Vec<InvocationResult>> {
        let mut join_set = JoinSet::new();
        for candidate in selected.iter().cloned() {
            let service = self.clone();
            let request = provider_request.clone();
            join_set.spawn(async move {
                service
                    .invoke_candidate(candidate, request, expected_json, timeout_cap)
                    .await
            });
        }

        let mut results = Vec::new();
        let mut early_deadline: Option<Instant> = None;
        while !join_set.is_empty() {
            let joined = if let Some(deadline) = early_deadline {
                match tokio::time::timeout_at(deadline, join_set.join_next()).await {
                    Ok(result) => result,
                    Err(_) => {
                        join_set.abort_all();
                        break;
                    }
                }
            } else {
                join_set.join_next().await
            };
            let Some(joined) = joined else {
                break;
            };
            let result = match joined {
                Ok(result) => result,
                Err(error) => InvocationResult {
                    candidate: ProviderCandidate::from_profile(ProviderProfile {
                        name: "join_error".to_string(),
                        provider_type: "join_error".to_string(),
                        source: Some("internal".to_string()),
                        ..ProviderProfile::default()
                    }),
                    response: None,
                    error: Some(error.to_string()),
                    latency_ms: None,
                    quality: -1.0,
                    score: f64::NEG_INFINITY,
                },
            };
            info!(candidate = %result.candidate.label(result.response.as_ref().map(|value| value.model.as_str())), status = if result.response.is_some() { "ok" } else { "error" }, latency_ms = ?result.latency_ms, quality = result.quality, error = ?result.error, "Gail candidate completed");
            let accepts_early =
                result.response.is_some() && result.quality >= early_success_min_quality;
            results.push(result);
            if !early_success_enabled || !accepts_early {
                continue;
            }
            if selection_mode == SelectionMode::Fastest {
                join_set.abort_all();
                break;
            }
            if early_deadline.is_none() {
                early_deadline = Some(
                    Instant::now() + Duration::from_secs_f64(early_success_settle_seconds.max(0.0)),
                );
            }
        }
        Ok(results)
    }

    async fn invoke_candidate(
        &self,
        candidate: ProviderCandidate,
        provider_request: ProviderCompletionRequest,
        expected_json: bool,
        timeout_cap: Option<u64>,
    ) -> InvocationResult {
        let quota_retries = env_int_any(&["LLM_RATE_LIMIT_RETRIES"], 2) as usize;
        let timeout_retries = env_int_any(&["LLM_TIMEOUT_RETRIES"], 1) as usize;
        let quota_backoff_base = env_float_any(&["LLM_RATE_LIMIT_BACKOFF_BASE"], 1.0).max(0.1);
        let timeout_backoff_base = env_float_any(&["LLM_TIMEOUT_BACKOFF_BASE"], 1.0).max(0.1);
        let retry_empty = env_bool_any(
            &["REFINER_AI_RETRY_EMPTY_OUTPUT", "GAIL_RETRY_EMPTY_OUTPUT"],
            true,
        );
        let mut quota_attempts = 0usize;
        let mut timeout_attempts = 0usize;
        let mut attempts = 0usize;
        loop {
            attempts += 1;
            let mut effective =
                provider_request_from_profile(&candidate.profile, &provider_request);
            if effective.timeout_seconds.is_none() {
                effective.timeout_seconds = timeout_cap;
            }
            let started = std::time::Instant::now();
            let adapter = match build_adapter(self.inner.client.clone(), &candidate.profile) {
                Ok(adapter) => adapter,
                Err(error) => {
                    return InvocationResult {
                        candidate,
                        response: None,
                        error: Some(error.to_string()),
                        latency_ms: None,
                        quality: -1.0,
                        score: f64::NEG_INFINITY,
                    };
                }
            };
            match adapter.complete(&effective).await {
                Ok(response) => {
                    let latency_ms = started.elapsed().as_millis() as u64;
                    if response.text.trim().is_empty() && retry_empty && attempts < 2 {
                        continue;
                    }
                    let quality = quality_score(&response.text, expected_json);
                    return InvocationResult {
                        candidate,
                        response: Some(response),
                        error: None,
                        latency_ms: Some(latency_ms),
                        quality,
                        score: f64::NEG_INFINITY,
                    };
                }
                Err(error) => {
                    let latency_ms = started.elapsed().as_millis() as u64;
                    if error.is_quota() && quota_attempts < quota_retries {
                        let delay = quota_backoff_base * 2_f64.powi(quota_attempts as i32);
                        quota_attempts += 1;
                        sleep(Duration::from_secs_f64(delay)).await;
                        continue;
                    }
                    if error.is_timeout() && timeout_attempts < timeout_retries {
                        let delay = timeout_backoff_base * 2_f64.powi(timeout_attempts as i32);
                        timeout_attempts += 1;
                        sleep(Duration::from_secs_f64(delay)).await;
                        continue;
                    }
                    return InvocationResult {
                        candidate,
                        response: None,
                        error: Some(error.to_string()),
                        latency_ms: Some(latency_ms),
                        quality: -1.0,
                        score: f64::NEG_INFINITY,
                    };
                }
            }
        }
    }

    fn select_specialist(&self, name: Option<&str>) -> Result<&SpecialistEngine> {
        if let Some(name) = name {
            self.inner
                .specialists
                .iter()
                .find(|engine| engine.matches_name(name))
                .ok_or_else(|| GailError::not_found(format!("unknown specialist engine: {name}")))
        } else {
            self.inner
                .specialists
                .first()
                .ok_or_else(|| GailError::not_found("no specialist engines are configured"))
        }
    }

    fn include_configured_candidates(&self) -> bool {
        env_bool_any(
            &[
                "GAIL_INCLUDE_CONFIGURED_CANDIDATES",
                "REFINER_AI_INCLUDE_CONFIGURED_CANDIDATES",
            ],
            self.inner
                .config
                .orchestration
                .include_configured_candidates,
        )
    }

    fn max_parallel_candidates(&self) -> usize {
        env_int_any(
            &[
                "GAIL_MAX_PARALLEL_CANDIDATES",
                "REFINER_AI_MAX_CONCURRENT_CANDIDATES",
            ],
            self.inner.config.orchestration.max_parallel_candidates as u64,
        ) as usize
    }

    fn health_ttl_seconds(&self) -> f64 {
        env_float_any(
            &["GAIL_HEALTH_TTL_SECONDS", "REFINER_AI_HEALTH_TTL_SECONDS"],
            self.inner.config.orchestration.health_ttl_seconds,
        )
        .max(30.0)
    }

    fn selection_mode(&self) -> SelectionMode {
        let env_value = env::var("GAIL_SELECTION_MODE")
            .ok()
            .or_else(|| env::var("REFINER_AI_SELECTION_MODE").ok())
            .unwrap_or_default();
        match env_value.trim().to_ascii_lowercase().as_str() {
            "fastest" => SelectionMode::Fastest,
            "best" => SelectionMode::Best,
            _ => self.inner.config.orchestration.selection_mode.clone(),
        }
    }

    fn early_success_enabled(
        &self,
        workflow: &str,
        role: &str,
        selection_mode: &SelectionMode,
    ) -> bool {
        if *selection_mode == SelectionMode::Fastest {
            return true;
        }
        if let Ok(value) = env::var("REFINER_AI_EARLY_SUCCESS_ENABLED") {
            return parse_bool(&value, false);
        }
        if let Ok(value) = env::var("GAIL_EARLY_SUCCESS_ENABLED") {
            return parse_bool(&value, false);
        }
        if self.inner.config.orchestration.early_success_enabled {
            return true;
        }
        is_interactive_workflow(workflow, role)
    }

    fn early_success_settle_seconds(
        &self,
        workflow: &str,
        role: &str,
        selection_mode: &SelectionMode,
    ) -> f64 {
        let default = if *selection_mode == SelectionMode::Fastest {
            0.0
        } else if is_interactive_workflow(workflow, role) {
            0.75
        } else {
            0.0
        };
        env_float_any(
            &[
                "GAIL_EARLY_SUCCESS_SETTLE_SECONDS",
                "REFINER_AI_EARLY_SUCCESS_SETTLE_SECONDS",
            ],
            if self.inner.config.orchestration.early_success_settle_seconds > 0.0 {
                self.inner.config.orchestration.early_success_settle_seconds
            } else {
                default
            },
        )
        .max(0.0)
    }

    fn early_success_min_quality(&self) -> f64 {
        env_float_any(
            &[
                "GAIL_EARLY_SUCCESS_MIN_QUALITY",
                "REFINER_AI_EARLY_SUCCESS_MIN_QUALITY",
            ],
            self.inner.config.orchestration.early_success_min_quality,
        )
    }

    fn candidate_timeout_cap(&self, workflow: &str, role: &str) -> Option<u64> {
        let default = if is_interactive_workflow(workflow, role) {
            45
        } else {
            self.inner
                .config
                .orchestration
                .candidate_timeout_cap_seconds
                .unwrap_or_default() as i64
        };
        let value = env_int_any(
            &[
                "GAIL_CANDIDATE_TIMEOUT_CAP_SECONDS",
                "REFINER_AI_CANDIDATE_TIMEOUT_CAP_SECONDS",
            ],
            default.max(0) as u64,
        );
        if value == 0 { None } else { Some(value.max(1)) }
    }

    fn always_route_specialists(&self) -> bool {
        self.inner.config.orchestration.always_route_specialists
            || env_bool_any(
                &[
                    "GAIL_ALWAYS_ROUTE_SPECIALISTS",
                    "REFINER_SPECIALIST_ENGINES_ALWAYS_ROUTE",
                    "REFINER_NEUROMORPHIC_ALWAYS_ROUTE",
                    "REFINER_AARNN_ALWAYS_ROUTE",
                ],
                false,
            )
    }
}

impl ProviderCandidate {
    fn from_profile(mut profile: ProviderProfile) -> Self {
        let provider_type = normalize_provider_type(profile.provider_type.as_str());
        if profile.name.trim().is_empty() {
            profile.name = provider_type.clone();
        }
        let configured_model = profile.model.clone().unwrap_or_default();
        let specialties = infer_specialties(
            provider_type.as_str(),
            configured_model.as_str(),
            profile.source.as_deref(),
            &profile.specialties,
        );
        let roles = profile
            .roles
            .iter()
            .map(|item| normalize_key(item, "general"))
            .collect::<HashSet<_>>();
        let source = profile
            .source
            .clone()
            .unwrap_or_else(|| "config".to_string());
        let weight = profile.weight;
        let preferred = profile.preferred;
        Self {
            profile,
            source,
            provider_type,
            configured_model,
            preferred,
            weight,
            specialties,
            roles,
        }
    }

    fn candidate_id(&self) -> String {
        format!(
            "{}/{}",
            self.provider_type,
            if self.configured_model.trim().is_empty() {
                "default"
            } else {
                self.configured_model.trim()
            }
        )
    }

    fn label(&self, resolved_model: Option<&str>) -> String {
        let resolved = resolved_model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                if self.configured_model.trim().is_empty() {
                    "default"
                } else {
                    self.configured_model.trim()
                }
            });
        if !self.configured_model.trim().is_empty() && self.configured_model.trim() != resolved {
            format!(
                "{}/{} (configured {})",
                self.provider_type,
                resolved,
                self.configured_model.trim()
            )
        } else {
            format!("{}/{}", self.provider_type, resolved)
        }
    }

    fn summary(&self, resolved_model: Option<&str>) -> CandidateSummary {
        let resolved = resolved_model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                if self.configured_model.trim().is_empty() {
                    "default"
                } else {
                    self.configured_model.trim()
                }
            })
            .to_string();
        CandidateSummary {
            candidate_id: self.candidate_id(),
            provider: self.provider_type.clone(),
            model: resolved.clone(),
            configured_model: if self.configured_model.trim().is_empty() {
                resolved.clone()
            } else {
                self.configured_model.clone()
            },
            resolved_model: resolved,
            source: self.source.clone(),
            specialties: sorted_strings(self.specialties.clone()),
            roles: sorted_strings(self.roles.clone()),
        }
    }
}

fn dedupe_candidates(candidates: Vec<ProviderCandidate>) -> Vec<ProviderCandidate> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for candidate in candidates {
        let key = format!(
            "{}::{}::{}",
            candidate.provider_type,
            candidate.configured_model,
            candidate.profile.base_url.clone().unwrap_or_default()
        );
        if seen.insert(key) {
            deduped.push(candidate);
        }
    }
    deduped
}

fn infer_specialties(
    provider_type: &str,
    model: &str,
    source: Option<&str>,
    configured: &[String],
) -> HashSet<String> {
    let mut specialties = default_routing_profiles().base_provider_specialties(provider_type);
    let lowered_model = model.to_ascii_lowercase();
    if lowered_model.contains("codex") {
        specialties.extend(
            ["code", "planning", "review"]
                .into_iter()
                .map(ToOwned::to_owned),
        );
    }
    if lowered_model.contains("flash")
        || lowered_model.contains("mini")
        || lowered_model.contains("small")
    {
        specialties.insert("fast".to_string());
    }
    if lowered_model.contains("pro") || lowered_model.contains("o3") || lowered_model.contains("o4")
    {
        specialties.insert("reasoning".to_string());
    }
    if lowered_model.contains("embed") {
        specialties.insert("retrieval".to_string());
    }
    if source
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("local")
    {
        specialties.insert("local".to_string());
    }
    specialties.extend(
        configured
            .iter()
            .map(|item| normalize_key(item, "general"))
            .filter(|item| !item.is_empty()),
    );
    specialties
}

fn workflow_tags(workflow: &str, role: &str, text: &str) -> HashSet<String> {
    default_routing_profiles().workflow_tags(workflow, role, text)
}

fn expected_json(messages: &[crate::models::ChatMessage], system: Option<&str>) -> bool {
    let text = flatten_prompt_text(messages, system).to_ascii_lowercase();
    [
        "return only valid json",
        "respond with json only",
        "valid json",
        "json with keys",
        "output only json",
        "schema",
    ]
    .iter()
    .any(|hint| text.contains(hint))
}

fn try_parse_json(text: &str) -> Option<Value> {
    let payload = text.trim();
    if payload.is_empty() {
        return None;
    }
    serde_json::from_str(payload).ok().or_else(|| {
        if payload.starts_with("```") && payload.ends_with("```") {
            let inner = payload
                .trim_matches('`')
                .trim()
                .strip_prefix("json")
                .unwrap_or(payload.trim_matches('`').trim())
                .trim();
            serde_json::from_str(inner).ok()
        } else {
            None
        }
    })
}

fn quality_score(text: &str, expected_json: bool) -> f64 {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return -3.0;
    }
    let mut score = 0.6;
    if cleaned.len() >= 40 {
        score += 0.35;
    }
    if expected_json {
        match try_parse_json(cleaned) {
            Some(Value::Object(_)) => score += 2.45,
            Some(Value::Array(_)) => score += 2.35,
            Some(_) => score += 2.2,
            None => score -= 2.0,
        }
    }
    if expected_json && cleaned.contains("```") {
        score -= 0.4;
    }
    score
}

fn flatten_prompt_text(messages: &[crate::models::ChatMessage], system: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(system) = system {
        let system = system.trim();
        if !system.is_empty() {
            parts.push(system.to_string());
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

fn normalize_key(value: &str, fallback: &str) -> String {
    let cleaned = value.trim().to_ascii_lowercase();
    if cleaned.is_empty() {
        fallback.to_string()
    } else {
        cleaned
    }
}

fn sorted_strings<T>(values: T) -> Vec<String>
where
    T: IntoIterator<Item = String>,
{
    let mut items = values.into_iter().collect::<Vec<_>>();
    items.sort();
    items
}

fn is_interactive_workflow(workflow: &str, role: &str) -> bool {
    workflow.starts_with("assistant_") || role == "assistant"
}

fn parse_bool(value: &str, default: bool) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}

fn env_bool_any(names: &[&str], default: bool) -> bool {
    for name in names {
        if let Ok(value) = env::var(name) {
            return parse_bool(&value, default);
        }
    }
    default
}

fn env_int_any(names: &[&str], default: u64) -> u64 {
    for name in names {
        if let Ok(value) = env::var(name) {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                return parsed;
            }
        }
    }
    default
}

fn env_float_any(names: &[&str], default: f64) -> f64 {
    for name in names {
        if let Ok(value) = env::var(name) {
            if let Ok(parsed) = value.trim().parse::<f64>() {
                return parsed;
            }
        }
    }
    default
}

fn preview_labels(mut labels: Vec<String>, limit: usize) -> String {
    labels.retain(|item| !item.trim().is_empty());
    if labels.is_empty() {
        return "none".to_string();
    }
    let preview = labels.into_iter().take(limit.max(1)).collect::<Vec<_>>();
    preview.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ChatMessage, MessageContent};

    #[test]
    fn quality_score_prefers_valid_json() {
        assert!(quality_score("{\"ok\":true}", true) > quality_score("not json", true));
    }

    #[test]
    fn workflow_tags_include_keyword_and_profile_tags() {
        let tags = workflow_tags(
            "assistant_requirements",
            "assistant",
            "Need JSON schema for a reading quiz",
        );
        assert!(tags.contains("json"));
        assert!(tags.contains("requirements"));
    }

    #[test]
    fn expected_json_detects_schema_prompt() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text("Return only valid JSON with keys: summary".to_string()),
        }];
        assert!(expected_json(&messages, None));
    }
}
