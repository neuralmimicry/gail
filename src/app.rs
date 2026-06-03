use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Multipart, Query, State},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures::stream;
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::signal;
use uuid::Uuid;

use crate::{
    adaptive_schema, api_issues,
    config::ProviderProfile,
    errors::{GailError, Result},
    models::{
        AerDecodeRequest, AerDecodeResponse, AerEncodeRequest, AerEncodeResponse, ChatMessage,
        CompletionRequest, CompletionResponse, ContentPart, HealthResponse, ImageUrlValue,
        MessageContent, NeuromorphicAnalyzeRequest, NeuromorphicPredictRequest,
        NeuromorphicPredictResponse, OpenAIChatCompletionRequest, OpenAIResponseFormat,
        OpenAIResponseRequest, ProviderCompletionRequest, SpecialistAnalysisResponse,
        TranscriptionResponse,
    },
    orchestration::GailService,
    providers::{TranscriptionInput, normalize_provider_type},
    trading::{
        config::TradingConfigOverride,
        state::{TradeAction, TradeOverride},
    },
};

#[derive(Debug, Default, Deserialize)]
pub struct StatusQuery {
    pub limit: Option<usize>,
    pub probe_engines: Option<bool>,
    pub probe_providers: Option<bool>,
}

#[derive(Clone, Debug)]
enum OpenAIResolvedRoute {
    Orchestrated {
        public_model: String,
        request_category: Option<String>,
        system_suffix: Option<String>,
    },
    Explicit {
        public_model: String,
        provider: String,
        model: Option<String>,
        profile: Option<ProviderProfile>,
    },
}

#[derive(Clone, Debug, Default)]
struct OpenAIToolContext {
    tool_names: Vec<String>,
}

impl OpenAIToolContext {
    fn from_tools(tools: Option<&Value>) -> Option<Self> {
        let tool_names = tools
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(openai_tool_name)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (!tool_names.is_empty()).then_some(Self { tool_names })
    }

    fn contains(&self, name: &str) -> bool {
        self.tool_names.iter().any(|tool| tool == name)
    }

    fn first_name(&self) -> Option<&str> {
        self.tool_names.first().map(String::as_str)
    }
}

#[derive(Clone, Debug, Default)]
struct OpenAIResponseSchemaContext {
    manager_tool_call: bool,
}

impl OpenAIResponseSchemaContext {
    fn from_chat_request(request: &OpenAIChatCompletionRequest) -> Self {
        let mut context = String::new();
        if let Some(instructions) = request.instructions.as_deref() {
            context.push_str(instructions);
            context.push('\n');
        }
        if let Some(format) = request.response_format.as_ref()
            && let Ok(text) = serde_json::to_string(format)
        {
            context.push_str(&text);
            context.push('\n');
        }
        for message in &request.messages {
            context.push_str(&message.flattened_text());
            context.push('\n');
        }
        let lowered = context.to_ascii_lowercase();
        let manager_tool_call = lowered.contains("managertoolcall")
            || (lowered.contains("tool_name")
                && lowered.contains("arguments")
                && (lowered.contains("agent_name") || lowered.contains("run_agent")));
        Self { manager_tool_call }
    }

    fn normalize_response_text(&self, text: &str) -> Option<String> {
        if self.manager_tool_call {
            normalize_manager_tool_call_text(text)
        } else {
            None
        }
    }
}

const EXECUTION_PLAN_CACHE_MAX_ENTRIES: usize = 64;
static EXECUTION_PLAN_CACHE: Lazy<Mutex<HashMap<String, Value>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn build_router(service: GailService) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/responses", post(openai_responses))
        .route(
            "/v1/audio/transcriptions",
            post(openai_audio_transcriptions),
        )
        .route("/v1/llm/complete", post(complete))
        .route("/v1/llm/direct-complete", post(direct_complete))
        .route("/v1/llm/transcribe", post(transcribe))
        .route("/v1/neuromorphic/analyze", post(analyze_neuromorphic))
        .route("/v1/neuromorphic/predict", post(predict_neuromorphic))
        .route("/v1/aer/encode", post(encode_aer))
        .route("/v1/aer/decode", post(decode_aer))
        .route("/v1/status/orchestration", get(orchestration_status))
        .route("/v1/status/api-schema", get(adaptive_api_schema_status))
        .route("/v1/status/api-issues", get(api_issues_status))
        .route("/metrics", get(prometheus_metrics))
        // Trading bridge endpoints
        .route("/v1/trading/status", get(trading_status))
        .route("/v1/trading/portfolio", get(trading_portfolio))
        .route("/v1/trading/positions", get(trading_positions))
        .route("/v1/trading/history", get(trading_history))
        .route("/v1/trading/logs", get(trading_logs))
        .route("/v1/trading/api-schema", get(trading_api_schema))
        .route("/v1/trading/exchanges", get(trading_exchanges))
        .route("/v1/trading/currencies", get(trading_currencies))
        .route(
            "/v1/trading/config",
            get(trading_get_config).post(trading_set_config),
        )
        .route("/v1/trading/pause", post(trading_pause))
        .route("/v1/trading/resume", post(trading_resume))
        .route("/v1/trading/override", post(trading_override))
        .route("/v1/trading/evaluate", post(trading_evaluate))
        .route(
            "/v1/trading/backtest",
            get(trading_backtest_result).post(trading_run_backtest),
        )
        .with_state(service)
}

async fn health(
    State(service): State<GailService>,
    headers: HeaderMap,
) -> Result<Json<HealthResponse>> {
    if !service.can_access_health_unauthenticated() {
        let _ = service.authorize(&headers, "health")?;
    }
    Ok(Json(service.health().await))
}

async fn openai_models(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Err(error) = service.authorize(&headers, "llm") {
        return openai_error_response(error);
    }
    Json(json!({
        "object": "list",
        "data": openai_model_cards(&service),
    }))
    .into_response()
}

async fn openai_chat_completions(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<OpenAIChatCompletionRequest>,
) -> Response {
    if let Err(error) = service.authorize(&headers, "llm") {
        return openai_error_response(error);
    }
    let stream_response = request.stream.unwrap_or(false);
    let tool_context = OpenAIToolContext::from_tools(request.tools.as_ref());
    let response_schema_context = OpenAIResponseSchemaContext::from_chat_request(&request);
    let request_for_fallback = request.clone();
    match dispatch_openai_chat_completion(&service, request).await {
        Ok((public_model, mut response)) => {
            if tool_context.is_none()
                && let Some(normalized) =
                    response_schema_context.normalize_response_text(response.text.as_str())
            {
                response.text = normalized;
            }
            maybe_cache_execution_plan_from_chat_success(&request_for_fallback, &response);
            if stream_response {
                openai_chat_completion_stream(public_model, response).into_response()
            } else {
                Json(openai_chat_completion_body(
                    &public_model,
                    &response,
                    tool_context.as_ref(),
                ))
                .into_response()
            }
        }
        Err(error) => {
            if let Some((public_model, fallback)) = degraded_chat_completion_for_upstream_error(
                &request_for_fallback,
                &response_schema_context,
                &error,
            ) {
                if stream_response {
                    openai_chat_completion_stream(public_model, fallback).into_response()
                } else {
                    Json(openai_chat_completion_body(
                        &public_model,
                        &fallback,
                        tool_context.as_ref(),
                    ))
                    .into_response()
                }
            } else {
                openai_error_response(error)
            }
        }
    }
}

async fn openai_responses(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<OpenAIResponseRequest>,
) -> Response {
    if let Err(error) = service.authorize(&headers, "llm") {
        return openai_error_response(error);
    }
    let stream_response = request.stream.unwrap_or(false);
    let request_for_fallback = request.clone();
    match dispatch_openai_responses(&service, request).await {
        Ok((public_model, response)) => {
            maybe_cache_execution_plan_from_responses_success(&request_for_fallback, &response);
            if stream_response {
                openai_responses_stream(public_model, response).into_response()
            } else {
                Json(openai_responses_body(&public_model, &response)).into_response()
            }
        }
        Err(error) => {
            if let Some((public_model, fallback)) =
                degraded_responses_completion_for_upstream_error(&request_for_fallback, &error)
            {
                if stream_response {
                    openai_responses_stream(public_model, fallback).into_response()
                } else {
                    Json(openai_responses_body(&public_model, &fallback)).into_response()
                }
            } else {
                openai_error_response(error)
            }
        }
    }
}

async fn openai_audio_transcriptions(
    State(service): State<GailService>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    if let Err(error) = service.authorize(&headers, "llm") {
        return openai_error_response(error);
    }

    let mut requested_model = Some("whisper-1".to_string());
    let mut explicit_provider = None;
    let mut api_key = None;
    let mut access_token = None;
    let mut base_url = None;
    let mut timeout_seconds = None;
    let mut file_name = None;
    let mut mime_type = None;
    let mut file_bytes = None;

    while let Some(field) = match multipart.next_field().await {
        Ok(field) => field,
        Err(error) => return openai_error_response(GailError::Multipart(error.to_string())),
    } {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(ToOwned::to_owned);
                mime_type = field.content_type().map(ToOwned::to_owned);
                match field.bytes().await {
                    Ok(bytes) => file_bytes = Some(bytes.to_vec()),
                    Err(error) => {
                        return openai_error_response(GailError::Multipart(error.to_string()));
                    }
                }
            }
            "model" => match field.text().await {
                Ok(value) => requested_model = Some(value),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            "provider" => match field.text().await {
                Ok(value) => explicit_provider = Some(value),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            "api_key" => match field.text().await {
                Ok(value) => api_key = Some(value),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            "access_token" => match field.text().await {
                Ok(value) => access_token = Some(value),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            "base_url" => match field.text().await {
                Ok(value) => base_url = Some(value),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            "timeout_seconds" => match field.text().await {
                Ok(value) => timeout_seconds = value.trim().parse::<u64>().ok(),
                Err(error) => {
                    return openai_error_response(GailError::Multipart(error.to_string()));
                }
            },
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let Some(requested_model) = requested_model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return openai_error_response(GailError::bad_request(
            "OpenAI transcription requests require `model`",
        ));
    };

    let input = TranscriptionInput {
        filename: file_name.unwrap_or_else(|| "upload.bin".to_string()),
        mime_type,
        bytes: match file_bytes {
            Some(bytes) => bytes,
            None => {
                return openai_error_response(GailError::bad_request(
                    "multipart field `file` is required",
                ));
            }
        },
        timeout_seconds,
    };

    let route = match resolve_openai_route(&service, &requested_model, explicit_provider.as_deref())
    {
        Ok(route) => route,
        Err(error) => return openai_error_response(error),
    };

    let OpenAIResolvedRoute::Explicit {
        public_model,
        provider,
        model,
        profile,
    } = route
    else {
        return openai_error_response(GailError::bad_request(
            "audio transcription requests require a provider-backed model such as `whisper-1`, `openai/whisper-1`, or `gemini/gemini-2.5-flash`",
        ));
    };

    match service
        .transcribe(
            provider,
            model.or_else(|| profile.as_ref().and_then(|item| item.model.clone())),
            api_key.or_else(|| profile.as_ref().and_then(|item| item.api_key.clone())),
            access_token.or_else(|| profile.as_ref().and_then(|item| item.access_token.clone())),
            base_url.or_else(|| profile.as_ref().and_then(|item| item.base_url.clone())),
            input,
        )
        .await
    {
        Ok(response) => Json(json!({
            "text": response.text,
            "model": public_model,
            "gail": {
                "provider": response.provider,
                "resolved_model": response.model,
                "request_id": response.request_id,
                "latency_ms": response.latency_ms,
            }
        }))
        .into_response(),
        Err(error) => openai_error_response(error),
    }
}

async fn complete(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>> {
    let _ = service.authorize(&headers, "llm")?;
    Ok(Json(service.complete(request).await?))
}

async fn direct_complete(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<ProviderCompletionRequest>,
) -> Result<Json<CompletionResponse>> {
    let _ = service.authorize(&headers, "llm")?;
    Ok(Json(service.direct_complete(request).await?))
}

async fn transcribe(
    State(service): State<GailService>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<TranscriptionResponse>> {
    let _ = service.authorize(&headers, "llm")?;
    let mut provider = None;
    let mut model = None;
    let mut api_key = None;
    let mut access_token = None;
    let mut base_url = None;
    let mut timeout_seconds = None;
    let mut file_name = None;
    let mut mime_type = None;
    let mut file_bytes = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| GailError::Multipart(error.to_string()))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(ToOwned::to_owned);
                mime_type = field.content_type().map(ToOwned::to_owned);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| GailError::Multipart(error.to_string()))?;
                file_bytes = Some(bytes.to_vec());
            }
            "provider" => {
                provider = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| GailError::Multipart(error.to_string()))?,
                )
            }
            "model" => {
                model = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| GailError::Multipart(error.to_string()))?,
                )
            }
            "api_key" => {
                api_key = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| GailError::Multipart(error.to_string()))?,
                )
            }
            "access_token" => {
                access_token = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| GailError::Multipart(error.to_string()))?,
                )
            }
            "base_url" => {
                base_url = Some(
                    field
                        .text()
                        .await
                        .map_err(|error| GailError::Multipart(error.to_string()))?,
                )
            }
            "timeout_seconds" => {
                let raw = field
                    .text()
                    .await
                    .map_err(|error| GailError::Multipart(error.to_string()))?;
                timeout_seconds = raw.trim().parse::<u64>().ok();
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let provider = provider.unwrap_or_else(|| "openai".to_string());
    let input = TranscriptionInput {
        filename: file_name.unwrap_or_else(|| "upload.bin".to_string()),
        mime_type,
        bytes: file_bytes
            .ok_or_else(|| GailError::bad_request("multipart field `file` is required"))?,
        timeout_seconds,
    };
    Ok(Json(
        service
            .transcribe(provider, model, api_key, access_token, base_url, input)
            .await?,
    ))
}

async fn analyze_neuromorphic(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<NeuromorphicAnalyzeRequest>,
) -> Result<Json<SpecialistAnalysisResponse>> {
    let _ = service.authorize(&headers, "neuromorphic")?;
    Ok(Json(service.analyze_neuromorphic(request).await?))
}

async fn predict_neuromorphic(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<NeuromorphicPredictRequest>,
) -> Result<Json<NeuromorphicPredictResponse>> {
    let _ = service.authorize(&headers, "neuromorphic")?;
    Ok(Json(service.predict_neuromorphic(request).await?))
}

async fn encode_aer(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<AerEncodeRequest>,
) -> Result<Json<AerEncodeResponse>> {
    let _ = service.authorize(&headers, "aer")?;
    Ok(Json(service.encode_aer(request)?))
}

async fn decode_aer(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(request): Json<AerDecodeRequest>,
) -> Result<Json<AerDecodeResponse>> {
    let _ = service.authorize(&headers, "aer")?;
    Ok(Json(service.decode_aer(request)?))
}

async fn orchestration_status(
    State(service): State<GailService>,
    headers: HeaderMap,
    Query(query): Query<StatusQuery>,
) -> Result<Json<Value>> {
    let _ = service.authorize(&headers, "status")?;
    Ok(Json(
        service
            .orchestration_status_value(
                query.limit.unwrap_or(20),
                query.probe_engines.unwrap_or(false),
                query.probe_providers.unwrap_or(false),
            )
            .await,
    ))
}

async fn adaptive_api_schema_status(
    State(service): State<GailService>,
    headers: HeaderMap,
) -> Result<Json<Value>> {
    let _ = service.authorize(&headers, "status")?;
    Ok(Json(json!({
        "adaptive_api_schema": adaptive_schema::snapshot().await,
    })))
}

async fn api_issues_status(
    State(service): State<GailService>,
    headers: HeaderMap,
) -> Result<Json<Value>> {
    let _ = service.authorize(&headers, "status")?;
    Ok(Json(json!({
        "api_issues": api_issues::snapshot().await,
    })))
}

async fn prometheus_metrics(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if !service.can_access_metrics_unauthenticated()
        && let Err(error) = service.authorize(&headers, "status")
    {
        return error.into_response();
    }
    let mut rendered = api_issues::prometheus_metrics().await;
    rendered.push_str(&service.provider_prometheus_metrics().await);
    ([(CONTENT_TYPE, "text/plain; version=0.0.4")], rendered).into_response()
}

// ---------------------------------------------------------------------------
// Trading bridge HTTP handlers
// ---------------------------------------------------------------------------

/// Helper: require the `trading` scope and return an error response if missing.
fn require_trading_scope(service: &GailService, headers: &HeaderMap) -> Option<Response> {
    match service.authorize(headers, "trading") {
        Ok(_) => None,
        Err(err) => Some(err.into_response()),
    }
}

/// Helper: require `trading` scope AND verify the client_id is in admin_client_ids.
fn require_trading_admin(service: &GailService, headers: &HeaderMap) -> Option<Response> {
    match service.authorize(headers, "trading") {
        Ok(ctx) => {
            let admin_ids = &service.config().trading.admin_client_ids;
            if admin_ids.is_empty() {
                return None; // no admin restriction configured
            }
            let client_id = ctx.client_id.as_deref().unwrap_or("");
            if admin_ids.iter().any(|id| id == client_id) {
                None
            } else {
                Some(GailError::unauthorized().into_response())
            }
        }
        Err(err) => Some(err.into_response()),
    }
}

/// Helper: return a 503 when the trading bridge is disabled/not configured.
fn trading_unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": "trading bridge is not enabled or configured" })),
    )
        .into_response()
}

async fn trading_status(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            let snapshot = state.status_snapshot(bridge.is_enabled());
            Json(snapshot).into_response()
        }
    }
}

async fn trading_portfolio(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            Json(json!({ "portfolio": state.current_portfolio })).into_response()
        }
    }
}

async fn trading_positions(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            Json(json!({ "positions": state.open_positions })).into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    limit: Option<usize>,
}

async fn trading_history(
    State(service): State<GailService>,
    headers: HeaderMap,
    Query(query): Query<HistoryQuery>,
) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let limit = query.limit.unwrap_or(50).min(500);
            let state = bridge.state.0.lock().await;
            let trades: Vec<_> = state.recent_trades.iter().rev().take(limit).collect();
            Json(json!({ "trades": trades, "total": state.trade_count })).into_response()
        }
    }
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    limit: Option<usize>,
}

async fn trading_logs(
    State(service): State<GailService>,
    headers: HeaderMap,
    Query(query): Query<LogsQuery>,
) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let limit = query.limit.unwrap_or(100).min(1000);
            let state = bridge.state.0.lock().await;
            let logs: Vec<_> = state.activity_log.iter().rev().take(limit).collect();
            Json(json!({ "logs": logs })).into_response()
        }
    }
}

async fn trading_api_schema(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let octobot_schema = {
                let state = bridge.state.0.lock().await;
                state.api_schema.clone()
            };
            let global_registry = adaptive_schema::snapshot().await;
            Json(json!({
                "api": "octobot",
                "api_schema": octobot_schema,
                "global_registry": global_registry,
            }))
            .into_response()
        }
    }
}

async fn trading_exchanges(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            Json(json!({ "exchanges": state.available_exchanges })).into_response()
        }
    }
}

async fn trading_currencies(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            let currencies: Vec<String> = state
                .available_exchanges
                .iter()
                .flat_map(|e| e.symbols.iter().cloned())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            Json(json!({ "currencies": currencies })).into_response()
        }
    }
}

async fn trading_get_config(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            // Return effective config: base config + active overrides.
            Json(json!({
                "config": *bridge.config,
                "overrides": state.config_overrides,
                "enabled": bridge.is_enabled()
            }))
            .into_response()
        }
    }
}

async fn trading_set_config(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(overrides): Json<TradingConfigOverride>,
) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let mut state = bridge.state.0.lock().await;
            state.config_overrides = Some(overrides);
            state.log_info("config", "Runtime config overrides updated via API");
            Json(json!({ "ok": true, "message": "config overrides applied" })).into_response()
        }
    }
}

async fn trading_pause(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let mut state = bridge.state.0.lock().await;
            state.paused = true;
            state.log_info("control", "Trading bridge PAUSED via API");
            Json(json!({ "ok": true, "paused": true })).into_response()
        }
    }
}

async fn trading_resume(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let mut state = bridge.state.0.lock().await;
            state.paused = false;
            state.log_info("control", "Trading bridge RESUMED via API");
            Json(json!({ "ok": true, "paused": false })).into_response()
        }
    }
}

async fn trading_override(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    let auth_ctx = match service.authorize(&headers, "trading") {
        Ok(ctx) => ctx,
        Err(err) => return err.into_response(),
    };
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let action_str = body.get("action").and_then(Value::as_str).unwrap_or("hold");
            let action = match action_str {
                "buy" => TradeAction::Buy,
                "sell" => TradeAction::Sell,
                "strong_buy" => TradeAction::StrongBuy,
                "strong_sell" => TradeAction::StrongSell,
                "cancel" => TradeAction::Cancel,
                _ => TradeAction::Hold,
            };
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let override_req = TradeOverride {
                action,
                exchange: body
                    .get("exchange")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                symbol: body
                    .get("symbol")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                amount_usd: body.get("amount_usd").and_then(Value::as_f64),
                reason: body
                    .get("reason")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                issued_at: now,
                issued_by: auth_ctx.client_id.unwrap_or_else(|| "unknown".to_string()),
            };
            let mut state = bridge.state.0.lock().await;
            state.pending_override = Some(override_req);
            state.log_info("control", format!("Trade override set: {action_str}"));
            Json(json!({ "ok": true, "message": "override queued for next evaluation" }))
                .into_response()
        }
    }
}

async fn trading_evaluate(State(service): State<GailService>, headers: HeaderMap) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            // We can't directly trigger the loop, but we can log it.
            // A full implementation would use a channel; for now return status.
            let state = bridge.state.0.lock().await;
            let snapshot = state.status_snapshot(bridge.is_enabled());
            Json(json!({
                "ok": true,
                "message": "evaluation will occur at next scheduled interval",
                "status": snapshot
            }))
            .into_response()
        }
    }
}

/// GET /v1/trading/backtest — return the most recent backtest summary and history.
async fn trading_backtest_result(
    State(service): State<GailService>,
    headers: HeaderMap,
) -> Response {
    if let Some(err_resp) = require_trading_scope(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let state = bridge.state.0.lock().await;
            Json(json!({
                "last_backtest": state.last_backtest,
                "backtest_history": state.backtest_history,
                "backtesting_enabled": bridge.config.backtesting_enabled,
                "backtest_interval_seconds": bridge.config.backtest_interval_seconds,
                "backtest_profitability_threshold": bridge.config.backtest_profitability_threshold,
            }))
            .into_response()
        }
    }
}

/// POST /v1/trading/backtest — trigger an immediate backtesting run (admin only).
///
/// Optional JSON body: `{"files": [...], "start_timestamp": ms, "end_timestamp": ms}`
/// If omitted, uses config defaults.
async fn trading_run_backtest(
    State(service): State<GailService>,
    headers: HeaderMap,
    body: Option<Json<serde_json::Value>>,
) -> Response {
    if let Some(err_resp) = require_trading_admin(&service, &headers) {
        return err_resp;
    }
    match service.trading_bridge() {
        None => trading_unavailable(),
        Some(bridge) => {
            let config = bridge.config.clone();
            let state = bridge.state.clone();
            // Run the backtest in a background task so the HTTP response is immediate.
            tokio::spawn(async move {
                use crate::trading::backtest::BacktestEngine;
                use crate::trading::octobot::{BacktestStartRequest as OctoReq, OctobotClient};
                let engine = BacktestEngine::new(
                    OctobotClient::new(
                        &config.octobot_base_url,
                        config.octobot_password.as_deref(),
                        config.octobot_timeout_seconds,
                    ),
                    config.backtest_profitability_threshold,
                );
                let summary = if let Some(Json(val)) = body {
                    let octo_req: OctoReq = serde_json::from_value(val).unwrap_or_default();
                    if octo_req.files.is_empty() {
                        engine.run_with_config(&config).await
                    } else {
                        engine.run(&octo_req).await
                    }
                } else {
                    engine.run_with_config(&config).await
                };
                let should_pause = config.backtest_pause_on_failure
                    && summary.assessment
                        == crate::trading::backtest::ApproachAssessment::Unprofitable;
                let mut s = state.0.lock().await;
                s.record_backtest(summary);
                if should_pause {
                    s.paused = true;
                }
            });
            Json(json!({
                "ok": true,
                "message": "backtesting run started in background; poll /v1/trading/backtest for results"
            }))
            .into_response()
        }
    }
}

async fn dispatch_openai_chat_completion(
    service: &GailService,
    request: OpenAIChatCompletionRequest,
) -> Result<(String, CompletionResponse)> {
    let route = resolve_openai_route(service, &request.model, request.provider.as_deref())?;
    let public_model = match &route {
        OpenAIResolvedRoute::Orchestrated { public_model, .. }
        | OpenAIResolvedRoute::Explicit { public_model, .. } => public_model.clone(),
    };
    let (system_from_messages, messages) = split_system_messages(request.messages);
    let request_category = route_request_category(&route, request.request_category);
    let role = role_for_request(request.role, request_category.as_deref());
    let system = combine_text_segments(vec![
        system_from_messages,
        request.instructions,
        response_format_system_hint(request.response_format.as_ref()),
        tool_call_system_hint(request.tools.as_ref()),
        route_system_suffix(&route),
    ]);

    let completion_request = build_completion_request(
        route.clone(),
        request.workflow.clone(),
        role.clone(),
        request_category.clone(),
        messages.clone(),
        system.clone(),
        request.max_tokens,
        request.temperature,
        request
            .reasoning
            .clone()
            .and_then(|reasoning| reasoning.effort),
        request.include_configured,
        request.selection_mode,
        request.max_candidates,
        request.api_key.clone(),
        request.access_token.clone(),
        request.base_url.clone(),
    );
    let response = match route {
        OpenAIResolvedRoute::Orchestrated { .. } => service.complete(completion_request).await?,
        OpenAIResolvedRoute::Explicit {
            provider,
            model,
            profile,
            ..
        } => {
            let direct_request = build_direct_provider_request(
                provider,
                model,
                profile,
                request.workflow,
                role,
                request_category,
                messages,
                system,
                request.max_tokens,
                request.temperature,
                request.reasoning.and_then(|reasoning| reasoning.effort),
                request.api_key,
                request.access_token,
                request.base_url,
            );
            service.direct_complete(direct_request).await?
        }
    };
    Ok((public_model, response))
}

async fn dispatch_openai_responses(
    service: &GailService,
    request: OpenAIResponseRequest,
) -> Result<(String, CompletionResponse)> {
    let route = resolve_openai_route(service, &request.model, request.provider.as_deref())?;
    let public_model = match &route {
        OpenAIResolvedRoute::Orchestrated { public_model, .. }
        | OpenAIResolvedRoute::Explicit { public_model, .. } => public_model.clone(),
    };
    let (system_from_input, messages) = openai_response_input_to_messages(&request.input)?;
    let request_category = route_request_category(&route, request.request_category);
    let role = role_for_request(request.role, request_category.as_deref());
    let system = combine_text_segments(vec![
        request.instructions,
        system_from_input,
        response_format_system_hint(request.text.as_ref().and_then(|item| item.format.as_ref())),
        route_system_suffix(&route),
    ]);

    let completion_request = build_completion_request(
        route.clone(),
        request.workflow.clone(),
        role.clone(),
        request_category.clone(),
        messages.clone(),
        system.clone(),
        request.max_output_tokens,
        request.temperature,
        request
            .reasoning
            .clone()
            .and_then(|reasoning| reasoning.effort),
        request.include_configured,
        request.selection_mode,
        request.max_candidates,
        request.api_key.clone(),
        request.access_token.clone(),
        request.base_url.clone(),
    );
    let response = match route {
        OpenAIResolvedRoute::Orchestrated { .. } => service.complete(completion_request).await?,
        OpenAIResolvedRoute::Explicit {
            provider,
            model,
            profile,
            ..
        } => {
            let direct_request = build_direct_provider_request(
                provider,
                model,
                profile,
                request.workflow,
                role,
                request_category,
                messages,
                system,
                request.max_output_tokens,
                request.temperature,
                request.reasoning.and_then(|reasoning| reasoning.effort),
                request.api_key,
                request.access_token,
                request.base_url,
            );
            service.direct_complete(direct_request).await?
        }
    };
    Ok((public_model, response))
}

#[allow(clippy::too_many_arguments)]
fn build_direct_provider_request(
    provider: String,
    model: Option<String>,
    profile: Option<ProviderProfile>,
    workflow: Option<String>,
    role: Option<String>,
    request_category: Option<String>,
    messages: Vec<ChatMessage>,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    api_key: Option<String>,
    access_token: Option<String>,
    base_url: Option<String>,
) -> ProviderCompletionRequest {
    ProviderCompletionRequest {
        provider,
        model: model.or_else(|| profile.as_ref().and_then(|item| item.model.clone())),
        api_key: api_key.or_else(|| profile.as_ref().and_then(|item| item.api_key.clone())),
        access_token: access_token
            .or_else(|| profile.as_ref().and_then(|item| item.access_token.clone())),
        base_url: base_url.or_else(|| profile.as_ref().and_then(|item| item.base_url.clone())),
        messages,
        system,
        max_tokens,
        temperature,
        timeout_seconds: None,
        reasoning_effort,
        request_category,
        workflow,
        role,
        min_model_size_b: None,
        strict_no_downgrade: None,
    }
}

fn build_completion_request(
    route: OpenAIResolvedRoute,
    workflow: Option<String>,
    role: Option<String>,
    request_category: Option<String>,
    messages: Vec<ChatMessage>,
    system: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    include_configured: Option<bool>,
    selection_mode: Option<crate::models::SelectionMode>,
    max_candidates: Option<usize>,
    api_key: Option<String>,
    access_token: Option<String>,
    base_url: Option<String>,
) -> CompletionRequest {
    match route {
        OpenAIResolvedRoute::Orchestrated { .. } => CompletionRequest {
            workflow,
            role,
            preferred_provider: None,
            preferred_model: None,
            preferred_api_key: None,
            preferred_access_token: None,
            fallback_provider: None,
            fallback_model: None,
            fallback_api_key: None,
            fallback_access_token: None,
            base_url,
            include_configured,
            selection_mode,
            max_candidates,
            messages,
            system,
            max_tokens,
            temperature,
            timeout_seconds: None,
            reasoning_effort,
            request_category,
        },
        OpenAIResolvedRoute::Explicit {
            provider,
            model,
            profile,
            ..
        } => CompletionRequest {
            workflow,
            role,
            preferred_provider: Some(provider),
            preferred_model: model.or_else(|| profile.as_ref().and_then(|item| item.model.clone())),
            preferred_api_key: api_key
                .or_else(|| profile.as_ref().and_then(|item| item.api_key.clone())),
            preferred_access_token: access_token
                .or_else(|| profile.as_ref().and_then(|item| item.access_token.clone())),
            fallback_provider: None,
            fallback_model: None,
            fallback_api_key: None,
            fallback_access_token: None,
            base_url: base_url.or_else(|| profile.as_ref().and_then(|item| item.base_url.clone())),
            include_configured: Some(false),
            selection_mode,
            max_candidates: Some(1),
            messages,
            system,
            max_tokens,
            temperature,
            timeout_seconds: None,
            reasoning_effort,
            request_category,
        },
    }
}

fn resolve_openai_route(
    service: &GailService,
    requested_model: &str,
    explicit_provider: Option<&str>,
) -> Result<OpenAIResolvedRoute> {
    let model = requested_model.trim();
    if model.is_empty() {
        return Err(GailError::bad_request(
            "OpenAI requests require a non-empty `model`",
        ));
    }

    let lowered = model.to_ascii_lowercase();
    if is_gail_auto_model(&lowered) {
        return Ok(OpenAIResolvedRoute::Orchestrated {
            public_model: "gail-auto".to_string(),
            request_category: None,
            system_suffix: None,
        });
    }
    if is_generic_specialist_alias(&lowered) {
        return Ok(OpenAIResolvedRoute::Orchestrated {
            public_model: model.to_string(),
            request_category: Some("neuromorphic".to_string()),
            system_suffix: Some("Use Gail's neuromorphic specialist context where it materially improves the answer.".to_string()),
        });
    }

    if let Some((prefix, routed_model)) = split_routed_model(model) {
        if matches!(prefix.as_str(), "gail" | "gateway") {
            if is_gail_auto_model(&routed_model.to_ascii_lowercase()) {
                return Ok(OpenAIResolvedRoute::Orchestrated {
                    public_model: "gail-auto".to_string(),
                    request_category: None,
                    system_suffix: None,
                });
            }
            return Err(GailError::bad_request(format!(
                "unsupported Gail model route `{model}`. Use `gail-auto` or `provider/model`"
            )));
        }
        if is_specialist_prefix(prefix.as_str()) {
            return Ok(OpenAIResolvedRoute::Orchestrated {
                public_model: format!("{}/{}", prefix, routed_model),
                request_category: Some(format!("neuromorphic {routed_model}")),
                system_suffix: Some(specialist_system_suffix(&routed_model)),
            });
        }
        let profile = select_provider_profile(service, prefix.as_str(), Some(&routed_model));
        let provider = profile
            .as_ref()
            .map(|item| normalize_provider_type(item.provider_type.as_str()))
            .unwrap_or_else(|| normalize_provider_type(prefix.as_str()));
        return Ok(OpenAIResolvedRoute::Explicit {
            public_model: model.to_string(),
            provider,
            model: (!routed_model.eq_ignore_ascii_case("default")).then_some(routed_model),
            profile,
        });
    }

    if let Some(provider_hint) = explicit_provider.filter(|value| !value.trim().is_empty()) {
        let normalized = provider_hint.trim().to_ascii_lowercase();
        if is_specialist_prefix(normalized.as_str()) {
            return Ok(OpenAIResolvedRoute::Orchestrated {
                public_model: model.to_string(),
                request_category: Some(format!("neuromorphic {model}")),
                system_suffix: Some(specialist_system_suffix(model)),
            });
        }
        let profile = select_provider_profile(service, provider_hint, Some(model));
        let provider = profile
            .as_ref()
            .map(|item| normalize_provider_type(item.provider_type.as_str()))
            .unwrap_or_else(|| normalize_provider_type(provider_hint));
        if !is_supported_provider(provider.as_str()) {
            return Err(GailError::bad_request(format!(
                "unsupported provider hint `{provider_hint}` for OpenAI-compatible routing"
            )));
        }
        return Ok(OpenAIResolvedRoute::Explicit {
            public_model: model.to_string(),
            provider,
            model: Some(model.to_string()),
            profile,
        });
    }

    if let Some((provider, profile)) = find_profile_for_model(service, model) {
        return Ok(OpenAIResolvedRoute::Explicit {
            public_model: model.to_string(),
            provider,
            model: Some(model.to_string()),
            profile: Some(profile),
        });
    }

    if let Some(provider) = infer_provider_from_model(model) {
        return Ok(OpenAIResolvedRoute::Explicit {
            public_model: model.to_string(),
            profile: select_provider_profile(service, &provider, Some(model)),
            provider,
            model: Some(model.to_string()),
        });
    }

    Err(GailError::bad_request(format!(
        "unable to route OpenAI model `{model}`. Use `gail-auto` for orchestration or `provider/model` for an explicit backend"
    )))
}

fn split_system_messages(messages: Vec<ChatMessage>) -> (Option<String>, Vec<ChatMessage>) {
    let mut system_segments = Vec::new();
    let mut non_system = Vec::new();
    for message in messages {
        if message.role.eq_ignore_ascii_case("system") {
            let text = message.flattened_text();
            if !text.trim().is_empty() {
                system_segments.push(Some(text));
            }
        } else {
            non_system.push(message);
        }
    }
    (combine_text_segments(system_segments), non_system)
}

fn openai_response_input_to_messages(input: &Value) -> Result<(Option<String>, Vec<ChatMessage>)> {
    match input {
        Value::String(text) => Ok((
            None,
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![user_text_message(text)]
            },
        )),
        Value::Array(items) => parse_openai_input_items(items),
        Value::Object(_) => parse_openai_input_items(std::slice::from_ref(input)),
        _ => Err(GailError::bad_request(
            "OpenAI responses input must be a string, object, or array",
        )),
    }
}

fn parse_openai_input_items(items: &[Value]) -> Result<(Option<String>, Vec<ChatMessage>)> {
    let mut system_segments = Vec::new();
    let mut messages = Vec::new();

    for item in items {
        match item {
            Value::String(text) if !text.trim().is_empty() => {
                messages.push(user_text_message(text));
            }
            Value::Object(object) => {
                if let Some(item_type) = object.get("type").and_then(Value::as_str) {
                    match item_type {
                        "input_text" | "text" | "output_text" => {
                            if let Some(text) = object.get("text").and_then(Value::as_str)
                                && !text.trim().is_empty()
                            {
                                messages.push(user_text_message(text));
                            }
                            continue;
                        }
                        "message" => {
                            let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
                            let (system, message) = openai_role_content(
                                role,
                                object.get("content").unwrap_or(&Value::Null),
                            )?;
                            if let Some(system) = system {
                                system_segments.push(Some(system));
                            }
                            if let Some(message) = message {
                                messages.push(message);
                            }
                            continue;
                        }
                        "input_image" => {
                            if let Some(content) = openai_message_content_from_value(item)? {
                                messages.push(ChatMessage {
                                    role: "user".to_string(),
                                    content,
                                });
                            }
                            continue;
                        }
                        _ => {}
                    }
                }

                if let Some(role) = object.get("role").and_then(Value::as_str) {
                    let (system, message) =
                        openai_role_content(role, object.get("content").unwrap_or(&Value::Null))?;
                    if let Some(system) = system {
                        system_segments.push(Some(system));
                    }
                    if let Some(message) = message {
                        messages.push(message);
                    }
                } else if let Some(text) = object.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    messages.push(user_text_message(text));
                }
            }
            _ => {}
        }
    }

    Ok((combine_text_segments(system_segments), messages))
}

fn openai_role_content(
    role: &str,
    content: &Value,
) -> Result<(Option<String>, Option<ChatMessage>)> {
    let Some(content) = openai_message_content_from_value(content)? else {
        return Ok((None, None));
    };
    if role.eq_ignore_ascii_case("system") {
        let text = content.flattened_text();
        if text.trim().is_empty() {
            return Ok((None, None));
        }
        return Ok((Some(text), None));
    }
    Ok((
        None,
        Some(ChatMessage {
            role: role.to_string(),
            content,
        }),
    ))
}

fn openai_message_content_from_value(value: &Value) -> Result<Option<MessageContent>> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) => {
            if text.trim().is_empty() {
                Ok(None)
            } else {
                Ok(Some(MessageContent::Text(text.to_string())))
            }
        }
        Value::Array(parts) => {
            let mut converted = Vec::new();
            for part in parts {
                if let Some(converted_part) = openai_content_part_from_value(part)? {
                    converted.push(converted_part);
                }
            }
            if converted.is_empty() {
                return Ok(None);
            }
            if converted.len() == 1
                && let ContentPart::Text { text } = &converted[0]
            {
                return Ok(Some(MessageContent::Text(text.clone())));
            }
            Ok(Some(MessageContent::Parts(converted)))
        }
        Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(Value::as_str)
                && !text.trim().is_empty()
            {
                return Ok(Some(MessageContent::Text(text.to_string())));
            }
            if let Some(url) = extract_image_url(object) {
                return Ok(Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrlValue { url },
                }])));
            }
            Ok(None)
        }
        _ => Ok(Some(MessageContent::Text(value.to_string()))),
    }
}

fn openai_content_part_from_value(value: &Value) -> Result<Option<ContentPart>> {
    match value {
        Value::String(text) => {
            if text.trim().is_empty() {
                Ok(None)
            } else {
                Ok(Some(ContentPart::Text {
                    text: text.to_string(),
                }))
            }
        }
        Value::Object(object) => {
            let part_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match part_type {
                "text" | "input_text" | "output_text" => Ok(object
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                    .map(|text| ContentPart::Text {
                        text: text.to_string(),
                    })),
                "image_url" | "input_image" => {
                    Ok(extract_image_url(object).map(|url| ContentPart::ImageUrl {
                        image_url: ImageUrlValue { url },
                    }))
                }
                _ => {
                    if let Some(text) = object.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        return Ok(Some(ContentPart::Text {
                            text: text.to_string(),
                        }));
                    }
                    Ok(extract_image_url(object).map(|url| ContentPart::ImageUrl {
                        image_url: ImageUrlValue { url },
                    }))
                }
            }
        }
        _ => Ok(None),
    }
}

fn extract_image_url(object: &serde_json::Map<String, Value>) -> Option<String> {
    object
        .get("image_url")
        .and_then(|value| {
            value.as_str().map(ToOwned::to_owned).or_else(|| {
                value
                    .get("url")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
        })
        .or_else(|| {
            object
                .get("url")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

fn openai_tool_name(tool: &Value) -> Option<String> {
    tool.get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| tool.get("name").and_then(Value::as_str))
        .filter(|name| !name.trim().is_empty())
        .map(|name| name.trim().to_string())
}

fn response_format_system_hint(format: Option<&OpenAIResponseFormat>) -> Option<String> {
    let format = format?;
    let kind = format
        .format_type
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if kind == "json_object" {
        return Some("Return only valid JSON with no markdown fences or commentary.".to_string());
    }
    if kind == "json_schema" || format.json_schema.is_some() {
        let schema = format
            .json_schema
            .as_ref()
            .and_then(|value| value.get("schema").cloned().or_else(|| Some(value.clone())))
            .unwrap_or(Value::Null);
        let schema_text = serde_json::to_string(&schema).unwrap_or_else(|_| "{}".to_string());
        return Some(format!(
            "Return only a JSON data instance that satisfies this schema: {schema_text}. The schema is not the answer; do not echo `$defs`, `properties`, `required`, `title`, `type`, or `additionalProperties` unless those are explicitly required as user data fields."
        ));
    }
    None
}

fn tool_call_system_hint(tools: Option<&Value>) -> Option<String> {
    let context = OpenAIToolContext::from_tools(tools)?;
    Some(format!(
        "The caller supplied OpenAI tools. If a tool should be used, return only valid JSON in this exact shape: {{\"tool_name\":\"<one of: {}>\",\"arguments\":{{...}}}}. Do not return bare tool arguments or prose.",
        context.tool_names.join(", ")
    ))
}

fn route_request_category(
    route: &OpenAIResolvedRoute,
    request_category: Option<String>,
) -> Option<String> {
    match route {
        OpenAIResolvedRoute::Orchestrated {
            request_category: route_category,
            ..
        } => merge_request_categories(request_category, route_category.clone()),
        OpenAIResolvedRoute::Explicit { .. } => request_category,
    }
}

fn route_system_suffix(route: &OpenAIResolvedRoute) -> Option<String> {
    match route {
        OpenAIResolvedRoute::Orchestrated { system_suffix, .. } => system_suffix.clone(),
        OpenAIResolvedRoute::Explicit { .. } => None,
    }
}

fn role_for_request(role: Option<String>, request_category: Option<&str>) -> Option<String> {
    if role.is_some() {
        return role;
    }
    if wants_specialist_support(request_category) {
        return Some("assistant".to_string());
    }
    None
}

fn wants_specialist_support(request_category: Option<&str>) -> bool {
    request_category
        .map(|value| {
            let lowered = value.to_ascii_lowercase();
            lowered.contains("neuromorphic") || lowered.contains("aarnn") || lowered.contains("snn")
        })
        .unwrap_or(false)
}

fn merge_request_categories(left: Option<String>, right: Option<String>) -> Option<String> {
    let mut merged = Vec::new();
    for value in [left, right].into_iter().flatten() {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        if merged
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(trimmed))
        {
            continue;
        }
        merged.push(trimmed.to_string());
    }
    if merged.is_empty() {
        None
    } else {
        Some(merged.join(" "))
    }
}

fn combine_text_segments(segments: Vec<Option<String>>) -> Option<String> {
    let mut merged = Vec::new();
    for value in segments.into_iter().flatten() {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        if merged.iter().any(|existing: &String| existing == trimmed) {
            continue;
        }
        merged.push(trimmed.to_string());
    }
    if merged.is_empty() {
        None
    } else {
        Some(merged.join("\n\n"))
    }
}

fn user_text_message(text: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: MessageContent::Text(text.to_string()),
    }
}

fn select_provider_profile(
    service: &GailService,
    provider_hint: &str,
    model_hint: Option<&str>,
) -> Option<ProviderProfile> {
    let provider_hint = provider_hint.trim();
    if provider_hint.is_empty() {
        return None;
    }
    let normalized = normalize_provider_type(provider_hint);
    let mut matches = service
        .config()
        .providers
        .iter()
        .filter(|profile| {
            profile.name.eq_ignore_ascii_case(provider_hint)
                || normalize_provider_type(profile.provider_type.as_str()) == normalized
        })
        .cloned()
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    if let Some(model_hint) = model_hint
        .filter(|value| !value.trim().is_empty() && !value.eq_ignore_ascii_case("default"))
    {
        let exact_matches = matches
            .iter()
            .filter(|profile| {
                profile
                    .model
                    .as_deref()
                    .is_some_and(|model| model.eq_ignore_ascii_case(model_hint))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !exact_matches.is_empty() {
            matches = exact_matches;
        }
    }
    matches.sort_by(|left, right| {
        right
            .preferred
            .cmp(&left.preferred)
            .then_with(|| left.name.cmp(&right.name))
    });
    matches.into_iter().next()
}

fn find_profile_for_model(service: &GailService, model: &str) -> Option<(String, ProviderProfile)> {
    let matches = service
        .config()
        .providers
        .iter()
        .filter(|profile| {
            profile
                .model
                .as_deref()
                .is_some_and(|configured| configured.eq_ignore_ascii_case(model))
        })
        .cloned()
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    let provider_types = matches
        .iter()
        .map(|profile| normalize_provider_type(profile.provider_type.as_str()))
        .collect::<HashSet<_>>();
    if provider_types.len() != 1 {
        return None;
    }
    let provider = provider_types.into_iter().next()?;
    let profile = select_provider_profile(service, &provider, Some(model))
        .unwrap_or_else(|| matches[0].clone());
    Some((provider, profile))
}

fn infer_provider_from_model(model: &str) -> Option<String> {
    let lowered = model.trim().to_ascii_lowercase();
    if lowered.starts_with("gpt-")
        || lowered.starts_with("chatgpt")
        || lowered.starts_with("o1")
        || lowered.starts_with("o3")
        || lowered.starts_with("o4")
        || lowered.starts_with("codex")
        || lowered.starts_with("whisper")
    {
        return Some("openai".to_string());
    }
    if lowered.starts_with("gemini") {
        return Some("gemini".to_string());
    }
    if [
        "llama",
        "mistral",
        "mixtral",
        "qwen",
        "phi",
        "deepseek",
        "gemma",
        "codellama",
        "dolphin",
        "orca",
        "nous",
    ]
    .iter()
    .any(|prefix| lowered.starts_with(prefix))
    {
        return Some("ollama".to_string());
    }
    None
}

fn is_supported_provider(provider: &str) -> bool {
    matches!(provider, "openai" | "nvidia" | "gemini" | "ollama")
}

fn is_specialist_prefix(prefix: &str) -> bool {
    matches!(prefix, "aarnn" | "snn" | "specialist" | "neuromorphic")
}

fn is_generic_specialist_alias(model: &str) -> bool {
    matches!(model, "aarnn" | "snn" | "specialist" | "neuromorphic")
}

fn is_known_route_prefix(prefix: &str) -> bool {
    is_supported_provider(prefix)
        || is_specialist_prefix(prefix)
        || matches!(prefix, "gail" | "gateway")
}

fn split_routed_model(model: &str) -> Option<(String, String)> {
    for delimiter in ['/', ':'] {
        if let Some((prefix, routed_model)) = model.split_once(delimiter) {
            let normalized = prefix.trim().to_ascii_lowercase();
            let routed_model = routed_model.trim();
            if is_known_route_prefix(normalized.as_str()) && !routed_model.is_empty() {
                return Some((normalized, routed_model.to_string()));
            }
        }
    }
    None
}

fn is_gail_auto_model(model: &str) -> bool {
    matches!(model.trim(), "gail-auto" | "auto" | "orchestrated")
}

fn specialist_system_suffix(label: &str) -> String {
    let label = label.trim();
    if label.is_empty() || label.eq_ignore_ascii_case("auto") {
        "Use Gail's neuromorphic specialist context where it materially improves the answer."
            .to_string()
    } else {
        format!(
            "Use Gail's neuromorphic specialist context and bias towards the `{label}` specialist when it is relevant."
        )
    }
}

fn openai_model_cards(service: &GailService) -> Vec<Value> {
    let created = current_unix_timestamp();
    let mut cards = Vec::new();
    let mut seen = HashSet::new();

    push_openai_model_card(
        &mut cards,
        &mut seen,
        "gail-auto".to_string(),
        created,
        json!({
            "routing": "orchestrated",
            "kind": "gateway",
            "provider": "gail",
        }),
    );

    let mut plain_model_counts = HashMap::new();
    for profile in &service.config().providers {
        if let Some(model) = profile
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            *plain_model_counts
                .entry(model.to_ascii_lowercase())
                .or_insert(0usize) += 1;
        }
    }

    for profile in &service.config().providers {
        let provider = normalize_provider_type(profile.provider_type.as_str());
        let model = profile
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("default");
        let provider_model_id = format!("{provider}/{model}");
        push_openai_model_card(
            &mut cards,
            &mut seen,
            provider_model_id,
            created,
            json!({
                "routing": "explicit",
                "provider": provider,
                "profile": profile.name,
                "preferred": profile.preferred,
                "specialties": profile.specialties,
                "roles": profile.roles,
                "base_url": profile.base_url,
            }),
        );

        if model != "default"
            && plain_model_counts
                .get(&model.to_ascii_lowercase())
                .copied()
                .unwrap_or_default()
                == 1
        {
            push_openai_model_card(
                &mut cards,
                &mut seen,
                model.to_string(),
                created,
                json!({
                    "routing": "explicit",
                    "provider": provider,
                    "profile": profile.name,
                    "preferred": profile.preferred,
                }),
            );
        }
    }

    for specialist in &service.config().specialists {
        let prefix = if specialist.engine_type.eq_ignore_ascii_case("aarnn") {
            "aarnn"
        } else {
            "specialist"
        };
        let slug = slugify_model_segment(&specialist.name);
        let model_id = format!(
            "{prefix}/{}",
            if slug.is_empty() {
                "auto"
            } else {
                slug.as_str()
            }
        );
        push_openai_model_card(
            &mut cards,
            &mut seen,
            model_id,
            created,
            json!({
                "routing": "specialist",
                "engine_name": specialist.name,
                "engine_type": specialist.engine_type,
                "roles": specialist.roles,
                "specialties": specialist.specialties,
            }),
        );
    }

    cards
}

fn push_openai_model_card(
    cards: &mut Vec<Value>,
    seen: &mut HashSet<String>,
    id: String,
    created: u64,
    metadata: Value,
) {
    if !seen.insert(id.clone()) {
        return;
    }
    cards.push(json!({
        "id": id,
        "object": "model",
        "created": created,
        "owned_by": "gail",
        "metadata": metadata,
    }));
}

fn slugify_model_segment(value: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in value.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn chunk_text_for_streaming(text: &str, target_chars: usize) -> Vec<String> {
    let chunk_size = target_chars.max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut count = 0usize;
    for ch in text.chars() {
        current.push(ch);
        count += 1;
        if count >= chunk_size {
            chunks.push(std::mem::take(&mut current));
            count = 0;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn openai_chat_completion_stream(
    public_model: String,
    response: CompletionResponse,
) -> Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = format!("chatcmpl_{}", response.request_id);
    let created = current_unix_timestamp();
    let chunks = chunk_text_for_streaming(&response.text, 56);
    let mut events: Vec<std::result::Result<Event, Infallible>> = Vec::new();

    let role_chunk = json!({
        "id": id.clone(),
        "object": "chat.completion.chunk",
        "created": created,
        "model": public_model.clone(),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null,
            }
        ]
    });
    events.push(Ok(Event::default().data(role_chunk.to_string())));

    for chunk in chunks {
        let content_chunk = json!({
            "id": id.clone(),
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_model.clone(),
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": Value::Null,
                }
            ]
        });
        events.push(Ok(Event::default().data(content_chunk.to_string())));
    }

    let terminal_chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": openai_usage_body(response.usage.as_ref()),
        "gail": gail_response_body(&response),
    });
    events.push(Ok(Event::default().data(terminal_chunk.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(events))
}

fn openai_responses_stream(
    public_model: String,
    response: CompletionResponse,
) -> Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = format!("resp_{}", response.request_id);
    let created = current_unix_timestamp();
    let chunks = chunk_text_for_streaming(&response.text, 56);
    let mut events: Vec<std::result::Result<Event, Infallible>> = Vec::new();

    let created_event = json!({
        "id": id.clone(),
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": public_model.clone(),
    });
    events.push(Ok(Event::default()
        .event("response.created")
        .data(created_event.to_string())));

    for chunk in chunks {
        let delta_event = json!({
            "id": id,
            "delta": chunk,
            "output_index": 0,
            "content_index": 0,
        });
        events.push(Ok(Event::default()
            .event("response.output_text.delta")
            .data(delta_event.to_string())));
    }

    let output_done = json!({
        "id": id,
        "text": response.text.clone(),
        "output_index": 0,
        "content_index": 0,
    });
    events.push(Ok(Event::default()
        .event("response.output_text.done")
        .data(output_done.to_string())));

    let completed = openai_responses_body(&public_model, &response);
    events.push(Ok(Event::default()
        .event("response.completed")
        .data(completed.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(events))
}

fn synthesize_openai_tool_call(
    request_id: &str,
    text: &str,
    context: &OpenAIToolContext,
) -> Option<Value> {
    let parsed = extract_tool_call_value(text).or_else(|| extract_json_value(text));
    let (tool_name, arguments) = match parsed {
        Some(Value::Object(object)) => tool_call_from_object(object, context)?,
        _ if text.contains("<finish>") || text.contains("</finish>") => {
            ("finish".to_string(), json!({}))
        }
        _ => return None,
    };
    if !context.contains(&tool_name) {
        return None;
    }
    let arguments = match arguments {
        Value::Object(_) => arguments,
        Value::String(raw) => serde_json::from_str::<Value>(&raw)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({})),
        Value::Null => json!({}),
        _ => json!({ "value": arguments }),
    };
    let arguments_text = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());
    Some(json!({
        "id": format!("call_{}_0", sanitize_openai_id_fragment(request_id)),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments_text,
        }
    }))
}

fn tool_call_from_object(
    object: serde_json::Map<String, Value>,
    context: &OpenAIToolContext,
) -> Option<(String, Value)> {
    if let Some(tool_name) = object.get("tool_name").and_then(Value::as_str) {
        let arguments = object
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        return Some((tool_name.to_string(), arguments));
    }
    if let Some(function) = object.get("function").and_then(Value::as_object)
        && let Some(tool_name) = function.get("name").and_then(Value::as_str)
    {
        let arguments = function
            .get("arguments")
            .cloned()
            .or_else(|| object.get("arguments").cloned())
            .unwrap_or_else(|| json!({}));
        return Some((tool_name.to_string(), arguments));
    }
    if let Some(tool_name) = object.get("name").and_then(Value::as_str) {
        let arguments = object
            .get("parameters")
            .cloned()
            .or_else(|| object.get("arguments").cloned())
            .unwrap_or_else(|| json!({}));
        return Some((tool_name.to_string(), arguments));
    }
    if let Some(tool_calls) = object.get("tool_calls").and_then(Value::as_array)
        && let Some(Value::Object(first)) = tool_calls.first()
    {
        return tool_call_from_object(first.clone(), context);
    }
    if object.is_empty() && context.contains("finish") {
        return Some(("finish".to_string(), json!({})));
    }
    if (object.contains_key("team_name") || object.contains_key("current_results"))
        && context.contains("finish")
    {
        return Some(("finish".to_string(), json!({})));
    }
    if object.contains_key("agent_name") && context.contains("run_agent") {
        return Some(("run_agent".to_string(), Value::Object(object)));
    }
    if (object.contains_key("debator_agent_names") || object.contains_key("judge_agent_name"))
        && context.contains("run_debate")
    {
        return Some(("run_debate".to_string(), Value::Object(object)));
    }
    let first_name = context.first_name()?;
    if context.tool_names.len() == 1 {
        return Some((first_name.to_string(), Value::Object(object)));
    }
    None
}

fn normalize_manager_tool_call_text(text: &str) -> Option<String> {
    let (value, already_transformed) = match extract_tool_call_value(text) {
        Some(value) => (value, true),
        None => (extract_json_value(text)?, false),
    };
    let Value::Object(mut object) = value else {
        return None;
    };
    let existing_tool_name = object
        .get("tool_name")
        .and_then(Value::as_str)
        .map(str::to_string);
    if !already_transformed
        && existing_tool_name.is_some()
        && object.get("arguments").is_some_and(Value::is_object)
    {
        return None;
    }

    let tool_name = object
        .remove("tool_name")
        .and_then(|value| value.as_str().map(str::to_string))
        .or_else(|| {
            object
                .remove("name")
                .and_then(|value| value.as_str().map(str::to_string))
        })
        .or_else(|| {
            if object.contains_key("agent_name") {
                Some("run_agent".to_string())
            } else if object.contains_key("debator_agent_names")
                || object.contains_key("judge_agent_name")
            {
                Some("run_debate".to_string())
            } else {
                None
            }
        })?;

    let arguments = match object.remove("arguments") {
        Some(Value::Object(arguments)) => Value::Object(arguments),
        Some(Value::String(raw)) => serde_json::from_str::<Value>(&raw)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({})),
        Some(Value::Null) | None => Value::Object(object),
        Some(value) => json!({ "value": value }),
    };
    let arguments = match arguments {
        Value::Object(mut arguments) if arguments.contains_key("parameters") => arguments
            .remove("parameters")
            .filter(Value::is_object)
            .unwrap_or(Value::Object(arguments)),
        Value::Object(mut arguments) if arguments.len() == 1 && arguments.contains_key("value") => {
            arguments
                .remove("value")
                .unwrap_or(Value::Object(arguments))
        }
        value => value,
    };

    serde_json::to_string(&json!({
        "tool_name": tool_name,
        "arguments": arguments,
    }))
    .ok()
}

fn extract_tool_call_value(text: &str) -> Option<Value> {
    extract_minimax_tool_call_value(text).or_else(|| extract_bracket_tool_call_value(text))
}

fn extract_minimax_tool_call_value(text: &str) -> Option<Value> {
    let invoke_start = text.find("<invoke")?;
    let invoke_tag_end = text[invoke_start..].find('>')? + invoke_start;
    let invoke_tag = &text[invoke_start..=invoke_tag_end];
    let tool_name = extract_xml_attr(invoke_tag, "name")?;
    let invoke_body_end = text[invoke_tag_end + 1..].find("</invoke>")? + invoke_tag_end + 1;
    let body = &text[invoke_tag_end + 1..invoke_body_end];
    let mut arguments = serde_json::Map::new();
    let mut cursor = body;
    while let Some(parameter_start) = cursor.find("<parameter") {
        let parameter_tag_end = cursor[parameter_start..].find('>')? + parameter_start;
        let parameter_tag = &cursor[parameter_start..=parameter_tag_end];
        let parameter_name = extract_xml_attr(parameter_tag, "name")?;
        let value_start = parameter_tag_end + 1;
        let value_end = cursor[value_start..].find("</parameter>")? + value_start;
        let value = cursor[value_start..value_end].trim();
        arguments.insert(parameter_name, Value::String(value.to_string()));
        cursor = &cursor[value_end + "</parameter>".len()..];
    }
    Some(json!({
        "tool_name": tool_name,
        "arguments": arguments,
    }))
}

fn extract_xml_attr(tag: &str, name: &str) -> Option<String> {
    let pattern = format!("{name}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn extract_bracket_tool_call_value(text: &str) -> Option<Value> {
    let start = text.find("[TOOL_CALL]")? + "[TOOL_CALL]".len();
    let end = text[start..].find("[/TOOL_CALL]")? + start;
    let body = text[start..end].trim();
    let tool_name = extract_arrow_field(body, "tool")?;
    let args_body = extract_braced_arrow_field(body, "args");
    let mut arguments = serde_json::Map::new();
    if let Some(args_body) = args_body {
        for (name, value) in extract_cli_style_args(args_body.as_str()) {
            arguments.insert(name, Value::String(value));
        }
        for (name, value) in extract_arrow_args(args_body.as_str()) {
            arguments.entry(name).or_insert(Value::String(value));
        }
    }
    Some(json!({
        "tool_name": tool_name,
        "arguments": arguments,
    }))
}

fn extract_arrow_field(text: &str, name: &str) -> Option<String> {
    let pattern = format!("{name} =>");
    let start = text.find(&pattern)? + pattern.len();
    let rest = text[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    let value = rest[..end].trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn extract_braced_arrow_field(text: &str, name: &str) -> Option<String> {
    let pattern = format!("{name} =>");
    let start = text.find(&pattern)? + pattern.len();
    let rest = text[start..].trim_start();
    let inner_start = rest.find('{')? + 1;
    let inner_end = rest[inner_start..].rfind('}')? + inner_start;
    Some(rest[inner_start..inner_end].trim().to_string())
}

fn extract_cli_style_args(text: &str) -> Vec<(String, String)> {
    let mut args = Vec::new();
    let mut tokens = text.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        let Some(name) = token.strip_prefix("--") else {
            continue;
        };
        let Some(value) = tokens.next() else {
            break;
        };
        args.push((
            name.trim_matches(|ch: char| ch == ',' || ch == '}')
                .to_string(),
            value
                .trim_matches(|ch: char| ch == '"' || ch == ',' || ch == '}')
                .to_string(),
        ));
    }
    args
}

fn extract_arrow_args(text: &str) -> Vec<(String, String)> {
    let mut args = Vec::new();
    for part in text.split(',') {
        let Some((name, value)) = part.split_once("=>") else {
            continue;
        };
        let name = name
            .trim()
            .trim_start_matches("--")
            .trim_matches(|ch: char| ch == '"' || ch == '{' || ch == '}');
        let value = value
            .trim()
            .trim_matches(|ch: char| ch == '"' || ch == '{' || ch == '}');
        if !name.is_empty() && !value.is_empty() {
            args.push((name.to_string(), value.to_string()));
        }
    }
    args
}

fn extract_json_value(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    serde_json::from_str::<Value>(trimmed).ok().or_else(|| {
        let start = trimmed.find(['{', '['])?;
        let end = trimmed.rfind(['}', ']'])?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Value>(&trimmed[start..=end]).ok()
    })
}

fn sanitize_openai_id_fragment(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .collect::<String>()
}

fn openai_chat_completion_body(
    public_model: &str,
    response: &CompletionResponse,
    tool_context: Option<&OpenAIToolContext>,
) -> Value {
    let synthesized_tool_call = tool_context.and_then(|context| {
        synthesize_openai_tool_call(&response.request_id, response.text.as_str(), context)
    });
    let (message, finish_reason) = match synthesized_tool_call {
        Some(tool_call) => (
            json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [tool_call],
            }),
            "tool_calls",
        ),
        None => (
            json!({
                "role": "assistant",
                "content": response.text,
            }),
            "stop",
        ),
    };
    json!({
        "id": format!("chatcmpl_{}", response.request_id),
        "object": "chat.completion",
        "created": current_unix_timestamp(),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": openai_usage_body(response.usage.as_ref()),
        "gail": gail_response_body(response),
    })
}

fn openai_responses_body(public_model: &str, response: &CompletionResponse) -> Value {
    let mut output = vec![json!({
        "type": "text",
        "text": response.text,
    })];
    if let Some(reasoning) = extract_reasoning_summary(response.raw.as_ref()) {
        output.push(json!({
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": reasoning,
                }
            ]
        }));
    }

    json!({
        "id": format!("resp_{}", response.request_id),
        "object": "response",
        "created_at": current_unix_timestamp(),
        "status": "completed",
        "model": public_model,
        "output": output,
        "output_text": response.text,
        "usage": openai_usage_body(response.usage.as_ref()),
        "gail": gail_response_body(response),
    })
}

fn extract_reasoning_summary(raw: Option<&Value>) -> Option<String> {
    let raw = raw?;
    raw.get("output")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                (item.get("type").and_then(Value::as_str) == Some("reasoning"))
                    .then(|| {
                        item.get("summary")
                            .and_then(Value::as_array)
                            .and_then(|summaries| summaries.first())
                            .and_then(|summary| summary.get("text").and_then(Value::as_str))
                            .map(ToOwned::to_owned)
                    })
                    .flatten()
            })
        })
}

fn openai_usage_body(usage: Option<&crate::models::TokenUsage>) -> Value {
    let Some(usage) = usage else {
        return json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        });
    };
    let mut payload = json!({
        "prompt_tokens": usage.prompt.unwrap_or(0),
        "completion_tokens": usage.completion.unwrap_or(0),
        "total_tokens": usage.total.unwrap_or_else(|| usage.prompt.unwrap_or(0) + usage.completion.unwrap_or(0)),
    });
    if let Some(cached) = usage.cached {
        payload["prompt_tokens_details"] = json!({"cached_tokens": cached});
    }
    if let Some(cost) = usage.cost.as_ref() {
        payload["cost"] = json!({
            "amount": cost.amount,
            "currency": cost.currency,
        });
    }
    payload
}

fn gail_response_body(response: &CompletionResponse) -> Value {
    let mut payload = json!({
        "provider": response.provider,
        "resolved_model": response.model,
        "request_id": response.request_id,
        "latency_ms": response.latency_ms,
    });
    if let Some(trace) = response.trace.as_ref() {
        payload["trace"] = serde_json::to_value(trace).unwrap_or(Value::Null);
    }
    payload
}

fn degraded_chat_completion_for_upstream_error(
    request: &OpenAIChatCompletionRequest,
    response_schema_context: &OpenAIResponseSchemaContext,
    error: &GailError,
) -> Option<(String, CompletionResponse)> {
    let reason = transient_openai_fallback_reason(error)?;
    let text = if let Some(schema) =
        schema_value_from_response_format(request.response_format.as_ref())
        && let Some(mut value) = minimal_json_from_schema(schema)
    {
        if response_format_targets_execution_plan(request.response_format.as_ref())
            && execution_plan_is_blank(&value)
            && let Some(cache_key) = execution_plan_cache_key_from_chat_request(request)
            && let Some(cached) = load_cached_execution_plan(cache_key.as_str(), schema)
        {
            value = cached;
        }
        serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
    } else if response_schema_context.manager_tool_call {
        json!({
            "tool_name": "finish",
            "arguments": {
                "status": "degraded",
                "decision": "hold",
                "action": "hold",
                "should_trade": false,
                "reason": reason,
            }
        })
        .to_string()
    } else if chat_request_expects_json(request) {
        json!({
            "status": "degraded",
            "decision": "hold",
            "action": "hold",
            "signal": "neutral",
            "confidence": 0.0,
            "should_trade": false,
            "orders": [],
            "trades": [],
            "risk": "provider_unavailable",
            "reason": reason,
        })
        .to_string()
    } else {
        format!(
            "HOLD / NO_TRADE: Gail returned a degraded response because upstream providers were unavailable. Reason: {reason}."
        )
    };

    let response = degraded_openai_completion_response(text, reason);
    Some((request.model.trim().to_string(), response))
}

fn degraded_responses_completion_for_upstream_error(
    request: &OpenAIResponseRequest,
    error: &GailError,
) -> Option<(String, CompletionResponse)> {
    let reason = transient_openai_fallback_reason(error)?;
    let text = if let Some(schema) = schema_value_from_response_format(
        request.text.as_ref().and_then(|text| text.format.as_ref()),
    ) && let Some(mut value) = minimal_json_from_schema(schema)
    {
        if response_format_targets_execution_plan(
            request.text.as_ref().and_then(|text| text.format.as_ref()),
        ) && execution_plan_is_blank(&value)
            && let Some(cache_key) = execution_plan_cache_key_from_responses_request(request)
            && let Some(cached) = load_cached_execution_plan(cache_key.as_str(), schema)
        {
            value = cached;
        }
        serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
    } else if responses_request_expects_json(request) {
        json!({
            "status": "degraded",
            "decision": "hold",
            "action": "hold",
            "signal": "neutral",
            "confidence": 0.0,
            "should_trade": false,
            "orders": [],
            "trades": [],
            "risk": "provider_unavailable",
            "reason": reason,
        })
        .to_string()
    } else {
        format!(
            "HOLD / NO_TRADE: Gail returned a degraded response because upstream providers were unavailable. Reason: {reason}."
        )
    };

    let response = degraded_openai_completion_response(text, reason);
    Some((request.model.trim().to_string(), response))
}

fn transient_openai_fallback_reason(error: &GailError) -> Option<String> {
    match error {
        GailError::Upstream {
            message,
            status,
            quota,
            timeout,
            ..
        } => {
            if *quota {
                return None;
            }
            let status_is_transient = status
                .map(|code| code.as_u16())
                .is_some_and(|code| matches!(code, 502 | 503 | 504));
            if *timeout || status_is_transient || message_indicates_transient_upstream(message) {
                return Some(truncate_openai_reason(message));
            }
            None
        }
        GailError::Reqwest(err) => {
            let text = err.to_string();
            if err.is_timeout() || message_indicates_transient_upstream(&text) {
                Some(truncate_openai_reason(text))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn message_indicates_transient_upstream(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("local ollama request queue is saturated")
        || lowered.contains("local model service is saturated")
        || lowered.contains("adaptive backoff")
        || lowered.contains("upstream error")
        || lowered.contains("gateway timeout")
        || lowered.contains("bad gateway")
        || lowered.contains("connection error")
        || lowered.contains("error sending request")
        || lowered.contains("timed out")
        || lowered.contains("timeout")
        || lowered.contains("status 502")
        || lowered.contains("status 503")
        || lowered.contains("status 504")
        || lowered.contains("http 502")
        || lowered.contains("http 503")
        || lowered.contains("http 504")
}

fn chat_request_expects_json(request: &OpenAIChatCompletionRequest) -> bool {
    if response_format_system_hint(request.response_format.as_ref()).is_some()
        || tool_call_system_hint(request.tools.as_ref()).is_some()
    {
        return true;
    }
    let mut context = String::new();
    if let Some(instructions) = request.instructions.as_deref() {
        context.push_str(instructions);
        context.push('\n');
    }
    for message in &request.messages {
        context.push_str(&message.flattened_text());
        context.push('\n');
    }
    let lowered = context.to_ascii_lowercase();
    lowered.contains("json")
        || lowered.contains("managertoolcall")
        || (lowered.contains("tool_name") && lowered.contains("arguments"))
}

fn responses_request_expects_json(request: &OpenAIResponseRequest) -> bool {
    if response_format_system_hint(request.text.as_ref().and_then(|text| text.format.as_ref()))
        .is_some()
    {
        return true;
    }
    let lowered = request.input.to_string().to_ascii_lowercase();
    lowered.contains("json")
}

fn maybe_cache_execution_plan_from_chat_success(
    request: &OpenAIChatCompletionRequest,
    response: &CompletionResponse,
) {
    if response.model.eq_ignore_ascii_case("degraded_safety") {
        return;
    }
    let Some(format) = request.response_format.as_ref() else {
        return;
    };
    maybe_cache_execution_plan_response(
        response.text.as_str(),
        Some(format),
        execution_plan_cache_key_from_chat_request(request),
    );
}

fn maybe_cache_execution_plan_from_responses_success(
    request: &OpenAIResponseRequest,
    response: &CompletionResponse,
) {
    if response.model.eq_ignore_ascii_case("degraded_safety") {
        return;
    }
    let Some(format) = request.text.as_ref().and_then(|text| text.format.as_ref()) else {
        return;
    };
    maybe_cache_execution_plan_response(
        response.text.as_str(),
        Some(format),
        execution_plan_cache_key_from_responses_request(request),
    );
}

fn maybe_cache_execution_plan_response(
    response_text: &str,
    format: Option<&OpenAIResponseFormat>,
    cache_key: Option<String>,
) {
    if !response_format_targets_execution_plan(format) {
        return;
    }
    let Some(cache_key) = cache_key else {
        return;
    };
    let Some(schema) = schema_value_from_response_format(format) else {
        return;
    };
    let Some(value) = extract_json_value(response_text) else {
        return;
    };
    if execution_plan_is_blank(&value) {
        return;
    }
    if !execution_plan_has_steps(&value) {
        return;
    }
    if !json_matches_object_schema_requirements(schema, &value) {
        return;
    }
    store_cached_execution_plan(cache_key, value);
}

fn schema_value_from_response_format(format: Option<&OpenAIResponseFormat>) -> Option<&Value> {
    let format = format?;
    let schema = format.json_schema.as_ref()?;
    if let Some(schema_value) = schema.get("schema") {
        Some(schema_value)
    } else {
        Some(schema)
    }
}

fn minimal_json_from_schema(schema: &Value) -> Option<Value> {
    minimal_json_from_schema_inner(schema, schema, 0)
}

fn minimal_json_from_schema_inner(root: &Value, schema: &Value, depth: usize) -> Option<Value> {
    if depth > 24 {
        return Some(Value::Null);
    }

    if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
        && reference.starts_with("#/")
        && let Some(target) = root.pointer(&reference[1..])
    {
        return minimal_json_from_schema_inner(root, target, depth + 1);
    }

    if let Some(value) = schema.get("const") {
        return Some(value.clone());
    }
    if let Some(value) = schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
    {
        return Some(value.clone());
    }
    if let Some(default) = schema.get("default") {
        return Some(default.clone());
    }
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array)
        && let Some(first) = any_of.first()
    {
        return minimal_json_from_schema_inner(root, first, depth + 1);
    }
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array)
        && let Some(first) = one_of.first()
    {
        return minimal_json_from_schema_inner(root, first, depth + 1);
    }
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array)
        && let Some(first) = all_of.first()
    {
        return minimal_json_from_schema_inner(root, first, depth + 1);
    }

    let mut selected_type = schema
        .get("type")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if selected_type.is_none()
        && let Some(types) = schema.get("type").and_then(Value::as_array)
    {
        selected_type = types
            .iter()
            .filter_map(Value::as_str)
            .find(|kind| *kind != "null")
            .map(ToOwned::to_owned);
    }
    if selected_type.is_none()
        && (schema.get("properties").is_some() || schema.get("required").is_some())
    {
        selected_type = Some("object".to_string());
    }

    match selected_type.as_deref().unwrap_or("object") {
        "object" => {
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required = schema
                .get("required")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            let mut object = serde_json::Map::new();
            for key in required.iter().filter_map(Value::as_str) {
                if let Some(prop_schema) = properties.get(key) {
                    if let Some(value) =
                        minimal_json_from_schema_inner(root, prop_schema, depth + 1)
                    {
                        object.insert(key.to_string(), value);
                    }
                } else if key.eq_ignore_ascii_case("steps") {
                    object.insert(key.to_string(), Value::Array(Vec::new()));
                }
            }
            Some(Value::Object(object))
        }
        "array" => Some(Value::Array(Vec::new())),
        "boolean" => Some(Value::Bool(false)),
        "integer" => Some(Value::from(0)),
        "number" => Some(Value::from(0.0)),
        "string" => Some(Value::String(String::new())),
        "null" => Some(Value::Null),
        _ => Some(Value::Null),
    }
}

fn response_format_targets_execution_plan(format: Option<&OpenAIResponseFormat>) -> bool {
    let Some(format) = format else {
        return false;
    };
    let Some(json_schema) = format.json_schema.as_ref() else {
        return false;
    };
    if json_schema
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| name.eq_ignore_ascii_case("executionplan"))
    {
        return true;
    }
    let schema = json_schema.get("schema").unwrap_or(json_schema);
    schema_value_looks_like_execution_plan(schema)
}

fn schema_value_looks_like_execution_plan(schema: &Value) -> bool {
    if schema
        .get("title")
        .and_then(Value::as_str)
        .is_some_and(|title| title.eq_ignore_ascii_case("executionplan"))
    {
        return true;
    }
    let has_steps_property = schema
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| properties.contains_key("steps"));
    let required_steps = schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .any(|key| key == "steps")
        });
    has_steps_property && required_steps
}

fn execution_plan_steps_len(value: &Value) -> Option<usize> {
    value
        .as_object()
        .and_then(|object| object.get("steps"))
        .and_then(Value::as_array)
        .map(Vec::len)
}

fn execution_plan_is_blank(value: &Value) -> bool {
    execution_plan_steps_len(value).is_some_and(|len| len == 0)
}

fn execution_plan_has_steps(value: &Value) -> bool {
    execution_plan_steps_len(value).is_some_and(|len| len > 0)
}

fn json_matches_object_schema_requirements(schema: &Value, value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            if !object.contains_key(key) {
                return false;
            }
        }
    }

    if schema
        .get("additionalProperties")
        .and_then(Value::as_bool)
        .is_some_and(|enabled| !enabled)
        && let Some(properties) = schema.get("properties").and_then(Value::as_object)
    {
        for key in object.keys() {
            if !properties.contains_key(key) {
                return false;
            }
        }
    }

    object.get("steps").is_none_or(Value::is_array)
}

fn execution_plan_cache_key_from_chat_request(
    request: &OpenAIChatCompletionRequest,
) -> Option<String> {
    if !response_format_targets_execution_plan(request.response_format.as_ref()) {
        return None;
    }
    execution_plan_cache_key_from_context(openai_chat_request_context_text(request).as_str())
}

fn execution_plan_cache_key_from_responses_request(
    request: &OpenAIResponseRequest,
) -> Option<String> {
    if !response_format_targets_execution_plan(
        request.text.as_ref().and_then(|text| text.format.as_ref()),
    ) {
        return None;
    }
    execution_plan_cache_key_from_context(openai_responses_request_context_text(request).as_str())
}

fn openai_chat_request_context_text(request: &OpenAIChatCompletionRequest) -> String {
    let mut context = String::new();
    if let Some(instructions) = request.instructions.as_deref() {
        context.push_str(instructions);
        context.push('\n');
    }
    for message in &request.messages {
        context.push_str(&message.flattened_text());
        context.push('\n');
    }
    context
}

fn openai_responses_request_context_text(request: &OpenAIResponseRequest) -> String {
    let mut context = String::new();
    if let Some(instructions) = request.instructions.as_deref() {
        context.push_str(instructions);
        context.push('\n');
    }
    append_json_string_values(&request.input, &mut context);
    context
}

fn append_json_string_values(value: &Value, context: &mut String) {
    match value {
        Value::String(text) => {
            context.push_str(text);
            context.push('\n');
        }
        Value::Array(items) => {
            for item in items {
                append_json_string_values(item, context);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                append_json_string_values(item, context);
            }
        }
        _ => {}
    }
}

fn execution_plan_cache_key_from_context(context: &str) -> Option<String> {
    let team = extract_labeled_line_value(context, "Team:")?;
    let agents_section = extract_labeled_section(
        context,
        "Agents:",
        &["Relations:", "Initial Data:", "Instructions:"],
    )?;
    let mut agent_names = extract_named_values_from_section(agents_section, "name")
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    agent_names.sort();
    agent_names.dedup();
    if agent_names.is_empty() {
        return None;
    }
    Some(format!(
        "team={}|agents={}",
        team.trim().to_ascii_lowercase(),
        agent_names.join(",")
    ))
}

fn extract_labeled_line_value(text: &str, label: &str) -> Option<String> {
    let start = text.find(label)?;
    let remainder = &text[start + label.len()..];
    let value = remainder
        .split(['\n', '\r', '\\', '"'])
        .next()
        .unwrap_or_default()
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn extract_labeled_section<'a>(
    text: &'a str,
    label: &str,
    end_markers: &[&str],
) -> Option<&'a str> {
    let start = text.find(label)?;
    let after_label = &text[start + label.len()..];
    let mut end = after_label.len();
    for marker in end_markers {
        if let Some(position) = after_label.find(marker) {
            end = end.min(position);
        }
    }
    Some(after_label[..end].trim())
}

fn extract_named_values_from_section(section: &str, field_name: &str) -> Vec<String> {
    let needle = format!("\"{field_name}\"");
    section
        .split(needle.as_str())
        .skip(1)
        .filter_map(|tail| {
            let (_, after_colon) = tail.split_once(':')?;
            let (_, after_start_quote) = after_colon.split_once('"')?;
            let (value, _) = after_start_quote.split_once('"')?;
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        })
        .collect()
}

fn load_cached_execution_plan(cache_key: &str, schema: &Value) -> Option<Value> {
    let cached = {
        let cache = EXECUTION_PLAN_CACHE.lock().ok()?;
        cache.get(cache_key).cloned()?
    };
    if !execution_plan_has_steps(&cached) {
        return None;
    }
    if !json_matches_object_schema_requirements(schema, &cached) {
        return None;
    }
    Some(cached)
}

fn store_cached_execution_plan(cache_key: String, execution_plan: Value) {
    let Ok(mut cache) = EXECUTION_PLAN_CACHE.lock() else {
        return;
    };
    if cache.len() >= EXECUTION_PLAN_CACHE_MAX_ENTRIES
        && !cache.contains_key(cache_key.as_str())
        && let Some(evicted_key) = cache.keys().next().cloned()
    {
        cache.remove(evicted_key.as_str());
    }
    cache.insert(cache_key, execution_plan);
}

fn degraded_openai_completion_response(text: String, reason: String) -> CompletionResponse {
    let request_id = format!("degraded_{}", Uuid::new_v4().simple());
    CompletionResponse {
        request_id,
        text,
        provider: "gail".to_string(),
        model: "degraded_safety".to_string(),
        latency_ms: 0,
        usage: None,
        trace: None,
        raw: Some(json!({
            "selected_source": "degraded_openai_gateway_fallback",
            "reason": reason,
            "safety_action": "hold_no_trade",
        })),
    }
}

fn truncate_openai_reason(reason: impl AsRef<str>) -> String {
    reason.as_ref().trim().chars().take(220).collect::<String>()
}

fn openai_error_response(error: GailError) -> Response {
    let status = openai_error_status(&error);
    let body = json!({
        "error": {
            "message": openai_error_message(&error),
            "type": openai_error_type(&error),
            "param": Value::Null,
            "code": openai_error_code(&error),
        }
    });
    (status, Json(body)).into_response()
}

fn openai_error_status(error: &GailError) -> StatusCode {
    match error {
        GailError::BadRequest(_) | GailError::Multipart(_) => StatusCode::BAD_REQUEST,
        GailError::Unauthorized => StatusCode::UNAUTHORIZED,
        GailError::NotFound(_) => StatusCode::NOT_FOUND,
        GailError::InvalidConfig(_) => StatusCode::INTERNAL_SERVER_ERROR,
        GailError::Upstream { quota: true, .. } => StatusCode::TOO_MANY_REQUESTS,
        GailError::Upstream { timeout: true, .. } => StatusCode::GATEWAY_TIMEOUT,
        GailError::Upstream {
            status: Some(status),
            ..
        } => *status,
        GailError::Upstream { .. } => StatusCode::BAD_GATEWAY,
        GailError::Io(_) | GailError::Json(_) | GailError::Yaml(_) | GailError::Reqwest(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

fn openai_error_message(error: &GailError) -> String {
    match error {
        GailError::BadRequest(message)
        | GailError::NotFound(message)
        | GailError::InvalidConfig(message) => message.to_string(),
        GailError::Unauthorized => "unauthorized".to_string(),
        GailError::Multipart(message) => message.clone(),
        GailError::Upstream { message, .. } => message.clone(),
        _ => error.to_string(),
    }
}

fn openai_error_type(error: &GailError) -> &'static str {
    match error {
        GailError::Unauthorized => "authentication_error",
        GailError::BadRequest(_) | GailError::Multipart(_) | GailError::NotFound(_) => {
            "invalid_request_error"
        }
        GailError::Upstream { quota: true, .. } => "rate_limit_error",
        GailError::Upstream { .. } => "api_error",
        _ => "server_error",
    }
}

fn openai_error_code(error: &GailError) -> Option<String> {
    match error {
        GailError::Unauthorized => Some("unauthorized".to_string()),
        GailError::Upstream { provider, .. } => Some(provider.clone()),
        _ => None,
    }
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request};
    use tower::ServiceExt;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string_contains, method, path},
    };

    use crate::config::{ApiTokenConfig, GailConfig, ProviderProfile, SpecialistProfile};

    async fn test_service_with_config(mut config: GailConfig) -> GailService {
        crate::providers::ollama::reset_test_runtime_state().await;
        config.security.allow_unauthenticated_health = false;
        config.storage.metrics_path = std::env::temp_dir()
            .join(format!(
                "gail-test-provider-metrics-{}.json",
                uuid::Uuid::new_v4()
            ))
            .to_string_lossy()
            .to_string();
        config.storage.adaptive_schema_path = std::env::temp_dir()
            .join(format!(
                "gail-test-adaptive-schema-{}.json",
                uuid::Uuid::new_v4()
            ))
            .to_string_lossy()
            .to_string();
        config.storage.api_issues_path = std::env::temp_dir()
            .join(format!(
                "gail-test-api-issues-{}.json",
                uuid::Uuid::new_v4()
            ))
            .to_string_lossy()
            .to_string();
        config.storage.postgres_dsn = None;
        config.security.api_tokens.push(ApiTokenConfig {
            client_id: "test".to_string(),
            token: "secret".to_string(),
            scopes: vec!["*".to_string()],
        });
        GailService::new(config).await.expect("service")
    }

    async fn read_json(response: Response) -> Value {
        let status = response.status();
        assert!(status.is_success(), "unexpected status: {status}");
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json")
    }

    async fn read_response_json(response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let payload = serde_json::from_slice(&bytes).expect("json");
        (status, payload)
    }

    async fn read_text(response: Response) -> String {
        let status = response.status();
        assert!(status.is_success(), "unexpected status: {status}");
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        String::from_utf8(bytes.to_vec()).expect("utf8")
    }

    #[tokio::test]
    async fn health_requires_auth_when_configured() {
        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_allows_valid_bearer_token() {
        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .header("authorization", "Bearer secret")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_issues_status_exposes_registry() {
        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/status/api-issues")
                    .header("authorization", "Bearer secret")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        assert!(payload.get("api_issues").is_some());
    }

    #[tokio::test]
    async fn prometheus_metrics_expose_api_issue_gauges() {
        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header("authorization", "Bearer secret")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("response");
        let body = read_text(response).await;
        assert!(body.contains("gail_api_issues_active"));
    }

    #[tokio::test]
    async fn openai_models_lists_gateway_provider_and_specialist_models() {
        let mut config = GailConfig::default();
        config.providers.push(ProviderProfile {
            name: "OpenAIPrimary".to_string(),
            provider_type: "openai".to_string(),
            model: Some("gpt-5.3-codex".to_string()),
            preferred: true,
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            base_url: Some("https://integrate.api.nvidia.com/v1".to_string()),
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "LocalOllama".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            base_url: Some("http://ollama.local".to_string()),
            ..ProviderProfile::default()
        });
        config.specialists.push(SpecialistProfile::default());
        let app = build_router(test_service_with_config(config).await);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer secret")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let ids = payload["data"]
            .as_array()
            .expect("model data")
            .iter()
            .filter_map(|item| item.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert!(ids.contains(&"gail-auto"));
        assert!(ids.contains(&"openai/gpt-5.3-codex"));
        assert!(ids.contains(&"gpt-5.3-codex"));
        assert!(ids.contains(&"nvidia/moonshotai/kimi-k2-instruct-0905"));
        assert!(ids.contains(&"moonshotai/kimi-k2-instruct-0905"));
        assert!(ids.contains(&"ollama/llama3.2"));
        assert!(ids.iter().any(|id| id.starts_with("aarnn/")));
    }

    #[tokio::test]
    async fn openai_chat_completions_route_explicit_ollama_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "mocked answer",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {"role": "user", "content": "hello"}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;

        assert_eq!(payload["model"], "ollama/llama3.2");
        assert_eq!(payload["choices"][0]["message"]["content"], "mocked answer");
        assert_eq!(payload["gail"]["provider"], "ollama");
        assert_eq!(payload["gail"]["resolved_model"], "llama3.2");
    }

    #[tokio::test]
    async fn openai_chat_completions_explicit_ollama_saturation_returns_degraded_success_payload() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": "local Ollama request queue is saturated; backing off before retrying in 14s"
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "response_format": { "type": "json_object" },
                            "messages": [
                                {"role": "user", "content": "Return only valid JSON for this evaluation."}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("completion content");
        let degraded: Value = serde_json::from_str(content).expect("degraded json payload");

        assert_eq!(payload["model"], "ollama/llama3.2");
        assert_eq!(degraded["status"], "degraded");
        assert_eq!(degraded["action"], "hold");
        assert_eq!(degraded["should_trade"], false);
        assert_eq!(payload["gail"]["provider"], "gail");
        assert_eq!(payload["gail"]["resolved_model"], "degraded_safety");
    }

    #[tokio::test]
    async fn openai_chat_completions_explicit_ollama_saturation_respects_json_schema_fallback() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": "local Ollama request queue is saturated; backing off before retrying in 14s"
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "ExecutionPlan",
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "loop": {"type": "boolean", "default": false},
                                            "loop_condition": {
                                                "anyOf": [{"type": "string"}, {"type": "null"}]
                                            },
                                            "max_iterations": {
                                                "anyOf": [{"type": "integer"}, {"type": "null"}]
                                            },
                                            "steps": {
                                                "type": "array",
                                                "items": {"type": "object"}
                                            }
                                        },
                                        "required": ["steps", "loop", "loop_condition", "max_iterations"],
                                        "additionalProperties": false
                                    }
                                }
                            },
                            "messages": [
                                {"role": "user", "content": "Return only valid execution plan JSON."}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("completion content");
        let degraded: Value = serde_json::from_str(content).expect("degraded json payload");
        let degraded_object = degraded.as_object().expect("degraded object");

        assert_eq!(payload["model"], "ollama/llama3.2");
        assert_eq!(degraded_object.len(), 4);
        assert_eq!(degraded["loop"], false);
        assert_eq!(degraded["loop_condition"], "");
        assert_eq!(degraded["max_iterations"], 0);
        assert!(degraded_object.contains_key("steps"));
        assert_eq!(
            degraded["steps"].as_array().expect("steps array").len(),
            0,
            "schema fallback should avoid non-schema keys like status/decision"
        );
        assert_eq!(payload["gail"]["provider"], "gail");
        assert_eq!(payload["gail"]["resolved_model"], "degraded_safety");
    }

    #[tokio::test]
    async fn openai_chat_completions_saturation_reuses_cached_execution_plan() {
        let success_server = MockServer::start().await;
        let success_base_url = format!("{}/chat-cache-success", success_server.uri());
        Mock::given(method("GET"))
            .and(path("/chat-cache-success/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&success_server)
            .await;
        let cached_execution_plan = json!({
            "steps": [
                {
                    "agent_name": "TechnicalAnalysisAIAgentProducer",
                    "instructions": null,
                    "wait_for": null,
                    "skip": false,
                    "step_type": "agent",
                    "debate_config": null
                }
            ],
            "loop": false,
            "loop_condition": null,
            "max_iterations": null
        });
        Mock::given(method("POST"))
            .and(path("/chat-cache-success/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": cached_execution_plan.to_string(),
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&success_server)
            .await;

        let saturated_server = MockServer::start().await;
        let saturated_base_url = format!("{}/chat-cache-saturated", saturated_server.uri());
        Mock::given(method("GET"))
            .and(path("/chat-cache-saturated/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&saturated_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat-cache-saturated/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": "local Ollama request queue is saturated; backing off before retrying in 14s"
            })))
            .mount(&saturated_server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let team_name = format!("ExecutionPlanCacheTeam{}", uuid::Uuid::new_v4().simple());
        let manager_prompt = format!(
            "Analyze the following team structure and create an execution plan:\n\
             Team: {team_name}\n\
             Agents: [{{\"name\":\"TechnicalAnalysisAIAgentProducer\",\"channel\":\"TechnicalAnalysisAIAgentChannel\"}},{{\"name\":\"SummarizationAIAgentProducer\",\"channel\":\"SummarizationAIAgentChannel\"}}]\n\
             Relations: [{{\"source\":\"TechnicalAnalysisAIAgentChannel\",\"target\":\"SummarizationAIAgentChannel\"}}]\n\
             Initial Data: {{}}\n\
             Instructions: None"
        );
        let request_payload = |base_url: String| {
            json!({
                "model": "ollama/llama3.2",
                "base_url": base_url,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ExecutionPlan",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                },
                                "loop": {"type": "boolean"},
                                "loop_condition": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "max_iterations": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                            },
                            "required": ["steps", "loop", "loop_condition", "max_iterations"],
                            "additionalProperties": false
                        }
                    }
                },
                "messages": [
                    {"role": "user", "content": manager_prompt}
                ]
            })
        };

        let first_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        request_payload(success_base_url.clone()).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("first response");
        let first_payload = read_json(first_response).await;
        let first_plan: Value = serde_json::from_str(
            first_payload["choices"][0]["message"]["content"]
                .as_str()
                .expect("first content"),
        )
        .expect("first plan json");
        assert_eq!(
            first_plan["steps"].as_array().expect("steps").len(),
            1,
            "first response should cache a working execution plan"
        );

        let second_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        request_payload(saturated_base_url.clone()).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("second response");
        let second_payload = read_json(second_response).await;
        let second_plan: Value = serde_json::from_str(
            second_payload["choices"][0]["message"]["content"]
                .as_str()
                .expect("second content"),
        )
        .expect("second plan json");

        assert_eq!(
            second_plan, first_plan,
            "degraded schema fallback should reuse the last non-empty execution plan"
        );
        assert_eq!(second_payload["gail"]["provider"], "gail");
        assert_eq!(second_payload["gail"]["resolved_model"], "degraded_safety");
    }

    #[tokio::test]
    async fn openai_chat_completions_synthesizes_tool_calls_for_octobot_tools() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "{\"agent_name\":\"SignalAIAgentProducer\"}",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {"role": "user", "content": "choose the next team tool"}
                            ],
                            "tools": [{
                                "type": "function",
                                "function": {
                                    "name": "run_agent",
                                    "description": "Run a specific agent",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "agent_name": {"type": "string"}
                                        },
                                        "required": ["agent_name"]
                                    }
                                }
                            }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let choice = &payload["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert!(choice["message"]["content"].is_null());
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "run_agent"
        );
        let args: Value = serde_json::from_str(
            choice["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .expect("arguments string"),
        )
        .expect("arguments json");
        assert_eq!(args["agent_name"], "SignalAIAgentProducer");
    }

    #[tokio::test]
    async fn openai_chat_completions_synthesizes_bracket_tool_call_for_octobot_tools() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "I will start with signals. [TOOL_CALL] {tool => \"run_agent\", args => { --agent_name \"SignalAIAgentProducer\" }} [/TOOL_CALL]",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {"role": "user", "content": "choose the next team tool"}
                            ],
                            "tools": [{
                                "type": "function",
                                "function": {
                                    "name": "run_agent",
                                    "description": "Run a specific agent",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "agent_name": {"type": "string"}
                                        },
                                        "required": ["agent_name"]
                                    }
                                }
                            }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let choice = &payload["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "run_agent"
        );
        let args: Value = serde_json::from_str(
            choice["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .expect("arguments string"),
        )
        .expect("arguments json");
        assert_eq!(args["agent_name"], "SignalAIAgentProducer");
    }

    #[tokio::test]
    async fn openai_chat_completions_wraps_bare_manager_tool_arguments() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "{\"agent_name\":\"SignalAIAgentProducer\"}",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return a ManagerToolCall JSON object with tool_name and arguments."
                                },
                                {
                                    "role": "user",
                                    "content": "choose the next run_agent call for agent_name"
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let manager_call: Value = serde_json::from_str(content).expect("manager call json");
        assert_eq!(manager_call["tool_name"], "run_agent");
        assert_eq!(
            manager_call["arguments"]["agent_name"],
            "SignalAIAgentProducer"
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_normalizes_name_parameters_manager_tool_call() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "{\"name\":\"run_agent\",\"parameters\":{\"agent_name\":\"SignalAIAgentProducer\"}}",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return a ManagerToolCall JSON object with tool_name and arguments."
                                },
                                {
                                    "role": "user",
                                    "content": "choose the next run_agent call for agent_name"
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let manager_call: Value = serde_json::from_str(content).expect("manager call json");
        assert_eq!(manager_call["tool_name"], "run_agent");
        assert_eq!(
            manager_call["arguments"]["agent_name"],
            "SignalAIAgentProducer"
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_normalizes_minimax_xml_manager_tool_call() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "<minimax:tool_call><invoke name=\"run_agent\"><parameter name=\"agent_name\">SignalAIAgentProducer</parameter></invoke></minimax:tool_call>",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return a ManagerToolCall JSON object with tool_name and arguments."
                                },
                                {
                                    "role": "user",
                                    "content": "choose the next run_agent call for agent_name"
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let manager_call: Value = serde_json::from_str(content).expect("manager call json");
        assert_eq!(manager_call["tool_name"], "run_agent");
        assert_eq!(
            manager_call["arguments"]["agent_name"],
            "SignalAIAgentProducer"
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_normalizes_bracket_manager_tool_call() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "Looking at the context, I need to coordinate the trading agents. [TOOL_CALL] {tool => \"run_agent\", args => { --agent_name \"SignalAIAgentProducer\" }} [/TOOL_CALL]",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return a ManagerToolCall JSON object with tool_name and arguments."
                                },
                                {
                                    "role": "user",
                                    "content": "choose the next run_agent call for agent_name"
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let manager_call: Value = serde_json::from_str(content).expect("manager call json");
        assert_eq!(manager_call["tool_name"], "run_agent");
        assert_eq!(
            manager_call["arguments"]["agent_name"],
            "SignalAIAgentProducer"
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_route_explicit_nvidia_models() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-nvidia",
                "model": "moonshotai/kimi-k2-instruct-0905",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "nvidia answer"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18
                }
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", server.uri())),
            ..ProviderProfile::default()
        });
        let app = build_router(test_service_with_config(config).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "nvidia/moonshotai/kimi-k2-instruct-0905",
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        assert_eq!(payload["model"], "nvidia/moonshotai/kimi-k2-instruct-0905");
        assert_eq!(payload["choices"][0]["message"]["content"], "nvidia answer");
        assert_eq!(payload["gail"]["provider"], "nvidia");
        assert_eq!(
            payload["gail"]["resolved_model"],
            "moonshotai/kimi-k2-instruct-0905"
        );
    }

    #[tokio::test]
    async fn openai_errors_preserve_nested_upstream_rate_limit_status() {
        let error = GailError::upstream(
            "gail",
            None,
            r#"nvidia upstream error: {"status":429,"title":"Too Many Requests"}"#,
        );
        let (status, payload) = read_response_json(openai_error_response(error)).await;

        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(payload["error"]["type"], "rate_limit_error");
        assert_eq!(payload["error"]["code"], "gail");
        assert_eq!(
            payload["error"]["message"],
            r#"nvidia upstream error: {"status":429,"title":"Too Many Requests"}"#
        );
    }

    #[tokio::test]
    async fn openai_errors_prefer_nested_rate_limit_over_gateway_status() {
        let error = GailError::upstream(
            "gail",
            Some(StatusCode::BAD_GATEWAY),
            r#"nvidia upstream error: {"status":429,"title":"Too Many Requests"}"#,
        );
        let (status, payload) = read_response_json(openai_error_response(error)).await;

        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(payload["error"]["type"], "rate_limit_error");
        assert_eq!(payload["error"]["code"], "gail");
    }

    #[test]
    fn json_schema_response_format_hint_rejects_schema_echo() {
        let hint = response_format_system_hint(Some(&OpenAIResponseFormat {
            format_type: Some("json_schema".to_string()),
            json_schema: Some(json!({
                "name": "ExecutionPlan",
                "schema": {
                    "type": "object",
                    "properties": {"steps": {"type": "array"}},
                    "required": ["steps"],
                    "additionalProperties": false
                }
            })),
        }))
        .expect("hint");

        assert!(hint.contains("schema is not the answer"));
        assert!(hint.contains("do not echo"));
    }

    #[tokio::test]
    async fn openai_chat_completions_falls_back_after_nvidia_rate_limit() {
        let nvidia = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "moonshotai/kimi-k2-instruct-0905"}]
            })))
            .mount(&nvidia)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "0")
                    .set_body_json(json!({"status": 429, "title": "Too Many Requests"})),
            )
            .mount(&nvidia)
            .await;

        let ollama = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "ollama fallback answer",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&ollama)
            .await;

        let mut config = GailConfig::default();
        config.orchestration.max_parallel_candidates = 1;
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", nvidia.uri())),
            roles: vec!["assistant".to_string()],
            specialties: vec!["reasoning".to_string()],
            weight: 10.0,
            preferred: true,
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "OllamaLocal".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            base_url: Some(ollama.uri()),
            roles: vec!["assistant".to_string()],
            specialties: vec!["local".to_string()],
            weight: 0.1,
            ..ProviderProfile::default()
        });

        let app = build_router(test_service_with_config(config).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "gail-auto",
                            "max_candidates": 1,
                            "messages": [{"role": "user", "content": "hello"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");

        let payload = read_json(response).await;
        assert_eq!(payload["model"], "gail-auto");
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            "ollama fallback answer"
        );
        assert_eq!(payload["gail"]["provider"], "ollama");
        let candidates = payload["gail"]["trace"]["candidates"]
            .as_array()
            .expect("trace candidates");
        assert!(candidates.iter().any(|candidate| {
            candidate["provider"] == "nvidia"
                && candidate["status"] == "error"
                && candidate["error"]
                    .as_str()
                    .is_some_and(|error| error.contains("Too Many Requests"))
        }));
        assert!(
            candidates.iter().any(|candidate| {
                candidate["provider"] == "ollama" && candidate["status"] == "ok"
            })
        );
    }

    #[tokio::test]
    async fn openai_chat_completions_returns_degraded_json_when_all_orchestrated_candidates_fail() {
        let nvidia = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "moonshotai/kimi-k2-instruct-0905"}]
            })))
            .mount(&nvidia)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({
                "error": {"message": "upstream transport failed"}
            })))
            .mount(&nvidia)
            .await;

        let ollama = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Reply OK."))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "OK",
                "prompt_eval_count": 1,
                "eval_count": 1
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Evaluate the next USDT microtrade"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({
                "error": "local generation unavailable"
            })))
            .mount(&ollama)
            .await;

        let mut config = GailConfig::default();
        config.orchestration.max_parallel_candidates = 1;
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", nvidia.uri())),
            roles: vec!["general".to_string()],
            specialties: vec!["json".to_string(), "trading".to_string()],
            weight: 1.0,
            preferred: true,
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "OllamaLocal".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            base_url: Some(ollama.uri()),
            roles: vec!["general".to_string()],
            specialties: vec![
                "local".to_string(),
                "json".to_string(),
                "trading".to_string(),
            ],
            weight: 0.1,
            preferred: false,
            ..ProviderProfile::default()
        });

        let app = build_router(test_service_with_config(config).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "gail-auto",
                            "workflow": "trading",
                            "request_category": "octobot trading evaluator",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return only valid JSON for the trading decision."
                                },
                                {
                                    "role": "user",
                                    "content": "Evaluate the next USDT microtrade."
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");

        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let decision: Value = serde_json::from_str(content).expect("degraded json");
        assert_eq!(decision["status"], "degraded");
        assert_eq!(decision["action"], "hold");
        assert_eq!(decision["should_trade"], false);
        assert_eq!(payload["gail"]["provider"], "gail");
        assert_eq!(payload["gail"]["resolved_model"], "degraded_safety");
        assert_eq!(payload["gail"]["trace"]["final_source"], "degraded_policy");
    }

    #[tokio::test]
    async fn openai_chat_completions_degraded_json_respects_automation_timeout() {
        let nvidia = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "moonshotai/kimi-k2-instruct-0905"}]
            })))
            .mount(&nvidia)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(5))
                    .set_body_json(json!({
                        "id": "chatcmpl-slow",
                        "model": "moonshotai/kimi-k2-instruct-0905",
                        "choices": [{
                            "message": {"role": "assistant", "content": "{\"action\":\"buy\"}"},
                            "finish_reason": "stop"
                        }]
                    })),
            )
            .mount(&nvidia)
            .await;

        let ollama = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Reply OK."))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "OK",
                "prompt_eval_count": 1,
                "eval_count": 1
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Evaluate the next USDT microtrade"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(5))
                    .set_body_json(json!({
                        "response": "{\"action\":\"buy\"}",
                        "prompt_eval_count": 2,
                        "eval_count": 3
                    })),
            )
            .mount(&ollama)
            .await;

        let mut config = GailConfig::default();
        config.orchestration.max_parallel_candidates = 2;
        config.orchestration.candidate_timeout_cap_seconds = Some(30);
        config
            .orchestration
            .automation_candidate_timeout_cap_seconds = Some(1);
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", nvidia.uri())),
            roles: vec!["general".to_string()],
            specialties: vec!["json".to_string(), "trading".to_string()],
            weight: 1.0,
            preferred: true,
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "OllamaLocal".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            base_url: Some(ollama.uri()),
            roles: vec!["general".to_string()],
            specialties: vec![
                "local".to_string(),
                "json".to_string(),
                "trading".to_string(),
            ],
            weight: 0.5,
            preferred: false,
            ..ProviderProfile::default()
        });

        let app = build_router(test_service_with_config(config).await);
        let started = std::time::Instant::now();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "gail-auto",
                            "workflow": "trading",
                            "request_category": "octobot trading evaluator",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "Return only valid JSON for the trading decision."
                                },
                                {
                                    "role": "user",
                                    "content": "Evaluate the next USDT microtrade."
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(4),
            "automation fallback took {:?}",
            started.elapsed()
        );

        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let decision: Value = serde_json::from_str(content).expect("degraded json");
        assert_eq!(decision["status"], "degraded");
        assert_eq!(decision["action"], "hold");
        assert_eq!(payload["gail"]["trace"]["final_source"], "degraded_policy");
    }

    #[tokio::test]
    async fn openai_chat_completions_json_tag_respects_automation_timeout() {
        let nvidia = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "moonshotai/kimi-k2-instruct-0905"}]
            })))
            .mount(&nvidia)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(5))
                    .set_body_json(json!({
                        "id": "chatcmpl-slow",
                        "model": "moonshotai/kimi-k2-instruct-0905",
                        "choices": [{
                            "message": {"role": "assistant", "content": "{\"action\":\"buy\"}"},
                            "finish_reason": "stop"
                        }]
                    })),
            )
            .mount(&nvidia)
            .await;

        let ollama = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Reply OK."))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "OK",
                "prompt_eval_count": 1,
                "eval_count": 1
            })))
            .mount(&ollama)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .and(body_string_contains("Choose the next portfolio action."))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(5))
                    .set_body_json(json!({
                        "response": "{\"action\":\"buy\"}",
                        "prompt_eval_count": 2,
                        "eval_count": 3
                    })),
            )
            .mount(&ollama)
            .await;

        let mut config = GailConfig::default();
        config.orchestration.max_parallel_candidates = 2;
        config.orchestration.candidate_timeout_cap_seconds = Some(30);
        config
            .orchestration
            .automation_candidate_timeout_cap_seconds = Some(1);
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", nvidia.uri())),
            roles: vec!["general".to_string()],
            specialties: vec!["json".to_string()],
            weight: 1.0,
            preferred: true,
            ..ProviderProfile::default()
        });
        config.providers.push(ProviderProfile {
            name: "OllamaLocal".to_string(),
            provider_type: "ollama".to_string(),
            model: Some("llama3.2".to_string()),
            base_url: Some(ollama.uri()),
            roles: vec!["general".to_string()],
            specialties: vec!["local".to_string(), "json".to_string()],
            weight: 0.5,
            preferred: false,
            ..ProviderProfile::default()
        });

        let app = build_router(test_service_with_config(config).await);
        let started = std::time::Instant::now();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "gail-auto",
                            "request_category": "json",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Choose the next portfolio action."
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(4),
            "json-tag fallback took {:?}",
            started.elapsed()
        );

        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let decision: Value = serde_json::from_str(content).expect("degraded json");
        assert_eq!(decision["status"], "degraded");
        assert_eq!(decision["action"], "hold");
        assert_eq!(payload["gail"]["trace"]["final_source"], "degraded_policy");
    }

    #[tokio::test]
    async fn openai_chat_completions_execution_plan_uses_standard_timeout_cap() {
        let nvidia = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "moonshotai/kimi-k2-instruct-0905"}]
            })))
            .mount(&nvidia)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(2))
                    .set_body_json(json!({
                        "id": "chatcmpl-plan",
                        "model": "moonshotai/kimi-k2-instruct-0905",
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": "{\"steps\":[{\"step\":\"summarize_market\",\"inputs\":{\"mode\":\"distribution\"}}]}"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 13,
                            "completion_tokens": 9,
                            "total_tokens": 22
                        }
                    })),
            )
            .mount(&nvidia)
            .await;

        let mut config = GailConfig::default();
        config.orchestration.max_parallel_candidates = 1;
        config.orchestration.candidate_timeout_cap_seconds = Some(8);
        config
            .orchestration
            .automation_candidate_timeout_cap_seconds = Some(1);
        config.providers.push(ProviderProfile {
            name: "NVIDIAKimi".to_string(),
            provider_type: "nvidia".to_string(),
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            api_key: Some("nvapi-test".to_string()),
            base_url: Some(format!("{}/v1", nvidia.uri())),
            roles: vec!["general".to_string()],
            specialties: vec!["json".to_string(), "planning".to_string()],
            weight: 1.0,
            preferred: true,
            ..ProviderProfile::default()
        });

        let app = build_router(test_service_with_config(config).await);
        let started = std::time::Instant::now();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "gail-auto",
                            "workflow": "trading",
                            "request_category": "octobot trading evaluator",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": r#"{"name":"ExecutionPlan","schema":{"type":"object","properties":{"steps":{"type":"array"}},"required":["steps"],"additionalProperties":false}}"#
                                },
                                {
                                    "role": "user",
                                    "content": "Generate the next execution plan."
                                }
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");

        assert!(
            started.elapsed() >= std::time::Duration::from_secs(2),
            "execution-plan request should not be clipped by automation timeout: {:?}",
            started.elapsed()
        );

        let payload = read_json(response).await;
        let content = payload["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        let parsed: Value = serde_json::from_str(content).expect("execution plan json");
        assert_eq!(parsed["steps"][0]["step"], "summarize_market");
        assert_ne!(payload["gail"]["trace"]["final_source"], "degraded_policy");
    }

    #[tokio::test]
    async fn openai_responses_route_input_text_items() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "mocked answer",
                "prompt_eval_count": 4,
                "eval_count": 3
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "input": [
                                {"type": "input_text", "text": "hello"}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;

        assert_eq!(payload["model"], "ollama/llama3.2");
        assert_eq!(payload["output_text"], "mocked answer");
        assert_eq!(payload["output"][0]["type"], "text");
        assert_eq!(payload["output"][0]["text"], "mocked answer");
    }

    #[tokio::test]
    async fn openai_responses_explicit_ollama_saturation_respects_json_schema_fallback() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": "local Ollama request queue is saturated; backing off before retrying in 14s"
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "text": {
                                "format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "ExecutionPlan",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "loop": {"type": "boolean", "default": false},
                                                "loop_condition": {
                                                    "anyOf": [{"type": "string"}, {"type": "null"}]
                                                },
                                                "max_iterations": {
                                                    "anyOf": [{"type": "integer"}, {"type": "null"}]
                                                },
                                                "steps": {
                                                    "type": "array",
                                                    "items": {"type": "object"}
                                                }
                                            },
                                            "required": ["steps", "loop", "loop_condition", "max_iterations"],
                                            "additionalProperties": false
                                        }
                                    }
                                }
                            },
                            "input": [
                                {"type": "input_text", "text": "Generate the next execution plan."}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");
        let payload = read_json(response).await;
        let degraded_text = payload["output_text"]
            .as_str()
            .expect("response output text");
        let degraded: Value =
            serde_json::from_str(degraded_text).expect("degraded json output_text payload");
        let degraded_object = degraded.as_object().expect("degraded object");

        assert_eq!(payload["model"], "ollama/llama3.2");
        assert_eq!(degraded_object.len(), 4);
        assert_eq!(degraded["loop"], false);
        assert_eq!(degraded["loop_condition"], "");
        assert_eq!(degraded["max_iterations"], 0);
        assert!(degraded_object.contains_key("steps"));
        assert_eq!(
            degraded["steps"].as_array().expect("steps array").len(),
            0,
            "schema fallback should avoid non-schema keys like status/decision"
        );
        assert_eq!(payload["output"][0]["type"], "text");
        assert_eq!(payload["gail"]["provider"], "gail");
        assert_eq!(payload["gail"]["resolved_model"], "degraded_safety");
    }

    #[tokio::test]
    async fn openai_responses_saturation_reuses_cached_execution_plan() {
        let success_server = MockServer::start().await;
        let success_base_url = format!("{}/responses-cache-success", success_server.uri());
        Mock::given(method("GET"))
            .and(path("/responses-cache-success/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&success_server)
            .await;
        let cached_execution_plan = json!({
            "steps": [
                {
                    "agent_name": "TechnicalAnalysisAIAgentProducer",
                    "instructions": null,
                    "wait_for": null,
                    "skip": false,
                    "step_type": "agent",
                    "debate_config": null
                }
            ],
            "loop": false,
            "loop_condition": null,
            "max_iterations": null
        });
        Mock::given(method("POST"))
            .and(path("/responses-cache-success/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": cached_execution_plan.to_string(),
                "prompt_eval_count": 4,
                "eval_count": 3
            })))
            .mount(&success_server)
            .await;

        let saturated_server = MockServer::start().await;
        let saturated_base_url = format!("{}/responses-cache-saturated", saturated_server.uri());
        Mock::given(method("GET"))
            .and(path("/responses-cache-saturated/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&saturated_server)
            .await;
        Mock::given(method("POST"))
            .and(path("/responses-cache-saturated/api/generate"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": "local Ollama request queue is saturated; backing off before retrying in 14s"
            })))
            .mount(&saturated_server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let team_name = format!("ExecutionPlanCacheTeam{}", uuid::Uuid::new_v4().simple());
        let manager_prompt = format!(
            "Analyze the following team structure and create an execution plan:\n\
             Team: {team_name}\n\
             Agents: [{{\"name\":\"TechnicalAnalysisAIAgentProducer\",\"channel\":\"TechnicalAnalysisAIAgentChannel\"}},{{\"name\":\"SummarizationAIAgentProducer\",\"channel\":\"SummarizationAIAgentChannel\"}}]\n\
             Relations: [{{\"source\":\"TechnicalAnalysisAIAgentChannel\",\"target\":\"SummarizationAIAgentChannel\"}}]\n\
             Initial Data: {{}}\n\
             Instructions: None"
        );
        let request_payload = |base_url: String| {
            json!({
                "model": "ollama/llama3.2",
                "base_url": base_url,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ExecutionPlan",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "steps": {
                                        "type": "array",
                                        "items": {"type": "object"}
                                    },
                                    "loop": {"type": "boolean"},
                                    "loop_condition": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "max_iterations": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                                },
                                "required": ["steps", "loop", "loop_condition", "max_iterations"],
                                "additionalProperties": false
                            }
                        }
                    }
                },
                "input": [
                    {"type": "input_text", "text": manager_prompt}
                ]
            })
        };

        let first_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        request_payload(success_base_url.clone()).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("first response");
        let first_payload = read_json(first_response).await;
        let first_plan: Value = serde_json::from_str(
            first_payload["output_text"]
                .as_str()
                .expect("first output text"),
        )
        .expect("first plan json");
        assert_eq!(
            first_plan["steps"].as_array().expect("steps").len(),
            1,
            "first response should cache a working execution plan"
        );

        let second_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        request_payload(saturated_base_url.clone()).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("second response");
        let second_payload = read_json(second_response).await;
        let second_plan: Value = serde_json::from_str(
            second_payload["output_text"]
                .as_str()
                .expect("second output text"),
        )
        .expect("second plan json");

        assert_eq!(
            second_plan, first_plan,
            "degraded schema fallback should reuse the last non-empty execution plan"
        );
        assert_eq!(second_payload["gail"]["provider"], "gail");
        assert_eq!(second_payload["gail"]["resolved_model"], "degraded_safety");
    }

    #[tokio::test]
    async fn openai_chat_completions_stream_route_returns_sse_chunks() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "mocked answer",
                "prompt_eval_count": 8,
                "eval_count": 5
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "stream": true,
                            "messages": [
                                {"role": "user", "content": "hello"}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"))
        );
        let body = read_text(response).await;
        assert!(body.contains("\"object\":\"chat.completion.chunk\""));
        assert!(body.contains("mocked answer"));
        assert!(body.contains("[DONE]"));
    }

    #[tokio::test]
    async fn openai_responses_stream_route_returns_sse_events() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"name": "llama3.2"}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "response": "mocked answer",
                "prompt_eval_count": 4,
                "eval_count": 3
            })))
            .mount(&server)
            .await;

        let app = build_router(test_service_with_config(GailConfig::default()).await);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer secret")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        json!({
                            "model": "ollama/llama3.2",
                            "base_url": server.uri(),
                            "stream": true,
                            "input": [
                                {"type": "input_text", "text": "hello"}
                            ]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"))
        );
        let body = read_text(response).await;
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains("mocked answer"));
        assert!(body.contains("[DONE]"));
    }
}
