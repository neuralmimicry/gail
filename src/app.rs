use axum::{
    Json, Router,
    extract::{Multipart, Query, State},
    http::HeaderMap,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::Value;
use tokio::signal;

use crate::{
    errors::{GailError, Result},
    models::{
        AerDecodeRequest, AerDecodeResponse, AerEncodeRequest, AerEncodeResponse,
        CompletionRequest, CompletionResponse, HealthResponse, NeuromorphicAnalyzeRequest,
        NeuromorphicPredictRequest, NeuromorphicPredictResponse, ProviderCompletionRequest,
        SpecialistAnalysisResponse, TranscriptionResponse,
    },
    orchestration::GailService,
    providers::TranscriptionInput,
};

#[derive(Debug, Default, Deserialize)]
pub struct StatusQuery {
    pub limit: Option<usize>,
    pub probe_engines: Option<bool>,
    pub probe_providers: Option<bool>,
}

pub fn build_router(service: GailService) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/llm/complete", post(complete))
        .route("/v1/llm/direct-complete", post(direct_complete))
        .route("/v1/llm/transcribe", post(transcribe))
        .route("/v1/neuromorphic/analyze", post(analyze_neuromorphic))
        .route("/v1/neuromorphic/predict", post(predict_neuromorphic))
        .route("/v1/aer/encode", post(encode_aer))
        .route("/v1/aer/decode", post(decode_aer))
        .route("/v1/status/orchestration", get(orchestration_status))
        .with_state(service)
}

async fn health(State(service): State<GailService>, headers: HeaderMap) -> Result<Json<HealthResponse>> {
    if !service.can_access_health_unauthenticated() {
        let _ = service.authorize(&headers, "health")?;
    }
    Ok(Json(service.health().await))
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
            "provider" => provider = Some(field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?),
            "model" => model = Some(field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?),
            "api_key" => api_key = Some(field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?),
            "access_token" => access_token = Some(field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?),
            "base_url" => base_url = Some(field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?),
            "timeout_seconds" => {
                let raw = field.text().await.map_err(|error| GailError::Multipart(error.to_string()))?;
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
        bytes: file_bytes.ok_or_else(|| GailError::bad_request("multipart field `file` is required"))?,
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
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use crate::config::{ApiTokenConfig, GailConfig};

    async fn test_service() -> GailService {
        let mut config = GailConfig::default();
        config.security.allow_unauthenticated_health = false;
        config.security.api_tokens.push(ApiTokenConfig {
            client_id: "test".to_string(),
            token: "secret".to_string(),
            scopes: vec!["*".to_string()],
        });
        GailService::new(config).await.expect("service")
    }

    #[tokio::test]
    async fn health_requires_auth_when_configured() {
        let app = build_router(test_service().await);
        let response = app
            .oneshot(Request::builder().uri("/healthz").body(axum::body::Body::empty()).unwrap())
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn health_allows_valid_bearer_token() {
        let app = build_router(test_service().await);
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
}
