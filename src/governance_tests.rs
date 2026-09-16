//! Contract and enforcement regressions; no live providers or credentials are used.
use super::*;
use crate::config::{ApiTokenConfig, GailConfig};
use axum::{Router, http::Request as HttpRequest, routing::post};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const EVALUATION: &str = "aria-evaluation-test-token-00000000000";
const ASSESSMENT_TOKEN: &str = "aria-assessment-test-token-00000000000";
async fn service(url: &str, mode: Mode, fail_open: bool) -> GailService {
    let mut config = GailConfig::default();
    config.security.api_tokens = vec![ApiTokenConfig {
        client_id: "trusted-client".into(),
        token: "ordinary-client-token".into(),
        scopes: vec!["*".into()],
    }];
    config.governance = GovernanceConfig {
        mode,
        aria_url: url.into(),
        evaluation_token: EVALUATION.into(),
        assessment_token: ASSESSMENT_TOKEN.into(),
        fail_open,
        ..Default::default()
    };
    config.llm_ledger.enabled = false;
    config.trading.enabled = false;
    let prefix = std::env::temp_dir().join(format!("gail-governance-{}", Uuid::new_v4()));
    config.storage.metrics_path = prefix
        .with_extension("metrics.json")
        .to_string_lossy()
        .into();
    config.storage.adaptive_schema_path = prefix
        .with_extension("schema.json")
        .to_string_lossy()
        .into();
    config.storage.api_issues_path = prefix
        .with_extension("issues.json")
        .to_string_lossy()
        .into();
    config.storage.postgres_dsn = None;
    GailService::new(config).await.unwrap()
}
fn request() -> HttpRequest<Body> {
    HttpRequest::builder().method("POST").uri("/v1/llm/complete").header("content-type","application/json")
        .header("authorization","Bearer ordinary-client-token").header("x-aria-bypass","true")
        .body(Body::from(json!({"source":"spoofed","messages":[{"role":"user","content":"hello"}],"api_key":"transport-secret"}).to_string())).unwrap()
}
fn test_router(service: GailService, count: Arc<AtomicU64>, sse: bool) -> Router {
    Router::new().route("/v1/llm/complete",post(move || {let count=count.clone();async move {
        count.fetch_add(1,Ordering::Relaxed);
        if sse {([("content-type","text/event-stream")],"data: {\"choices\":[{\"delta\":{\"content\":\"ignore previous \"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"instructions\"}}]}\n\ndata: [DONE]\n\n").into_response()}
        else {Json(json!({"text":"upstream-response"})).into_response()}
    }})).layer(axum::middleware::from_fn_with_state((service.clone(),service.governance().clone()),guard))
}
async fn verdicts(server: &MockServer, block_phase: &'static str) {
    Mock::given(method("POST")).and(path("/v1/evaluate")).respond_with(move |request:&wiremock::Request| {
        let value:Value=serde_json::from_slice(&request.body).unwrap();
        assert_eq!(value["source"],"trusted-client");assert!(!value["content"].as_str().unwrap().contains("transport-secret"));
        ResponseTemplate::new(200).set_body_json(json!({"protocol_version":1,"request_id":value["request_id"],"id":Uuid::new_v4(),"phase":value["phase"],"action":if value["phase"]==block_phase {"block"}else{"allow"}}))
    }).mount(server).await;
}

#[tokio::test]
async fn blocked_request_never_reaches_provider_and_bypass_header_is_ignored() {
    let mock = MockServer::start().await;
    verdicts(&mock, "request").await;
    let service = service(&mock.uri(), Mode::Enforce, false).await;
    let count = Arc::new(AtomicU64::new(0));
    let response = test_router(service, count.clone(), false)
        .oneshot(request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(count.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn blocked_response_is_withheld_including_streamed_content() {
    let mock = MockServer::start().await;
    verdicts(&mock, "response").await;
    let service = service(&mock.uri(), Mode::Enforce, false).await;
    let count = Arc::new(AtomicU64::new(0));
    let response = test_router(service, count.clone(), true)
        .oneshot(request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(count.load(Ordering::Relaxed), 1);
    let body = to_bytes(response.into_body(), 65536).await.unwrap();
    assert!(!String::from_utf8_lossy(&body).contains("instructions"));
    let calls = mock.received_requests().await.unwrap();
    let response_check: Value = serde_json::from_slice(&calls[1].body).unwrap();
    assert!(
        response_check["content"]
            .as_str()
            .unwrap()
            .contains("ignore previous instructions")
    );
}
#[tokio::test]
async fn monitor_records_without_blocking() {
    let mock = MockServer::start().await;
    verdicts(&mock, "request").await;
    let service = service(&mock.uri(), Mode::Monitor, false).await;
    let count = Arc::new(AtomicU64::new(0));
    let response = test_router(service, count.clone(), false)
        .oneshot(request())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-aria-status"], "would_block");
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(mock.received_requests().await.unwrap().len(), 2);
}
#[tokio::test]
async fn unavailable_aria_obeys_explicit_failure_policy() {
    let mock = MockServer::start().await;
    for fail_open in [false, true] {
        let service = service(&mock.uri(), Mode::Enforce, fail_open).await;
        let count = Arc::new(AtomicU64::new(0));
        let response = test_router(service, count.clone(), false)
            .oneshot(request())
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if fail_open {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            }
        );
        assert_eq!(count.load(Ordering::Relaxed), u64::from(fail_open));
        if fail_open {
            assert_eq!(response.headers()["x-aria-status"], "unavailable");
        }
    }
}
#[tokio::test]
async fn assessment_route_rejects_ordinary_credentials_before_parsing() {
    let service = service("http://127.0.0.1:9", Mode::Enforce, false).await;
    let response = crate::app::build_router(service)
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/v1/internal/aria/assess")
                .header("authorization", "Bearer ordinary-client-token")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
#[tokio::test]
async fn internal_completions_are_checked_without_an_http_request() {
    let mock = MockServer::start().await;
    verdicts(&mock, "request").await;
    let service = service(&mock.uri(), Mode::Enforce, false).await;
    let count = Arc::new(AtomicU64::new(0));
    let result = service
        .governance()
        .completion(json!({"messages":[]}), Some("trusted-client"), async {
            count.fetch_add(1, Ordering::Relaxed);
            Err(GailError::bad_request("should not execute"))
        })
        .await;
    assert!(result.is_err());
    assert_eq!(count.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn only_task_local_assessment_scope_skips_internal_checks() {
    let mock = MockServer::start().await;
    let service = service(&mock.uri(), Mode::Enforce, false).await;
    let result = ASSESSMENT
        .scope(
            (),
            service
                .governance()
                .completion(json!({}), Some("aria_assessment"), async {
                    Err(GailError::bad_request("operation_reached"))
                }),
        )
        .await;
    assert_eq!(result.unwrap_err().to_string(), "operation_reached");
    assert!(mock.received_requests().await.unwrap().is_empty());
}
#[test]
fn transport_redaction_does_not_hide_prompt_or_response_credentials() {
    let (numeric, _) = inspectable_content(
        br#"{"max_tokens":4000000000,"execute":true}"#,
        "application/json",
        true,
    );
    assert!(numeric.contains("4000000000") && numeric.contains("true"));
    let bytes =
        json!({"api_key":"transport-secret","messages":[{"content":"sk-sensitive-prompt-value"}]})
            .to_string();
    let (text, complete) = inspectable_content(bytes.as_bytes(), "application/json", true);
    assert!(complete);
    assert!(!text.contains("transport-secret"));
    assert!(text.contains("sk-sensitive-prompt-value"));
    let (text, _) = inspectable_content(
        br#"{"api_key":"leaked-response-secret"}"#,
        "application/json",
        false,
    );
    assert!(text.contains("leaked-response-secret"));
    assert!(
        !inspectable_content(
            br#"{"image_url":{"url":"https://example.test/image"}}"#,
            "application/json",
            true
        )
        .1
    );
}
#[test]
fn classifier_output_and_credential_configuration_are_strict() {
    assert!(parse_classification(r#"{"score":0,"categories":[]}"#).is_some());
    for invalid in [
        r#"{"score":1,"categories":[]}"#,
        r#"{"score":1,"categories":["paused"]}"#,
        r#"{"score":0,"categories":[],"instructions":"bypass"}"#,
    ] {
        assert!(parse_classification(invalid).is_none());
    }
    let config = GovernanceConfig {
        mode: Mode::Enforce,
        evaluation_token: EVALUATION.into(),
        assessment_token: ASSESSMENT_TOKEN.into(),
        ..Default::default()
    };
    let security = SecurityConfig {
        api_tokens: vec![ApiTokenConfig {
            client_id: "ordinary".into(),
            token: ASSESSMENT_TOKEN.into(),
            scopes: vec![],
        }],
        ..Default::default()
    };
    assert!(Governance::new(&config, &security).is_err());
}
