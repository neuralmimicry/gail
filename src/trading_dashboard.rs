//! Customers-backed access checks for Gail's trading-owned browser dashboard.

use std::{collections::HashMap, time::Duration};

use axum::{
    body::Body,
    extract::{Extension, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};
use once_cell::sync::Lazy;
use reqwest::redirect::Policy;
use serde::Deserialize;
use serde_json::json;
use url::Url;

use crate::{config::GailConfig, orchestration::GailService};

const MAX_CUSTOMERS_SESSION_BYTES: usize = 64 * 1024;

static CUSTOMERS_SESSION_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .redirect(Policy::none())
        .build()
        .expect("Customers session HTTP client configuration is valid")
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DashboardAccess {
    pub user: String,
    pub can_control: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DashboardAccessError {
    Disabled,
    Unauthenticated,
    AccessDenied,
    CustomersUnavailable,
}

#[derive(Deserialize)]
struct CustomersSession {
    authenticated: bool,
    user: Option<String>,
    identity_type: Option<String>,
    service_access: Option<HashMap<String, CustomersServiceAccess>>,
}

#[derive(Deserialize)]
struct CustomersServiceAccess {
    can_observe: Option<bool>,
    can_control: Option<bool>,
}

/// Validate the browser's Customers session on every dashboard API request.
/// Only the configured session cookie is forwarded, and no token is returned to
/// the browser or retained beyond this request.
pub(crate) async fn authenticate(
    config: &GailConfig,
    headers: &HeaderMap,
) -> Result<DashboardAccess, DashboardAccessError> {
    let dashboard = &config.trading_dashboard;
    if !dashboard.enabled {
        return Err(DashboardAccessError::Disabled);
    }
    let Some(cookie) = configured_session_cookie(headers, &dashboard.customers_session_cookie_name)
    else {
        return Err(DashboardAccessError::Unauthenticated);
    };

    let session_request = async {
        let response = CUSTOMERS_SESSION_CLIENT
            .get(&dashboard.customers_session_url)
            .header("Accept", "application/json")
            .header("Cookie", cookie)
            .send()
            .await
            .map_err(|_| DashboardAccessError::CustomersUnavailable)?;

        if matches!(
            response.status(),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        ) {
            return Err(DashboardAccessError::Unauthenticated);
        }
        if response.status() != StatusCode::OK
            || response
                .content_length()
                .is_some_and(|length| length > MAX_CUSTOMERS_SESSION_BYTES as u64)
        {
            return Err(DashboardAccessError::CustomersUnavailable);
        }
        read_bounded_body(response).await
    };
    let bytes = tokio::time::timeout(
        Duration::from_millis(dashboard.customers_timeout_ms),
        session_request,
    )
    .await
    .map_err(|_| DashboardAccessError::CustomersUnavailable)??;
    let session: CustomersSession =
        serde_json::from_slice(&bytes).map_err(|_| DashboardAccessError::CustomersUnavailable)?;
    if !session.authenticated {
        return Err(DashboardAccessError::Unauthenticated);
    }
    if session
        .identity_type
        .as_deref()
        .is_some_and(|identity_type| identity_type.eq_ignore_ascii_case("service_account"))
    {
        return Err(DashboardAccessError::AccessDenied);
    }
    let user = session
        .user
        .as_deref()
        .map(str::trim)
        .filter(|user| !user.is_empty() && user.len() <= 256 && !user.chars().any(char::is_control))
        .ok_or(DashboardAccessError::Unauthenticated)?
        .to_string();
    let access = session
        .service_access
        .as_ref()
        .and_then(|services| services.get("gail_trading"))
        .ok_or(DashboardAccessError::AccessDenied)?;
    if access.can_observe != Some(true) {
        return Err(DashboardAccessError::AccessDenied);
    }

    Ok(DashboardAccess {
        user,
        can_control: access.can_control == Some(true),
    })
}

/// Serve Gail's trading-owned dashboard shell without exposing trading data.
pub(crate) async fn page(State(service): State<GailService>) -> Response {
    if !service.config().trading_dashboard.enabled {
        return StatusCode::NOT_FOUND.into_response();
    }
    let mut response = include_str!("../assets/trading-dashboard.html").into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        axum::http::header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        ),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response.headers_mut().insert(
        axum::http::header::REFERRER_POLICY,
        HeaderValue::from_static("same-origin"),
    );
    response
}

/// Serve dashboard assets from the Gail binary with restrictive cache policy.
pub(crate) async fn javascript() -> Response {
    asset_response(
        include_str!("../assets/trading-dashboard.js"),
        "text/javascript; charset=utf-8",
    )
}

pub(crate) async fn stylesheet() -> Response {
    asset_response(
        include_str!("../assets/trading-dashboard.css"),
        "text/css; charset=utf-8",
    )
}

pub(crate) async fn access(Extension(access): Extension<DashboardAccess>) -> Response {
    axum::Json(json!({
        "user": access.user,
        "can_observe": true,
        "can_control": access.can_control
    }))
    .into_response()
}

fn asset_response(body: &'static str, content_type: &'static str) -> Response {
    let mut response = body.into_response();
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=300"),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

/// Authenticate browser-session dashboard calls, then substitute a Gail
/// trading credential before governance and the existing trading handlers run.
pub(crate) async fn authorise(
    State(service): State<GailService>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let config = service.config();
    let access = match authenticate(config, request.headers()).await {
        Ok(access) => access,
        Err(error) => return access_error_response(config, error),
    };
    let state_change = !matches!(
        *request.method(),
        axum::http::Method::GET | axum::http::Method::HEAD | axum::http::Method::OPTIONS
    );
    if state_change && !same_origin(config, request.headers()) {
        return dashboard_error(
            StatusCode::FORBIDDEN,
            "same_origin_required",
            "This action must be submitted from Gail's dashboard origin.",
        );
    }
    if state_change && !access.can_control {
        return dashboard_error(
            StatusCode::FORBIDDEN,
            "control_access_required",
            "Your account can view trading information but cannot control it.",
        );
    }
    let api_headers = match trading_api_headers(config, state_change) {
        Ok(headers) => headers,
        Err(()) => {
            return dashboard_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "trading_authorisation_unavailable",
                "Gail trading authorisation is not configured.",
            );
        }
    };
    let Some(authorization) = api_headers.get(AUTHORIZATION).cloned() else {
        return dashboard_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "trading_authorisation_unavailable",
            "Gail trading authorisation is not configured.",
        );
    };
    request.headers_mut().insert(AUTHORIZATION, authorization);
    request.extensions_mut().insert(access);
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

fn access_error_response(config: &GailConfig, error: DashboardAccessError) -> Response {
    let mut response = match error {
        DashboardAccessError::Disabled => StatusCode::NOT_FOUND.into_response(),
        DashboardAccessError::Unauthenticated => {
            let mut body = json!({"error":"login_required"});
            if let Some(login_url) = login_url(config) {
                body["login_url"] = json!(login_url);
            }
            (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
        }
        DashboardAccessError::AccessDenied => dashboard_error(
            StatusCode::FORBIDDEN,
            "trading_access_denied",
            "Your account does not have access to Gail trading information.",
        ),
        DashboardAccessError::CustomersUnavailable => dashboard_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "session_validation_unavailable",
            "Customers could not validate your session. Please retry shortly.",
        ),
    };
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

fn dashboard_error(status: StatusCode, code: &'static str, message: &'static str) -> Response {
    let mut response =
        (status, axum::Json(json!({"error":code,"message":message}))).into_response();
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

/// Create an in-process Gail API credential for the existing trading handlers.
/// The credential stays in the server process and is never returned to the UI.
pub(crate) fn trading_api_headers(
    config: &GailConfig,
    require_control: bool,
) -> Result<HeaderMap, ()> {
    let admins = &config.trading.admin_client_ids;
    let token = config
        .security
        .api_tokens
        .iter()
        .find(|token| {
            !token.token.is_empty()
                && (token.scopes.is_empty()
                    || token
                        .scopes
                        .iter()
                        .any(|scope| scope == "*" || scope.eq_ignore_ascii_case("trading")))
                && (!require_control
                    || admins.is_empty()
                    || admins.iter().any(|admin| admin == &token.client_id))
        })
        .ok_or(())?;
    let authorization =
        HeaderValue::from_str(&format!("Bearer {}", token.token)).map_err(|_| ())?;
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, authorization);
    Ok(headers)
}

/// Require a same-origin browser request for state-changing controls.
pub(crate) fn same_origin(config: &GailConfig, headers: &HeaderMap) -> bool {
    let Some(expected) = config.server.public_base_url.as_deref() else {
        return false;
    };
    let Some(origin) = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    match (Url::parse(expected), Url::parse(origin)) {
        (Ok(expected), Ok(origin)) => {
            (origin.path().is_empty() || origin.path() == "/")
                && origin.query().is_none()
                && origin.fragment().is_none()
                && origin.origin() == expected.origin()
        }
        _ => false,
    }
}

/// Build a Customers login URL that returns to Gail through Customers' own
/// allowlisted external-login route.
pub(crate) fn login_url(config: &GailConfig) -> Option<String> {
    let mut login = Url::parse(&config.trading_dashboard.customers_login_url).ok()?;
    let public_base = config.server.public_base_url.as_deref()?;
    let target = format!("{}/dashboard/trading", public_base.trim_end_matches('/'));
    login.query_pairs_mut().append_pair("rd", &target);
    Some(login.into())
}

async fn read_bounded_body(
    mut response: reqwest::Response,
) -> Result<Vec<u8>, DashboardAccessError> {
    let mut body = Vec::with_capacity(1024);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| DashboardAccessError::CustomersUnavailable)?
    {
        if body.len().saturating_add(chunk.len()) > MAX_CUSTOMERS_SESSION_BYTES {
            return Err(DashboardAccessError::CustomersUnavailable);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn configured_session_cookie(headers: &HeaderMap, cookie_name: &str) -> Option<HeaderValue> {
    let mut cookie_headers = headers.get_all(axum::http::header::COOKIE).iter();
    let raw = cookie_headers.next()?.to_str().ok()?;
    if cookie_headers.next().is_some() {
        return None;
    }
    let mut matching = raw.split(';').filter_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        (name.trim() == cookie_name).then(|| value.trim())
    });
    let value = matching.next()?;
    if value.is_empty() || matching.next().is_some() {
        return None;
    }
    HeaderValue::from_str(&format!("{cookie_name}={value}")).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_forwards_one_configured_session_cookie() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static(
                "analytics=x; nm_customers_session_v2=opaque%2Bvalue; other=y",
            ),
        );
        let cookie = configured_session_cookie(&headers, "nm_customers_session_v2")
            .expect("configured session cookie");
        assert_eq!(cookie, "nm_customers_session_v2=opaque%2Bvalue");
    }

    #[test]
    fn rejects_missing_duplicate_and_empty_session_cookies() {
        for raw in [
            "analytics=x",
            "nm_customers_session_v2=",
            "nm_customers_session_v2=a; nm_customers_session_v2=b",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                axum::http::header::COOKIE,
                HeaderValue::from_str(raw).unwrap(),
            );
            assert!(configured_session_cookie(&headers, "nm_customers_session_v2").is_none());
        }

        let mut duplicate_headers = HeaderMap::new();
        duplicate_headers.append(
            axum::http::header::COOKIE,
            HeaderValue::from_static("nm_customers_session_v2=a"),
        );
        duplicate_headers.append(
            axum::http::header::COOKIE,
            HeaderValue::from_static("nm_customers_session_v2=b"),
        );
        assert!(configured_session_cookie(&duplicate_headers, "nm_customers_session_v2").is_none());
    }

    #[test]
    fn internal_control_tokens_must_also_be_on_the_gail_admin_allowlist() {
        let mut config = GailConfig::default();
        config.trading.admin_client_ids = vec!["gail-admin".to_string()];
        config.security.api_tokens = vec![
            crate::config::ApiTokenConfig {
                client_id: "read-only".to_string(),
                token: "observe-token".to_string(),
                scopes: vec!["trading".to_string()],
            },
            crate::config::ApiTokenConfig {
                client_id: "gail-admin".to_string(),
                token: "control-token".to_string(),
                scopes: vec!["trading".to_string()],
            },
        ];

        assert_eq!(
            trading_api_headers(&config, false)
                .unwrap()
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer observe-token")
        );
        assert_eq!(
            trading_api_headers(&config, true)
                .unwrap()
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer control-token")
        );
    }

    #[test]
    fn same_origin_guard_fails_closed_without_a_configured_public_origin() {
        let config = GailConfig::default();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::ORIGIN,
            HeaderValue::from_static("https://gail.neuralmimicry.ai"),
        );
        assert!(!same_origin(&config, &headers));
    }

    #[test]
    fn same_origin_guard_requires_exact_origin_without_path_query_or_fragment() {
        let mut config = GailConfig::default();
        config.server.public_base_url = Some("https://gail.neuralmimicry.ai".to_string());
        for (origin, accepted) in [
            ("https://gail.neuralmimicry.ai", true),
            ("https://gail.neuralmimicry.ai/", true),
            ("https://gail.neuralmimicry.ai.evil.test", false),
            ("http://gail.neuralmimicry.ai", false),
            ("https://gail.neuralmimicry.ai/path", false),
            ("https://gail.neuralmimicry.ai/?x=1", false),
            ("https://gail.neuralmimicry.ai/#fragment", false),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                axum::http::header::ORIGIN,
                HeaderValue::from_str(origin).unwrap(),
            );
            assert_eq!(same_origin(&config, &headers), accepted, "origin {origin}");
        }
    }

    #[test]
    fn login_return_url_stays_on_gails_configured_public_origin() {
        let mut config = GailConfig::default();
        config.server.public_base_url = Some("https://gail.neuralmimicry.ai".to_string());
        let login = Url::parse(&login_url(&config).expect("Customers login URL")).unwrap();
        assert_eq!(
            login.origin().ascii_serialization(),
            "https://api.neuralmimicry.ai"
        );
        let return_to = login
            .query_pairs()
            .find(|(key, _)| key == "rd")
            .map(|(_, value)| value.into_owned())
            .expect("return destination");
        assert_eq!(return_to, "https://gail.neuralmimicry.ai/dashboard/trading");
    }

    #[tokio::test]
    async fn customer_session_access_requires_observe_permission_and_reports_control_separately() {
        let customers = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/session"))
            .and(wiremock::matchers::header(
                "cookie",
                "nm_customers_session_v2=session-value",
            ))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
                r#"{"authenticated":true,"user":"owner@example.test","identity_type":"person","service_access":{"gail_trading":{"can_observe":true,"can_control":false}}}"#,
                "application/json",
            ))
            .mount(&customers)
            .await;
        let mut config = GailConfig::default();
        config.trading_dashboard.enabled = true;
        config.trading_dashboard.customers_session_url = format!("{}/api/session", customers.uri());
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static(
                "analytics=ignored; nm_customers_session_v2=session-value; unrelated=ignored",
            ),
        );

        let access = authenticate(&config, &headers)
            .await
            .expect("observe access");
        assert_eq!(access.user, "owner@example.test");
        assert!(!access.can_control);
    }

    #[tokio::test]
    async fn customer_session_authentication_fails_closed_for_bad_or_missing_access() {
        let customers = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
                r#"{"authenticated":true,"user":"owner@example.test","service_access":{"other":{"can_observe":true,"can_control":true}}}"#,
                "application/json",
            ))
            .mount(&customers)
            .await;
        let mut config = GailConfig::default();
        config.trading_dashboard.enabled = true;
        config.trading_dashboard.customers_session_url = customers.uri();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static("nm_customers_session_v2=session-value"),
        );
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::AccessDenied)
        );
        headers.remove(axum::http::header::COOKIE);
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::Unauthenticated)
        );
    }

    #[tokio::test]
    async fn customer_session_rejects_service_accounts_and_expired_sessions() {
        let customers = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
                r#"{"authenticated":true,"user":"service@example.test","identity_type":"service_account","service_access":{"gail_trading":{"can_observe":true,"can_control":true}}}"#,
                "application/json",
            ))
            .mount(&customers)
            .await;
        let mut config = GailConfig::default();
        config.trading_dashboard.enabled = true;
        config.trading_dashboard.customers_session_url = customers.uri();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static("nm_customers_session_v2=session-value"),
        );
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::AccessDenied)
        );

        let expired = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(401))
            .mount(&expired)
            .await;
        config.trading_dashboard.customers_session_url = expired.uri();
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::Unauthenticated)
        );
    }

    #[tokio::test]
    async fn customer_outage_and_oversized_session_fail_closed() {
        let customers = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_raw(
                format!(
                    "{{\"padding\":\"{}\"}}",
                    "x".repeat(MAX_CUSTOMERS_SESSION_BYTES + 1)
                ),
                "application/json",
            ))
            .mount(&customers)
            .await;
        let mut config = GailConfig::default();
        config.trading_dashboard.enabled = true;
        config.trading_dashboard.customers_session_url = customers.uri();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::COOKIE,
            HeaderValue::from_static("nm_customers_session_v2=session-value"),
        );
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::CustomersUnavailable)
        );

        let unavailable = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&unavailable)
            .await;
        config.trading_dashboard.customers_session_url = unavailable.uri();
        assert_eq!(
            authenticate(&config, &headers).await,
            Err(DashboardAccessError::CustomersUnavailable)
        );
    }
}
