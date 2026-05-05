use std::borrow::Cow;

use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum GailError {
    #[error("{0}")]
    BadRequest(Cow<'static, str>),
    #[error("unauthorized")]
    Unauthorized,
    #[error("{0}")]
    NotFound(Cow<'static, str>),
    #[error("{0}")]
    InvalidConfig(Cow<'static, str>),
    #[error("multipart error: {0}")]
    Multipart(String),
    #[error("{provider} upstream error: {message}")]
    Upstream {
        provider: String,
        message: String,
        status: Option<StatusCode>,
        quota: bool,
        timeout: bool,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, GailError>;

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
    message: String,
    provider: Option<&'a str>,
    quota: bool,
    timeout: bool,
}

impl GailError {
    pub fn bad_request(message: impl Into<Cow<'static, str>>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn invalid_config(message: impl Into<Cow<'static, str>>) -> Self {
        Self::InvalidConfig(message.into())
    }

    pub fn not_found(message: impl Into<Cow<'static, str>>) -> Self {
        Self::NotFound(message.into())
    }

    pub fn unauthorized() -> Self {
        Self::Unauthorized
    }

    pub fn upstream(
        provider: impl Into<String>,
        status: Option<StatusCode>,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        let quota =
            status == Some(StatusCode::TOO_MANY_REQUESTS) || message_indicates_quota(&message);
        let lowered = message.to_ascii_lowercase();
        let timeout = lowered.contains("timeout") || lowered.contains("timed out");
        Self::Upstream {
            provider: provider.into(),
            message,
            status,
            quota,
            timeout,
        }
    }

    pub fn is_quota(&self) -> bool {
        matches!(self, Self::Upstream { quota: true, .. })
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Upstream { timeout: true, .. })
            || matches!(self, Self::Reqwest(err) if err.is_timeout())
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) | Self::Multipart(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::InvalidConfig(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Upstream {
                status: Some(status),
                ..
            } => *status,
            Self::Upstream { quota: true, .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::Upstream { timeout: true, .. } => StatusCode::GATEWAY_TIMEOUT,
            Self::Upstream { .. } => StatusCode::BAD_GATEWAY,
            Self::Io(_) | Self::Json(_) | Self::Yaml(_) | Self::Reqwest(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }
}

pub fn message_indicates_quota(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    let compact = lowered
        .chars()
        .filter(|char| !char.is_ascii_whitespace())
        .collect::<String>();
    lowered.contains("quota")
        || lowered.contains("rate limit")
        || lowered.contains("rate_limit")
        || lowered.contains("too many requests")
        || compact.contains("status\":429")
        || compact.contains("status:429")
        || lowered.contains("status 429")
        || lowered.contains("http 429")
}

impl IntoResponse for GailError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = match &self {
            Self::Unauthorized => ErrorBody {
                error: "unauthorized",
                message: self.to_string(),
                provider: None,
                quota: false,
                timeout: false,
            },
            Self::Upstream {
                provider,
                message,
                quota,
                timeout,
                ..
            } => ErrorBody {
                error: "upstream_error",
                message: message.clone(),
                provider: Some(provider.as_str()),
                quota: *quota,
                timeout: *timeout,
            },
            Self::BadRequest(message) | Self::NotFound(message) | Self::InvalidConfig(message) => {
                ErrorBody {
                    error: "request_error",
                    message: message.to_string(),
                    provider: None,
                    quota: false,
                    timeout: false,
                }
            }
            Self::Multipart(message) => ErrorBody {
                error: "multipart_error",
                message: message.clone(),
                provider: None,
                quota: false,
                timeout: false,
            },
            _ => ErrorBody {
                error: "internal_error",
                message: self.to_string(),
                provider: None,
                quota: false,
                timeout: false,
            },
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_detects_nested_too_many_requests_as_quota() {
        let error = GailError::upstream(
            "gail",
            None,
            r#"nvidia upstream error: {"status":429,"title":"Too Many Requests"}"#,
        );
        assert!(error.is_quota());
        assert!(message_indicates_quota(r#"{"status": 429}"#));
        assert!(message_indicates_quota("HTTP 429 from upstream"));
        assert!(message_indicates_quota("rate_limit_exceeded"));
    }
}
