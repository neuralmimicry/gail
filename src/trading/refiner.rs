/// Refiner RAG research client.
///
/// Calls `POST /api/rag/query` on the Refiner service to obtain external
/// market information and sentiment context for AI advisors.
use std::time::Duration;

use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{adaptive_schema, api_issues};

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagMatch {
    pub id: String,
    pub score: f64,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResearchContext {
    pub query: String,
    /// Rendered context block suitable for injecting into an LLM prompt.
    pub context: String,
    pub matches: Vec<RagMatch>,
    pub source: String,
}

impl ResearchContext {
    pub fn empty(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            context: String::new(),
            matches: Vec::new(),
            source: "none".to_string(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.context.trim().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RefinerClient {
    client: Client,
    base_url: String,
    api_token: Option<String>,
}

impl RefinerClient {
    pub fn new(base_url: &str, api_token: Option<&str>, timeout_seconds: f64) -> Self {
        let client = ClientBuilder::new()
            .use_rustls_tls()
            .timeout(Duration::from_secs_f64(timeout_seconds))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_token: api_token.map(str::to_string),
        }
    }

    fn with_auth(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref token) = self.api_token {
            request.header("Authorization", format!("Bearer {token}"))
        } else {
            request
        }
    }

    async fn bootstrap_index(&self, index_name: &str) -> Result<(), String> {
        let url = format!("{}/api/rag/index", self.base_url);
        let source_id = format!("gail-bootstrap-{}", index_name.trim());
        let body = json!({
            "name": index_name,
            "sources": [
                {
                    "id": source_id,
                    "title": "gail-trading-bootstrap",
                    "text": "Bootstrap placeholder for Gail trading research index."
                }
            ]
        });
        let response = self
            .with_auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|err| format!("Refiner RAG index bootstrap request failed: {err}"))?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let payload: serde_json::Value = response.json().await.unwrap_or_else(|_| json!({}));
        let error = payload
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        Err(format!(
            "Refiner RAG index bootstrap failed (HTTP {}): {}",
            status.as_u16(),
            error
        ))
    }

    /// Query the Refiner RAG index for market research context.
    pub async fn research(
        &self,
        index_name: &str,
        query: &str,
        top_k: usize,
    ) -> Result<ResearchContext, String> {
        let mut bootstrap_attempted = false;
        loop {
            let url = format!("{}/api/rag/query", self.base_url);
            let body = json!({
                "name": index_name,
                "query": query,
                "top_k": top_k,
                "min_score": 0.3
            });
            let resp = match self.with_auth(self.client.post(&url).json(&body)).send().await {
                Ok(resp) => resp,
                Err(err) => {
                    adaptive_schema::observe_failure(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                        None,
                        &err.to_string(),
                    )
                    .await;
                    api_issues::observe_api_failure(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                        None,
                        &err.to_string(),
                    )
                    .await;
                    return Err(format!("Refiner RAG request failed: {err}"));
                }
            };

            let status = resp.status();
            let data: serde_json::Value = match resp.json().await {
                Ok(data) => data,
                Err(err) => {
                    let message = format!("Refiner RAG response parse failed: {err}");
                    adaptive_schema::observe_failure(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                        Some(status.as_u16()),
                        &message,
                    )
                    .await;
                    api_issues::observe_api_failure(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                        Some(status.as_u16()),
                        &message,
                    )
                    .await;
                    return Err(message);
                }
            };

            if !status.is_success() {
                let error = data
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                if status.as_u16() == 404 && error == "index_not_found" {
                    if !bootstrap_attempted {
                        bootstrap_attempted = true;
                        match self.bootstrap_index(index_name).await {
                            Ok(()) => {
                                tracing::info!(
                                    index_name = index_name,
                                    "trading: bootstrapped missing Refiner RAG index"
                                );
                                continue;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    index_name = index_name,
                                    error = %err,
                                    "trading: failed to bootstrap Refiner RAG index"
                                );
                            }
                        }
                    }
                    adaptive_schema::observe_success(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                        &data,
                    )
                    .await;
                    api_issues::observe_api_recovery("refiner", "POST", "/api/rag/query", "rag query")
                        .await;
                    tracing::debug!(
                        index_name = index_name,
                        "trading: Refiner RAG index not found; using empty context"
                    );
                    return Ok(ResearchContext::empty(query));
                }
                adaptive_schema::observe_failure(
                    "refiner",
                    "POST",
                    "/api/rag/query",
                    "rag query",
                    Some(status.as_u16()),
                    error,
                )
                .await;
                api_issues::observe_api_failure(
                    "refiner",
                    "POST",
                    "/api/rag/query",
                    "rag query",
                    Some(status.as_u16()),
                    error,
                )
                .await;
                return Err(format!(
                    "Refiner RAG error (HTTP {}): {}",
                    status.as_u16(),
                    error
                ));
            }
            adaptive_schema::observe_success("refiner", "POST", "/api/rag/query", "rag query", &data)
                .await;
            api_issues::observe_api_recovery("refiner", "POST", "/api/rag/query", "rag query").await;

            let matches = data
                .get("matches")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            Some(RagMatch {
                                id: m.get("id").and_then(serde_json::Value::as_str)?.to_string(),
                                score: m
                                    .get("score")
                                    .and_then(serde_json::Value::as_f64)
                                    .unwrap_or(0.0),
                                content: m
                                    .get("content")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let context = data
                .get("context")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();

            return Ok(ResearchContext {
                query: query.to_string(),
                context,
                matches,
                source: format!("refiner/{}", index_name),
            });
        }
    }

    /// Best-effort research — returns empty context on failure instead of error.
    pub async fn research_best_effort(
        &self,
        index_name: &str,
        query: &str,
        top_k: usize,
    ) -> ResearchContext {
        match self.research(index_name, query, top_k).await {
            Ok(ctx) => ctx,
            Err(err) => {
                tracing::warn!(
                    "trading: Refiner research failed (using empty context): {}",
                    err
                );
                ResearchContext::empty(query)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn research_uses_empty_context_when_optional_index_is_missing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/rag/query"))
            .and(header("authorization", "Bearer refiner-token"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({
                "error": "index_not_found"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/rag/index"))
            .and(header("authorization", "Bearer refiner-token"))
            .respond_with(ResponseTemplate::new(500).set_body_json(json!({
                "error": "bootstrap_failed"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = RefinerClient::new(&server.uri(), Some("refiner-token"), 1.0);
        let context = client
            .research("crypto", "bitcoin market sentiment", 5)
            .await
            .expect("missing optional RAG index should not fail trading research");

        assert_eq!(context.query, "bitcoin market sentiment");
        assert!(context.is_empty());
        assert!(context.matches.is_empty());
    }

    #[tokio::test]
    async fn research_still_fails_on_refiner_auth_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/rag/query"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": "unauthorized"
            })))
            .mount(&server)
            .await;

        let client = RefinerClient::new(&server.uri(), None, 1.0);
        let err = client
            .research("crypto", "bitcoin market sentiment", 5)
            .await
            .expect_err("auth failures must remain visible");

        assert!(err.contains("HTTP 401"));
        assert!(err.contains("unauthorized"));
    }
}
