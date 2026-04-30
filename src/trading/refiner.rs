/// Refiner RAG research client.
///
/// Calls `POST /api/rag/query` on the Refiner service to obtain external
/// market information and sentiment context for AI advisors.
use std::time::Duration;

use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;

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

    /// Query the Refiner RAG index for market research context.
    pub async fn research(
        &self,
        index_name: &str,
        query: &str,
        top_k: usize,
    ) -> Result<ResearchContext, String> {
        let url = format!("{}/api/rag/query", self.base_url);
        let body = json!({
            "name": index_name,
            "query": query,
            "top_k": top_k,
            "min_score": 0.3
        });
        let mut req = self.client.post(&url).json(&body);
        if let Some(ref token) = self.api_token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("Refiner RAG request failed: {e}"))?;

        let status = resp.status();
        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Refiner RAG response parse failed: {e}"))?;

        if !status.is_success() {
            return Err(format!(
                "Refiner RAG error (HTTP {}): {}",
                status.as_u16(),
                data.get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            ));
        }

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

        Ok(ResearchContext {
            query: query.to_string(),
            context,
            matches,
            source: format!("refiner/{}", index_name),
        })
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
