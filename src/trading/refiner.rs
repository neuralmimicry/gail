/// Refiner RAG research client.
///
/// Calls `POST /api/rag/query` on the Refiner service to obtain external
/// market information and sentiment context for AI advisors.
use std::time::Duration;

use futures::future::join_all;
use reqwest::{Client, ClientBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use url::Url;

use crate::{adaptive_schema, api_issues};

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagMatch {
    pub id: String,
    pub score: f64,
    pub content: String,
    pub source: Option<String>,
    pub citation: Option<String>,
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

fn normalize_site_hint(hint: &str) -> Option<String> {
    let trimmed = hint.trim();
    if trimmed.is_empty() {
        return None;
    }
    let without_site_prefix = trimmed
        .strip_prefix("site:")
        .or_else(|| trimmed.strip_prefix("SITE:"))
        .unwrap_or(trimmed)
        .trim();
    if without_site_prefix.is_empty() {
        return None;
    }
    let url_like = if without_site_prefix.contains("://") {
        without_site_prefix.to_string()
    } else {
        format!("https://{without_site_prefix}")
    };
    let parsed = Url::parse(&url_like).ok()?;
    let mut host = parsed
        .host_str()?
        .trim()
        .trim_start_matches("*.")
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if let Some(stripped) = host.strip_prefix("www.") {
        host = stripped.to_string();
    }
    if host.is_empty()
        || !host
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '-')
    {
        return None;
    }
    Some(host)
}

fn normalize_site_hints(site_hints: &[String]) -> Vec<String> {
    let mut normalized: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for hint in site_hints {
        if let Some(host) = normalize_site_hint(hint) {
            if seen.insert(host.clone()) {
                normalized.push(host);
            }
        }
    }
    normalized
}

fn build_site_queries(
    base_query: &str,
    site_hints: &[String],
    max_parallel_queries: usize,
) -> Vec<String> {
    let mut queries: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let max_parallel_queries = max_parallel_queries.max(1);
    let base_query = base_query.trim();
    if !base_query.is_empty() && seen.insert(base_query.to_ascii_lowercase()) {
        queries.push(base_query.to_string());
    }
    for site in site_hints {
        if queries.len() >= max_parallel_queries {
            break;
        }
        let query = if base_query.is_empty() {
            format!("site:{site}")
        } else {
            format!("{base_query} site:{site}")
        };
        if seen.insert(query.to_ascii_lowercase()) {
            queries.push(query);
        }
    }
    if queries.is_empty() {
        queries.push(base_query.to_string());
    }
    queries
}

fn parse_rag_match(raw: &serde_json::Value) -> Option<RagMatch> {
    let id = raw
        .get("id")
        .or_else(|| raw.get("chunk_id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let content = raw
        .get("content")
        .or_else(|| raw.get("text"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if id.is_empty() && content.is_empty() {
        return None;
    }
    let source = raw
        .get("source")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let citation = raw
        .get("citation")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(RagMatch {
        id: if id.is_empty() {
            format!("match:{}", content.chars().take(32).collect::<String>())
        } else {
            id
        },
        score: raw
            .get("score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        content,
        source,
        citation,
    })
}

fn merge_research_contexts(
    base_query: &str,
    index_name: &str,
    site_hints: &[String],
    top_k: usize,
    contexts: Vec<ResearchContext>,
) -> ResearchContext {
    let mut deduped_matches: Vec<RagMatch> = Vec::new();
    let mut seen_match_keys = std::collections::HashSet::new();
    let mut context_blocks = Vec::new();
    let mut sources = std::collections::BTreeSet::new();

    for ctx in contexts {
        if !ctx.context.trim().is_empty() {
            context_blocks.push(format!("Query: {}\n{}", ctx.query, ctx.context.trim()));
        }
        for m in ctx.matches {
            if let Some(source) = m.source.as_deref() {
                sources.insert(source.to_string());
            }
            if let Some(citation) = m.citation.as_deref() {
                sources.insert(citation.to_string());
            }
            let key = format!(
                "{}|{}|{}",
                m.id.to_ascii_lowercase(),
                m.source.as_deref().unwrap_or_default().to_ascii_lowercase(),
                m.content
                    .chars()
                    .take(128)
                    .collect::<String>()
                    .to_ascii_lowercase()
            );
            if seen_match_keys.insert(key) {
                deduped_matches.push(m);
            }
        }
    }

    deduped_matches.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if top_k > 0 && deduped_matches.len() > top_k {
        deduped_matches.truncate(top_k);
    }

    if context_blocks.is_empty() {
        let fallback = deduped_matches
            .iter()
            .map(|m| m.content.trim())
            .filter(|content| !content.is_empty())
            .take(3)
            .collect::<Vec<_>>()
            .join("\n\n");
        if !fallback.is_empty() {
            context_blocks.push(fallback);
        }
    }

    let mut header_lines: Vec<String> = Vec::new();
    if !site_hints.is_empty() {
        header_lines.push(format!("Preferred sites: {}", site_hints.join(", ")));
    }
    if !sources.is_empty() {
        header_lines.push(format!(
            "Matched sources: {}",
            sources.into_iter().take(12).collect::<Vec<_>>().join(", ")
        ));
    }
    let header = if header_lines.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", header_lines.join("\n"))
    };

    let context = format!("{}{}", header, context_blocks.join("\n\n---\n\n"));

    ResearchContext {
        query: base_query.to_string(),
        context,
        matches: deduped_matches,
        source: format!("refiner/{index_name}"),
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
            let resp = match self
                .with_auth(self.client.post(&url).json(&body))
                .send()
                .await
            {
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
                    api_issues::observe_api_recovery(
                        "refiner",
                        "POST",
                        "/api/rag/query",
                        "rag query",
                    )
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

            let matches = data
                .get("matches")
                .and_then(serde_json::Value::as_array)
                .map(|arr| arr.iter().filter_map(parse_rag_match).collect::<Vec<_>>())
                .unwrap_or_default();

            let mut context = data
                .get("context")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            if context.trim().is_empty() && !matches.is_empty() {
                context = matches
                    .iter()
                    .map(|m| m.content.trim())
                    .filter(|content| !content.is_empty())
                    .take(top_k.max(1))
                    .collect::<Vec<_>>()
                    .join("\n\n");
            }

            return Ok(ResearchContext {
                query: query.to_string(),
                context,
                matches,
                source: format!("refiner/{}", index_name),
            });
        }
    }

    /// Query Refiner using the base query plus optional `site:<domain>` variants in parallel.
    pub async fn research_with_site_hints(
        &self,
        index_name: &str,
        query: &str,
        site_hints: &[String],
        top_k: usize,
        max_parallel_queries: usize,
    ) -> Result<ResearchContext, String> {
        let normalized_site_hints = normalize_site_hints(site_hints);
        let queries = build_site_queries(query, &normalized_site_hints, max_parallel_queries);
        let results = join_all(
            queries
                .iter()
                .map(|q| self.research(index_name, q, top_k.max(1))),
        )
        .await;
        let mut contexts: Vec<ResearchContext> = Vec::new();
        let mut errors: Vec<String> = Vec::new();
        for result in results {
            match result {
                Ok(context) => contexts.push(context),
                Err(err) => errors.push(err),
            }
        }
        if contexts.is_empty() {
            let details = if errors.is_empty() {
                "unknown error".to_string()
            } else {
                errors.join(" | ")
            };
            return Err(format!(
                "Refiner site-hinted research failed for index '{index_name}': {details}"
            ));
        }
        Ok(merge_research_contexts(
            query,
            index_name,
            &normalized_site_hints,
            top_k.max(1),
            contexts,
        ))
    }

    /// Best-effort source-aware research — returns empty context on failure instead of error.
    pub async fn research_with_site_hints_best_effort(
        &self,
        index_name: &str,
        query: &str,
        site_hints: &[String],
        top_k: usize,
        max_parallel_queries: usize,
    ) -> ResearchContext {
        match self
            .research_with_site_hints(index_name, query, site_hints, top_k, max_parallel_queries)
            .await
        {
            Ok(ctx) => ctx,
            Err(err) => {
                tracing::warn!(
                    "trading: Refiner source-aware research failed (using empty context): {}",
                    err
                );
                ResearchContext::empty(query)
            }
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

    #[test]
    fn site_hint_normalization_handles_urls_and_site_prefix() {
        let hints = vec![
            " bloomberg.com ".to_string(),
            "site:Reuters.com/markets".to_string(),
            "https://www.Bloomberg.com/markets".to_string(),
            "".to_string(),
        ];
        let normalized = normalize_site_hints(&hints);
        assert_eq!(
            normalized,
            vec!["bloomberg.com".to_string(), "reuters.com".to_string()]
        );
    }

    #[test]
    fn build_site_queries_respects_parallel_limit() {
        let hints = vec![
            "bloomberg.com".to_string(),
            "reuters.com".to_string(),
            "cnbc.com".to_string(),
        ];
        let queries = build_site_queries("btc market momentum", &hints, 2);
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0], "btc market momentum");
        assert_eq!(queries[1], "btc market momentum site:bloomberg.com");
    }

    #[tokio::test]
    async fn research_parses_openapi_match_shape_and_falls_back_to_match_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/rag/query"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "name": "crypto",
                "query": "btc market momentum",
                "context": "",
                "matches": [
                    {
                        "chunk_id": "chunk-1",
                        "score": 0.91,
                        "text": "Bloomberg reports BTC inflow acceleration.",
                        "source": "https://www.bloomberg.com/markets/cryptocurrencies",
                        "citation": "bloomberg:2026-05-15"
                    }
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = RefinerClient::new(&server.uri(), None, 1.0);
        let context = client
            .research("crypto", "btc market momentum", 5)
            .await
            .expect("query should succeed");
        assert!(!context.is_empty());
        assert_eq!(context.matches.len(), 1);
        assert_eq!(context.matches[0].id, "chunk-1");
        assert_eq!(
            context.matches[0].source.as_deref(),
            Some("https://www.bloomberg.com/markets/cryptocurrencies")
        );
        assert_eq!(
            context.matches[0].citation.as_deref(),
            Some("bloomberg:2026-05-15")
        );
        assert!(
            context
                .context
                .contains("Bloomberg reports BTC inflow acceleration")
        );
    }

    #[tokio::test]
    async fn research_with_site_hints_fans_out_queries_and_merges_results() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/rag/query"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "context": "Market breadth improving",
                "matches": [
                    {
                        "chunk_id": "chunk-1",
                        "score": 0.81,
                        "text": "Market breadth improving",
                        "source": "https://www.bloomberg.com"
                    }
                ]
            })))
            .expect(3)
            .mount(&server)
            .await;

        let client = RefinerClient::new(&server.uri(), None, 1.0);
        let context = client
            .research_with_site_hints(
                "crypto",
                "btc market sentiment",
                &vec![
                    "bloomberg.com".to_string(),
                    "reuters.com".to_string(),
                    "cnbc.com".to_string(),
                ],
                5,
                3,
            )
            .await
            .expect("site-hinted research should succeed");
        assert_eq!(context.query, "btc market sentiment");
        assert!(
            context
                .context
                .contains("Preferred sites: bloomberg.com, reuters.com, cnbc.com")
        );
        assert!(context.context.contains("Query: btc market sentiment"));
        assert_eq!(context.matches.len(), 1, "deduped match list expected");
    }
}
