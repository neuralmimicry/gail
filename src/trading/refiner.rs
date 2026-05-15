/// Refiner RAG research client.
///
/// Calls `POST /api/rag/query` on the Refiner service to obtain external
/// market information and sentiment context for AI advisors.
use std::{
    cmp::Ordering,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

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
    pub published_at_unix: Option<i64>,
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
    let published_at_unix = infer_match_timestamp(raw, source.as_deref(), citation.as_deref())
        .and_then(normalize_unix_ts);
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
        published_at_unix,
    })
}

fn normalize_unix_ts(ts: i64) -> Option<i64> {
    let seconds = if ts.abs() >= 1_000_000_000_000 {
        ts / 1000
    } else {
        ts
    };
    if (946_684_800..=4_102_444_800).contains(&seconds) {
        Some(seconds)
    } else {
        None
    }
}

fn parse_timestamp_value(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_u64().and_then(|v| i64::try_from(v).ok()))
            .or_else(|| n.as_f64().map(|v| v.round() as i64))
            .and_then(normalize_unix_ts),
        serde_json::Value::String(s) => parse_timestamp_string(s),
        _ => None,
    }
}

fn parse_timestamp_string(raw: &str) -> Option<i64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if (10..=16).contains(&trimmed.len()) && trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        if let Ok(parsed) = trimmed.parse::<i64>() {
            if let Some(ts) = normalize_unix_ts(parsed) {
                return Some(ts);
            }
        }
    }
    parse_yyyy_mm_dd_timestamp(trimmed)
}

fn parse_yyyy_mm_dd_timestamp(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.len() < 10 {
        return None;
    }
    for idx in 0..=bytes.len() - 10 {
        let Some(year) = parse_digits(bytes, idx, 4) else {
            continue;
        };
        let Some(sep) = bytes.get(idx + 4).copied() else {
            continue;
        };
        if sep != b'-' && sep != b'/' {
            continue;
        }
        let Some(month) = parse_digits(bytes, idx + 5, 2) else {
            continue;
        };
        if bytes.get(idx + 7).copied() != Some(sep) {
            continue;
        }
        let Some(day) = parse_digits(bytes, idx + 8, 2) else {
            continue;
        };
        if !(1..=12).contains(&month) {
            continue;
        }
        let day_max = days_in_month(year as i32, month as u32) as i64;
        if !(1..=day_max).contains(&day) {
            continue;
        }
        return unix_ts_from_ymd(year as i32, month as u32, day as u32);
    }
    None
}

fn parse_digits(bytes: &[u8], start: usize, len: usize) -> Option<i64> {
    let mut value: i64 = 0;
    for offset in 0..len {
        let ch = *bytes.get(start + offset)?;
        if !ch.is_ascii_digit() {
            return None;
        }
        value = value * 10 + i64::from(ch - b'0');
    }
    Some(value)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let mut y = i64::from(year);
    let m = i64::from(month);
    let d = i64::from(day);
    if m <= 2 {
        y -= 1;
    }
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn unix_ts_from_ymd(year: i32, month: u32, day: u32) -> Option<i64> {
    let days = days_from_civil(year, month, day);
    normalize_unix_ts(days.saturating_mul(86_400))
}

fn infer_match_timestamp(
    raw: &serde_json::Value,
    source: Option<&str>,
    citation: Option<&str>,
) -> Option<i64> {
    let mut candidates: Vec<i64> = Vec::new();
    let timestamp_fields = [
        "published_at",
        "publishedAt",
        "updated_at",
        "updatedAt",
        "created_at",
        "createdAt",
        "timestamp",
        "ts",
        "time",
        "date",
        "datetime",
        "fetched_at",
        "last_modified",
    ];
    if let Some(obj) = raw.as_object() {
        for key in &timestamp_fields {
            if let Some(value) = obj.get(*key) {
                if let Some(ts) = parse_timestamp_value(value) {
                    candidates.push(ts);
                }
            }
        }
        if let Some(metadata) = obj.get("metadata").and_then(serde_json::Value::as_object) {
            for (key, value) in metadata {
                let lowered = key.to_ascii_lowercase();
                if lowered.contains("time")
                    || lowered.contains("date")
                    || lowered.contains("publish")
                    || lowered.contains("update")
                    || lowered.contains("create")
                    || lowered.contains("stamp")
                {
                    if let Some(ts) = parse_timestamp_value(value) {
                        candidates.push(ts);
                    }
                }
            }
        }
    }
    for text in [source, citation].into_iter().flatten() {
        if let Some(ts) = parse_timestamp_string(text) {
            candidates.push(ts);
        }
    }
    candidates.into_iter().max()
}

fn now_unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn utc_date_from_unix_days(days_since_epoch: i64) -> String {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    format!("{year:04}-{month:02}-{day:02}")
}

fn format_match_timestamp(ts: i64) -> String {
    utc_date_from_unix_days(ts.div_euclid(86_400))
}

fn recency_order(left: &RagMatch, right: &RagMatch) -> Ordering {
    match (left.published_at_unix, right.published_at_unix) {
        (Some(left_ts), Some(right_ts)) => right_ts.cmp(&left_ts).then_with(|| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal),
    }
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
    let mut fallback_context_blocks = Vec::new();
    let mut sources = std::collections::BTreeSet::new();
    let now_ts = now_unix_ts();

    for ctx in contexts {
        if !ctx.context.trim().is_empty() {
            fallback_context_blocks.push(format!("Query: {}\n{}", ctx.query, ctx.context.trim()));
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

    deduped_matches.sort_by(recency_order);
    let dated_match_count = deduped_matches
        .iter()
        .filter(|m| m.published_at_unix.is_some())
        .count();

    if top_k > 0 && deduped_matches.len() > top_k {
        deduped_matches.truncate(top_k);
    }

    let recency_blocks = deduped_matches
        .iter()
        .filter_map(|m| {
            let content = m.content.trim();
            if content.is_empty() {
                return None;
            }
            let source_label = m
                .source
                .as_deref()
                .or(m.citation.as_deref())
                .unwrap_or("unknown_source");
            let line = match m.published_at_unix {
                Some(ts) => {
                    let age_days = now_ts.saturating_sub(ts).max(0) / 86_400;
                    format!(
                        "score={:.3} date={} age_days={} source={}",
                        m.score,
                        format_match_timestamp(ts),
                        age_days,
                        source_label
                    )
                }
                None => format!(
                    "score={:.3} date=unknown age_days=unknown source={}",
                    m.score, source_label
                ),
            };
            Some(format!("[{line}]\n{content}"))
        })
        .take(top_k.max(1))
        .collect::<Vec<_>>();

    let context_body = if !recency_blocks.is_empty() {
        recency_blocks.join("\n\n")
    } else if !fallback_context_blocks.is_empty() {
        fallback_context_blocks.join("\n\n---\n\n")
    } else {
        String::new()
    };

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
    header_lines.push(
        "Time relevance policy: newest timestamped evidence ranks above older or undated matches."
            .to_string(),
    );
    header_lines.push(format!(
        "Reference date (UTC): {}",
        format_match_timestamp(now_ts)
    ));
    header_lines.push(format!(
        "Timestamp coverage: {dated_match_count}/{} matches carried parseable dates.",
        deduped_matches.len()
    ));
    let header = if header_lines.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", header_lines.join("\n"))
    };

    let context = format!("{}{}", header, context_body);

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

    #[test]
    fn parse_timestamp_string_supports_iso_date_and_unix_seconds() {
        let iso = parse_timestamp_string("published on 2026-05-15 by Bloomberg")
            .expect("must parse ISO date");
        assert_eq!(format_match_timestamp(iso), "2026-05-15");

        let unix = parse_timestamp_string("1715731200").expect("must parse unix seconds");
        assert_eq!(format_match_timestamp(unix), "2024-05-15");
    }

    #[test]
    fn merge_research_contexts_prioritizes_newer_dated_matches() {
        let contexts = vec![ResearchContext {
            query: "btc market sentiment".to_string(),
            context: "legacy context".to_string(),
            matches: vec![
                RagMatch {
                    id: "old-high-score".to_string(),
                    score: 0.99,
                    content: "Older article".to_string(),
                    source: Some("https://example.com/2024/01/01/report".to_string()),
                    citation: None,
                    published_at_unix: unix_ts_from_ymd(2024, 1, 1),
                },
                RagMatch {
                    id: "newer-lower-score".to_string(),
                    score: 0.61,
                    content: "More recent article".to_string(),
                    source: Some("https://example.com/2026/05/15/report".to_string()),
                    citation: None,
                    published_at_unix: unix_ts_from_ymd(2026, 5, 15),
                },
                RagMatch {
                    id: "undated-high-score".to_string(),
                    score: 1.0,
                    content: "Undated article".to_string(),
                    source: Some("https://example.com/undated".to_string()),
                    citation: None,
                    published_at_unix: None,
                },
            ],
            source: "refiner/crypto".to_string(),
        }];
        let merged = merge_research_contexts(
            "btc market sentiment",
            "crypto",
            &vec!["bloomberg.com".to_string()],
            5,
            contexts,
        );

        assert_eq!(merged.matches[0].id, "newer-lower-score");
        assert_eq!(merged.matches[1].id, "old-high-score");
        assert_eq!(merged.matches[2].id, "undated-high-score");
        assert!(merged.context.contains("Time relevance policy"));
        assert!(merged.context.contains("Timestamp coverage:"));
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
        assert_eq!(
            context.matches[0].published_at_unix,
            unix_ts_from_ymd(2026, 5, 15)
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
        assert!(
            context
                .context
                .contains("Time relevance policy: newest timestamped evidence ranks above older or undated matches.")
        );
        assert!(context.context.contains("Market breadth improving"));
        assert_eq!(context.matches.len(), 1, "deduped match list expected");
    }
}
