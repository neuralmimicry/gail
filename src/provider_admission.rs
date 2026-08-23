//! Durable comparative-validation state for providers that are not trusted by
//! generic routing until they have demonstrated useful responses.

use serde_json::Value;
use tokio_postgres::NoTls;

use crate::config::ComparativeValidationConfig;

#[derive(Clone, Debug)]
pub struct AdmissionObservation {
    pub kind: String,
    pub endpoint: String,
    pub model: String,
    pub model_version: String,
    pub useful: bool,
    pub candidate_quality: f64,
    pub baseline_quality: f64,
    pub similarity: f64,
    pub latency_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProviderAdmission {
    pub kind: String,
    pub endpoint: String,
    pub model: String,
    pub model_version: String,
}

#[derive(Clone, Debug, Default)]
pub struct AdmissionDecision {
    pub admitted: bool,
    pub sample_count: u64,
    pub useful_count: u64,
    pub average_quality: f64,
    pub average_baseline_quality: f64,
    pub average_similarity: f64,
}

pub async fn record_observation(
    dsn: &str,
    observation: &AdmissionObservation,
    config: &ComparativeValidationConfig,
) -> Result<AdmissionDecision, tokio_postgres::Error> {
    let client = connect(dsn).await?;
    let endpoint = normalize_endpoint(&observation.endpoint);
    let latency_ms = observation
        .latency_ms
        .map(|value| value.min(i64::MAX as u64) as i64);
    let error = observation.error.as_deref().map(truncate_error);
    let row = client
        .query_one(
            r#"
            INSERT INTO gail_provider_admissions (
                kind, endpoint, model, model_version,
                sample_count, useful_count,
                average_quality, average_baseline_quality, average_similarity,
                last_quality, last_baseline_quality, last_similarity,
                last_latency_ms, last_error, last_sample_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, 1, $5, $6, $7, $8, $6, $7, $8,
                $9, $10, now(), now()
            )
            ON CONFLICT (kind, endpoint, model) DO UPDATE SET
                model_version = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN gail_provider_admissions.model_version
                    ELSE EXCLUDED.model_version
                END,
                sample_count = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN gail_provider_admissions.sample_count + 1 ELSE 1 END,
                useful_count = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN gail_provider_admissions.useful_count + EXCLUDED.useful_count ELSE EXCLUDED.useful_count END,
                average_quality = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN (gail_provider_admissions.average_quality * gail_provider_admissions.sample_count + EXCLUDED.average_quality)
                         / (gail_provider_admissions.sample_count + 1)
                    ELSE EXCLUDED.average_quality END,
                average_baseline_quality = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN (gail_provider_admissions.average_baseline_quality * gail_provider_admissions.sample_count + EXCLUDED.average_baseline_quality)
                         / (gail_provider_admissions.sample_count + 1)
                    ELSE EXCLUDED.average_baseline_quality END,
                average_similarity = CASE
                    WHEN gail_provider_admissions.model_version = EXCLUDED.model_version
                    THEN (gail_provider_admissions.average_similarity * gail_provider_admissions.sample_count + EXCLUDED.average_similarity)
                         / (gail_provider_admissions.sample_count + 1)
                    ELSE EXCLUDED.average_similarity END,
                last_quality = EXCLUDED.last_quality,
                last_baseline_quality = EXCLUDED.last_baseline_quality,
                last_similarity = EXCLUDED.last_similarity,
                last_latency_ms = EXCLUDED.last_latency_ms,
                last_error = EXCLUDED.last_error,
                last_sample_at = now(),
                updated_at = now(),
                admitted = false
            RETURNING sample_count, useful_count, average_quality,
                      average_baseline_quality, average_similarity
            "#,
            &[
                &observation.kind,
                &endpoint,
                &observation.model,
                &observation.model_version,
                &(observation.useful as i64),
                &observation.candidate_quality,
                &observation.baseline_quality,
                &observation.similarity,
                &latency_ms,
                &error,
            ],
        )
        .await?;
    let sample_count = row.get::<_, i64>("sample_count").max(0) as u64;
    let useful_count = row.get::<_, i64>("useful_count").max(0) as u64;
    let average_quality = row.get::<_, f64>("average_quality");
    let average_baseline_quality = row.get::<_, f64>("average_baseline_quality");
    let average_similarity = row.get::<_, f64>("average_similarity");
    let useful_rate = useful_count as f64 / sample_count.max(1) as f64;
    let comparable = average_similarity >= config.minimum_similarity
        && average_quality + config.quality_tolerance >= average_baseline_quality;
    let better = average_quality >= average_baseline_quality + config.minimum_quality_improvement;
    let admitted = sample_count >= config.minimum_samples as u64
        && useful_rate >= config.minimum_useful_rate
        && average_quality >= config.minimum_candidate_quality
        && (comparable || better);
    client
        .execute(
            "UPDATE gail_provider_admissions SET admitted = $4, updated_at = now() WHERE kind = $1 AND endpoint = $2 AND model = $3",
            &[&observation.kind, &endpoint, &observation.model, &admitted],
        )
        .await?;
    Ok(AdmissionDecision {
        admitted,
        sample_count,
        useful_count,
        average_quality,
        average_baseline_quality,
        average_similarity,
    })
}

pub async fn admitted_for_model(
    dsn: &str,
    model_version: &str,
    max_age_seconds: u64,
) -> Result<Vec<ProviderAdmission>, tokio_postgres::Error> {
    let client = connect(dsn).await?;
    let rows = client
        .query(
            r#"
            SELECT kind, endpoint, model, model_version
            FROM gail_provider_admissions
            WHERE admitted
              AND model_version = $1
              AND last_sample_at >= now() - make_interval(secs => $2::int)
            "#,
            &[
                &model_version,
                &(max_age_seconds.min(i32::MAX as u64) as i32),
            ],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| ProviderAdmission {
            kind: row.get("kind"),
            endpoint: row.get("endpoint"),
            model: row.get("model"),
            model_version: row.get("model_version"),
        })
        .collect())
}

pub async fn admitted_for_kind(
    dsn: &str,
    kind: &str,
    endpoint: &str,
    model: &str,
    max_age_seconds: u64,
) -> Result<bool, tokio_postgres::Error> {
    let client = connect(dsn).await?;
    let found = client
        .query_opt(
            r#"
            SELECT 1
            FROM gail_provider_admissions
            WHERE admitted
              AND kind = $1
              AND endpoint = $2
              AND model = $3
              AND last_sample_at >= now() - make_interval(secs => $4::int)
            "#,
            &[
                &kind,
                &normalize_endpoint(endpoint),
                &model,
                &(max_age_seconds.min(i32::MAX as u64) as i32),
            ],
        )
        .await?;
    Ok(found.is_some())
}

pub fn admission_endpoint_matches(left: &str, right: &str) -> bool {
    normalize_endpoint(left) == normalize_endpoint(right)
}

pub fn admission_model_matches(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}

pub fn normalize_endpoint(endpoint: &str) -> String {
    endpoint.trim().trim_end_matches('/').to_ascii_lowercase()
}

/// Conservative, transport-independent usefulness score.  Empty, refusal,
/// placeholder, and reasoning-only responses must never become admitted just
/// because their endpoint was fast.
pub fn response_quality(text: &str, expected_json: bool) -> f64 {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return 0.0;
    }
    let lowered = cleaned.to_ascii_lowercase();
    if [
        "i can't help",
        "i cannot help",
        "i'm unable",
        "i am unable",
        "cannot answer",
        "no answer",
        "service unavailable",
        "model not found",
        "upstream error",
        "as an ai language model",
        "<think>",
    ]
    .iter()
    .any(|marker| lowered.contains(marker))
        || matches!(lowered.as_str(), "n/a" | "null" | "none" | "..." | "-")
    {
        return 0.0;
    }
    if expected_json {
        let parsed = serde_json::from_str::<Value>(cleaned).or_else(|_| {
            cleaned
                .strip_prefix("```")
                .and_then(|value| value.strip_suffix("```"))
                .map(|value| value.trim().strip_prefix("json").unwrap_or(value).trim())
                .ok_or_else(|| serde_json::Error::io(std::io::Error::other("not json")))
                .and_then(serde_json::from_str::<Value>)
        });
        let valid = match parsed {
            Ok(Value::Object(value)) => !value.is_empty(),
            Ok(Value::Array(value)) => !value.is_empty(),
            Ok(Value::Null) | Err(_) => false,
            Ok(_) => true,
        };
        if !valid {
            return 0.0;
        }
    }
    let length_score: f64 = if cleaned.chars().count() >= 40 {
        0.25
    } else {
        0.10
    };
    (0.65 + length_score).min(1.0)
}

pub fn token_similarity(left: &str, right: &str) -> f64 {
    use std::collections::HashSet;
    let left = left
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let right = right
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    left.intersection(&right).count() as f64 / left.union(&right).count() as f64
}

fn truncate_error(error: &str) -> String {
    error.chars().take(4000).collect()
}

#[cfg(test)]
mod tests {
    use super::{response_quality, token_similarity};

    #[test]
    fn usefulness_rejects_empty_refusal_and_reasoning_only_text() {
        assert_eq!(response_quality("", false), 0.0);
        assert_eq!(response_quality("I cannot help with that.", false), 0.0);
        assert_eq!(
            response_quality("<think>internal reasoning</think>", false),
            0.0
        );
        assert!(
            response_quality("A useful answer with enough detail for the caller.", false) > 0.5
        );
    }

    #[test]
    fn usefulness_requires_valid_nonempty_json_when_requested() {
        assert_eq!(response_quality("{}", true), 0.0);
        assert_eq!(response_quality("not json", true), 0.0);
        assert!(response_quality(r#"{"ok":true}"#, true) > 0.5);
    }

    #[test]
    fn similarity_is_order_independent_token_overlap() {
        let score = token_similarity("A useful answer for Gail", "Gail answer useful");
        assert!(score > 0.5);
        assert!(token_similarity("one", "different") < score);
    }
}

async fn connect(dsn: &str) -> Result<tokio_postgres::Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            tracing::debug!(error = %error, "provider admission Postgres connection closed");
        }
    });
    Ok(client)
}
