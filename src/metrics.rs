use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{fs, sync::Mutex};

use crate::{errors::Result, models::CandidateSummary};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MetricsData {
    pub candidates: HashMap<String, CandidateBucket>,
    pub updated_at: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CandidateBucket {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub configured_model: Option<String>,
    pub resolved_model: Option<String>,
    pub specialties: Vec<String>,
    pub stats: StatsBucket,
    pub roles: HashMap<String, StatsBucket>,
    pub health: HealthBucket,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StatsBucket {
    pub successes: u64,
    pub failures: u64,
    pub total: u64,
    pub ewma_latency_ms: Option<f64>,
    pub ewma_quality: f64,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    pub updated_at: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HealthBucket {
    pub ok: Option<bool>,
    pub mode: Option<String>,
    pub checked_at: Option<f64>,
    pub latency_ms: Option<u64>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateMetricsSummary {
    pub candidate_id: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub configured_model: Option<String>,
    pub resolved_model: Option<String>,
    pub specialties: Vec<String>,
    pub successes: u64,
    pub failures: u64,
    pub total: u64,
    pub success_rate: Option<f64>,
    pub ewma_latency_ms: Option<f64>,
    pub ewma_quality: f64,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    pub updated_at: Option<f64>,
    pub health_ok: Option<bool>,
    pub health_mode: Option<String>,
    pub health_checked_at: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub path: String,
    pub exists: bool,
    pub updated_at: f64,
    pub candidate_count: usize,
    pub healthy_candidates: usize,
    pub degraded_candidates: usize,
    pub candidates: Vec<CandidateMetricsSummary>,
}

#[derive(Clone)]
pub struct MetricsStore {
    path: PathBuf,
    inner: Arc<Mutex<MetricsData>>,
}

impl MetricsStore {
    pub async fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let data = match fs::read_to_string(&path).await {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
            Err(_) => MetricsData::default(),
        };
        Ok(Self {
            path,
            inner: Arc::new(Mutex::new(data)),
        })
    }

    pub fn path(&self) -> String {
        self.path.to_string_lossy().to_string()
    }

    async fn save(&self, data: &MetricsData) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let rendered = serde_json::to_string_pretty(data)?;
        fs::write(&self.path, rendered).await?;
        Ok(())
    }

    pub async fn should_probe(&self, candidate_id: &str, ttl_seconds: f64) -> bool {
        let data = self.inner.lock().await;
        let checked_at = data
            .candidates
            .get(candidate_id)
            .and_then(|bucket| bucket.health.checked_at)
            .unwrap_or(0.0);
        checked_at <= 0.0 || (now_ts() - checked_at) >= ttl_seconds
    }

    pub async fn health_snapshot(&self, candidate_id: &str) -> HealthBucket {
        let data = self.inner.lock().await;
        data.candidates
            .get(candidate_id)
            .map(|bucket| bucket.health.clone())
            .unwrap_or_default()
    }

    pub async fn provider_in_quota_backoff(&self, provider: &str, ttl_seconds: f64) -> bool {
        let provider = provider.trim();
        if provider.is_empty() {
            return false;
        }
        let now = now_ts();
        let data = self.inner.lock().await;
        data.candidates
            .iter()
            .filter(|(candidate_id, bucket)| {
                bucket
                    .provider
                    .as_deref()
                    .is_some_and(|item| item.eq_ignore_ascii_case(provider))
                    || candidate_id
                        .split_once('/')
                        .map(|(prefix, _)| prefix.eq_ignore_ascii_case(provider))
                        .unwrap_or(false)
            })
            .any(|(_, bucket)| {
                bucket
                    .health
                    .mode
                    .as_deref()
                    .is_some_and(|mode| mode.eq_ignore_ascii_case("quota"))
                    && bucket
                        .health
                        .checked_at
                        .is_some_and(|checked_at| now - checked_at < ttl_seconds)
            })
    }

    pub async fn record_health(
        &self,
        summary: &CandidateSummary,
        health: HealthBucket,
    ) -> Result<()> {
        let mut data = self.inner.lock().await;
        let bucket = data
            .candidates
            .entry(summary.candidate_id.clone())
            .or_default();
        bucket.provider = Some(summary.provider.clone());
        bucket.model = Some(summary.model.clone());
        bucket.configured_model = Some(summary.configured_model.clone());
        bucket.resolved_model = Some(summary.resolved_model.clone());
        bucket.specialties = summary.specialties.clone();
        bucket.health = HealthBucket {
            checked_at: Some(now_ts()),
            ..health
        };
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    fn merge_stats(
        bucket: &mut StatsBucket,
        success: bool,
        latency_ms: Option<u64>,
        quality: f64,
        error: Option<&str>,
    ) {
        if success {
            bucket.successes += 1;
        } else {
            bucket.failures += 1;
        }
        bucket.total = bucket.successes + bucket.failures;
        if let Some(latency_ms) = latency_ms {
            bucket.ewma_latency_ms = Some(match bucket.ewma_latency_ms {
                Some(previous) => (previous * 0.75) + (latency_ms as f64 * 0.25),
                None => latency_ms as f64,
            });
        }
        bucket.ewma_quality = (bucket.ewma_quality * 0.75) + (quality * 0.25);
        bucket.last_status = Some(if success { "success" } else { "failure" }.to_string());
        bucket.last_error = error.map(|value| value.to_string());
        bucket.updated_at = Some(now_ts());
    }

    pub async fn record_result(
        &self,
        summary: &CandidateSummary,
        workflow: &str,
        role: &str,
        success: bool,
        latency_ms: Option<u64>,
        quality: f64,
        error: Option<&str>,
    ) -> Result<()> {
        let mut data = self.inner.lock().await;
        let bucket = data
            .candidates
            .entry(summary.candidate_id.clone())
            .or_default();
        bucket.provider = Some(summary.provider.clone());
        bucket.model = Some(summary.model.clone());
        bucket.configured_model = Some(summary.configured_model.clone());
        bucket.resolved_model = Some(summary.resolved_model.clone());
        bucket.specialties = summary.specialties.clone();
        Self::merge_stats(&mut bucket.stats, success, latency_ms, quality, error);
        let role_key = format!("{workflow}:{role}");
        let role_bucket = bucket.roles.entry(role_key).or_default();
        Self::merge_stats(role_bucket, success, latency_ms, quality, error);
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn score_bonus(&self, candidate_id: &str, workflow: &str, role: &str) -> f64 {
        let data = self.inner.lock().await;
        let Some(bucket) = data.candidates.get(candidate_id) else {
            return 0.0;
        };
        let role_key = format!("{workflow}:{role}");
        let stats = bucket.roles.get(&role_key).unwrap_or(&bucket.stats);
        if stats.total == 0 {
            return 0.0;
        }
        let success_rate = stats.successes as f64 / stats.total as f64;
        let latency_bonus = stats
            .ewma_latency_ms
            .map(|latency| ((1500.0 - latency) / 3000.0).clamp(-0.35, 0.35))
            .unwrap_or(0.0);
        ((success_rate - 0.5) + stats.ewma_quality + latency_bonus * 1.0).round_to(6)
    }

    pub async fn summary(&self, limit: usize) -> MetricsSummary {
        let data = self.inner.lock().await;
        let mut candidates = data
            .candidates
            .iter()
            .map(|(candidate_id, bucket)| CandidateMetricsSummary {
                candidate_id: candidate_id.clone(),
                provider: bucket.provider.clone(),
                model: bucket.model.clone(),
                configured_model: bucket.configured_model.clone(),
                resolved_model: bucket
                    .resolved_model
                    .clone()
                    .or_else(|| bucket.model.clone()),
                specialties: bucket.specialties.clone(),
                successes: bucket.stats.successes,
                failures: bucket.stats.failures,
                total: bucket.stats.total,
                success_rate: if bucket.stats.total > 0 {
                    Some((bucket.stats.successes as f64 / bucket.stats.total as f64).round_to(6))
                } else {
                    None
                },
                ewma_latency_ms: bucket.stats.ewma_latency_ms,
                ewma_quality: bucket.stats.ewma_quality.round_to(6),
                last_status: bucket.stats.last_status.clone(),
                last_error: bucket.stats.last_error.clone(),
                updated_at: bucket.stats.updated_at,
                health_ok: bucket.health.ok,
                health_mode: bucket.health.mode.clone(),
                health_checked_at: bucket.health.checked_at,
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .health_ok
                .unwrap_or(false)
                .cmp(&left.health_ok.unwrap_or(false))
                .then_with(|| {
                    right
                        .success_rate
                        .partial_cmp(&left.success_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    right
                        .ewma_quality
                        .partial_cmp(&left.ewma_quality)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    left.ewma_latency_ms
                        .partial_cmp(&right.ewma_latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let limited = candidates
            .into_iter()
            .take(limit.max(1))
            .collect::<Vec<_>>();
        MetricsSummary {
            path: self.path(),
            exists: self.path.exists(),
            updated_at: data.updated_at,
            candidate_count: data.candidates.len(),
            healthy_candidates: data
                .candidates
                .values()
                .filter(|bucket| bucket.health.ok == Some(true))
                .count(),
            degraded_candidates: data
                .candidates
                .values()
                .filter(|bucket| bucket.health.ok == Some(false))
                .count(),
            candidates: limited,
        }
    }
}

trait RoundTo {
    fn round_to(self, precision: i32) -> Self;
}

impl RoundTo for f64 {
    fn round_to(self, precision: i32) -> Self {
        let factor = 10_f64.powi(precision);
        (self * factor).round() / factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(provider: &str, model: &str) -> CandidateSummary {
        CandidateSummary {
            candidate_id: format!("{provider}/{model}"),
            provider: provider.to_string(),
            model: model.to_string(),
            configured_model: model.to_string(),
            resolved_model: model.to_string(),
            source: "test".to_string(),
            specialties: Vec::new(),
            roles: Vec::new(),
        }
    }

    #[tokio::test]
    async fn provider_quota_backoff_matches_provider_family() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        store
            .record_health(
                &summary("nvidia", "moonshotai/kimi-k2-instruct-0905"),
                HealthBucket {
                    ok: Some(false),
                    mode: Some("quota".to_string()),
                    checked_at: None,
                    latency_ms: Some(10),
                    message: Some("Too Many Requests".to_string()),
                },
            )
            .await
            .expect("record health");

        assert!(store.provider_in_quota_backoff("nvidia", 1800.0).await);
        assert!(!store.provider_in_quota_backoff("ollama", 1800.0).await);
    }
}
