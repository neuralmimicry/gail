use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{fs, sync::Mutex};

use crate::{errors::Result, models::CandidateSummary};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

/// Reject malformed or future worker timestamps before they reach Prometheus.
/// A five-minute allowance accommodates small clock skew between hosts.
fn valid_unix_timestamp(value: Option<f64>) -> Option<f64> {
    let now = now_ts();
    value.filter(|timestamp| timestamp.is_finite() && *timestamp > 0.0 && *timestamp <= now + 300.0)
}

fn reset_routing_metrics_on_startup() -> bool {
    let requested = env::var("GAIL_METRICS_RESET_ROUTING_ON_STARTUP")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if !requested {
        return false;
    }
    // MetricsStore is also used by trainer_worker to read historical
    // throughput for trained-model placement. Only the serving process may
    // reset the shared file; worker roles must retain that history.
    matches!(
        env::var("GAIL_ROLE").ok().as_deref(),
        None | Some("") | Some("serve") | Some("api")
    )
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MetricsData {
    pub candidates: HashMap<String, CandidateBucket>,
    #[serde(default)]
    pub ai_response_times: HashMap<String, AiResponseTimeStats>,
    #[serde(default)]
    pub api_response_times: HashMap<String, AiResponseTimeStats>,
    #[serde(default)]
    pub orchestration_events: OrchestrationEventMetrics,
    #[serde(default)]
    pub request_flow: RequestFlowMetrics,
    #[serde(default)]
    pub trading_semantic: TradingSemanticMetrics,
    pub updated_at: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TradingSemanticMetrics {
    pub responses: u64,
    pub parsed_valid: u64,
    pub invalid_json: u64,
    pub invalid_shape: u64,
    pub incomplete: u64,
    pub provider_failures: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OrchestrationEventMetrics {
    pub candidate_selections: u64,
    pub capacity_races: u64,
    pub queue_waits: u64,
    pub queue_wait_total_ms: u64,
    pub queue_wait_timeouts: u64,
    pub timeouts: u64,
    pub fallbacks: u64,
    pub empty_plans: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RequestFlowMetrics {
    pub received: u64,
    pub queued: u64,
    pub in_progress: u64,
    pub replied: u64,
    pub failed: u64,
    pub timed_out: u64,
}

/// One completed or failed training run.  Training is written by a separate
/// worker process, so observations live in their own append-only state file
/// and cannot overwrite provider routing metrics.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TrainingRunObservation {
    pub snapshot_id: String,
    pub backend: String,
    pub status: String,
    /// Stable category for dashboards; detailed stderr remains in the
    /// snapshot pipeline artifact rather than becoming a Prometheus label.
    #[serde(default)]
    pub failure_reason: String,
    pub base_model: String,
    pub slurm_job_id: Option<String>,
    pub nodelist: Option<String>,
    pub world_size: Option<u64>,
    pub samples: u64,
    pub total_tokens: u64,
    pub non_padding_tokens: u64,
    pub optimizer_steps: u64,
    pub runtime_seconds: f64,
    pub tokens_per_second: f64,
    pub non_padding_tokens_per_second: f64,
    pub started_ts: Option<f64>,
    pub finished_ts: Option<f64>,
    /// True when the trainer resumed from the previously published adapter.
    /// This is recorded explicitly so dashboards do not infer cumulative
    /// learning from timestamps or snapshot names.
    #[serde(default)]
    pub cumulative_training: bool,
    /// True when a requested QLoRA run intentionally executed as CPU LoRA.
    #[serde(default)]
    pub cpu_fallback: bool,
    /// Effective DataLoader pinned-memory setting for the run.
    #[serde(default)]
    pub pin_memory: bool,
    #[serde(default)]
    pub quantisation_backend: String,
}

impl TrainingRunObservation {
    /// Convert the stable training report contract into dashboard telemetry.
    /// Slurm and local trainers write the same report shape, so keeping this
    /// conversion here also makes startup backfill and live observations
    /// consistent.
    pub fn from_report(
        snapshot_id: &str,
        report: &Value,
        status: &str,
        default_base_model: &str,
    ) -> Self {
        let metrics = report.get("metrics").cloned().unwrap_or(Value::Null);
        let distributed = report.get("distributed");
        let rank_reports = report
            .get("rank_reports")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let rank_metric = |key: &str| {
            rank_reports
                .iter()
                .filter_map(|rank| {
                    rank.get("metrics")
                        .and_then(|value| value.get(key))
                        .and_then(Value::as_u64)
                        .or_else(|| rank.get(key).and_then(Value::as_u64))
                })
                .sum::<u64>()
        };
        let rank_metric_max = |key: &str| {
            rank_reports
                .iter()
                .filter_map(|rank| {
                    rank.get("metrics")
                        .and_then(|value| value.get(key))
                        .and_then(Value::as_f64)
                        .or_else(|| rank.get(key).and_then(Value::as_f64))
                })
                .fold(0.0, f64::max)
        };
        let runtime_seconds = metrics
            .get("runtime_seconds")
            .and_then(Value::as_f64)
            .or_else(|| report.get("runtime_seconds").and_then(Value::as_f64))
            .or_else(|| {
                report
                    .get("training_runtime_seconds")
                    .and_then(Value::as_f64)
            })
            .unwrap_or_else(|| rank_metric_max("runtime_seconds"));
        let total_tokens = metrics
            .get("total_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                distributed
                    .and_then(|value| value.get("total_tokens"))
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                let total = rank_metric("total_tokens");
                (total > 0).then_some(total)
            })
            // Older distributed Python reports only recorded evaluation
            // tokens. Keep the fallback bounded while old snapshots are
            // being backfilled, rather than leaving dashboards blank.
            .or_else(|| {
                report
                    .get("evaluation")
                    .and_then(|value| value.get("tokens"))
                    .and_then(Value::as_u64)
                    .map(|value| value.min(10_000_000))
            })
            .unwrap_or(0);
        let non_padding_tokens = metrics
            .get("non_padding_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                distributed
                    .and_then(|value| value.get("non_padding_tokens"))
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                let total = rank_metric("non_padding_tokens");
                (total > 0).then_some(total)
            })
            .unwrap_or(0);
        let tokens_per_second = metrics
            .get("aggregate_tokens_per_second")
            .and_then(Value::as_f64)
            .or_else(|| metrics.get("tokens_per_second").and_then(Value::as_f64))
            .unwrap_or_else(|| rate(total_tokens, runtime_seconds));
        let non_padding_tokens_per_second = metrics
            .get("non_padding_tokens_per_second")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| rate(non_padding_tokens, runtime_seconds));

        Self {
            snapshot_id: snapshot_id.to_string(),
            backend: report
                .get("backend")
                .and_then(Value::as_str)
                .or_else(|| report.get("training_backend").and_then(Value::as_str))
                .unwrap_or("unknown")
                .to_string(),
            status: status.to_string(),
            failure_reason: report
                .get("failure_reason")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_default(),
            base_model: report
                .get("base_model")
                .and_then(Value::as_str)
                .unwrap_or(default_base_model)
                .to_string(),
            slurm_job_id: distributed
                .and_then(|value| value.get("slurm_job_id"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            nodelist: distributed
                .and_then(|value| value.get("slurm_nodelist"))
                .and_then(Value::as_str)
                .or_else(|| {
                    distributed
                        .and_then(|value| value.get("nodelist"))
                        .and_then(Value::as_str)
                })
                .map(ToOwned::to_owned),
            world_size: distributed
                .and_then(|value| value.get("world_size"))
                .and_then(Value::as_u64),
            samples: metrics
                .get("samples")
                .and_then(Value::as_u64)
                .or_else(|| {
                    distributed
                        .and_then(|value| value.get("total_samples"))
                        .and_then(Value::as_u64)
                })
                .or_else(|| {
                    let total = rank_metric("samples");
                    (total > 0).then_some(total)
                })
                .or_else(|| report.get("samples").and_then(Value::as_u64))
                .unwrap_or(0),
            total_tokens,
            non_padding_tokens,
            optimizer_steps: metrics
                .get("total_optimizer_steps")
                .and_then(Value::as_u64)
                .or_else(|| {
                    let total = rank_metric("total_optimizer_steps");
                    (total > 0).then_some(total)
                })
                .unwrap_or(0),
            runtime_seconds,
            tokens_per_second,
            non_padding_tokens_per_second,
            started_ts: valid_unix_timestamp(report.get("started_ts").and_then(Value::as_f64)),
            finished_ts: valid_unix_timestamp(report.get("finished_ts").and_then(Value::as_f64)),
            cumulative_training: report
                .get("cumulative_training")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            cpu_fallback: report
                .get("cpu_fallback")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            pin_memory: report
                .get("pin_memory")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            quantisation_backend: report
                .get("quantisation_backend")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
        }
    }
}

fn rate(tokens: u64, runtime_seconds: f64) -> f64 {
    if runtime_seconds > 0.0 {
        tokens as f64 / runtime_seconds
    } else {
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
struct TrainingMetricsData {
    runs: Vec<TrainingRunObservation>,
    updated_at: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
struct TrainingProgressObservation {
    snapshot_id: String,
    status: String,
    backend: String,
    slurm_job_id: Option<String>,
    completed_steps: u64,
    total_steps: u64,
    progress_ratio: f64,
    progress_per_hour: f64,
    eta_seconds: f64,
    elapsed_seconds: f64,
    started_ts: Option<f64>,
    updated_ts: Option<f64>,
}

/// Unified response-time statistics for user-visible AI work.
///
/// Candidate metrics remain provider-specific for routing. These buckets are
/// deliberately broader so callers can estimate work before Gail has selected
/// a particular provider or specialist engine.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AiResponseTimeStats {
    pub requests: u64,
    pub successes: u64,
    pub failures: u64,
    pub total_latency_ms: u64,
    pub average_latency_ms: Option<f64>,
    pub min_latency_ms: Option<u64>,
    pub max_latency_ms: Option<u64>,
    pub ewma_latency_ms: Option<f64>,
    /// Latency distribution for successful responses only.
    #[serde(default)]
    pub successful_latency_samples: u64,
    #[serde(default)]
    pub successful_latency_total_ms: u64,
    #[serde(default)]
    pub successful_latency_average_ms: Option<f64>,
    #[serde(default)]
    pub successful_latency_min_ms: Option<u64>,
    #[serde(default)]
    pub successful_latency_max_ms: Option<u64>,
    #[serde(default)]
    pub successful_latency_ewma_ms: Option<f64>,
    /// Latency distribution for failed responses only.
    #[serde(default)]
    pub failed_latency_samples: u64,
    #[serde(default)]
    pub failed_latency_total_ms: u64,
    #[serde(default)]
    pub failed_latency_average_ms: Option<f64>,
    #[serde(default)]
    pub failed_latency_min_ms: Option<u64>,
    #[serde(default)]
    pub failed_latency_max_ms: Option<u64>,
    #[serde(default)]
    pub failed_latency_ewma_ms: Option<f64>,
    pub last_latency_ms: Option<u64>,
    pub total_prompt_tokens: u64,
    pub average_prompt_tokens: Option<f64>,
    pub ewma_prompt_tokens: Option<f64>,
    pub updated_at: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CandidateBucket {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub configured_model: Option<String>,
    pub resolved_model: Option<String>,
    #[serde(default)]
    pub host_id: Option<String>,
    pub specialties: Vec<String>,
    pub stats: StatsBucket,
    pub roles: HashMap<String, StatsBucket>,
    pub health: HealthBucket,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct StatsBucket {
    #[serde(default)]
    pub successes: u64,
    #[serde(default)]
    pub failures: u64,
    #[serde(default)]
    pub total: u64,
    /// Backward-compatible successful latency fields. New callers should use
    /// the explicit `successful_*` fields below.
    #[serde(default)]
    pub total_latency_ms: u64,
    #[serde(default)]
    pub average_latency_ms: Option<f64>,
    #[serde(default)]
    pub min_latency_ms: Option<u64>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub ewma_latency_ms: Option<f64>,
    #[serde(default)]
    pub successful_latency_samples: u64,
    #[serde(default)]
    pub successful_latency_total_ms: u64,
    #[serde(default)]
    pub successful_latency_average_ms: Option<f64>,
    #[serde(default)]
    pub successful_latency_min_ms: Option<u64>,
    #[serde(default)]
    pub successful_latency_max_ms: Option<u64>,
    #[serde(default)]
    pub successful_latency_ewma_ms: Option<f64>,
    #[serde(default)]
    pub failed_latency_samples: u64,
    #[serde(default)]
    pub failed_latency_total_ms: u64,
    #[serde(default)]
    pub failed_latency_average_ms: Option<f64>,
    #[serde(default)]
    pub failed_latency_min_ms: Option<u64>,
    #[serde(default)]
    pub failed_latency_max_ms: Option<u64>,
    #[serde(default)]
    pub failed_latency_ewma_ms: Option<f64>,
    #[serde(default)]
    pub stale_failures: u64,
    #[serde(default)]
    pub queue_wait_total_ms: u64,
    #[serde(default)]
    pub queue_wait_samples: u64,
    #[serde(default)]
    pub queue_wait_average_ms: Option<f64>,
    #[serde(default)]
    pub queue_wait_min_ms: Option<u64>,
    #[serde(default)]
    pub queue_wait_max_ms: Option<u64>,
    pub ewma_queue_wait_ms: Option<f64>,
    pub ewma_inference_ms: Option<f64>,
    pub ewma_prompt_tokens: Option<f64>,
    pub ewma_tokens_estimate: Option<f64>,
    #[serde(default)]
    pub ewma_completion_tokens: Option<f64>,
    /// Successful generated-token throughput. Queue wait and failed requests
    /// are intentionally excluded from this distribution.
    #[serde(default)]
    pub generation_tokens_per_second_samples: u64,
    #[serde(default)]
    pub generation_tokens_per_second_total: f64,
    #[serde(default)]
    pub generation_tokens_per_second_average: Option<f64>,
    #[serde(default)]
    pub generation_tokens_per_second_min: Option<f64>,
    #[serde(default)]
    pub generation_tokens_per_second_max: Option<f64>,
    #[serde(default)]
    pub generation_tokens_per_second_ewma: Option<f64>,
    pub ewma_quality: f64,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    pub updated_at: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LocalUsageTelemetry {
    pub prompt_tokens_estimate: Option<u32>,
    pub queue_wait_ms: Option<u64>,
    pub inference_ms: Option<u64>,
    pub total_tokens_estimate: Option<u32>,
    pub completion_tokens_estimate: Option<u32>,
}

/// Historical capacity observations for one provider/model/request profile.
/// These values are deliberately returned as a small routing-facing view of
/// the persisted statistics rather than exposing the mutable metrics buckets.
#[derive(Clone, Debug, Default)]
pub struct CandidateCapacityEstimate {
    pub samples: u64,
    pub success_rate: f64,
    pub quality: f64,
    pub useful_rate: f64,
    pub generation_tokens_per_second: Option<f64>,
    pub service_time_ms: Option<f64>,
    pub queue_wait_ms: Option<f64>,
    pub completion_tokens: Option<f64>,
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
    pub host_id: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub configured_model: Option<String>,
    pub resolved_model: Option<String>,
    pub specialties: Vec<String>,
    pub successes: u64,
    pub failures: u64,
    pub total: u64,
    pub average_latency_ms: Option<f64>,
    pub min_latency_ms: Option<u64>,
    pub max_latency_ms: Option<u64>,
    pub success_rate: Option<f64>,
    pub ewma_latency_ms: Option<f64>,
    pub successful_latency_samples: u64,
    pub successful_latency_average_ms: Option<f64>,
    pub successful_latency_min_ms: Option<u64>,
    pub successful_latency_max_ms: Option<u64>,
    pub successful_latency_ewma_ms: Option<f64>,
    pub failed_latency_samples: u64,
    pub failed_latency_average_ms: Option<f64>,
    pub failed_latency_min_ms: Option<u64>,
    pub failed_latency_max_ms: Option<u64>,
    pub failed_latency_ewma_ms: Option<f64>,
    pub stale_failures: u64,
    pub queue_wait_average_ms: Option<f64>,
    pub queue_wait_min_ms: Option<u64>,
    pub queue_wait_max_ms: Option<u64>,
    pub ewma_queue_wait_ms: Option<f64>,
    pub ewma_inference_ms: Option<f64>,
    pub ewma_prompt_tokens: Option<f64>,
    pub ewma_tokens_estimate: Option<f64>,
    #[serde(default)]
    pub ewma_completion_tokens: Option<f64>,
    pub generation_tokens_per_second_average: Option<f64>,
    pub generation_tokens_per_second_min: Option<f64>,
    pub generation_tokens_per_second_max: Option<f64>,
    pub generation_tokens_per_second_ewma: Option<f64>,
    pub ewma_quality: f64,
    pub last_status: Option<String>,
    pub last_error: Option<String>,
    pub updated_at: Option<f64>,
    pub health_ok: Option<bool>,
    pub health_mode: Option<String>,
    pub health_checked_at: Option<f64>,
    /// Persistent source/profile-specific observations used by the ranker.
    pub routing_profiles: HashMap<String, StatsBucket>,
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
    pub ai_response_times: HashMap<String, AiResponseTimeStats>,
    pub api_response_times: HashMap<String, AiResponseTimeStats>,
}

#[derive(Clone)]
pub struct MetricsStore {
    path: PathBuf,
    inner: Arc<Mutex<MetricsData>>,
    persist_lock: Arc<Mutex<()>>,
}

/// Persist one Slurm/local training observation without touching the routing
/// metrics file owned by the API process.
pub async fn append_training_observation(
    path: impl Into<PathBuf>,
    observation: TrainingRunObservation,
) -> Result<()> {
    upsert_training_observation(path, observation).await
}

/// Persist a training observation idempotently. Retries and worker restarts
/// must not duplicate a snapshot in Prometheus or inflate dashboard totals.
pub async fn upsert_training_observation(
    path: impl Into<PathBuf>,
    observation: TrainingRunObservation,
) -> Result<()> {
    let path = path.into();
    let data = read_training_metrics(&path).await;
    write_training_metrics(&path, merge_training_observations(data, [observation])).await
}

fn merge_training_observations<I>(
    mut data: TrainingMetricsData,
    observations: I,
) -> TrainingMetricsData
where
    I: IntoIterator<Item = TrainingRunObservation>,
{
    for observation in observations {
        data.runs
            .retain(|run| run.snapshot_id != observation.snapshot_id);
        data.runs.push(observation);
    }
    // Bound label/cardinality growth while retaining a useful operational
    // history for dashboards and postmortems.
    const MAX_TRAINING_RUNS: usize = 512;
    if data.runs.len() > MAX_TRAINING_RUNS {
        let keep_from = data.runs.len() - MAX_TRAINING_RUNS;
        data.runs.drain(..keep_from);
    }
    data.updated_at = now_ts();
    data
}

async fn read_training_metrics(path: &Path) -> TrainingMetricsData {
    match fs::read_to_string(path).await {
        Ok(raw) => serde_json::from_str::<TrainingMetricsData>(&raw).unwrap_or_default(),
        Err(_) => TrainingMetricsData::default(),
    }
}

async fn write_training_metrics(path: &PathBuf, data: TrainingMetricsData) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_string_pretty(&data)? + "\n").await?;
    fs::rename(temporary, path).await?;
    Ok(())
}

/// Discover reports produced before the telemetry file existed. Reports are
/// treated as completed training artifacts; promotion status is recorded by
/// the live worker when it is known.
pub async fn discover_training_observations(
    snapshot_root: impl Into<PathBuf>,
    default_base_model: &str,
) -> Result<Vec<TrainingRunObservation>> {
    let mut observations = Vec::new();
    let mut entries = match fs::read_dir(snapshot_root.into()).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(observations),
        Err(error) => return Err(error.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        if !entry.file_type().await?.is_dir() {
            continue;
        }
        let snapshot_id = entry.file_name().to_string_lossy().to_string();
        let report_path = entry.path().join("training_report.json");
        let report = match fs::read_to_string(report_path).await {
            Ok(raw) => match serde_json::from_str::<Value>(&raw) {
                Ok(report) => report,
                Err(_) => continue,
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({}),
            Err(_) => continue,
        };
        let report = merge_pipeline_provenance(entry.path().join("pipeline.json"), report).await;
        let has_metrics = report.get("metrics").is_some();
        let failure_reason = report
            .get("failure_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty());
        if !has_metrics && failure_reason.is_none() {
            continue;
        }
        let status = if failure_reason.is_some() {
            if report.get("ollama_registration_error").is_some()
                || report.get("health_check_error").is_some()
            {
                "promotion_failed"
            } else {
                "failed"
            }
        } else {
            "historical_completed"
        };
        let mut observation = TrainingRunObservation::from_report(
            snapshot_id.as_str(),
            &report,
            status,
            default_base_model,
        );
        if let Some(reason) = failure_reason {
            observation.failure_reason = reason.to_string();
        }
        observations.push(observation);
    }
    observations.sort_by(|left, right| {
        left.finished_ts
            .partial_cmp(&right.finished_ts)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(observations)
}

async fn merge_pipeline_provenance(pipeline_path: PathBuf, mut report: Value) -> Value {
    let Ok(raw) = fs::read_to_string(pipeline_path).await else {
        return report;
    };
    let Ok(pipeline) = serde_json::from_str::<Value>(&raw) else {
        return report;
    };
    let Some(report_object) = report.as_object_mut() else {
        return report;
    };
    let Some(pipeline_object) = pipeline.as_object() else {
        return report;
    };
    for key in [
        "started_ts",
        "finished_ts",
        "cumulative_training",
        "training_runtime_seconds",
        "slurm_job_id",
    ] {
        if let Some(value) = pipeline_object.get(key) {
            report_object.insert(key.to_string(), value.clone());
        }
    }
    if report_object.get("failure_reason").is_none() {
        for key in [
            "qualification_error",
            "ollama_registration_error",
            "health_check_error",
            "training_error",
        ] {
            if let Some(error) = pipeline_object.get(key).and_then(Value::as_str) {
                report_object.insert(
                    "failure_reason".to_string(),
                    Value::String(classify_training_failure(error).to_string()),
                );
                break;
            }
        }
    }
    report
}

pub fn classify_training_failure(error: &str) -> &'static str {
    let error = error.to_ascii_lowercase();
    if error.contains("unsupported architecture")
        || error.contains("cannot be registered as safetensors")
        || error.contains("adapter conversion")
    {
        "unsupported_serving_architecture"
    } else if error.contains("stale") || error.contains("heartbeat") {
        "stale_training_lifecycle"
    } else if error.contains("slurm") || error.contains("scancel") {
        "slurm_infrastructure"
    } else if error.contains("timeout") || error.contains("timed out") {
        "training_timeout"
    } else if error.contains("cuda") || error.contains("nvidia") || error.contains("out of memory")
    {
        "gpu_or_memory"
    } else if error.contains("dataset") {
        "dataset_build"
    } else if error.contains("qualification") || error.contains("baseline") {
        "qualification_gate"
    } else if error.contains("ollama") || error.contains("promotion") {
        "promotion_or_registration"
    } else {
        "training_execution"
    }
}

/// Backfill the persistent metrics file from existing training reports.
pub async fn backfill_training_observations(
    metrics_path: impl Into<PathBuf>,
    snapshot_root: impl Into<PathBuf>,
    default_base_model: &str,
) -> Result<usize> {
    let metrics_path = metrics_path.into();
    let observations = discover_training_observations(snapshot_root, default_base_model).await?;
    let count = observations.len();
    if count > 0 {
        let data = read_training_metrics(&metrics_path).await;
        write_training_metrics(
            &metrics_path,
            merge_training_observations(data, observations),
        )
        .await?;
    }
    Ok(count)
}

async fn discover_training_progress(snapshot_root: PathBuf) -> Vec<TrainingProgressObservation> {
    let training_root = snapshot_root.parent().map(Path::to_path_buf);
    let slurm_results_root = training_root
        .as_ref()
        .map(|root| root.join("slurm").join("spool").join("results"));

    // Slurm is the normal production backend.  There can be thousands of
    // historical snapshot directories on the shared filesystem, so scanning
    // every one of them for progress on every Prometheus scrape makes
    // `/metrics` block long enough for Prometheus to mark Gail down.  The
    // trainer already maintains one active lifecycle marker and the Slurm
    // dispatcher maintains one status file per active request; use those
    // bounded files as the authoritative active-task index.
    if let Some(training_root) = training_root.as_ref() {
        let active_marker = training_root.join("active_training.json");
        if let Ok(raw) = fs::read_to_string(&active_marker).await {
            if let Ok(marker) = serde_json::from_str::<Value>(&raw) {
                if let Some(snapshot_id) = marker.get("snapshot_id").and_then(Value::as_str) {
                    if let Some(results_root) = slurm_results_root.as_ref() {
                        let status_path = results_root.join(format!("{snapshot_id}.status"));
                        if let Ok(raw_status) = fs::read_to_string(status_path).await
                            && let Ok(status) = serde_json::from_str::<Value>(&raw_status)
                            && let Some(progress) =
                                slurm_progress_from_status(&snapshot_root, snapshot_id, &status)
                                    .await
                        {
                            return vec![progress];
                        }
                    }
                }
            }
        }

        if let Some(results_root) = slurm_results_root.as_ref()
            && fs::metadata(results_root).await.is_ok()
        {
            // Recover gracefully if the marker was lost during a trainer pod
            // restart. This scans status files, not the historical snapshot
            // tree, and therefore remains bounded by the Slurm spool.
            if let Ok(mut entries) = fs::read_dir(results_root).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.extension().and_then(|value| value.to_str()) != Some("status") {
                        continue;
                    }
                    let Some(snapshot_id) = path
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .filter(|value| !value.is_empty())
                    else {
                        continue;
                    };
                    let Ok(raw_status) = fs::read_to_string(&path).await else {
                        continue;
                    };
                    let Ok(status) = serde_json::from_str::<Value>(&raw_status) else {
                        continue;
                    };
                    if let Some(progress) =
                        slurm_progress_from_status(&snapshot_root, snapshot_id, &status).await
                    {
                        return vec![progress];
                    }
                }
            }
            return Vec::new();
        }
    }

    // Local-development fallback: retain the previous directory scan when
    // no Slurm spool is present.
    let mut observations = Vec::new();
    let Ok(mut entries) = fs::read_dir(snapshot_root).await else {
        return observations;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path().join("progress.json");
        let Ok(raw) = fs::read_to_string(path).await else {
            continue;
        };
        let Ok(mut progress) = serde_json::from_str::<TrainingProgressObservation>(&raw) else {
            continue;
        };
        let progress_fresh = progress
            .updated_ts
            .and_then(|updated| valid_unix_timestamp(Some(updated)))
            .is_some_and(|updated| now_ts() - updated <= 900.0);
        // A long CPU training step can leave progress.json unchanged for
        // longer than the normal reporting TTL.  The Slurm dispatcher still
        // refreshes its RUNNING status heartbeat every few seconds, which is
        // the authoritative liveness signal for that case.  Keep the last
        // known progress/ETA, but advance updated_ts so Grafana does not
        // incorrectly show zero active tasks while the job is making work.
        let slurm_fresh = if !progress_fresh {
            let status_path = slurm_results_root
                .as_ref()
                .map(|root| root.join(format!("{}.status", progress.snapshot_id)));
            let status = match status_path {
                Some(path) => fs::read_to_string(path)
                    .await
                    .ok()
                    .and_then(|raw| serde_json::from_str::<Value>(&raw).ok()),
                None => None,
            };
            let state_is_running = status
                .as_ref()
                .and_then(|value| value.get("state"))
                .and_then(Value::as_str)
                .is_some_and(|state| state.eq_ignore_ascii_case("running"));
            let heartbeat = status
                .as_ref()
                .and_then(|value| value.get("heartbeat_ts"))
                .and_then(Value::as_f64)
                .and_then(|value| valid_unix_timestamp(Some(value)));
            let fresh = state_is_running
                && heartbeat
                    .is_some_and(|value| now_ts() - value <= training_heartbeat_stale_seconds());
            if fresh {
                if progress.slurm_job_id.is_none() {
                    progress.slurm_job_id = status
                        .as_ref()
                        .and_then(|value| value.get("slurm_job_id"))
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                }
                progress.updated_ts = heartbeat;
            }
            fresh
        } else {
            false
        };
        if (progress_fresh || slurm_fresh)
            && !progress.snapshot_id.is_empty()
            && progress.status != "completed"
            && progress.status != "failed"
        {
            // progress.json is written at optimizer-step boundaries.  A
            // long CPU step can therefore leave its persisted elapsed value
            // at the initial value for many minutes even while the Slurm
            // heartbeat proves the task is alive.  Export wall-clock elapsed
            // time from the task start so Grafana reflects the real run.
            if let Some(started) = valid_unix_timestamp(progress.started_ts) {
                progress.elapsed_seconds =
                    progress.elapsed_seconds.max((now_ts() - started).max(0.0));
            }
            progress.progress_ratio = progress.progress_ratio.clamp(0.0, 1.0);
            progress.progress_per_hour = progress.progress_per_hour.max(0.0);
            progress.eta_seconds = progress.eta_seconds.max(0.0);
            progress.elapsed_seconds = progress.elapsed_seconds.max(0.0);
            progress.started_ts = valid_unix_timestamp(progress.started_ts);
            progress.updated_ts = valid_unix_timestamp(progress.updated_ts);
            observations.push(progress);
        }
    }
    observations
}

async fn slurm_progress_from_status(
    snapshot_root: &Path,
    snapshot_id: &str,
    status: &Value,
) -> Option<TrainingProgressObservation> {
    let state_is_running = status
        .get("state")
        .and_then(Value::as_str)
        .is_some_and(|state| state.eq_ignore_ascii_case("running"));
    let heartbeat = status
        .get("heartbeat_ts")
        .and_then(Value::as_f64)
        .and_then(|value| valid_unix_timestamp(Some(value)));
    let raw_progress = fs::read_to_string(snapshot_root.join(snapshot_id).join("progress.json"))
        .await
        .ok()?;
    let mut progress = serde_json::from_str::<TrainingProgressObservation>(&raw_progress).ok()?;
    let progress_fresh = progress
        .updated_ts
        .and_then(|updated| valid_unix_timestamp(Some(updated)))
        .is_some_and(|updated| now_ts() - updated <= 900.0);
    if !progress_fresh && state_is_running {
        progress.updated_ts = heartbeat;
        if progress.slurm_job_id.is_none() {
            progress.slurm_job_id = status
                .get("slurm_job_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
    }
    let still_active = progress_fresh
        || (state_is_running
            && heartbeat
                .is_some_and(|value| now_ts() - value <= training_heartbeat_stale_seconds()));
    if !still_active
        || progress.snapshot_id.is_empty()
        || progress.status == "completed"
        || progress.status == "failed"
    {
        return None;
    }
    if let Some(started) = valid_unix_timestamp(progress.started_ts) {
        progress.elapsed_seconds = progress.elapsed_seconds.max((now_ts() - started).max(0.0));
    }
    progress.progress_ratio = progress.progress_ratio.clamp(0.0, 1.0);
    progress.progress_per_hour = progress.progress_per_hour.max(0.0);
    progress.eta_seconds = progress.eta_seconds.max(0.0);
    progress.elapsed_seconds = progress.elapsed_seconds.max(0.0);
    progress.started_ts = valid_unix_timestamp(progress.started_ts);
    progress.updated_ts = valid_unix_timestamp(progress.updated_ts);
    Some(progress)
}

fn training_heartbeat_stale_seconds() -> f64 {
    env::var("GAIL_TRAINING_HEARTBEAT_STALE_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 60.0)
        .unwrap_or(7_200.0)
}

impl AiResponseTimeStats {
    fn normalize_split_latency_fields(&mut self) {
        // The split fields were introduced after the legacy aggregate file
        // format. A zero sample count means any populated split values came
        // from the pre-count implementation and cannot be averaged reliably.
        if self.successful_latency_samples == 0 {
            self.successful_latency_total_ms = 0;
            self.successful_latency_average_ms = None;
            self.successful_latency_min_ms = None;
            self.successful_latency_max_ms = None;
            self.successful_latency_ewma_ms = None;
        }
        if self.failed_latency_samples == 0 {
            self.failed_latency_total_ms = 0;
            self.failed_latency_average_ms = None;
            self.failed_latency_min_ms = None;
            self.failed_latency_max_ms = None;
            self.failed_latency_ewma_ms = None;
        }
    }
}

impl StatsBucket {
    fn normalize_split_latency_fields(&mut self) {
        if self.successful_latency_samples == 0 {
            self.successful_latency_total_ms = 0;
            self.successful_latency_average_ms = None;
            self.successful_latency_min_ms = None;
            self.successful_latency_max_ms = None;
            self.successful_latency_ewma_ms = None;
        }
        if self.failed_latency_samples == 0 {
            self.failed_latency_total_ms = 0;
            self.failed_latency_average_ms = None;
            self.failed_latency_min_ms = None;
            self.failed_latency_max_ms = None;
            self.failed_latency_ewma_ms = None;
        }
    }
}

impl MetricsStore {
    pub async fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let reset_routing = reset_routing_metrics_on_startup();
        let mut data = match fs::read_to_string(&path).await {
            Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
            Err(_) => MetricsData::default(),
        };
        if reset_routing {
            // Provider health and routing observations describe the previous
            // Gail process. A node can have rebooted, changed model, or
            // recovered from a transient fault since those observations were
            // written. Keep candidate identity metadata, but make the first
            // requests after restart establish a fresh baseline instead of
            // inheriting stale failures, latency, or token-rate values.
            for candidate in data.candidates.values_mut() {
                candidate.stats = StatsBucket::default();
                candidate.roles.clear();
                candidate.health = HealthBucket::default();
            }
            data.ai_response_times.clear();
            data.updated_at = now_ts();
        }
        for stats in data.ai_response_times.values_mut() {
            stats.normalize_split_latency_fields();
        }
        for stats in data.api_response_times.values_mut() {
            stats.normalize_split_latency_fields();
        }
        for candidate in data.candidates.values_mut() {
            candidate.stats.normalize_split_latency_fields();
            for stats in candidate.roles.values_mut() {
                stats.normalize_split_latency_fields();
            }
        }
        // Queue and in-progress gauges describe this Gail process, not
        // historical work from a previous process lifetime. Avoid exposing
        // stale values after a restart. Request-flow counters are reset along
        // with routing observations when explicitly requested: older Gail
        // versions did not record terminal outcomes for every received HTTP
        // request, so carrying those counters forward would create a
        // permanently false unaccounted backlog and invalid success rate.
        data.request_flow.queued = 0;
        data.request_flow.in_progress = 0;
        if reset_routing {
            data.request_flow = RequestFlowMetrics::default();
        }
        let store = Self {
            path,
            inner: Arc::new(Mutex::new(data)),
            persist_lock: Arc::new(Mutex::new(())),
        };
        if reset_routing {
            let snapshot = store.inner.lock().await.clone();
            store.save(&snapshot).await?;
        }
        Ok(store)
    }

    pub fn path(&self) -> String {
        self.path.to_string_lossy().to_string()
    }

    fn normalize_ai_source(source: &str) -> &'static str {
        match source.trim().to_ascii_lowercase().as_str() {
            "snn" | "aarnn" | "neuromorphic" => "snn",
            "llm" | "language_model" | "language-model" => "llm",
            _ => "all",
        }
    }

    fn merge_ai_response_time(
        bucket: &mut AiResponseTimeStats,
        latency_ms: u64,
        success: bool,
        prompt_tokens_estimate: Option<u32>,
    ) {
        bucket.requests = bucket.requests.saturating_add(1);
        if success {
            bucket.successes = bucket.successes.saturating_add(1);
        } else {
            bucket.failures = bucket.failures.saturating_add(1);
        }
        let sample_count = if success {
            bucket.successful_latency_samples = bucket.successful_latency_samples.saturating_add(1);
            bucket.successful_latency_samples
        } else {
            bucket.failed_latency_samples = bucket.failed_latency_samples.saturating_add(1);
            bucket.failed_latency_samples
        };
        let (total, average, minimum, maximum, ewma) = if success {
            (
                &mut bucket.successful_latency_total_ms,
                &mut bucket.successful_latency_average_ms,
                &mut bucket.successful_latency_min_ms,
                &mut bucket.successful_latency_max_ms,
                &mut bucket.successful_latency_ewma_ms,
            )
        } else {
            (
                &mut bucket.failed_latency_total_ms,
                &mut bucket.failed_latency_average_ms,
                &mut bucket.failed_latency_min_ms,
                &mut bucket.failed_latency_max_ms,
                &mut bucket.failed_latency_ewma_ms,
            )
        };
        *total = total.saturating_add(latency_ms);
        *average = Some(*total as f64 / sample_count as f64);
        *minimum = Some(minimum.map_or(latency_ms, |value| value.min(latency_ms)));
        *maximum = Some(maximum.map_or(latency_ms, |value| value.max(latency_ms)));
        *ewma = Some(match *ewma {
            Some(previous) => (previous * 0.75) + (latency_ms as f64 * 0.25),
            None => latency_ms as f64,
        });
        // The original aggregate fields remain backward-compatible for
        // callers that need total request latency; the explicit successful
        // and failed fields above are used for routing and dashboards.
        bucket.total_latency_ms = bucket.total_latency_ms.saturating_add(latency_ms);
        bucket.average_latency_ms =
            Some(bucket.total_latency_ms as f64 / bucket.requests.max(1) as f64);
        bucket.min_latency_ms = Some(
            bucket
                .min_latency_ms
                .map_or(latency_ms, |value| value.min(latency_ms)),
        );
        bucket.max_latency_ms = Some(
            bucket
                .max_latency_ms
                .map_or(latency_ms, |value| value.max(latency_ms)),
        );
        bucket.ewma_latency_ms = Some(match bucket.ewma_latency_ms {
            Some(previous) => (previous * 0.75) + (latency_ms as f64 * 0.25),
            None => latency_ms as f64,
        });
        if let Some(prompt_tokens) = prompt_tokens_estimate {
            bucket.total_prompt_tokens = bucket
                .total_prompt_tokens
                .saturating_add(prompt_tokens as u64);
            bucket.average_prompt_tokens =
                Some(bucket.total_prompt_tokens as f64 / bucket.requests.max(1) as f64);
            bucket.ewma_prompt_tokens = Some(match bucket.ewma_prompt_tokens {
                Some(previous) => (previous * 0.75) + (prompt_tokens as f64 * 0.25),
                None => prompt_tokens as f64,
            });
        }
        bucket.last_latency_ms = Some(latency_ms);
        bucket.updated_at = Some(now_ts());
    }

    /// Record one user-visible AI operation in its modality bucket and the
    /// all-modalities bucket used for estimates before routing is known.
    pub async fn record_ai_response_time(
        &self,
        source: &str,
        latency_ms: u64,
        success: bool,
    ) -> Result<()> {
        self.record_ai_response_time_with_prompt(source, latency_ms, success, None)
            .await
    }

    pub async fn record_ai_response_time_with_prompt(
        &self,
        source: &str,
        latency_ms: u64,
        success: bool,
        prompt_tokens_estimate: Option<u32>,
    ) -> Result<()> {
        let source = Self::normalize_ai_source(source);
        let mut data = self.inner.lock().await;
        Self::merge_ai_response_time(
            data.ai_response_times
                .entry(source.to_string())
                .or_default(),
            latency_ms,
            success,
            prompt_tokens_estimate,
        );
        if source != "all" {
            Self::merge_ai_response_time(
                data.ai_response_times.entry("all".to_string()).or_default(),
                latency_ms,
                success,
                prompt_tokens_estimate,
            );
        }
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    /// Return the average observed response time for a modality, falling back
    /// to the aggregate bucket when the modality has no samples yet.
    pub async fn ai_response_time_estimate_ms(&self, source: &str) -> Option<u64> {
        let source = Self::normalize_ai_source(source);
        let data = self.inner.lock().await;
        data.ai_response_times
            .get(source)
            .or_else(|| data.ai_response_times.get("all"))
            .and_then(|stats| {
                stats
                    .successful_latency_average_ms
                    .or(stats.average_latency_ms)
            })
            .map(|value| value.max(1.0).round() as u64)
    }

    pub async fn ai_response_time_summary(&self) -> HashMap<String, AiResponseTimeStats> {
        self.inner.lock().await.ai_response_times.clone()
    }

    pub async fn record_api_source_response_time(
        &self,
        source: &str,
        latency_ms: u64,
        success: bool,
        prompt_tokens_estimate: Option<u32>,
    ) -> Result<()> {
        let source = source
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>();
        let source = if source.is_empty() {
            "unknown".to_string()
        } else {
            source
        };
        let mut data = self.inner.lock().await;
        Self::merge_ai_response_time(
            data.api_response_times.entry(source).or_default(),
            latency_ms,
            success,
            prompt_tokens_estimate,
        );
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn record_request_received(&self) -> Result<()> {
        let mut data = self.inner.lock().await;
        data.request_flow.received = data.request_flow.received.saturating_add(1);
        data.request_flow.queued = data.request_flow.queued.saturating_add(1);
        data.request_flow.in_progress = data.request_flow.in_progress.saturating_add(1);
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn record_request_terminal(&self, success: bool, timed_out: bool) -> Result<()> {
        let mut data = self.inner.lock().await;
        data.request_flow.queued = data.request_flow.queued.saturating_sub(1);
        if success {
            data.request_flow.replied = data.request_flow.replied.saturating_add(1);
        } else {
            data.request_flow.failed = data.request_flow.failed.saturating_add(1);
            if timed_out {
                data.request_flow.timed_out = data.request_flow.timed_out.saturating_add(1);
            }
        }
        data.request_flow.in_progress = data.request_flow.in_progress.saturating_sub(1);
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn record_request_replied(&self) -> Result<()> {
        self.record_request_terminal(true, false).await
    }

    pub async fn record_trading_semantic(&self, outcome: &str) -> Result<()> {
        let mut data = self.inner.lock().await;
        let metrics = &mut data.trading_semantic;
        metrics.responses = metrics.responses.saturating_add(1);
        match outcome {
            "parsed_valid" => metrics.parsed_valid = metrics.parsed_valid.saturating_add(1),
            "invalid_json" => metrics.invalid_json = metrics.invalid_json.saturating_add(1),
            "invalid_shape" => metrics.invalid_shape = metrics.invalid_shape.saturating_add(1),
            "incomplete_json" => metrics.incomplete = metrics.incomplete.saturating_add(1),
            "provider_failure" => {
                metrics.provider_failures = metrics.provider_failures.saturating_add(1)
            }
            _ => {}
        }
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn record_orchestration_event(
        &self,
        event: &str,
        duration_ms: Option<u64>,
    ) -> Result<()> {
        let mut data = self.inner.lock().await;
        match event {
            "candidate_selection" => {
                data.orchestration_events.candidate_selections = data
                    .orchestration_events
                    .candidate_selections
                    .saturating_add(1)
            }
            "capacity_race" => {
                data.orchestration_events.capacity_races =
                    data.orchestration_events.capacity_races.saturating_add(1)
            }
            "queue_wait" => {
                data.orchestration_events.queue_waits =
                    data.orchestration_events.queue_waits.saturating_add(1);
                data.orchestration_events.queue_wait_total_ms = data
                    .orchestration_events
                    .queue_wait_total_ms
                    .saturating_add(duration_ms.unwrap_or(0));
            }
            "queue_wait_timeout" => {
                data.orchestration_events.queue_wait_timeouts = data
                    .orchestration_events
                    .queue_wait_timeouts
                    .saturating_add(1)
            }
            "timeout" => {
                data.orchestration_events.timeouts =
                    data.orchestration_events.timeouts.saturating_add(1)
            }
            "fallback" => {
                data.orchestration_events.fallbacks =
                    data.orchestration_events.fallbacks.saturating_add(1)
            }
            "empty_plan" => {
                data.orchestration_events.empty_plans =
                    data.orchestration_events.empty_plans.saturating_add(1)
            }
            _ => return Ok(()),
        }
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    async fn save(&self, data: &MetricsData) -> Result<()> {
        let _persist_guard = self.persist_lock.lock().await;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let rendered = serde_json::to_string_pretty(data)?;
        // A scrape or restart must never observe a partially-written JSON
        // snapshot. The old direct write could truncate the file while
        // multiple request completions persisted telemetry concurrently.
        let temporary = self.path.with_extension(format!(
            "json.tmp-{}-{}",
            std::process::id(),
            now_ts().to_bits()
        ));
        fs::write(&temporary, rendered + "\n").await?;
        fs::rename(temporary, &self.path).await?;
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
        self.provider_in_health_backoff(provider, &["quota"], ttl_seconds)
            .await
    }

    pub async fn candidate_in_health_backoff(
        &self,
        candidate_id: &str,
        modes: &[&str],
        ttl_seconds: f64,
    ) -> bool {
        let now = now_ts();
        let data = self.inner.lock().await;
        let Some(bucket) = data.candidates.get(candidate_id) else {
            return false;
        };
        bucket
            .health
            .mode
            .as_deref()
            .is_some_and(|mode| modes.iter().any(|item| mode.eq_ignore_ascii_case(item)))
            && bucket
                .health
                .checked_at
                .is_some_and(|checked_at| now - checked_at < ttl_seconds)
    }

    pub async fn provider_in_health_backoff(
        &self,
        provider: &str,
        modes: &[&str],
        ttl_seconds: f64,
    ) -> bool {
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
                    .is_some_and(|mode| modes.iter().any(|item| mode.eq_ignore_ascii_case(item)))
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
        bucket.host_id = summary.host_id.clone();
        bucket.specialties = summary.specialties.clone();
        if health.ok == Some(true) {
            // A successful tags/health/completion probe proves that a prior
            // model-not-found observation is no longer actionable. Preserve
            // the historical count separately, but remove it from the live
            // success-rate denominator and routing error state.
            Self::decay_stale_model_failures(&mut bucket.stats);
            for role_bucket in bucket.roles.values_mut() {
                Self::decay_stale_model_failures(role_bucket);
            }
        }
        bucket.health = HealthBucket {
            checked_at: Some(now_ts()),
            ..health
        };
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    fn decay_stale_model_failures(bucket: &mut StatsBucket) {
        let Some(error) = bucket.last_error.as_deref() else {
            return;
        };
        let error = error.to_ascii_lowercase();
        // Ollama commonly reports this as `model '<name>' not found`, while
        // other adapters use `model not found: <name>`. Treat both forms as
        // stale once an authoritative health probe confirms the candidate.
        let named_model_not_found = error.contains("model '") && error.contains("' not found")
            || error.contains("model \"") && error.contains("\" not found");
        if !(error.contains("model not found")
            || named_model_not_found
            || error.contains("no such model"))
        {
            return;
        }
        bucket.stale_failures = bucket.stale_failures.saturating_add(bucket.failures);
        bucket.failures = 0;
        bucket.total = bucket.successes;
        bucket.last_error = None;
        bucket.last_status = Some("health_recovered".to_string());
    }

    fn merge_stats(
        bucket: &mut StatsBucket,
        success: bool,
        latency_ms: Option<u64>,
        telemetry: Option<&LocalUsageTelemetry>,
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
            let sample_count = if success {
                bucket.successful_latency_samples =
                    bucket.successful_latency_samples.saturating_add(1);
                bucket.successful_latency_samples
            } else {
                bucket.failed_latency_samples = bucket.failed_latency_samples.saturating_add(1);
                bucket.failed_latency_samples
            };
            let (total, average, minimum, maximum, ewma) = if success {
                (
                    &mut bucket.successful_latency_total_ms,
                    &mut bucket.successful_latency_average_ms,
                    &mut bucket.successful_latency_min_ms,
                    &mut bucket.successful_latency_max_ms,
                    &mut bucket.successful_latency_ewma_ms,
                )
            } else {
                (
                    &mut bucket.failed_latency_total_ms,
                    &mut bucket.failed_latency_average_ms,
                    &mut bucket.failed_latency_min_ms,
                    &mut bucket.failed_latency_max_ms,
                    &mut bucket.failed_latency_ewma_ms,
                )
            };
            *total = total.saturating_add(latency_ms);
            *average = Some(*total as f64 / sample_count as f64);
            *minimum = Some(minimum.map_or(latency_ms, |value| value.min(latency_ms)));
            *maximum = Some(maximum.map_or(latency_ms, |value| value.max(latency_ms)));
            *ewma = Some(match *ewma {
                Some(previous) => (previous * 0.75) + (latency_ms as f64 * 0.25),
                None => latency_ms as f64,
            });
            if success {
                // Preserve the pre-split JSON/Prometheus field names as
                // aliases for successful completion latency.
                bucket.total_latency_ms = bucket.successful_latency_total_ms;
                bucket.average_latency_ms = bucket.successful_latency_average_ms;
                bucket.min_latency_ms = bucket.successful_latency_min_ms;
                bucket.max_latency_ms = bucket.successful_latency_max_ms;
                bucket.ewma_latency_ms = bucket.successful_latency_ewma_ms;
            }
        }
        if let Some(telemetry) = telemetry {
            if let Some(prompt_tokens_estimate) = telemetry.prompt_tokens_estimate {
                bucket.ewma_prompt_tokens = Some(match bucket.ewma_prompt_tokens {
                    Some(previous) => (previous * 0.75) + (prompt_tokens_estimate as f64 * 0.25),
                    None => prompt_tokens_estimate as f64,
                });
            }
            if let Some(queue_wait_ms) = telemetry.queue_wait_ms {
                bucket.queue_wait_total_ms =
                    bucket.queue_wait_total_ms.saturating_add(queue_wait_ms);
                bucket.queue_wait_samples = bucket.queue_wait_samples.saturating_add(1);
                bucket.queue_wait_average_ms = Some(
                    bucket.queue_wait_total_ms as f64 / bucket.queue_wait_samples.max(1) as f64,
                );
                bucket.queue_wait_min_ms = Some(
                    bucket
                        .queue_wait_min_ms
                        .map_or(queue_wait_ms, |value| value.min(queue_wait_ms)),
                );
                bucket.queue_wait_max_ms = Some(
                    bucket
                        .queue_wait_max_ms
                        .map_or(queue_wait_ms, |value| value.max(queue_wait_ms)),
                );
                bucket.ewma_queue_wait_ms = Some(match bucket.ewma_queue_wait_ms {
                    Some(previous) => (previous * 0.75) + (queue_wait_ms as f64 * 0.25),
                    None => queue_wait_ms as f64,
                });
            }
            if let Some(inference_ms) = telemetry.inference_ms {
                bucket.ewma_inference_ms = Some(match bucket.ewma_inference_ms {
                    Some(previous) => (previous * 0.75) + (inference_ms as f64 * 0.25),
                    None => inference_ms as f64,
                });
            }
            if let Some(total_tokens_estimate) = telemetry.total_tokens_estimate {
                bucket.ewma_tokens_estimate = Some(match bucket.ewma_tokens_estimate {
                    Some(previous) => (previous * 0.75) + (total_tokens_estimate as f64 * 0.25),
                    None => total_tokens_estimate as f64,
                });
            }
            if let Some(completion_tokens_estimate) = telemetry.completion_tokens_estimate {
                bucket.ewma_completion_tokens = Some(match bucket.ewma_completion_tokens {
                    Some(previous) => {
                        (previous * 0.75) + (completion_tokens_estimate as f64 * 0.25)
                    }
                    None => completion_tokens_estimate as f64,
                });
            }
            // OpenAI-compatible llama.cpp nodes do not always report a
            // provider-side inference duration. In that case latency is the
            // effective end-to-end throughput denominator; Ollama nodes use
            // their precise inference duration above.
            if success
                && let (Some(completion_tokens), Some(duration_ms)) = (
                    telemetry.completion_tokens_estimate,
                    telemetry.inference_ms.or(latency_ms),
                )
                && completion_tokens > 0
                && duration_ms > 0
            {
                let tokens_per_second = completion_tokens as f64 * 1000.0 / duration_ms as f64;
                bucket.generation_tokens_per_second_samples = bucket
                    .generation_tokens_per_second_samples
                    .saturating_add(1);
                bucket.generation_tokens_per_second_total += tokens_per_second;
                bucket.generation_tokens_per_second_average = Some(
                    bucket.generation_tokens_per_second_total
                        / bucket.generation_tokens_per_second_samples.max(1) as f64,
                );
                bucket.generation_tokens_per_second_min = Some(
                    bucket
                        .generation_tokens_per_second_min
                        .map_or(tokens_per_second, |value| value.min(tokens_per_second)),
                );
                bucket.generation_tokens_per_second_max = Some(
                    bucket
                        .generation_tokens_per_second_max
                        .map_or(tokens_per_second, |value| value.max(tokens_per_second)),
                );
                bucket.generation_tokens_per_second_ewma = Some(
                    bucket
                        .generation_tokens_per_second_ewma
                        .map_or(tokens_per_second, |previous| {
                            (previous * 0.75) + (tokens_per_second * 0.25)
                        }),
                );
            }
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
        telemetry: Option<LocalUsageTelemetry>,
        quality: f64,
        error: Option<&str>,
    ) -> Result<()> {
        self.record_result_with_context(
            summary,
            "unknown",
            "unclassified",
            workflow,
            role,
            None,
            success,
            latency_ms,
            telemetry,
            quality,
            error,
        )
        .await
    }

    pub async fn record_result_with_context(
        &self,
        summary: &CandidateSummary,
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
        success: bool,
        latency_ms: Option<u64>,
        telemetry: Option<LocalUsageTelemetry>,
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
        bucket.host_id = summary.host_id.clone();
        bucket.specialties = summary.specialties.clone();
        Self::merge_stats(
            &mut bucket.stats,
            success,
            latency_ms,
            telemetry.as_ref(),
            quality,
            error,
        );
        let role_key =
            routing_profile_key(source, request_profile, workflow, role, request_category);
        let role_bucket = bucket.roles.entry(role_key).or_default();
        Self::merge_stats(
            role_bucket,
            success,
            latency_ms,
            telemetry.as_ref(),
            quality,
            error,
        );
        data.updated_at = now_ts();
        let snapshot = data.clone();
        drop(data);
        self.save(&snapshot).await
    }

    pub async fn score_bonus(&self, candidate_id: &str, workflow: &str, role: &str) -> f64 {
        self.score_bonus_for_context(
            candidate_id,
            "unknown",
            "unclassified",
            workflow,
            role,
            None,
        )
        .await
    }

    pub async fn score_bonus_for_context(
        &self,
        candidate_id: &str,
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
    ) -> f64 {
        let data = self.inner.lock().await;
        let Some(bucket) = data.candidates.get(candidate_id) else {
            return 0.0;
        };
        let role_key =
            routing_profile_key(source, request_profile, workflow, role, request_category);
        let legacy_key = format!("{workflow}:{role}");
        let stats = bucket
            .roles
            .get(&role_key)
            .filter(|stats| stats.total > 0)
            .or_else(|| bucket.roles.get(&legacy_key))
            .filter(|stats| stats.total > 0)
            .unwrap_or(&bucket.stats);
        if stats.total == 0 {
            return 0.0;
        }
        let success_rate = stats.successes as f64 / stats.total as f64;
        let latency_bonus = stats
            .successful_latency_ewma_ms
            .map(|latency| ((1500.0 - latency) / 3000.0).clamp(-0.35, 0.35))
            .unwrap_or(0.0);
        let range_penalty = stats
            .successful_latency_max_ms
            .zip(stats.successful_latency_min_ms)
            .map(|(max, min)| ((max.saturating_sub(min)) as f64 / 20_000.0).clamp(0.0, 0.25))
            .unwrap_or(0.0);
        let queue_wait_penalty = stats
            .ewma_queue_wait_ms
            .map(|queue_wait| (queue_wait / 1600.0).clamp(0.0, 0.45))
            .unwrap_or(0.0);
        let inference_penalty = stats
            .ewma_inference_ms
            .map(|inference| (inference / 8000.0).clamp(0.0, 0.45))
            .unwrap_or(0.0);
        let token_pressure_penalty = stats
            .ewma_tokens_estimate
            .map(|tokens| ((tokens - 1200.0).max(0.0) / 8000.0).clamp(0.0, 0.2))
            .unwrap_or(0.0);
        ((success_rate - 0.5) + stats.ewma_quality + latency_bonus
            - queue_wait_penalty
            - inference_penalty
            - token_pressure_penalty
            - range_penalty)
            .round_to(6)
    }

    /// Return the observed successful generation throughput for a candidate.
    /// Prefer the request-context EWMA when available, then fall back to the
    /// candidate-wide EWMA so newly introduced request profiles still benefit
    /// from endpoint observations already collected by Gail.
    pub async fn generation_tokens_per_second_for_context(
        &self,
        candidate_id: &str,
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
    ) -> Option<f64> {
        let data = self.inner.lock().await;
        let bucket = data.candidates.get(candidate_id)?;
        let role_key =
            routing_profile_key(source, request_profile, workflow, role, request_category);
        let legacy_key = format!("{workflow}:{role}");
        bucket
            .roles
            .get(&role_key)
            .or_else(|| bucket.roles.get(&legacy_key))
            .and_then(|stats| stats.generation_tokens_per_second_ewma)
            .or(bucket.stats.generation_tokens_per_second_ewma)
    }

    /// Estimate useful service capacity for one provider/model/request
    /// profile. A fast endpoint with poor success/quality history is worth
    /// less than a slightly slower endpoint that reliably returns usable
    /// output.
    pub async fn candidate_capacity_estimate_for_context(
        &self,
        candidate_id: &str,
        source: &str,
        request_profile: &str,
        workflow: &str,
        role: &str,
        request_category: Option<&str>,
        requested_output_tokens: u32,
    ) -> CandidateCapacityEstimate {
        let data = self.inner.lock().await;
        let Some(bucket) = data.candidates.get(candidate_id) else {
            return CandidateCapacityEstimate::default();
        };
        let role_key =
            routing_profile_key(source, request_profile, workflow, role, request_category);
        let legacy_key = format!("{workflow}:{role}");
        let stats = bucket
            .roles
            .get(&role_key)
            .filter(|stats| stats.total > 0)
            .or_else(|| bucket.roles.get(&legacy_key))
            .filter(|stats| stats.total > 0)
            .unwrap_or(&bucket.stats);
        // A request profile can contain failures before it contains a single
        // successful completion (for example immediately after a node
        // reboot).  Do not turn that sparse failure-only profile into the
        // optimistic default service time below.  Keep its success/quality
        // values for the useful-rate penalty, but borrow measured performance
        // from the candidate-wide bucket until this profile has a successful
        // sample.  Otherwise a known-slow endpoint can beat a known-fast
        // endpoint simply because its profile has not succeeded yet.
        let performance_stats = if stats.successes == 0 && bucket.stats.successes > 0 {
            &bucket.stats
        } else {
            stats
        };
        let samples = stats.total;
        let success_rate = if samples == 0 {
            0.75
        } else {
            stats.successes as f64 / samples as f64
        };
        let quality = if samples == 0 {
            0.7
        } else {
            stats.ewma_quality
        };
        let quality_factor = ((quality + 1.0) / 2.0).clamp(0.1, 1.0);
        let useful_rate = (success_rate * quality_factor).clamp(0.0, 1.0);
        let queue_wait_ms = performance_stats
            .ewma_queue_wait_ms
            .filter(|value| value.is_finite());
        let inference_ms = performance_stats
            .ewma_inference_ms
            .filter(|value| value.is_finite());
        let observed_latency_ms = performance_stats
            .successful_latency_ewma_ms
            .or(performance_stats.ewma_latency_ms)
            .filter(|value| value.is_finite());
        // Provider inference timing excludes the provider queue. When only
        // end-to-end latency is available, remove the learned queue portion
        // before adding Gail's current queue estimate.
        let service_base_ms = inference_ms.or_else(|| {
            observed_latency_ms.map(|latency| (latency - queue_wait_ms.unwrap_or(0.0)).max(0.0))
        });
        let generation_tokens_per_second = performance_stats
            .generation_tokens_per_second_ewma
            .filter(|value| value.is_finite() && *value > 0.0);
        let completion_tokens = performance_stats
            .ewma_completion_tokens
            .filter(|value| value.is_finite() && *value > 0.0);
        let requested_output_tokens = requested_output_tokens.max(1) as f64;
        let service_time_ms = match (service_base_ms, generation_tokens_per_second) {
            (Some(base), Some(tokens_per_second)) => {
                let generated_ms = requested_output_tokens * 1000.0 / tokens_per_second;
                let historical_generated_ms = completion_tokens
                    .map(|tokens| tokens * 1000.0 / tokens_per_second)
                    .unwrap_or(0.0);
                Some((base - historical_generated_ms).max(0.0) + generated_ms)
            }
            (None, Some(tokens_per_second)) => {
                Some(requested_output_tokens * 1000.0 / tokens_per_second)
            }
            (Some(base), None) => Some(base.max(1.0)),
            (None, None) => None,
        };
        CandidateCapacityEstimate {
            samples,
            success_rate,
            quality,
            useful_rate,
            generation_tokens_per_second,
            service_time_ms,
            queue_wait_ms,
            completion_tokens,
        }
    }

    pub async fn recent_usage_penalty(
        &self,
        candidate_id: &str,
        workflow: &str,
        role: &str,
        decay_seconds: f64,
    ) -> f64 {
        let decay_seconds = decay_seconds.max(30.0);
        let data = self.inner.lock().await;
        let Some(bucket) = data.candidates.get(candidate_id) else {
            return 0.0;
        };
        let role_key = format!("{workflow}:{role}");
        let stats = bucket.roles.get(&role_key).unwrap_or(&bucket.stats);
        if stats.total == 0 {
            return 0.0;
        }
        let updated_at = stats.updated_at.unwrap_or(0.0);
        if updated_at <= 0.0 {
            return 0.0;
        }
        let age_seconds = (now_ts() - updated_at).max(0.0);
        if age_seconds > decay_seconds * 8.0 {
            return 0.0;
        }
        let recency = (-(age_seconds / decay_seconds)).exp();
        let intensity = ((stats.total as f64).ln_1p() / 6.0).clamp(0.0, 1.5);
        (intensity * recency).round_to(6)
    }

    pub async fn summary(&self, limit: usize) -> MetricsSummary {
        let data = self.inner.lock().await;
        let mut candidates = data
            .candidates
            .iter()
            .map(|(candidate_id, bucket)| CandidateMetricsSummary {
                candidate_id: candidate_id.clone(),
                host_id: bucket.host_id.clone(),
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
                average_latency_ms: bucket.stats.average_latency_ms,
                min_latency_ms: bucket.stats.min_latency_ms,
                max_latency_ms: bucket.stats.max_latency_ms,
                success_rate: if bucket.stats.total > 0 {
                    Some((bucket.stats.successes as f64 / bucket.stats.total as f64).round_to(6))
                } else {
                    None
                },
                ewma_latency_ms: bucket.stats.ewma_latency_ms,
                successful_latency_samples: bucket.stats.successful_latency_samples,
                successful_latency_average_ms: bucket.stats.successful_latency_average_ms,
                successful_latency_min_ms: bucket.stats.successful_latency_min_ms,
                successful_latency_max_ms: bucket.stats.successful_latency_max_ms,
                successful_latency_ewma_ms: bucket.stats.successful_latency_ewma_ms,
                failed_latency_samples: bucket.stats.failed_latency_samples,
                failed_latency_average_ms: bucket.stats.failed_latency_average_ms,
                failed_latency_min_ms: bucket.stats.failed_latency_min_ms,
                failed_latency_max_ms: bucket.stats.failed_latency_max_ms,
                failed_latency_ewma_ms: bucket.stats.failed_latency_ewma_ms,
                stale_failures: bucket.stats.stale_failures,
                queue_wait_average_ms: bucket.stats.queue_wait_average_ms,
                queue_wait_min_ms: bucket.stats.queue_wait_min_ms,
                queue_wait_max_ms: bucket.stats.queue_wait_max_ms,
                ewma_queue_wait_ms: bucket.stats.ewma_queue_wait_ms,
                ewma_inference_ms: bucket.stats.ewma_inference_ms,
                ewma_prompt_tokens: bucket.stats.ewma_prompt_tokens,
                ewma_tokens_estimate: bucket.stats.ewma_tokens_estimate,
                ewma_completion_tokens: bucket.stats.ewma_completion_tokens,
                generation_tokens_per_second_average: bucket
                    .stats
                    .generation_tokens_per_second_average,
                generation_tokens_per_second_min: bucket.stats.generation_tokens_per_second_min,
                generation_tokens_per_second_max: bucket.stats.generation_tokens_per_second_max,
                generation_tokens_per_second_ewma: bucket.stats.generation_tokens_per_second_ewma,
                ewma_quality: bucket.stats.ewma_quality.round_to(6),
                last_status: bucket.stats.last_status.clone(),
                last_error: bucket.stats.last_error.clone(),
                updated_at: bucket.stats.updated_at,
                health_ok: bucket.health.ok,
                health_mode: bucket.health.mode.clone(),
                health_checked_at: bucket.health.checked_at,
                routing_profiles: bucket.roles.clone(),
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
            ai_response_times: data.ai_response_times.clone(),
            api_response_times: data.api_response_times.clone(),
        }
    }

    pub async fn prometheus_metrics(&self) -> String {
        let data = self.inner.lock().await.clone();
        let training_path = self
            .path
            .parent()
            .map(|parent| parent.join("training_metrics.json"))
            .unwrap_or_else(|| PathBuf::from("training_metrics.json"));
        let persisted_training = fs::read_to_string(&training_path)
            .await
            .ok()
            .and_then(|raw| serde_json::from_str::<TrainingMetricsData>(&raw).ok());
        let training = match persisted_training {
            Some(training) if !training.runs.is_empty() => training,
            _ => {
                let snapshot_root = training_path
                    .parent()
                    .map(|parent| parent.join("training").join("snapshots"))
                    .unwrap_or_else(|| PathBuf::from("training/snapshots"));
                TrainingMetricsData {
                    runs: discover_training_observations(snapshot_root, "unknown")
                        .await
                        .unwrap_or_default(),
                    updated_at: now_ts(),
                }
            }
        };
        let snapshot_root = training_path
            .parent()
            .map(|parent| parent.join("training").join("snapshots"))
            .unwrap_or_else(|| PathBuf::from("training/snapshots"));
        let progress = discover_training_progress(snapshot_root).await;
        let mut rendered = render_prometheus_metrics(&data);
        render_training_prometheus_metrics(&mut rendered, &training, &progress);
        rendered
    }
}

fn routing_profile_key(
    source: &str,
    request_profile: &str,
    workflow: &str,
    role: &str,
    request_category: Option<&str>,
) -> String {
    fn clean(value: &str) -> String {
        value
            .trim()
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                    ch.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim_matches('_')
            .chars()
            .take(96)
            .collect()
    }
    format!(
        "source={};profile={};workflow={};role={};category={}",
        clean(source),
        clean(request_profile),
        clean(workflow),
        clean(role),
        clean(request_category.unwrap_or("")),
    )
}

fn render_prometheus_metrics(data: &MetricsData) -> String {
    let mut out = String::new();
    out.push_str(
        "# HELP gail_orchestration_events_total Gail orchestration lifecycle events by outcome.\n",
    );
    out.push_str("# TYPE gail_orchestration_events_total counter\n");
    let events = &data.orchestration_events;
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"candidate_selection\"}} {}\n",
        events.candidate_selections
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"capacity_race\"}} {}\n",
        events.capacity_races
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"queue_wait\"}} {}\n",
        events.queue_waits
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"queue_wait_timeout\"}} {}\n",
        events.queue_wait_timeouts
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"timeout\"}} {}\n",
        events.timeouts
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"fallback\"}} {}\n",
        events.fallbacks
    ));
    out.push_str(&format!(
        "gail_orchestration_events_total{{event=\"empty_plan\"}} {}\n",
        events.empty_plans
    ));
    out.push_str("# HELP gail_orchestration_queue_wait_total_ms Gail orchestration queue wait time in milliseconds.\n");
    out.push_str("# TYPE gail_orchestration_queue_wait_total_ms counter\n");
    out.push_str(&format!(
        "gail_orchestration_queue_wait_total_ms {}\n",
        events.queue_wait_total_ms
    ));
    out.push_str("# HELP gail_requests_received_total Gail LLM requests received.\n");
    out.push_str("# TYPE gail_requests_received_total counter\n");
    out.push_str(&format!(
        "gail_requests_received_total {}\n",
        data.request_flow.received
    ));
    out.push_str("# HELP gail_requests_in_progress Gail LLM requests currently being processed.\n");
    out.push_str("# TYPE gail_requests_in_progress gauge\n");
    out.push_str(&format!(
        "gail_requests_in_progress {}\n",
        data.request_flow.in_progress
    ));
    out.push_str("# HELP gail_requests_queued Gail LLM requests queued or awaiting terminal accounting.\n# TYPE gail_requests_queued gauge\n");
    out.push_str(&format!(
        "gail_requests_queued {}\n",
        data.request_flow.queued
    ));
    out.push_str("# HELP gail_requests_replied_total Gail LLM requests that completed a reply.\n");
    out.push_str("# TYPE gail_requests_replied_total counter\n");
    out.push_str(&format!(
        "gail_requests_replied_total {}\n",
        data.request_flow.replied
    ));
    out.push_str("# HELP gail_requests_failed_total Gail LLM requests that failed.\n# TYPE gail_requests_failed_total counter\n");
    out.push_str(&format!(
        "gail_requests_failed_total {}\n",
        data.request_flow.failed
    ));
    let terminal = data
        .request_flow
        .replied
        .saturating_add(data.request_flow.failed);
    let unaccounted = data.request_flow.received.saturating_sub(terminal);
    out.push_str("# HELP gail_requests_terminal_total Gail requests with a recorded terminal outcome.\n# TYPE gail_requests_terminal_total counter\n");
    out.push_str(&format!("gail_requests_terminal_total {terminal}\n"));
    out.push_str("# HELP gail_requests_unaccounted_total Received requests without a recorded terminal outcome.\n# TYPE gail_requests_unaccounted_total gauge\n");
    out.push_str(&format!("gail_requests_unaccounted_total {unaccounted}\n"));
    out.push_str("# HELP gail_requests_timed_out_total Gail LLM requests that timed out.\n# TYPE gail_requests_timed_out_total counter\n");
    out.push_str(&format!(
        "gail_requests_timed_out_total {}\n",
        data.request_flow.timed_out
    ));
    out.push_str("# HELP gail_trading_responses_total Trading advisory responses classified by semantic validity.\n# TYPE gail_trading_responses_total counter\n");
    out.push_str(&format!(
        "gail_trading_responses_total {}\n",
        data.trading_semantic.responses
    ));
    for (name, value) in [
        ("parsed_valid", data.trading_semantic.parsed_valid),
        ("invalid_json", data.trading_semantic.invalid_json),
        ("invalid_shape", data.trading_semantic.invalid_shape),
        ("incomplete_json", data.trading_semantic.incomplete),
        ("provider_failure", data.trading_semantic.provider_failures),
    ] {
        out.push_str(&format!(
            "gail_trading_responses_total{{outcome=\"{name}\"}} {value}\n"
        ));
    }
    out.push_str("# HELP gail_provider_candidate_successes_total Gail provider candidate successful completions.\n");
    out.push_str("# TYPE gail_provider_candidate_successes_total counter\n");
    out.push_str("# HELP gail_provider_candidate_failures_total Gail provider candidate failed completions.\n");
    out.push_str("# TYPE gail_provider_candidate_failures_total counter\n");
    out.push_str("# HELP gail_provider_candidate_health_ok Gail provider candidate health, 1 for healthy and 0 for degraded.\n");
    out.push_str("# TYPE gail_provider_candidate_health_ok gauge\n");
    out.push_str("# HELP gail_provider_candidate_latency_ms Gail provider candidate EWMA latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_latency_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_latency_average_ms Average observed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_latency_average_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_latency_min_ms Minimum observed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_latency_min_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_latency_max_ms Maximum observed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_latency_max_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_success_latency_average_ms Average successful provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_success_latency_average_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_success_latency_min_ms Minimum successful provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_success_latency_min_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_success_latency_max_ms Maximum successful provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_success_latency_max_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_success_latency_ewma_ms EWMA successful provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_success_latency_ewma_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_failure_latency_average_ms Average failed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_failure_latency_average_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_failure_latency_min_ms Minimum failed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_failure_latency_min_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_failure_latency_max_ms Maximum failed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_failure_latency_max_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_failure_latency_ewma_ms EWMA failed provider candidate latency in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_failure_latency_ewma_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_stale_failures_total Failed observations removed from active routing after health recovery.\n");
    out.push_str("# TYPE gail_provider_candidate_stale_failures_total counter\n");
    out.push_str("# HELP gail_provider_candidate_queue_wait_ms Gail provider candidate EWMA local queue wait in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_queue_wait_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_queue_wait_average_ms Average local queue wait in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_queue_wait_average_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_queue_wait_min_ms Minimum local queue wait in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_queue_wait_min_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_queue_wait_max_ms Maximum local queue wait in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_queue_wait_max_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_inference_ms Gail provider candidate EWMA local inference duration in milliseconds.\n");
    out.push_str("# TYPE gail_provider_candidate_inference_ms gauge\n");
    out.push_str("# HELP gail_provider_candidate_tokens_estimate Gail provider candidate EWMA token estimate.\n");
    out.push_str("# TYPE gail_provider_candidate_tokens_estimate gauge\n");
    out.push_str("# HELP gail_provider_candidate_generation_tokens_per_second Successful generated-token throughput by serving host.\n");
    out.push_str("# TYPE gail_provider_candidate_generation_tokens_per_second gauge\n");
    out.push_str("# HELP gail_provider_candidate_generation_tokens_per_second_average Average successful generated-token throughput by serving host.\n");
    out.push_str("# TYPE gail_provider_candidate_generation_tokens_per_second_average gauge\n");
    out.push_str("# HELP gail_provider_candidate_generation_tokens_per_second_min Minimum successful generated-token throughput by serving host.\n");
    out.push_str("# TYPE gail_provider_candidate_generation_tokens_per_second_min gauge\n");
    out.push_str("# HELP gail_provider_candidate_generation_tokens_per_second_max Maximum successful generated-token throughput by serving host.\n");
    out.push_str("# TYPE gail_provider_candidate_generation_tokens_per_second_max gauge\n");
    out.push_str("# HELP gail_ai_response_time_average_ms Average user-visible AI response time in milliseconds.\n");
    out.push_str("# TYPE gail_ai_response_time_average_ms gauge\n");
    out.push_str("# HELP gail_ai_response_time_requests_total User-visible AI requests observed by modality.\n");
    out.push_str("# TYPE gail_ai_response_time_requests_total counter\n");
    out.push_str("# HELP gail_ai_response_time_min_ms Minimum user-visible AI response time in milliseconds.\n");
    out.push_str("# TYPE gail_ai_response_time_min_ms gauge\n");
    out.push_str("# HELP gail_ai_response_time_max_ms Maximum user-visible AI response time in milliseconds.\n");
    out.push_str("# TYPE gail_ai_response_time_max_ms gauge\n");
    out.push_str("# HELP gail_ai_response_time_success_average_ms Average successful user-visible AI response time.\n");
    out.push_str("# TYPE gail_ai_response_time_success_average_ms gauge\n");
    out.push_str("# HELP gail_ai_response_time_failure_average_ms Average failed user-visible AI response time.\n");
    out.push_str("# TYPE gail_ai_response_time_failure_average_ms gauge\n");
    out.push_str("# HELP gail_api_source_response_time_average_ms Average response time by authenticated API source.\n");
    out.push_str("# TYPE gail_api_source_response_time_average_ms gauge\n");
    out.push_str("# HELP gail_api_source_response_time_min_ms Minimum response time by authenticated API source.\n");
    out.push_str("# TYPE gail_api_source_response_time_min_ms gauge\n");
    out.push_str("# HELP gail_api_source_response_time_max_ms Maximum response time by authenticated API source.\n");
    out.push_str("# TYPE gail_api_source_response_time_max_ms gauge\n");
    out.push_str(
        "# HELP gail_api_source_requests_total Requests observed by authenticated API source.\n",
    );
    out.push_str("# TYPE gail_api_source_requests_total counter\n");
    for (source, stats) in &data.ai_response_times {
        let labels = format!("source=\"{}\"", escape_label(source));
        if let Some(average) = stats.average_latency_ms {
            out.push_str(&format!(
                "gail_ai_response_time_average_ms{{{labels}}} {:.3}\n",
                average
            ));
        }
        out.push_str(&format!(
            "gail_ai_response_time_requests_total{{{labels}}} {}\n",
            stats.requests
        ));
        if let Some(minimum) = stats.min_latency_ms {
            out.push_str(&format!(
                "gail_ai_response_time_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = stats.max_latency_ms {
            out.push_str(&format!(
                "gail_ai_response_time_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        if let Some(average) = stats.successful_latency_average_ms {
            out.push_str(&format!(
                "gail_ai_response_time_success_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(average) = stats.failed_latency_average_ms {
            out.push_str(&format!(
                "gail_ai_response_time_failure_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
    }
    for (source, stats) in &data.api_response_times {
        let labels = format!("source=\"{}\"", escape_label(source));
        if let Some(average) = stats.average_latency_ms {
            out.push_str(&format!(
                "gail_api_source_response_time_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = stats.min_latency_ms {
            out.push_str(&format!(
                "gail_api_source_response_time_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = stats.max_latency_ms {
            out.push_str(&format!(
                "gail_api_source_response_time_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        out.push_str(&format!(
            "gail_api_source_requests_total{{{labels}}} {}\n",
            stats.requests
        ));
    }
    for (candidate_id, bucket) in &data.candidates {
        let labels = format!(
            "candidate_id=\"{}\",host_id=\"{}\",provider=\"{}\",model=\"{}\",health_mode=\"{}\"",
            escape_label(candidate_id),
            escape_label(bucket.host_id.as_deref().unwrap_or("unknown")),
            escape_label(bucket.provider.as_deref().unwrap_or("")),
            escape_label(
                bucket
                    .resolved_model
                    .as_deref()
                    .or(bucket.model.as_deref())
                    .unwrap_or("")
            ),
            escape_label(bucket.health.mode.as_deref().unwrap_or("unknown")),
        );
        out.push_str(&format!(
            "gail_provider_candidate_successes_total{{{labels}}} {}\n",
            bucket.stats.successes
        ));
        out.push_str(&format!(
            "gail_provider_candidate_failures_total{{{labels}}} {}\n",
            bucket.stats.failures
        ));
        out.push_str(&format!(
            "gail_provider_candidate_stale_failures_total{{{labels}}} {}\n",
            bucket.stats.stale_failures
        ));
        if let Some(ok) = bucket.health.ok {
            out.push_str(&format!(
                "gail_provider_candidate_health_ok{{{labels}}} {}\n",
                if ok { 1 } else { 0 }
            ));
        }
        if let Some(latency) = bucket.stats.ewma_latency_ms {
            out.push_str(&format!(
                "gail_provider_candidate_latency_ms{{{labels}}} {:.3}\n",
                latency
            ));
        }
        if let Some(average) = bucket.stats.average_latency_ms {
            out.push_str(&format!(
                "gail_provider_candidate_latency_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = bucket.stats.min_latency_ms {
            out.push_str(&format!(
                "gail_provider_candidate_latency_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = bucket.stats.max_latency_ms {
            out.push_str(&format!(
                "gail_provider_candidate_latency_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        if let Some(average) = bucket.stats.successful_latency_average_ms {
            out.push_str(&format!(
                "gail_provider_candidate_success_latency_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = bucket.stats.successful_latency_min_ms {
            out.push_str(&format!(
                "gail_provider_candidate_success_latency_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = bucket.stats.successful_latency_max_ms {
            out.push_str(&format!(
                "gail_provider_candidate_success_latency_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        if let Some(ewma) = bucket.stats.successful_latency_ewma_ms {
            out.push_str(&format!(
                "gail_provider_candidate_success_latency_ewma_ms{{{labels}}} {ewma:.3}\n"
            ));
        }
        if let Some(average) = bucket.stats.failed_latency_average_ms {
            out.push_str(&format!(
                "gail_provider_candidate_failure_latency_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = bucket.stats.failed_latency_min_ms {
            out.push_str(&format!(
                "gail_provider_candidate_failure_latency_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = bucket.stats.failed_latency_max_ms {
            out.push_str(&format!(
                "gail_provider_candidate_failure_latency_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        if let Some(ewma) = bucket.stats.failed_latency_ewma_ms {
            out.push_str(&format!(
                "gail_provider_candidate_failure_latency_ewma_ms{{{labels}}} {ewma:.3}\n"
            ));
        }
        if let Some(ewma) = bucket.stats.generation_tokens_per_second_ewma {
            out.push_str(&format!(
                "gail_provider_candidate_generation_tokens_per_second{{{labels}}} {ewma:.3}\n"
            ));
        }
        if let Some(average) = bucket.stats.generation_tokens_per_second_average {
            out.push_str(&format!(
                "gail_provider_candidate_generation_tokens_per_second_average{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = bucket.stats.generation_tokens_per_second_min {
            out.push_str(&format!(
                "gail_provider_candidate_generation_tokens_per_second_min{{{labels}}} {minimum:.3}\n"
            ));
        }
        if let Some(maximum) = bucket.stats.generation_tokens_per_second_max {
            out.push_str(&format!(
                "gail_provider_candidate_generation_tokens_per_second_max{{{labels}}} {maximum:.3}\n"
            ));
        }
        if let Some(queue_wait) = bucket.stats.ewma_queue_wait_ms {
            out.push_str(&format!(
                "gail_provider_candidate_queue_wait_ms{{{labels}}} {:.3}\n",
                queue_wait
            ));
        }
        if let Some(average) = bucket.stats.queue_wait_average_ms {
            out.push_str(&format!(
                "gail_provider_candidate_queue_wait_average_ms{{{labels}}} {average:.3}\n"
            ));
        }
        if let Some(minimum) = bucket.stats.queue_wait_min_ms {
            out.push_str(&format!(
                "gail_provider_candidate_queue_wait_min_ms{{{labels}}} {minimum}\n"
            ));
        }
        if let Some(maximum) = bucket.stats.queue_wait_max_ms {
            out.push_str(&format!(
                "gail_provider_candidate_queue_wait_max_ms{{{labels}}} {maximum}\n"
            ));
        }
        if let Some(inference) = bucket.stats.ewma_inference_ms {
            out.push_str(&format!(
                "gail_provider_candidate_inference_ms{{{labels}}} {:.3}\n",
                inference
            ));
        }
        if let Some(tokens) = bucket.stats.ewma_tokens_estimate {
            out.push_str(&format!(
                "gail_provider_candidate_tokens_estimate{{{labels}}} {:.3}\n",
                tokens
            ));
        }
        for (profile, stats) in &bucket.roles {
            let profile_labels = format!("{labels},request_profile=\"{}\"", escape_label(profile));
            if let Some(average) = stats.average_latency_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_latency_average_ms{{{profile_labels}}} {average:.3}\n"
                ));
            }
            if let Some(minimum) = stats.min_latency_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_latency_min_ms{{{profile_labels}}} {minimum}\n"
                ));
            }
            if let Some(maximum) = stats.max_latency_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_latency_max_ms{{{profile_labels}}} {maximum}\n"
                ));
            }
            if let Some(average) = stats.successful_latency_average_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_success_latency_average_ms{{{profile_labels}}} {average:.3}\n"
                ));
            }
            if let Some(average) = stats.failed_latency_average_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_failure_latency_average_ms{{{profile_labels}}} {average:.3}\n"
                ));
            }
            if let Some(average) = stats.queue_wait_average_ms {
                out.push_str(&format!(
                    "gail_provider_request_profile_queue_wait_average_ms{{{profile_labels}}} {average:.3}\n"
                ));
            }
            out.push_str(&format!(
                "gail_provider_request_profile_requests_total{{{profile_labels}}} {}\n",
                stats.total
            ));
        }
    }
    out
}

fn render_training_prometheus_metrics(
    out: &mut String,
    data: &TrainingMetricsData,
    progress: &[TrainingProgressObservation],
) {
    out.push_str(
        "# HELP gail_training_runs_total Gail training runs observed by backend and status.\n",
    );
    out.push_str("# TYPE gail_training_runs_total counter\n");
    out.push_str("# HELP gail_training_task_progress_ratio Current progress ratio for active training tasks.\n");
    out.push_str("# TYPE gail_training_task_progress_ratio gauge\n");
    out.push_str("# HELP gail_training_task_progress_per_hour Training optimizer steps completed per hour.\n");
    out.push_str("# TYPE gail_training_task_progress_per_hour gauge\n");
    out.push_str("# HELP gail_training_task_eta_seconds Estimated seconds remaining for each active training task.\n");
    out.push_str("# TYPE gail_training_task_eta_seconds gauge\n");
    out.push_str("# HELP gail_training_task_elapsed_seconds Elapsed seconds for each active training task.\n");
    out.push_str("# TYPE gail_training_task_elapsed_seconds gauge\n");
    out.push_str("# HELP gail_training_task_updated_timestamp_seconds Last progress update timestamp for each active training task.\n");
    out.push_str("# TYPE gail_training_task_updated_timestamp_seconds gauge\n");
    out.push_str("# HELP gail_training_active_tasks Number of active Gail training tasks.\n");
    out.push_str("# TYPE gail_training_active_tasks gauge\n");
    out.push_str("# HELP gail_training_average_eta_seconds Average estimated seconds remaining across active training tasks.\n");
    out.push_str("# TYPE gail_training_average_eta_seconds gauge\n");
    let mut eta_total = 0.0;
    for task in progress {
        let labels = format!(
            "snapshot_id=\"{}\",backend=\"{}\",slurm_job_id=\"{}\"",
            escape_label(&task.snapshot_id),
            escape_label(&task.backend),
            escape_label(task.slurm_job_id.as_deref().unwrap_or("")),
        );
        out.push_str(&format!(
            "gail_training_task_progress_ratio{{{labels}}} {:.6}\n",
            task.progress_ratio.clamp(0.0, 1.0)
        ));
        out.push_str(&format!(
            "gail_training_task_progress_per_hour{{{labels}}} {:.3}\n",
            task.progress_per_hour
        ));
        out.push_str(&format!(
            "gail_training_task_eta_seconds{{{labels}}} {:.3}\n",
            task.eta_seconds
        ));
        out.push_str(&format!(
            "gail_training_task_elapsed_seconds{{{labels}}} {:.3}\n",
            task.elapsed_seconds
        ));
        if let Some(updated) = task.updated_ts {
            out.push_str(&format!(
                "gail_training_task_updated_timestamp_seconds{{{labels}}} {:.3}\n",
                updated
            ));
        }
        eta_total += task.eta_seconds;
    }
    out.push_str(&format!("gail_training_active_tasks {}\n", progress.len()));
    out.push_str(&format!(
        "gail_training_average_eta_seconds {:.3}\n",
        if progress.is_empty() {
            0.0
        } else {
            eta_total / progress.len() as f64
        }
    ));
    out.push_str("# HELP gail_training_tokens_total Tokens processed by training runs.\n");
    out.push_str("# TYPE gail_training_tokens_total counter\n");
    out.push_str("# HELP gail_training_tokens_per_second Training throughput for each run.\n");
    out.push_str("# TYPE gail_training_tokens_per_second gauge\n");
    out.push_str("# HELP gail_training_runtime_seconds Training runtime for each run.\n");
    out.push_str("# TYPE gail_training_runtime_seconds gauge\n");
    out.push_str("# HELP gail_training_samples_total Samples processed by each training run.\n");
    out.push_str("# TYPE gail_training_samples_total gauge\n");
    out.push_str("# HELP gail_training_optimizer_steps_total Optimizer steps completed by each training run.\n");
    out.push_str("# TYPE gail_training_optimizer_steps_total gauge\n");
    out.push_str("# HELP gail_training_last_finished_timestamp_seconds Unix timestamp of the latest training observation.\n");
    out.push_str("# TYPE gail_training_last_finished_timestamp_seconds gauge\n");
    out.push_str("# HELP gail_training_cumulative_training Whether the run resumed from the previous adapter.\n");
    out.push_str("# TYPE gail_training_cumulative_training gauge\n");
    out.push_str("# HELP gail_training_cpu_fallback Whether requested QLoRA intentionally ran as CPU LoRA.\n");
    out.push_str("# TYPE gail_training_cpu_fallback gauge\n");
    out.push_str("# HELP gail_training_pin_memory Whether DataLoader pinned memory was enabled.\n");
    out.push_str("# TYPE gail_training_pin_memory gauge\n");
    out.push_str("# HELP gail_training_quantisation_backend_info Effective quantisation backend for the run.\n");
    out.push_str("# TYPE gail_training_quantisation_backend_info gauge\n");
    out.push_str("# HELP gail_training_failure_reason_total Training runs grouped by stable failure category.\n");
    out.push_str("# TYPE gail_training_failure_reason_total gauge\n");
    for run in &data.runs {
        let labels = format!(
            "snapshot_id=\"{}\",backend=\"{}\",status=\"{}\",failure_reason=\"{}\",model=\"{}\",slurm_job_id=\"{}\",nodelist=\"{}\"",
            escape_label(&run.snapshot_id),
            escape_label(&run.backend),
            escape_label(&run.status),
            escape_label(&run.failure_reason),
            escape_label(&run.base_model),
            escape_label(run.slurm_job_id.as_deref().unwrap_or("")),
            escape_label(run.nodelist.as_deref().unwrap_or("")),
        );
        out.push_str(&format!(
            "gail_training_tokens_total{{{labels}}} {}\n",
            run.total_tokens
        ));
        out.push_str(&format!(
            "gail_training_tokens_per_second{{{labels}}} {:.3}\n",
            run.tokens_per_second
        ));
        out.push_str(&format!(
            "gail_training_runtime_seconds{{{labels}}} {:.3}\n",
            run.runtime_seconds
        ));
        out.push_str(&format!(
            "gail_training_samples_total{{{labels}}} {}\n",
            run.samples
        ));
        out.push_str(&format!(
            "gail_training_optimizer_steps_total{{{labels}}} {}\n",
            run.optimizer_steps
        ));
        if let Some(finished) = run.finished_ts {
            out.push_str(&format!(
                "gail_training_last_finished_timestamp_seconds{{{labels}}} {:.3}\n",
                finished
            ));
        }
        out.push_str(&format!(
            "gail_training_cumulative_training{{{labels}}} {}\n",
            if run.cumulative_training { 1 } else { 0 }
        ));
        out.push_str(&format!(
            "gail_training_cpu_fallback{{{labels}}} {}\n",
            if run.cpu_fallback { 1 } else { 0 }
        ));
        out.push_str(&format!(
            "gail_training_pin_memory{{{labels}}} {}\n",
            if run.pin_memory { 1 } else { 0 }
        ));
        out.push_str(&format!(
            "gail_training_quantisation_backend_info{{{},quantisation_backend=\"{}\"}} 1\n",
            labels,
            escape_label(&run.quantisation_backend)
        ));
    }
    let mut totals: HashMap<(String, String), u64> = HashMap::new();
    for run in &data.runs {
        *totals
            .entry((run.backend.clone(), run.status.clone()))
            .or_default() += 1;
    }
    for ((backend, status), count) in totals {
        let labels = format!(
            "backend=\"{}\",status=\"{}\"",
            escape_label(&backend),
            escape_label(&status)
        );
        out.push_str(&format!("gail_training_runs_total{{{labels}}} {count}\n"));
    }
    let mut failures: HashMap<(String, String), u64> = HashMap::new();
    for run in &data.runs {
        if run.status == "failed" || run.status == "promotion_failed" {
            *failures
                .entry((
                    run.backend.clone(),
                    if run.failure_reason.is_empty() {
                        "unknown".to_string()
                    } else {
                        run.failure_reason.clone()
                    },
                ))
                .or_default() += 1;
        }
    }
    for ((backend, reason), count) in failures {
        let labels = format!(
            "backend=\"{}\",reason=\"{}\"",
            escape_label(&backend),
            escape_label(&reason)
        );
        out.push_str(&format!(
            "gail_training_failure_reason_total{{{labels}}} {count}\n"
        ));
    }
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
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
            host_id: Some("test-host".to_string()),
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

    #[tokio::test]
    async fn candidate_health_backoff_matches_exact_candidate() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let first = summary("ollama", "qwen2.5-coder:1.5b@openai_compat");
        let second = summary("ollama", "qwen2.5-coder:1.5b@native");
        store
            .record_health(
                &first,
                HealthBucket {
                    ok: Some(false),
                    mode: Some("ollama_saturated".to_string()),
                    checked_at: None,
                    latency_ms: Some(10),
                    message: Some("queue saturated".to_string()),
                },
            )
            .await
            .expect("record first");
        assert!(
            store
                .candidate_in_health_backoff(&first.candidate_id, &["ollama_saturated"], 1800.0)
                .await
        );
        assert!(
            !store
                .candidate_in_health_backoff(&second.candidate_id, &["ollama_saturated"], 1800.0)
                .await
        );
    }

    #[tokio::test]
    async fn prometheus_metrics_include_provider_candidate_counters() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let summary = summary("ollama", "llama3.2");
        store
            .record_result(
                &summary,
                "project_solver",
                "planner",
                true,
                Some(42),
                Some(LocalUsageTelemetry {
                    prompt_tokens_estimate: None,
                    queue_wait_ms: Some(10),
                    inference_ms: Some(32),
                    total_tokens_estimate: Some(128),
                    completion_tokens_estimate: Some(64),
                }),
                1.0,
                None,
            )
            .await
            .expect("record result");
        let rendered = store.prometheus_metrics().await;
        assert!(rendered.contains("gail_provider_candidate_successes_total"));
        assert!(rendered.contains("gail_provider_candidate_latency_average_ms"));
        assert!(rendered.contains("candidate_id=\"ollama/llama3.2\""));
        assert!(rendered.contains("host_id=\"test-host\""));
        assert!(rendered.contains("gail_provider_candidate_generation_tokens_per_second"));
    }

    #[tokio::test]
    async fn serving_throughput_and_training_observations_are_exported() {
        let directory = tempfile::tempdir().expect("metrics directory");
        let provider_path = directory.path().join("provider_metrics.json");
        let store = MetricsStore::new(provider_path).await.expect("store");
        store
            .record_result(
                &summary("ollama", "qwen3.5:9b"),
                "interactive",
                "general",
                true,
                Some(1_100),
                Some(LocalUsageTelemetry {
                    inference_ms: Some(1_000),
                    total_tokens_estimate: Some(120),
                    completion_tokens_estimate: Some(100),
                    ..LocalUsageTelemetry::default()
                }),
                1.0,
                None,
            )
            .await
            .expect("record serving result");
        append_training_observation(
            directory.path().join("training_metrics.json"),
            TrainingRunObservation {
                snapshot_id: "1786023597".to_string(),
                backend: "slurm".to_string(),
                status: "promotion_failed".to_string(),
                failure_reason: "unsupported_serving_architecture".to_string(),
                base_model: "qwen3.5:4b".to_string(),
                slurm_job_id: Some("1786023597".to_string()),
                nodelist: Some("qc[00-05]".to_string()),
                world_size: Some(6),
                samples: 128,
                total_tokens: 4_000,
                non_padding_tokens: 3_200,
                optimizer_steps: 12,
                runtime_seconds: 20.0,
                tokens_per_second: 200.0,
                non_padding_tokens_per_second: 160.0,
                started_ts: Some(1.0),
                finished_ts: Some(2.0),
                cumulative_training: false,
                cpu_fallback: false,
                pin_memory: false,
                quantisation_backend: "none".to_string(),
            },
        )
        .await
        .expect("record training observation");
        let rendered = store.prometheus_metrics().await;
        assert!(rendered.contains("gail_provider_candidate_generation_tokens_per_second{"));
        assert!(rendered.contains("gail_training_tokens_per_second{"));
        assert!(rendered.contains("status=\"promotion_failed\""));
    }

    #[test]
    fn training_report_conversion_uses_report_fields_and_rate_fallbacks() {
        let report = serde_json::json!({
            "backend": "slurm_distributed_cpu_lora",
            "base_model": "qwen3.5:4b",
            "metrics": {
                "samples": 7,
                "total_tokens": 900,
                "non_padding_tokens": 600,
                "runtime_seconds": 3.0,
                "total_optimizer_steps": 4
            },
            "distributed": {
                "slurm_job_id": "1786023597",
                "slurm_nodelist": "qc[00-05]",
                "world_size": 6
            },
            "started_ts": 10.0,
            "finished_ts": 13.0
        });
        let observation = TrainingRunObservation::from_report(
            "1786023597",
            &report,
            "historical_completed",
            "fallback-model",
        );

        assert_eq!(observation.backend, "slurm_distributed_cpu_lora");
        assert_eq!(observation.total_tokens, 900);
        assert_eq!(observation.tokens_per_second, 300.0);
        assert_eq!(observation.non_padding_tokens_per_second, 200.0);
        assert_eq!(observation.slurm_job_id.as_deref(), Some("1786023597"));
        assert_eq!(observation.nodelist.as_deref(), Some("qc[00-05]"));
        assert_eq!(observation.world_size, Some(6));
        assert_eq!(observation.started_ts, Some(10.0));
        assert_eq!(observation.finished_ts, Some(13.0));
        assert!(!observation.cumulative_training);
    }

    #[test]
    fn training_report_conversion_does_not_invent_finish_time() {
        let report = serde_json::json!({
            "metrics": { "total_tokens": 10, "runtime_seconds": 1.0 },
            "cumulative_training": true
        });
        let observation = TrainingRunObservation::from_report(
            "snapshot-without-finish",
            &report,
            "historical_completed",
            "fallback-model",
        );

        assert_eq!(observation.finished_ts, None);
        assert!(observation.cumulative_training);
    }

    #[test]
    fn training_report_conversion_supports_legacy_distributed_python_reports() {
        let report = serde_json::json!({
            "backend": "slurm_distributed_peft",
            "base_model": "qwen3.5:4b",
            "distributed": {
                "world_size": 2,
                "slurm_job_id": "326",
                "total_samples": 204
            },
            "evaluation": { "tokens": 11826 },
            "rank_reports": [
                { "samples": 102, "runtime_seconds": 200.0 },
                { "samples": 102, "runtime_seconds": 210.0 }
            ],
            "training_runtime_seconds": 4083.0
        });
        let observation = TrainingRunObservation::from_report(
            "1787592446830",
            &report,
            "historical_completed",
            "fallback-model",
        );

        assert_eq!(observation.samples, 204);
        assert_eq!(observation.total_tokens, 11826);
        assert_eq!(observation.runtime_seconds, 4083.0);
        assert_eq!(observation.tokens_per_second, 11826.0 / 4083.0);
        assert_eq!(observation.slurm_job_id.as_deref(), Some("326"));
        assert_eq!(observation.world_size, Some(2));
    }

    #[tokio::test]
    async fn training_backfill_is_idempotent_and_replaces_stale_observations() {
        let directory = tempfile::tempdir().expect("metrics directory");
        let snapshot_root = directory.path().join("training/snapshots");
        let snapshot = snapshot_root.join("snapshot-a");
        fs::create_dir_all(&snapshot)
            .await
            .expect("snapshot directory");
        fs::write(
            snapshot.join("training_report.json"),
            serde_json::to_string(&serde_json::json!({
                "backend": "slurm",
                "metrics": { "total_tokens": 100, "runtime_seconds": 2.0 }
            }))
            .expect("report JSON"),
        )
        .await
        .expect("training report");
        fs::write(
            snapshot.join("pipeline.json"),
            serde_json::to_string(&serde_json::json!({
                "started_ts": 100.0,
                "finished_ts": 102.5,
                "cumulative_training": true
            }))
            .expect("pipeline JSON"),
        )
        .await
        .expect("pipeline provenance");

        let metrics_path = directory.path().join("training_metrics.json");
        let first = backfill_training_observations(&metrics_path, &snapshot_root, "fallback-model")
            .await
            .expect("first backfill");
        let second =
            backfill_training_observations(&metrics_path, &snapshot_root, "fallback-model")
                .await
                .expect("second backfill");
        assert_eq!(first, 1);
        assert_eq!(second, 1);
        let persisted = read_training_metrics(&metrics_path).await;
        assert_eq!(persisted.runs[0].started_ts, Some(100.0));
        assert_eq!(persisted.runs[0].finished_ts, Some(102.5));
        assert!(persisted.runs[0].cumulative_training);

        let mut replacement = TrainingRunObservation::default();
        replacement.snapshot_id = "snapshot-a".to_string();
        replacement.status = "promotion_failed".to_string();
        replacement.total_tokens = 123;
        upsert_training_observation(&metrics_path, replacement)
            .await
            .expect("replace observation");
        let persisted = read_training_metrics(&metrics_path).await;
        assert_eq!(persisted.runs.len(), 1);
        assert_eq!(persisted.runs[0].status, "promotion_failed");
        assert_eq!(persisted.runs[0].total_tokens, 123);
    }

    #[tokio::test]
    async fn ai_response_time_metrics_track_modalities_and_estimates() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        store
            .record_ai_response_time("llm", 100, true)
            .await
            .expect("record llm");
        store
            .record_ai_response_time("snn", 300, true)
            .await
            .expect("record snn");

        assert_eq!(store.ai_response_time_estimate_ms("llm").await, Some(100));
        assert_eq!(store.ai_response_time_estimate_ms("snn").await, Some(300));
        assert_eq!(store.ai_response_time_estimate_ms("all").await, Some(200));
        let summary = store.ai_response_time_summary().await;
        assert_eq!(summary["all"].requests, 2);
        assert_eq!(summary["all"].average_latency_ms, Some(200.0));
        assert!(
            store
                .prometheus_metrics()
                .await
                .contains("gail_ai_response_time_average_ms")
        );
    }

    #[tokio::test]
    async fn request_flow_metrics_track_received_in_progress_and_replied() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");

        store.record_request_received().await.expect("received");
        store.record_request_received().await.expect("received");
        let during = store.prometheus_metrics().await;
        assert!(during.contains("gail_requests_received_total 2"));
        assert!(during.contains("gail_requests_in_progress 2"));
        assert!(during.contains("gail_requests_replied_total 0"));
        assert!(during.contains("gail_requests_terminal_total 0"));
        assert!(during.contains("gail_requests_unaccounted_total 2"));

        store.record_request_replied().await.expect("replied");
        let after = store.prometheus_metrics().await;
        assert!(after.contains("gail_requests_received_total 2"));
        assert!(after.contains("gail_requests_in_progress 1"));
        assert!(after.contains("gail_requests_replied_total 1"));
        assert!(after.contains("gail_requests_terminal_total 1"));
        assert!(after.contains("gail_requests_unaccounted_total 1"));
    }

    #[tokio::test]
    async fn recent_usage_penalty_decays_with_age() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let summary = summary("ollama", "llama3.2");
        store
            .record_result(
                &summary,
                "project_solver",
                "planner",
                true,
                Some(42),
                None,
                1.0,
                None,
            )
            .await
            .expect("record result");
        let penalty = store
            .recent_usage_penalty(&summary.candidate_id, "project_solver", "planner", 600.0)
            .await;
        assert!(penalty > 0.0);
    }

    #[tokio::test]
    async fn score_bonus_penalizes_queue_and_inference_pressure() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let fast = summary("ollama", "qwen2.5-coder:1.5b");
        let slow = summary("ollama", "qwen2.5-coder:0.5b");

        store
            .record_result(
                &fast,
                "project_solver",
                "reviewer",
                true,
                Some(120),
                Some(LocalUsageTelemetry {
                    prompt_tokens_estimate: None,
                    queue_wait_ms: Some(5),
                    inference_ms: Some(110),
                    total_tokens_estimate: Some(900),
                    completion_tokens_estimate: Some(800),
                }),
                1.0,
                None,
            )
            .await
            .expect("record fast result");
        store
            .record_result(
                &slow,
                "project_solver",
                "reviewer",
                true,
                Some(1800),
                Some(LocalUsageTelemetry {
                    prompt_tokens_estimate: None,
                    queue_wait_ms: Some(850),
                    inference_ms: Some(3900),
                    total_tokens_estimate: Some(3400),
                    completion_tokens_estimate: Some(2500),
                }),
                1.0,
                None,
            )
            .await
            .expect("record slow result");

        let fast_score = store
            .score_bonus(&fast.candidate_id, "project_solver", "reviewer")
            .await;
        let slow_score = store
            .score_bonus(&slow.candidate_id, "project_solver", "reviewer")
            .await;
        assert!(fast_score > slow_score);
    }

    #[tokio::test]
    async fn candidate_capacity_estimate_uses_successful_throughput_and_global_fallback() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let candidate = summary("ollama", "qwen3.5:9b");
        for _ in 0..5 {
            store
                .record_result(
                    &candidate,
                    "project_solver",
                    "planner",
                    true,
                    Some(1_000),
                    Some(LocalUsageTelemetry {
                        queue_wait_ms: Some(200),
                        inference_ms: Some(800),
                        completion_tokens_estimate: Some(100),
                        ..LocalUsageTelemetry::default()
                    }),
                    1.0,
                    None,
                )
                .await
                .expect("record result");
        }

        let estimate = store
            .candidate_capacity_estimate_for_context(
                &candidate.candidate_id,
                "different-client",
                "new-profile",
                "project_solver",
                "planner",
                None,
                200,
            )
            .await;
        assert_eq!(estimate.samples, 5);
        assert_eq!(estimate.completion_tokens, Some(100.0));
        assert_eq!(estimate.generation_tokens_per_second, Some(125.0));
        assert_eq!(estimate.queue_wait_ms, Some(200.0));
        assert!(estimate.useful_rate > 0.8);
        assert_eq!(estimate.service_time_ms, Some(1_600.0));
    }

    #[tokio::test]
    async fn candidate_capacity_uses_global_performance_for_failure_only_profile() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let candidate = summary("openai", "qwen3.5:9b");

        // The endpoint has a measured successful history, but this newly
        // introduced request profile has only observed failures so far.
        store
            .record_result(
                &candidate,
                "project_solver",
                "planner",
                true,
                Some(1_200),
                Some(LocalUsageTelemetry {
                    queue_wait_ms: Some(200),
                    inference_ms: Some(1_000),
                    completion_tokens_estimate: Some(100),
                    ..LocalUsageTelemetry::default()
                }),
                1.0,
                None,
            )
            .await
            .expect("record global success");
        store
            .record_result_with_context(
                &candidate,
                "refiner",
                "code",
                "project_solver",
                "general",
                None,
                false,
                Some(30_000),
                None,
                -1.0,
                Some("profile-only failure"),
            )
            .await
            .expect("record profile failure");

        let estimate = store
            .candidate_capacity_estimate_for_context(
                &candidate.candidate_id,
                "refiner",
                "code",
                "project_solver",
                "general",
                None,
                200,
            )
            .await;

        assert_eq!(estimate.samples, 1);
        assert_eq!(estimate.generation_tokens_per_second, Some(100.0));
        assert_eq!(estimate.queue_wait_ms, Some(200.0));
        assert_eq!(estimate.service_time_ms, Some(2_000.0));
        assert!(estimate.useful_rate < 0.1 + f64::EPSILON);
    }

    #[tokio::test]
    async fn successful_and_failed_latency_are_kept_separate() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let candidate = summary("ollama", "qwen3.5:9b");
        store
            .record_result(
                &candidate,
                "project_solver",
                "planner",
                false,
                Some(9_000),
                Some(LocalUsageTelemetry {
                    queue_wait_ms: Some(8_000),
                    ..LocalUsageTelemetry::default()
                }),
                -1.0,
                Some("model not found"),
            )
            .await
            .expect("record failure");
        store
            .record_result(
                &candidate,
                "project_solver",
                "planner",
                true,
                Some(200),
                Some(LocalUsageTelemetry {
                    queue_wait_ms: Some(20),
                    ..LocalUsageTelemetry::default()
                }),
                1.0,
                None,
            )
            .await
            .expect("record success");

        let candidate_metrics = &store.summary(10).await.candidates[0];
        assert_eq!(candidate_metrics.successful_latency_average_ms, Some(200.0));
        assert_eq!(candidate_metrics.successful_latency_samples, 1);
        assert_eq!(candidate_metrics.failed_latency_average_ms, Some(9_000.0));
        assert_eq!(candidate_metrics.failed_latency_samples, 1);
        assert_eq!(candidate_metrics.queue_wait_average_ms, Some(4_010.0));
        assert_eq!(candidate_metrics.successes, 1);
        assert_eq!(candidate_metrics.failures, 1);
    }

    #[test]
    fn legacy_split_latency_without_sample_counts_is_discarded() {
        let mut stats = StatsBucket {
            successful_latency_total_ms: 900,
            successful_latency_average_ms: Some(225.0),
            failed_latency_total_ms: 400,
            failed_latency_average_ms: Some(200.0),
            ..StatsBucket::default()
        };
        stats.normalize_split_latency_fields();
        assert_eq!(stats.successful_latency_total_ms, 0);
        assert_eq!(stats.successful_latency_average_ms, None);
        assert_eq!(stats.failed_latency_total_ms, 0);
        assert_eq!(stats.failed_latency_average_ms, None);
    }

    #[tokio::test]
    async fn successful_health_probe_decays_stale_model_not_found_failures() {
        let path = tempfile::NamedTempFile::new()
            .expect("temp file")
            .into_temp_path();
        let store = MetricsStore::new(path.to_path_buf()).await.expect("store");
        let candidate = summary("ollama", "qwen3.5:9b");
        store
            .record_result(
                &candidate,
                "project_solver",
                "planner",
                false,
                Some(100),
                None,
                -1.0,
                Some("ollama upstream error: model 'qwen3.5:9b' not found"),
            )
            .await
            .expect("record failure");
        store
            .record_health(
                &candidate,
                HealthBucket {
                    ok: Some(true),
                    mode: Some("runtime_completion".to_string()),
                    ..HealthBucket::default()
                },
            )
            .await
            .expect("record recovery");

        let candidate_metrics = &store.summary(10).await.candidates[0];
        assert_eq!(candidate_metrics.failures, 0);
        assert_eq!(candidate_metrics.stale_failures, 1);
        assert_eq!(candidate_metrics.last_error, None);
    }

    #[tokio::test]
    async fn current_slurm_heartbeat_keeps_long_training_progress_visible() {
        let directory = tempfile::tempdir().expect("metrics directory");
        let snapshot_root = directory.path().join("training/snapshots");
        let snapshot = snapshot_root.join("snapshot-running");
        let results = directory.path().join("training/slurm/spool/results");
        fs::create_dir_all(&snapshot)
            .await
            .expect("snapshot directory");
        fs::create_dir_all(&results)
            .await
            .expect("results directory");
        fs::write(
            snapshot.join("progress.json"),
            serde_json::to_string(&serde_json::json!({
                "snapshot_id": "snapshot-running",
                "status": "running",
                "backend": "slurm",
                "slurm_job_id": "305",
                "completed_steps": 10,
                "total_steps": 26,
                "progress_ratio": 10.0 / 26.0,
                "progress_per_hour": 5.0,
                "eta_seconds": 11_000.0,
                "elapsed_seconds": 7_000.0,
                "started_ts": now_ts() - 7_000.0,
                "updated_ts": now_ts() - 3_600.0
            }))
            .expect("progress JSON"),
        )
        .await
        .expect("progress file");
        fs::write(
            results.join("snapshot-running.status"),
            serde_json::to_string(&serde_json::json!({
                "state": "running",
                "slurm_job_id": "305",
                "heartbeat_ts": now_ts()
            }))
            .expect("status JSON"),
        )
        .await
        .expect("status file");

        let observations = discover_training_progress(snapshot_root).await;

        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].slurm_job_id.as_deref(), Some("305"));
        assert!(observations[0].updated_ts.expect("heartbeat") > now_ts() - 60.0);
    }
}
