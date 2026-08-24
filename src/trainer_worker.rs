use std::{
    collections::HashSet,
    env,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

use crate::{
    config::{GailConfig, TrainerConfig, TrainerServingTarget},
    errors::{GailError, Result},
    hardware::{HardwareProfile, detect_hardware, log_hardware_profile},
    llm_ledger,
    metrics::{
        MetricsStore, TrainingRunObservation, backfill_training_observations,
        classify_training_failure, upsert_training_observation,
    },
};

pub async fn run(config: GailConfig) -> Result<()> {
    let Some(dsn) = config.storage.postgres_dsn.clone() else {
        return Err(GailError::invalid_config(
            "trainer worker requires storage.postgres_dsn (or GAIL_POSTGRES_DSN)",
        ));
    };
    llm_ledger::initialize_schema(&dsn).await.map_err(|error| {
        GailError::invalid_config(format!("failed to initialise LLM ledger schema: {error}"))
    })?;
    let _training_worker_lock = llm_ledger::acquire_training_worker_lock(&dsn)
        .await
        .map_err(|error| {
            GailError::invalid_config(format!("failed to acquire trainer lock: {error}"))
        })?
        .ok_or_else(|| {
            GailError::invalid_config("another Gail trainer worker is already running")
        })?;
    let trainer = config.trainer.clone();
    match llm_ledger::recover_incomplete_training_registrations(&dsn, trainer.recovery_batch_size)
        .await
    {
        Ok(recovered) if recovered > 0 => tracing::warn!(
            recovered,
            "requeued rows that were previously finalized before model registration completed"
        ),
        Ok(_) => {}
        Err(error) => tracing::warn!(
            error = %error,
            "failed to reconcile incomplete trainer registrations"
        ),
    }
    if trainer.recover_infrastructure_failures {
        match llm_ledger::recover_training_infrastructure_failures(
            &dsn,
            trainer.recovery_batch_size,
        )
        .await
        {
            Ok(recovered) => tracing::info!(
                recovered,
                recovery_batch_size = trainer.recovery_batch_size,
                "requeued terminal training rows caused by missing trainer infrastructure"
            ),
            Err(error) => tracing::warn!(
                error = %error,
                "failed to recover terminal training infrastructure rows"
            ),
        }
    }
    let hardware = detect_hardware().await;
    log_hardware_profile("trainer_worker", &hardware);
    tracing::info!(
        poll_interval_seconds = trainer.poll_interval_seconds,
        min_samples = trainer.min_samples,
        max_samples_per_snapshot = trainer.max_samples_per_snapshot,
        include_degraded = trainer.include_degraded,
        algorithm = %trainer.algorithm,
        output_root = %trainer.output_root,
        register_with_ollama = trainer.register_with_ollama,
        "Gail trainer worker started"
    );
    let training_metrics_path = training_metrics_path(&trainer);
    match backfill_training_observations(
        training_metrics_path.clone(),
        PathBuf::from(&trainer.output_root).join("snapshots"),
        trainer.ollama_base_model.as_str(),
    )
    .await
    {
        Ok(backfilled) if backfilled > 0 => tracing::info!(
            backfilled,
            path = %training_metrics_path.display(),
            "backfilled historical training telemetry"
        ),
        Ok(_) => {}
        Err(error) => tracing::warn!(
            error = %error,
            path = %training_metrics_path.display(),
            "failed to backfill historical training telemetry"
        ),
    }
    if !trainer.serving_targets.is_empty()
        && let Err(error) =
            ensure_active_serving_target(&trainer, config.storage.metrics_path.as_str()).await
    {
        tracing::warn!(error = %error, "failed to reconcile the active trained-model serving target");
    }
    let poll_interval = Duration::from_secs(trainer.poll_interval_seconds);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("trainer worker received shutdown signal");
                break;
            }
        _ = tokio::time::sleep(poll_interval) => {}
        }
        if let Some(spool) = env_string("GAIL_TRAIN_SLURM_SPOOL") {
            if let Err(error) =
                requeue_stale_slurm_requests(&dsn, Path::new(&spool), &trainer).await
            {
                tracing::warn!(error = %error, "failed to reconcile stale Slurm training requests");
            }
        }
        match active_training_snapshot(&trainer, &dsn).await {
            Ok(true) => {
                tracing::info!(
                    "trainer worker is waiting for the unresolved snapshot lifecycle to finish"
                );
                continue;
            }
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(error = %error, "trainer worker could not inspect active snapshot lifecycle");
                continue;
            }
        }
        let mut entries = match llm_ledger::fetch_pending_training(
            &dsn,
            trainer.max_samples_per_snapshot,
            trainer.include_degraded,
        )
        .await
        {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(error = %error, "trainer worker failed to fetch pending ledger rows");
                continue;
            }
        };
        if entries.len() < trainer.min_samples {
            continue;
        }
        entries.truncate(trainer.max_samples_per_snapshot);
        let snapshot_id = snapshot_id();
        let snapshot_root = PathBuf::from(trainer.output_root.clone());
        let dataset_path = snapshot_root
            .join("datasets")
            .join(format!("{snapshot_id}.jsonl"));
        let snapshot_dir = snapshot_root.join("snapshots").join(snapshot_id.as_str());
        if let Err(error) = write_dataset(entries.as_slice(), dataset_path.as_path()).await {
            tracing::warn!(error = %error, path = %dataset_path.display(), "trainer worker failed to build dataset snapshot");
            let error_text = error.to_string();
            if let Err(metrics_error) = record_training_observation(
                &trainer,
                &snapshot_id,
                snapshot_dir.as_path(),
                None,
                Some(error_text.as_str()),
            )
            .await
            {
                tracing::warn!(
                    error = %metrics_error,
                    snapshot = %snapshot_id,
                    "failed to persist dataset failure telemetry"
                );
            }
            for entry in entries {
                let _ = llm_ledger::mark_training_retry(
                    &dsn,
                    entry.id,
                    format!("dataset_write_failed: {error_text}").as_str(),
                    trainer.max_attempts,
                    trainer.retry_backoff_seconds,
                )
                .await;
            }
            continue;
        }
        let ids = entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
        if let Err(error) =
            write_active_training_marker(&trainer, &snapshot_id, &ids, "submitted").await
        {
            tracing::warn!(error = %error, snapshot = %snapshot_id, "failed to persist active training lifecycle marker");
            continue;
        }
        let train_outcome = run_training_pipeline(
            &trainer,
            &hardware,
            &snapshot_id,
            dataset_path.as_path(),
            snapshot_dir.as_path(),
            ids.as_slice(),
            config.storage.metrics_path.as_str(),
            dsn.as_str(),
        )
        .await;
        if let Err(error) = remove_active_training_marker(&trainer).await {
            tracing::warn!(error = %error, snapshot = %snapshot_id, "failed to clear active training lifecycle marker");
        }
        let training_error = train_outcome.as_ref().err().map(ToString::to_string);
        if let Err(error) = record_training_observation(
            &trainer,
            &snapshot_id,
            snapshot_dir.as_path(),
            train_outcome
                .as_ref()
                .ok()
                .map(|outcome| outcome.status.as_str()),
            training_error.as_deref(),
        )
        .await
        {
            tracing::warn!(error = %error, snapshot = %snapshot_id, "failed to persist training metrics");
        }
        match train_outcome {
            Ok(outcome) => {
                if let Err(error) = llm_ledger::mark_training_success(
                    &dsn,
                    ids.as_slice(),
                    outcome.snapshot_tag.as_str(),
                    outcome.status.as_str(),
                )
                .await
                {
                    tracing::warn!(
                        error = %error,
                        snapshot = %outcome.snapshot_tag,
                        "trainer worker failed to mark ledger rows as trained"
                    );
                }
            }
            Err(error) => {
                let error_text = error.to_string();
                tracing::warn!(error = %error_text, "trainer worker snapshot failed");
                // An incompatible serving artifact is deterministic. Retrying
                // it burns the same training work every poll and obscures the
                // actual remediation (a real model export or GGUF converter).
                let max_attempts = if is_non_retryable_training_error(&error_text) {
                    1
                } else {
                    trainer.max_attempts
                };
                for id in ids {
                    let _ = llm_ledger::mark_training_retry(
                        &dsn,
                        id,
                        error_text.as_str(),
                        max_attempts,
                        trainer.retry_backoff_seconds,
                    )
                    .await;
                }
            }
        }
    }
    Ok(())
}

fn training_metrics_path(trainer: &TrainerConfig) -> PathBuf {
    PathBuf::from(&trainer.output_root)
        .parent()
        .map(|path| path.join("training_metrics.json"))
        .unwrap_or_else(|| PathBuf::from("training_metrics.json"))
}

fn active_training_marker_path(trainer: &TrainerConfig) -> PathBuf {
    PathBuf::from(&trainer.output_root).join("active_training.json")
}

async fn write_active_training_marker(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    ledger_ids: &[i64],
    state: &str,
) -> Result<()> {
    write_json(
        &active_training_marker_path(trainer),
        &json!({
            "version": 1,
            "snapshot_id": snapshot_id,
            "ledger_ids": ledger_ids,
            "state": state,
            "started_ts": now_ts(),
            "heartbeat_ts": now_ts(),
        }),
    )
    .await
}

/// Keep the lifecycle marker alive while a Slurm request is being serviced.
///
/// The trainer worker uses this marker to prevent a second snapshot from
/// being created for the same pending ledger rows.  A Slurm training step can
/// legitimately take longer than the worker's stale-marker timeout, so the
/// marker heartbeat must follow the request heartbeat rather than only being
/// written at submission time.
async fn refresh_active_training_marker(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    ledger_ids: &[i64],
) -> Result<()> {
    let path = active_training_marker_path(trainer);
    let started_ts = fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .filter(|value| value.get("snapshot_id").and_then(Value::as_str) == Some(snapshot_id))
        .and_then(|value| value.get("started_ts").and_then(Value::as_f64))
        .unwrap_or_else(now_ts);
    write_json(
        &path,
        &json!({
            "version": 1,
            "snapshot_id": snapshot_id,
            "ledger_ids": ledger_ids,
            "state": "submitted",
            "started_ts": started_ts,
            "heartbeat_ts": now_ts(),
        }),
    )
    .await
}

async fn remove_active_training_marker(trainer: &TrainerConfig) -> Result<()> {
    match fs::remove_file(active_training_marker_path(trainer)).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn terminal_slurm_result_exit_code(raw: &str) -> Option<i32> {
    serde_json::from_str::<Value>(raw)
        .ok()?
        .get("exit_code")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
}

/// Release an active marker when the dispatcher has already written a
/// terminal result. This can happen when the trainer pod is restarted after
/// Slurm finished but before the worker consumed the result. The result is
/// intentionally retained for auditability; snapshot ids are unique, so it
/// cannot be mistaken for a newly submitted request.
async fn recover_terminal_slurm_result(
    trainer: &TrainerConfig,
    dsn: &str,
    spool: &Path,
    snapshot_id: &str,
    ledger_ids: &[i64],
) -> Result<bool> {
    let result_path = spool.join("results").join(format!("{snapshot_id}.result"));
    let raw = match fs::read_to_string(&result_path).await {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let exit_code = terminal_slurm_result_exit_code(&raw);
    let reason = match exit_code {
        Some(0) => {
            "terminal Slurm result was written before trainer restart; snapshot will be retried"
        }
        Some(code) => {
            tracing::warn!(
                snapshot = snapshot_id,
                exit_code = code,
                "recovering terminal failed Slurm snapshot"
            );
            "terminal Slurm result reported failure; snapshot will be retried"
        }
        None => "terminal Slurm result was invalid; snapshot will be retried",
    };
    for id in ledger_ids {
        let _ = llm_ledger::mark_training_retry(
            dsn,
            *id,
            reason,
            trainer.max_attempts,
            trainer.retry_backoff_seconds,
        )
        .await;
    }
    remove_active_training_marker(trainer).await?;
    tracing::warn!(
        snapshot = snapshot_id,
        result = %result_path.display(),
        "released active training lifecycle after terminal Slurm result"
    );
    Ok(true)
}

/// Return true while a previous snapshot is still unresolved.  A stale
/// marker is recovered explicitly so a worker restart cannot create a second
/// snapshot for the same ledger rows or leave a dead queue request forever.
async fn active_training_snapshot(trainer: &TrainerConfig, dsn: &str) -> Result<bool> {
    let path = active_training_marker_path(trainer);
    let raw = match fs::read_to_string(&path).await {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let value: Value = serde_json::from_str(&raw).map_err(|error| {
        GailError::invalid_config(format!("invalid active training marker: {error}"))
    })?;
    let heartbeat = value
        .get("heartbeat_ts")
        .and_then(Value::as_f64)
        .or_else(|| value.get("started_ts").and_then(Value::as_f64))
        .unwrap_or(0.0);
    let snapshot_id = value
        .get("snapshot_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let ids = value
        .get("ledger_ids")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_i64).collect::<Vec<_>>())
        .unwrap_or_default();
    if let Some(spool) = env_string("GAIL_TRAIN_SLURM_SPOOL") {
        if recover_terminal_slurm_result(trainer, dsn, Path::new(&spool), snapshot_id, &ids).await?
        {
            return Ok(false);
        }

        // A request can legitimately wait behind another Slurm allocation for
        // longer than command_timeout_seconds.  The dispatcher owns queue
        // ordering and will eventually submit it, so the trainer must not
        // delete a still-present request merely because its marker heartbeat
        // was not refreshed during a worker restart.
        let spool = Path::new(&spool);
        let queue_request = spool.join("queue").join(format!("{snapshot_id}.request"));
        let running_request = spool.join("running").join(format!("{snapshot_id}.request"));
        if queue_request.exists() || running_request.exists() {
            return Ok(true);
        }
    }
    let stale_after = trainer.command_timeout_seconds.max(60) as f64;
    if heartbeat > 0.0 && now_ts() - heartbeat <= stale_after {
        return Ok(true);
    }
    if let Some(spool) = env_string("GAIL_TRAIN_SLURM_SPOOL") {
        let request = Path::new(&spool)
            .join("queue")
            .join(format!("{snapshot_id}.request"));
        if request.exists() {
            fs::remove_file(&request).await.map_err(|error| {
                GailError::invalid_config(format!("failed to cancel stale Slurm request: {error}"))
            })?;
        }
    }
    for id in ids {
        let _ = llm_ledger::mark_training_retry(
            dsn,
            id,
            "stale unresolved training lifecycle; request cancelled and requeued",
            trainer.max_attempts,
            trainer.retry_backoff_seconds,
        )
        .await;
    }
    remove_active_training_marker(trainer).await?;
    tracing::warn!(
        snapshot = snapshot_id,
        "recovered stale unresolved training lifecycle"
    );
    Ok(false)
}

async fn requeue_stale_slurm_requests(
    dsn: &str,
    spool: &Path,
    trainer: &TrainerConfig,
) -> Result<()> {
    let active_snapshot = fs::read_to_string(active_training_marker_path(trainer))
        .await
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("snapshot_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        });
    let queue = spool.join("queue");
    let mut entries = match fs::read_dir(&queue).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("request") {
            continue;
        }
        let raw = match fs::read_to_string(&path).await {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        let value: Value = match serde_json::from_str(&raw) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let requested_at = value
            .get("requested_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if requested_at <= 0.0
            || now_ts() - requested_at <= trainer.command_timeout_seconds.max(60) as f64
        {
            continue;
        }
        let snapshot_id = value
            .get("snapshot_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        // The active trainer lifecycle may be older than the queue timeout
        // while Slurm is busy with an earlier request.  Keep its request for
        // the dispatcher; active_training_snapshot will release it only when
        // the request is genuinely gone.
        if active_snapshot.as_deref() == Some(snapshot_id) {
            continue;
        }
        if let Some(job_id) = value.get("slurm_job_id").and_then(Value::as_str) {
            cancel_slurm_job(job_id).await;
        }
        let ids = value
            .get("ledger_ids")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_i64).collect::<Vec<_>>())
            .unwrap_or_default();
        fs::remove_file(&path).await.map_err(|error| {
            GailError::invalid_config(format!("failed to remove stale Slurm request: {error}"))
        })?;
        let cancelled = spool
            .join("results")
            .join(format!("{snapshot_id}.cancelled.json"));
        write_json(
            &cancelled,
            &json!({"snapshot_id": snapshot_id, "state": "cancelled", "reason": "stale queue request", "cancelled_at": now_ts()}),
        )
        .await?;
        for id in ids {
            let _ = llm_ledger::mark_training_retry(
                dsn,
                id,
                "stale Slurm queue request cancelled and requeued",
                trainer.max_attempts,
                trainer.retry_backoff_seconds,
            )
            .await;
        }
        tracing::warn!(
            snapshot = snapshot_id,
            "cancelled and requeued stale Slurm training request"
        );
    }
    Ok(())
}

async fn cancel_slurm_job(job_id: &str) {
    if job_id.trim().is_empty() {
        return;
    }
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        Command::new("scancel").arg(job_id).status(),
    )
    .await;
    match result {
        Ok(Ok(status)) if status.success() => tracing::info!(job_id, "cancelled stale Slurm job"),
        Ok(Ok(status)) => tracing::warn!(job_id, code = ?status.code(), "scancel returned failure"),
        Ok(Err(error)) => {
            tracing::debug!(job_id, error = %error, "scancel unavailable while cancelling stale job")
        }
        Err(_) => tracing::warn!(job_id, "timed out cancelling stale Slurm job"),
    }
}

async fn record_training_observation(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    snapshot_dir: &Path,
    outcome_status: Option<&str>,
    training_error: Option<&str>,
) -> Result<()> {
    let report_path = snapshot_dir.join("training_report.json");
    let report = match fs::read_to_string(&report_path).await {
        Ok(raw) => Some(serde_json::from_str::<Value>(&raw).map_err(|error| {
            GailError::invalid_config(format!("invalid training report: {error}"))
        })?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut report = report;
    // The command writes training metrics while Gail writes lifecycle timing
    // to pipeline.json. Merge the latter into the former so provenance is
    // stable across backfill/restart and is not replaced with backfill time.
    let pipeline_path = snapshot_dir.join("pipeline.json");
    if let Ok(raw) = fs::read_to_string(&pipeline_path).await
        && let Ok(pipeline) = serde_json::from_str::<Value>(&raw)
    {
        if let (Some(metrics), Some(pipeline_object)) = (report.as_mut(), pipeline.as_object())
            && let Some(metrics_object) = metrics.as_object_mut()
        {
            for key in ["started_ts", "finished_ts", "cumulative_training"] {
                if let Some(value) = pipeline_object.get(key) {
                    metrics_object.insert(key.to_string(), value.clone());
                }
            }
            if let Some(job_id) = pipeline_object
                .get("slurm_job_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
            {
                let distributed = metrics_object
                    .get_mut("distributed")
                    .and_then(Value::as_object_mut)
                    .map(|_| ())
                    .is_some();
                if !distributed {
                    metrics_object.insert("distributed".to_string(), json!({}));
                }
                if let Some(distributed) = metrics_object
                    .get_mut("distributed")
                    .and_then(Value::as_object_mut)
                {
                    distributed.insert("slurm_job_id".to_string(), json!(job_id));
                }
            }
        }
    }
    let status = outcome_status.unwrap_or_else(|| {
        if training_error.is_some() {
            if report.is_some() {
                "promotion_failed"
            } else {
                "failed"
            }
        } else {
            "completed"
        }
    });
    let mut observation = report
        .as_ref()
        .map(|report| {
            TrainingRunObservation::from_report(
                snapshot_id,
                report,
                status,
                trainer.ollama_base_model.as_str(),
            )
        })
        .unwrap_or_else(|| TrainingRunObservation {
            snapshot_id: snapshot_id.to_string(),
            backend: if env_string("GAIL_TRAIN_SLURM_SPOOL").is_some() {
                "slurm"
            } else {
                "local"
            }
            .to_string(),
            status: status.to_string(),
            failure_reason: training_error
                .map(classify_training_failure)
                .unwrap_or_default()
                .to_string(),
            base_model: trainer.ollama_base_model.clone(),
            finished_ts: Some(now_ts()),
            ..TrainingRunObservation::default()
        });
    if observation.failure_reason.is_empty() {
        if let Some(error) = training_error {
            observation.failure_reason = classify_training_failure(error).to_string();
        }
    }
    upsert_training_observation(training_metrics_path(trainer), observation).await
}

fn is_non_retryable_training_error(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("unsupported architecture")
        || error.contains("cannot be registered as safetensors")
        || error.contains("synthetic bootstrap")
        || error.contains("not a production")
}

struct TrainingOutcome {
    snapshot_tag: String,
    status: String,
}

/// End-to-end trainer execution plan shared with the child training process.
///
/// The worker derives this plan from runtime hardware and exposes it as
/// environment variables and a JSON artifact (`training_execution_plan.json`).
#[derive(Debug, Clone, Serialize)]
struct TrainingExecutionPlan {
    profile: String,
    backend: String,
    device: String,
    device_index: Option<usize>,
    gpu_count: usize,
    gpu_memory_mb: u64,
    gpu_free_memory_mb: u64,
    cpu_threads_available: usize,
    cpu_intraop_threads: usize,
    cpu_interop_threads: usize,
    tokenizer_threads: usize,
    async_worker_threads: usize,
    prefetch_batches: usize,
    compute_dtype: String,
    quantisation_backend: String,
    dynamic_padding: bool,
    sequence_packing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TrainingArtifactMode {
    Production,
    DevelopmentFixture,
}

fn training_artifact_mode() -> TrainingArtifactMode {
    let configured = env::var("GAIL_TRAIN_ARTIFACT_MODE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "production".to_string());
    if configured == "development_fixture" || env_bool("GAIL_TRAIN_ALLOW_SYNTHETIC_MODEL", false) {
        return TrainingArtifactMode::DevelopmentFixture;
    }
    TrainingArtifactMode::Production
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn build_training_execution_plan(
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
) -> TrainingExecutionPlan {
    let gpu_count = hardware.gpu_count();
    let use_gpu = gpu_count > 0;
    let cpu_threads_available = hardware.preferred_worker_threads().max(1);
    let cpu_intraop_threads = env_usize(
        "GAIL_TRAIN_CPU_INTRAOP_THREADS",
        cpu_threads_available.saturating_sub(2),
        1,
        256,
    );
    let cpu_interop_threads = env_usize(
        "GAIL_TRAIN_CPU_INTEROP_THREADS",
        if cpu_intraop_threads >= 24 { 1 } else { 2 },
        1,
        32,
    );
    let tokenizer_threads = env_usize(
        "GAIL_TRAIN_TOKENIZER_THREADS",
        (cpu_intraop_threads / 3).clamp(2, 16),
        1,
        64,
    );
    let async_worker_threads = env_usize(
        "GAIL_TRAIN_ASYNC_WORKER_THREADS",
        (cpu_threads_available / 12).clamp(2, 4),
        1,
        32,
    );
    let prefetch_batches = env_usize("GAIL_TRAIN_PREFETCH_BATCHES", 2, 1, 32);
    let dynamic_padding = !env_bool("GAIL_TRAIN_DISABLE_DYNAMIC_PADDING", false);
    let sequence_packing = env_bool("GAIL_TRAIN_SEQUENCE_PACKING", true);
    let quantisation_backend = if env_bool("GAIL_TCH_BASE_PREQUANTISED", false) {
        "prequantised_base".to_string()
    } else {
        "none".to_string()
    };
    let compute_dtype = env_string("GAIL_TRAIN_COMPUTE_DTYPE").unwrap_or_else(|| {
        if use_gpu {
            "fp16".to_string()
        } else {
            "fp32".to_string()
        }
    });
    let profile = if hardware.cpu_arch.eq_ignore_ascii_case("aarch64") && use_gpu {
        "centriq_rtx3060_12gb".to_string()
    } else if hardware.cpu_arch.eq_ignore_ascii_case("aarch64") {
        "centriq_cpu_armv8".to_string()
    } else if use_gpu {
        "generic_cuda".to_string()
    } else {
        "generic_cpu".to_string()
    };
    let backend = if trainer.algorithm.eq_ignore_ascii_case("qlora_sft") && use_gpu {
        "cuda_qlora".to_string()
    } else if use_gpu {
        "cuda_lora".to_string()
    } else {
        "cpu_lora".to_string()
    };
    TrainingExecutionPlan {
        profile,
        backend,
        device: if use_gpu {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        },
        device_index: if use_gpu { Some(0) } else { None },
        gpu_count,
        gpu_memory_mb: hardware.total_gpu_memory_mb(),
        gpu_free_memory_mb: hardware.total_gpu_free_memory_mb(),
        cpu_threads_available,
        cpu_intraop_threads,
        cpu_interop_threads,
        tokenizer_threads,
        async_worker_threads,
        prefetch_batches,
        compute_dtype,
        quantisation_backend,
        dynamic_padding,
        sequence_packing,
    }
}

async fn run_training_pipeline(
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
    ledger_ids: &[i64],
    metrics_path: &str,
    postgres_dsn: &str,
) -> Result<TrainingOutcome> {
    fs::create_dir_all(snapshot_dir).await.map_err(|error| {
        GailError::invalid_config(format!("failed to create snapshot output path: {error}"))
    })?;
    let execution_plan = build_training_execution_plan(trainer, hardware);
    let artifact_mode = training_artifact_mode();
    let resume_adapter = active_adapter_path(trainer)?;
    tracing::info!(
        snapshot = snapshot_id,
        requested_algorithm = %trainer.algorithm,
        effective_backend = %execution_plan.backend,
        device = %execution_plan.device,
        quantisation_backend = %execution_plan.quantisation_backend,
        pin_memory = execution_plan.device == "cuda",
        cpu_fallback = execution_plan.device == "cpu"
            && trainer.algorithm.eq_ignore_ascii_case("qlora_sft"),
        "training execution path selected"
    );
    write_json(
        snapshot_dir.join("training_execution_plan.json").as_path(),
        &serde_json::to_value(&execution_plan).unwrap_or(Value::Null),
    )
    .await?;
    let mut pipeline_report = json!({
        "snapshot_id": snapshot_id,
        "algorithm": trainer.algorithm,
        "dataset_path": dataset_path.to_string_lossy().to_string(),
        "snapshot_dir": snapshot_dir.to_string_lossy().to_string(),
        "artifact_mode": artifact_mode,
        "cpu_cores": hardware.cpu_cores,
        "cpu_arch": hardware.cpu_arch,
        "cpu_model": hardware.cpu_model,
        "total_memory_mb": hardware.total_memory_mb,
        "available_memory_mb": hardware.available_memory_mb,
        "gpu_count": hardware.gpu_count(),
        "gpu_memory_mb": hardware.total_gpu_memory_mb(),
        "gpu_free_memory_mb": hardware.total_gpu_free_memory_mb(),
        "execution_plan": execution_plan,
        "started_ts": now_ts(),
        "resume_adapter": resume_adapter.as_ref().map(|path| path.to_string_lossy().to_string()),
        "cumulative_training": resume_adapter.is_some(),
    });
    let training_invocation = resolve_training_invocation(
        trainer,
        hardware,
        snapshot_id,
        dataset_path,
        snapshot_dir,
        resume_adapter.as_deref(),
    )
    .await?;
    let mut training_executed = false;
    if let Some(command_line) = training_invocation {
        let command_output = if let Some(spool) = env_string("GAIL_TRAIN_SLURM_SPOOL") {
            pipeline_report["training_backend"] = json!("slurm");
            execute_slurm_training_request(
                Path::new(&spool),
                trainer,
                snapshot_id,
                dataset_path,
                snapshot_dir,
                ledger_ids,
            )
            .await?
        } else {
            pipeline_report["training_backend"] = json!("local");
            execute_training_command(
                command_line.as_str(),
                trainer,
                hardware,
                &execution_plan,
                snapshot_id,
                dataset_path,
                snapshot_dir,
            )
            .await?
        };
        pipeline_report["training_command"] = if env_string("GAIL_TRAIN_SLURM_SPOOL").is_some() {
            json!("submitted through the Gail Slurm spool")
        } else {
            json!(command_line)
        };
        pipeline_report["training_stdout_tail"] = json!(command_output.stdout);
        pipeline_report["training_stderr_tail"] = json!(command_output.stderr);
        pipeline_report["training_exit_code"] = json!(command_output.exit_code);
        pipeline_report["training_runtime_seconds"] = json!(command_output.runtime_seconds);
        pipeline_report["slurm_job_id"] = json!(command_output.backend_job_id);
        pipeline_report["heartbeat_ts"] = json!(command_output.heartbeat_ts);
        pipeline_report["lifecycle"] = json!(["submitted", "trained"]);
        training_executed = true;
    } else {
        pipeline_report["training_command"] = json!(
            "skipped: trainer command unresolved (unsupported algorithm, command_template unset, or Rust qlora model artifacts missing)"
        );
    }
    let mut snapshot_tag = format!("{}:{}", trainer.model_prefix, snapshot_id);
    let mut registration_succeeded = false;
    let mut registration_error = None;
    if training_executed {
        match qualify_training_snapshot(snapshot_dir).await {
            Ok(qualification) => {
                pipeline_report["evaluation"] = qualification;
                pipeline_report["lifecycle"] =
                    json!(["submitted", "trained", "evaluated", "qualified"]);
            }
            Err(error) => {
                pipeline_report["lifecycle"] = json!(["submitted", "trained", "evaluated"]);
                pipeline_report["qualification_error"] = json!(error.to_string());
                pipeline_report["finished_ts"] = json!(now_ts());
                write_json(
                    snapshot_dir.join("pipeline.json").as_path(),
                    &pipeline_report,
                )
                .await?;
                return Err(GailError::invalid_config(format!(
                    "training snapshot failed qualification; model was not registered or promoted: {error}"
                )));
            }
        }
    }
    let serving_target = if trainer.serving_targets.is_empty() || !training_executed {
        None
    } else {
        Some(select_serving_target(trainer, metrics_path).await?)
    };
    if trainer.register_with_ollama && training_executed {
        let previous_snapshot = active_snapshot_id(trainer)?;
        let previous_pointer =
            fs::read_to_string(PathBuf::from(&trainer.output_root).join("active_snapshot.json"))
                .await
                .ok();
        match register_snapshot_with_ollama(trainer, snapshot_id, snapshot_dir).await {
            Ok(registration_mode) => {
                registration_succeeded = true;
                snapshot_tag = trainer.model_alias.clone();
                pipeline_report["lifecycle"] =
                    json!(["submitted", "trained", "evaluated", "qualified", "promoted"]);
                match registration_mode {
                    OllamaRegistrationMode::Adapter | OllamaRegistrationMode::BaseModel => {
                        pipeline_report["ollama_registration"] = json!("registered");
                    }
                }
                if let Err(error) = publish_active_snapshot(
                    trainer,
                    snapshot_id,
                    snapshot_dir,
                    serving_target.as_ref(),
                )
                .await
                {
                    if let Err(rollback_error) =
                        rollback_ollama_alias(trainer, previous_snapshot.as_deref()).await
                    {
                        tracing::error!(error = %rollback_error, snapshot = snapshot_id, "failed to roll back serving alias after active snapshot publication failure");
                    }
                    registration_succeeded = false;
                    registration_error =
                        Some(format!("active snapshot publication failed: {error}"));
                    pipeline_report["ollama_registration"] = json!("rolled_back");
                    pipeline_report["ollama_registration_error"] = json!(error.to_string());
                } else if let Err(error) = health_check_promoted_model(trainer).await {
                    if let Err(rollback_error) =
                        rollback_ollama_alias(trainer, previous_snapshot.as_deref()).await
                    {
                        tracing::error!(error = %rollback_error, snapshot = snapshot_id, "failed to roll back unhealthy serving alias");
                    }
                    restore_active_snapshot_pointer(trainer, previous_pointer.as_deref()).await;
                    registration_succeeded = false;
                    registration_error =
                        Some(format!("promoted model health check failed: {error}"));
                    pipeline_report["lifecycle"] = json!([
                        "submitted",
                        "trained",
                        "evaluated",
                        "qualified",
                        "promoted",
                        "rolled_back"
                    ]);
                    pipeline_report["health_check_error"] = json!(error.to_string());
                } else if let Some(target) = serving_target.as_ref()
                    && let Err(error) = health_check_serving_target(target).await
                {
                    if let Err(rollback_error) =
                        rollback_ollama_alias(trainer, previous_snapshot.as_deref()).await
                    {
                        tracing::error!(error = %rollback_error, snapshot = snapshot_id, "failed to roll back alias after serving-target readiness failure");
                    }
                    restore_active_snapshot_pointer(trainer, previous_pointer.as_deref()).await;
                    registration_succeeded = false;
                    registration_error =
                        Some(format!("serving target health check failed: {error}"));
                    pipeline_report["lifecycle"] = json!([
                        "submitted",
                        "trained",
                        "evaluated",
                        "qualified",
                        "promoted",
                        "rolled_back"
                    ]);
                    pipeline_report["health_check_error"] = json!(error.to_string());
                } else {
                    pipeline_report["lifecycle"] = json!([
                        "submitted",
                        "trained",
                        "evaluated",
                        "qualified",
                        "promoted",
                        "health_checked"
                    ]);
                }
                if registration_succeeded {
                    match llm_ledger::schedule_validation_now(postgres_dsn).await {
                        Ok(rows) => tracing::info!(
                            snapshot = snapshot_id,
                            rows,
                            "scheduled immediate comparative validation for promoted snapshot"
                        ),
                        Err(error) => tracing::error!(
                            snapshot = snapshot_id,
                            error = %error,
                            "failed to schedule immediate comparative validation for promoted snapshot"
                        ),
                    }
                }
                if registration_succeeded && let Err(error) = rotate_ollama_models(trainer).await {
                    tracing::warn!(
                        error = %error,
                        snapshot = snapshot_id,
                        "trained snapshot registered, but old-model rotation failed"
                    );
                    pipeline_report["ollama_rotation_error"] = json!(error.to_string());
                }
            }
            Err(error) => {
                if let Err(rollback_error) =
                    rollback_ollama_alias(trainer, previous_snapshot.as_deref()).await
                {
                    tracing::error!(
                        error = %rollback_error,
                        snapshot = snapshot_id,
                        "failed to roll back serving alias after snapshot registration failure"
                    );
                }
                tracing::warn!(
                    error = %error,
                    snapshot = snapshot_id,
                    "training completed, but serving-model registration failed"
                );
                pipeline_report["ollama_registration"] = json!("failed_retryable");
                pipeline_report["ollama_registration_error"] = json!(error.to_string());
                registration_error = Some(error.to_string());
            }
        }
    } else if trainer.register_with_ollama {
        pipeline_report["ollama_registration"] =
            json!("skipped: no training command executed for this snapshot");
    }
    pipeline_report["snapshot_tag"] = json!(snapshot_tag);
    pipeline_report["finished_ts"] = json!(now_ts());
    write_json(
        snapshot_dir.join("pipeline.json").as_path(),
        &pipeline_report,
    )
    .await?;
    if let Some(error) = registration_error {
        return Err(GailError::invalid_config(format!(
            "trained snapshot was retained but not promoted: {error}"
        )));
    }
    Ok(TrainingOutcome {
        snapshot_tag,
        status: if pipeline_report
            .get("training_exit_code")
            .and_then(|value| value.as_i64())
            .unwrap_or(-1)
            == 0
        {
            if trainer.register_with_ollama && registration_succeeded {
                "trained".to_string()
            } else {
                "snapshotted".to_string()
            }
        } else {
            "snapshotted".to_string()
        },
    })
}

async fn qualify_training_snapshot(snapshot_dir: &Path) -> Result<Value> {
    let report_path = snapshot_dir.join("training_report.json");
    let report_raw = fs::read_to_string(&report_path).await.map_err(|error| {
        GailError::invalid_config(format!(
            "training result artifact is missing training_report.json: {error}"
        ))
    })?;
    let report: Value = serde_json::from_str(&report_raw).map_err(|error| {
        GailError::invalid_config(format!("training result report is invalid JSON: {error}"))
    })?;
    let has_artifact = [
        "adapter",
        "adapter.gguf",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    .iter()
    .any(|name| std::fs::metadata(snapshot_dir.join(name)).is_ok());
    if !has_artifact {
        return Err(GailError::invalid_config(
            "training completed without a valid model artifact",
        ));
    }
    let evaluation = if let Some(value) = report.get("evaluation") {
        value.clone()
    } else {
        let path = snapshot_dir.join("evaluation.json");
        let raw = fs::read_to_string(&path).await.map_err(|_| {
            GailError::invalid_config(
                "evaluation metrics are missing; provide training_report.json.evaluation or evaluation.json",
            )
        })?;
        serde_json::from_str(&raw).map_err(|error| {
            GailError::invalid_config(format!("evaluation metrics are invalid JSON: {error}"))
        })?
    };
    let metric_name = ["score", "accuracy", "f1", "loss", "perplexity"]
        .iter()
        .find(|name| evaluation.get(*name).and_then(Value::as_f64).is_some())
        .copied()
        .ok_or_else(|| {
            GailError::invalid_config("evaluation metrics contain no finite score/loss metric")
        })?;
    let candidate = evaluation
        .get(metric_name)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| GailError::invalid_config("candidate evaluation metric is not finite"))?;
    let baseline = if let Some(value) = evaluation.get("baseline") {
        value.clone()
    } else if let Some(value) = report.get("baseline") {
        value.clone()
    } else {
        let path = snapshot_dir.join("baseline.json");
        let raw = fs::read_to_string(&path).await.map_err(|_| {
            GailError::invalid_config(
                "baseline comparison is missing; provide evaluation.baseline, baseline, or baseline.json",
            )
        })?;
        serde_json::from_str(&raw).map_err(|error| {
            GailError::invalid_config(format!("baseline metrics are invalid JSON: {error}"))
        })?
    };
    let baseline_value = baseline
        .get(metric_name)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| GailError::invalid_config("baseline comparison metric is not finite"))?;
    let lower_is_better = matches!(metric_name, "loss" | "perplexity");
    let improvement = if lower_is_better {
        baseline_value - candidate
    } else {
        candidate - baseline_value
    };
    let minimum_improvement = env::var("GAIL_TRAIN_MIN_BASELINE_IMPROVEMENT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    if improvement < minimum_improvement {
        return Err(GailError::invalid_config(format!(
            "candidate {metric_name}={candidate:.6} did not beat baseline {baseline_value:.6} by {minimum_improvement:.6}"
        )));
    }
    Ok(json!({
        "metric": metric_name,
        "candidate": candidate,
        "baseline": baseline_value,
        "improvement": improvement,
        "minimum_improvement": minimum_improvement,
    }))
}

async fn health_check_promoted_model(trainer: &TrainerConfig) -> Result<()> {
    let response = ollama_api_post(
        &ollama_api_client(),
        trainer,
        "generate",
        &json!({"model": trainer.model_alias, "prompt": "health check", "stream": false, "options": {"num_predict": 2}}),
    )
    .await?;
    if response
        .get("response")
        .and_then(Value::as_str)
        .is_none_or(|text| text.trim().is_empty())
    {
        return Err(GailError::invalid_config(
            "promoted model returned no health-check response",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ServingTargetSelection {
    target: TrainerServingTarget,
    throughput_tokens_per_second: f64,
}

/// Select only a ready target with the exact training base model. Capacity is
/// the primary ordering; historical successful generation throughput breaks
/// equal-capacity (including CPU-only) ties. The final host-id ordering keeps
/// selection deterministic when no history exists.
async fn select_serving_target(
    trainer: &TrainerConfig,
    metrics_path: &str,
) -> Result<ServingTargetSelection> {
    let metrics = MetricsStore::new(metrics_path).await.ok();
    let summaries = match metrics.as_ref() {
        Some(store) => store.summary(1024).await.candidates,
        None => Vec::new(),
    };
    let mut ready = Vec::new();
    for target in trainer
        .serving_targets
        .iter()
        .filter(|target| target.enabled && target.base_model == trainer.ollama_base_model)
    {
        if !target.endpoint.trim().is_empty() && target_is_launchable(target).await {
            let throughput = summaries
                .iter()
                .filter(|summary| {
                    serving_target_metric_matches(
                        target,
                        summary.host_id.as_deref(),
                        summary.candidate_id.as_str(),
                    )
                })
                .filter_map(|summary| summary.generation_tokens_per_second_ewma)
                .fold(0.0_f64, f64::max);
            ready.push(ServingTargetSelection {
                target: target.clone(),
                throughput_tokens_per_second: throughput,
            });
        }
    }
    rank_serving_targets(&mut ready);
    ready.into_iter().next().ok_or_else(|| {
        GailError::invalid_config(format!(
            "no ready trained-model serving target is compatible with base model {}",
            trainer.ollama_base_model
        ))
    })
}

/// Provider metrics may identify one serving host by its configured logical
/// ID, its URL, or the URL-derived candidate ID. Treat all three forms as the
/// same target so historical throughput remains effective for promotion ties.
fn serving_target_metric_matches(
    target: &TrainerServingTarget,
    metric_host_id: Option<&str>,
    candidate_id: &str,
) -> bool {
    let endpoint = target.endpoint.trim_end_matches('/');
    let endpoint_without_scheme = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let endpoint_fingerprint = endpoint_without_scheme.replace(['/', ':', '.'], "_");
    let endpoint_host = reqwest::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned));

    metric_host_id.is_some_and(|host| {
        host == target.host_id
            || host == endpoint
            || endpoint_host
                .as_deref()
                .is_some_and(|endpoint_host| host.contains(endpoint_host))
    }) || candidate_id.contains(target.host_id.as_str())
        || candidate_id.contains(endpoint_fingerprint.as_str())
}

fn rank_serving_targets(targets: &mut [ServingTargetSelection]) {
    targets.sort_by(|left, right| {
        right
            .target
            .vram_mb
            .cmp(&left.target.vram_mb)
            .then_with(|| {
                right
                    .throughput_tokens_per_second
                    .total_cmp(&left.throughput_tokens_per_second)
            })
            .then_with(|| left.target.host_id.cmp(&right.target.host_id))
    });
}

async fn target_is_ready(target: &TrainerServingTarget) -> bool {
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(6))
        .build()
        .unwrap_or_else(|_| Client::new());
    let url = format!("{}/models", target.endpoint.trim_end_matches('/'));
    let Ok(response) = client.get(url).send().await else {
        return false;
    };
    if !response.status().is_success() {
        return false;
    }
    let Ok(body) = response.json::<Value>().await else {
        return false;
    };
    let expected_model = target
        .model_alias
        .as_deref()
        .unwrap_or(target.base_model.as_str());
    body.get("data")
        .or_else(|| body.get("models"))
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                model
                    .get("id")
                    .or_else(|| model.get("name"))
                    .or_else(|| model.get("model"))
                    .and_then(Value::as_str)
                    .is_some_and(|id| id == expected_model)
            })
        })
}

/// A target is normally already serving the trained alias.  During a
/// promotion, however, the native selector deliberately stops that service
/// everywhere except the currently selected host.  Treat the target as
/// launchable when its exact base model is present in the host's Ollama model
/// store; the post-publication readiness check below still requires the
/// trained llama.cpp endpoint and alias to answer successfully.
async fn target_is_launchable(target: &TrainerServingTarget) -> bool {
    if target_is_ready(target).await {
        return true;
    }
    let Ok(endpoint) = reqwest::Url::parse(target.endpoint.trim_end_matches('/')) else {
        return false;
    };
    let Some(host) = endpoint.host_str() else {
        return false;
    };
    let ollama_url = format!("{}://{}:11434/api/tags", endpoint.scheme(), host);
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(6))
        .build()
        .unwrap_or_else(|_| Client::new());
    let Ok(response) = client.get(ollama_url).send().await else {
        return false;
    };
    if !response.status().is_success() {
        return false;
    }
    let Ok(body) = response.json::<Value>().await else {
        return false;
    };
    body.get("models")
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                model
                    .get("name")
                    .or_else(|| model.get("model"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == target.base_model)
            })
        })
}

async fn health_check_serving_target(selection: &ServingTargetSelection) -> Result<()> {
    let timeout_seconds = env::var("GAIL_TRAIN_SERVING_READINESS_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(300)
        .clamp(30, 900);
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
    loop {
        if target_is_ready(&selection.target).await {
            tracing::info!(
                host = %selection.target.host_id,
                vram_mb = selection.target.vram_mb,
                throughput_tokens_per_second = selection.throughput_tokens_per_second,
                "promoted trained model serving target is ready"
            );
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(GailError::invalid_config(format!(
                "serving target {} did not become ready for base model {} within {} seconds",
                selection.target.host_id, selection.target.base_model, timeout_seconds
            )));
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn ensure_active_serving_target(trainer: &TrainerConfig, metrics_path: &str) -> Result<()> {
    let pointer = PathBuf::from(&trainer.output_root).join("active_snapshot.json");
    let raw = match fs::read_to_string(&pointer).await {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let mut value: Value = serde_json::from_str(&raw).map_err(|error| {
        GailError::invalid_config(format!("invalid active training snapshot pointer: {error}"))
    })?;
    let selection = select_serving_target(trainer, metrics_path).await?;
    let target = json!({
        "host_id": selection.target.host_id,
        "endpoint": selection.target.endpoint,
        "model_alias": selection.target.model_alias,
        "base_model": selection.target.base_model,
        "vram_mb": selection.target.vram_mb,
        "throughput_tokens_per_second": selection.throughput_tokens_per_second,
    });
    let target_changed = value
        .get("serving_target")
        .and_then(Value::as_object)
        .is_none_or(|current| {
            current.get("host_id").and_then(Value::as_str)
                != Some(selection.target.host_id.as_str())
                || current.get("endpoint").and_then(Value::as_str)
                    != Some(selection.target.endpoint.as_str())
                || current.get("model_alias").and_then(Value::as_str)
                    != selection.target.model_alias.as_deref()
                || current.get("vram_mb").and_then(Value::as_u64) != Some(selection.target.vram_mb)
        });
    if !target_changed {
        return Ok(());
    }
    value["serving_target"] = target;
    let temporary = pointer.with_extension(format!("json.tmp-{}", std::process::id()));
    write_json(&temporary, &value).await?;
    fs::rename(&temporary, &pointer).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to publish reconciled serving target: {error}"
        ))
    })
}

async fn restore_active_snapshot_pointer(trainer: &TrainerConfig, previous: Option<&str>) {
    let pointer = PathBuf::from(&trainer.output_root).join("active_snapshot.json");
    match previous {
        Some(previous) => {
            if let Err(error) = fs::write(pointer, previous).await {
                tracing::error!(error = %error, "failed to restore active snapshot pointer after rollback");
            }
        }
        None => {
            let _ = fs::remove_file(pointer).await;
        }
    }
}

struct CommandOutcome {
    stdout: String,
    stderr: String,
    exit_code: i32,
    runtime_seconds: f64,
    backend_job_id: Option<String>,
    heartbeat_ts: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SlurmTrainingResult {
    exit_code: i32,
    runtime_seconds: f64,
    log_file: Option<String>,
    message: Option<String>,
    #[serde(default)]
    slurm_job_id: Option<String>,
    #[serde(default)]
    heartbeat_ts: Option<f64>,
}

async fn execute_slurm_training_request(
    spool: &Path,
    trainer: &TrainerConfig,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
    ledger_ids: &[i64],
) -> Result<CommandOutcome> {
    if snapshot_id.is_empty()
        || !snapshot_id
            .chars()
            .all(|value| value.is_ascii_alphanumeric() || matches!(value, '-' | '_'))
    {
        return Err(GailError::invalid_config(format!(
            "unsafe Slurm snapshot id: {snapshot_id}"
        )));
    }
    let queue = spool.join("queue");
    let results = spool.join("results");
    fs::create_dir_all(&queue).await.map_err(|error| {
        GailError::invalid_config(format!("failed to create Slurm request queue: {error}"))
    })?;
    fs::create_dir_all(&results).await.map_err(|error| {
        GailError::invalid_config(format!("failed to create Slurm result directory: {error}"))
    })?;
    let request_path = queue.join(format!("{snapshot_id}.request"));
    let temporary_path = queue.join(format!(".{snapshot_id}.request-{}", std::process::id()));
    let result_path = results.join(format!("{snapshot_id}.result"));
    let status_path = results.join(format!("{snapshot_id}.status"));
    let heartbeat_path = results.join(format!("{snapshot_id}.heartbeat"));
    let request = json!({
        "version": 1,
        "snapshot_id": snapshot_id,
        "algorithm": trainer.algorithm,
        "dataset_path": dataset_path.to_string_lossy(),
        "snapshot_dir": snapshot_dir.to_string_lossy(),
        "ledger_ids": ledger_ids,
        "slurm_job_id": Value::Null,
        "requested_at": now_ts(),
    });
    fs::write(
        &temporary_path,
        serde_json::to_string_pretty(&request).unwrap_or_else(|_| "{}".to_string()) + "\n",
    )
    .await
    .map_err(|error| {
        GailError::invalid_config(format!("failed to stage Slurm training request: {error}"))
    })?;
    fs::rename(&temporary_path, &request_path)
        .await
        .map_err(|error| {
            GailError::invalid_config(format!("failed to publish Slurm training request: {error}"))
        })?;
    tracing::info!(
        snapshot = snapshot_id,
        request = %request_path.display(),
        "submitted Gail training snapshot to Slurm"
    );

    let started = tokio::time::Instant::now();
    let timeout = Duration::from_secs(trainer.command_timeout_seconds.max(1));
    let poll_seconds = env_usize("GAIL_TRAIN_SLURM_POLL_SECONDS", 2, 1, 60) as u64;
    loop {
        let heartbeat_ts = now_ts();
        let _ = fs::write(&heartbeat_path, format!("{heartbeat_ts:.6}\n")).await;
        refresh_active_training_marker(trainer, snapshot_id, ledger_ids).await?;
        if result_path.exists() {
            let body = fs::read_to_string(&result_path).await.map_err(|error| {
                GailError::invalid_config(format!("failed to read Slurm training result: {error}"))
            })?;
            let result: SlurmTrainingResult = serde_json::from_str(&body).map_err(|error| {
                GailError::invalid_config(format!("invalid Slurm training result: {error}"))
            })?;
            let log = if let Some(log_file) = result.log_file.as_deref() {
                if Path::new(log_file)
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
                {
                    let candidate = spool.join(log_file);
                    fs::read_to_string(candidate).await.unwrap_or_default()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            let log_tail = truncate_chars(&log, 8_000);
            let message = result.message.unwrap_or_default();
            if result.exit_code != 0 {
                return Err(GailError::invalid_config(format!(
                    "Slurm training exited with status {}: {} {}",
                    result.exit_code,
                    message,
                    truncate_chars(&log, 1200)
                )));
            }
            return Ok(CommandOutcome {
                stdout: log_tail,
                stderr: message,
                exit_code: result.exit_code,
                runtime_seconds: result.runtime_seconds,
                backend_job_id: result.slurm_job_id.or_else(|| {
                    std::fs::read_to_string(&status_path)
                        .ok()
                        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
                        .and_then(|value| {
                            value
                                .get("slurm_job_id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                }),
                heartbeat_ts: result.heartbeat_ts.or(Some(heartbeat_ts)),
            });
        }
        if started.elapsed() >= timeout {
            if let Ok(body) = fs::read_to_string(&status_path).await
                && let Ok(value) = serde_json::from_str::<Value>(&body)
                && let Some(job_id) = value.get("slurm_job_id").and_then(Value::as_str)
            {
                cancel_slurm_job(job_id).await;
            }
            let _ = fs::remove_file(&request_path).await;
            return Err(GailError::invalid_config(format!(
                "Slurm training timed out after {}s waiting for {}",
                trainer.command_timeout_seconds,
                result_path.display()
            )));
        }
        tokio::time::sleep(Duration::from_secs(poll_seconds)).await;
    }
}

async fn execute_training_command(
    command_line: &str,
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    execution_plan: &TrainingExecutionPlan,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
) -> Result<CommandOutcome> {
    let started = tokio::time::Instant::now();

    let mut command = Command::new("bash");
    command
        .arg("-lc")
        .arg(command_line)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("GAIL_TRAIN_SNAPSHOT_ID", snapshot_id)
        .env("GAIL_TRAIN_ALGORITHM", trainer.algorithm.as_str())
        .env(
            "GAIL_TRAIN_DATASET_PATH",
            dataset_path.to_string_lossy().to_string(),
        )
        .env(
            "GAIL_TRAIN_OUTPUT_DIR",
            snapshot_dir.to_string_lossy().to_string(),
        )
        .env(
            "GAIL_TRAIN_CPU_THREADS",
            execution_plan.cpu_intraop_threads.to_string(),
        )
        .env(
            "GAIL_TRAIN_CPU_INTRAOP_THREADS",
            execution_plan.cpu_intraop_threads.to_string(),
        )
        .env(
            "GAIL_TRAIN_CPU_INTEROP_THREADS",
            execution_plan.cpu_interop_threads.to_string(),
        )
        .env(
            "GAIL_TRAIN_TOKENIZER_THREADS",
            execution_plan.tokenizer_threads.to_string(),
        )
        .env(
            "GAIL_TRAIN_ASYNC_WORKER_THREADS",
            execution_plan.async_worker_threads.to_string(),
        )
        .env(
            "GAIL_TRAIN_PREFETCH_BATCHES",
            execution_plan.prefetch_batches.to_string(),
        )
        .env(
            "GAIL_TRAIN_DYNAMIC_PADDING",
            if execution_plan.dynamic_padding {
                "1"
            } else {
                "0"
            },
        )
        .env(
            "GAIL_TRAIN_SEQUENCE_PACKING",
            if execution_plan.sequence_packing {
                "1"
            } else {
                "0"
            },
        )
        .env("GAIL_TRAIN_COMPUTE_DTYPE", &execution_plan.compute_dtype)
        .env("GAIL_TRAIN_DEVICE", &execution_plan.device)
        .env("GAIL_TRAIN_EXECUTION_PROFILE", &execution_plan.profile)
        .env("GAIL_TRAIN_BACKEND", &execution_plan.backend)
        .env("GAIL_TRAIN_GPU_COUNT", hardware.gpu_count().to_string())
        .env(
            "GAIL_TRAIN_GPU_MEMORY_MB",
            hardware.total_gpu_memory_mb().to_string(),
        )
        .env(
            "GAIL_TRAIN_GPU_FREE_MEMORY_MB",
            hardware.total_gpu_free_memory_mb().to_string(),
        )
        .env(
            "GAIL_TRAIN_ARTIFACT_MODE",
            match training_artifact_mode() {
                TrainingArtifactMode::Production => "production",
                TrainingArtifactMode::DevelopmentFixture => "development_fixture",
            },
        )
        // Make the child process GPU/CPU-aware for common Rust, BLAS and Python backends.
        .env(
            "RAYON_NUM_THREADS",
            execution_plan.tokenizer_threads.to_string(),
        )
        .env(
            "TOKIO_WORKER_THREADS",
            execution_plan.async_worker_threads.to_string(),
        )
        .env(
            "OMP_NUM_THREADS",
            execution_plan.cpu_intraop_threads.to_string(),
        )
        .env(
            "MKL_NUM_THREADS",
            execution_plan.cpu_intraop_threads.to_string(),
        )
        .env(
            "OPENBLAS_NUM_THREADS",
            execution_plan.cpu_intraop_threads.to_string(),
        )
        .env(
            "NUMEXPR_NUM_THREADS",
            execution_plan.tokenizer_threads.to_string(),
        );

    if hardware.gpu_count() == 0 {
        command.env("CUDA_VISIBLE_DEVICES", "");
    } else if let Some(index) = execution_plan.device_index {
        command.env("CUDA_VISIBLE_DEVICES", index.to_string());
    }

    let mut child = command.spawn().map_err(|error| {
        GailError::invalid_config(format!("failed to spawn trainer command: {error}"))
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| GailError::invalid_config("failed to capture trainer stdout".to_string()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| GailError::invalid_config("failed to capture trainer stderr".to_string()))?;

    let stdout_task = tokio::spawn(stream_child_output("trainer.stdout", stdout));
    let stderr_task = tokio::spawn(stream_child_output("trainer.stderr", stderr));

    let timeout_duration = Duration::from_secs(trainer.command_timeout_seconds.max(1));
    let status = match tokio::time::timeout(timeout_duration, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            return Err(GailError::invalid_config(format!(
                "trainer command failed to execute: {error}"
            )));
        }
        Err(_) => {
            let _ = child.kill().await;
            return Err(GailError::invalid_config(format!(
                "trainer command timed out after {}s",
                trainer.command_timeout_seconds
            )));
        }
    };

    let stdout = stdout_task.await.map_err(|error| {
        GailError::invalid_config(format!("trainer stdout reader failed: {error}"))
    })?;
    let stderr = stderr_task.await.map_err(|error| {
        GailError::invalid_config(format!("trainer stderr reader failed: {error}"))
    })?;
    let exit_code = status.code().unwrap_or(-1);

    if !status.success() {
        return Err(GailError::invalid_config(format!(
            "trainer command exited with status {exit_code}: {}",
            truncate_chars(&stderr, 1200)
        )));
    }

    Ok(CommandOutcome {
        stdout: truncate_chars(&stdout, 8_000),
        stderr: truncate_chars(&stderr, 8_000),
        exit_code,
        runtime_seconds: started.elapsed().as_secs_f64(),
        backend_job_id: None,
        heartbeat_ts: None,
    })
}

async fn stream_child_output<R>(target: &'static str, reader: R) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    let mut tail = String::new();
    while let Ok(Some(line)) = lines.next_line().await {
        if target.ends_with(".stderr") {
            tracing::warn!(target = target, "{}", line);
        } else {
            tracing::info!(target = target, "{}", line);
        }
        tail.push_str(&line);
        tail.push('\n');
        if tail.len() > 16_000 {
            tail = tail
                .chars()
                .rev()
                .take(12_000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
        }
    }
    tail
}

async fn resolve_training_invocation(
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
    resume_adapter: Option<&Path>,
) -> Result<Option<String>> {
    if let Some(command_template) = trainer.command_template.as_deref() {
        return Ok(Some(render_training_command(
            command_template,
            trainer,
            hardware,
            snapshot_id,
            dataset_path,
            snapshot_dir,
            resume_adapter,
        )));
    }

    if matches!(trainer.algorithm.as_str(), "qlora_sft" | "lora_sft") {
        let python_runner = env_string("GAIL_PYTHON_QLORA_SFT_BIN")
            .unwrap_or_else(|| "/usr/local/libexec/gail-qlora-sft-python".to_string());
        let prefer_python = env::var("GAIL_TRAINER_BACKEND")
            .map(|value| !value.trim().eq_ignore_ascii_case("rust_torchscript"))
            .unwrap_or(true);
        if prefer_python && Path::new(&python_runner).is_file() {
            let python = bootstrap_python_binary();
            let ollama_base_model = trainer.ollama_base_model.as_str();
            let hf_base_model = env_string("GAIL_TRAIN_HF_BASE_MODEL")
                .or_else(|| mapped_hf_model(ollama_base_model).map(ToOwned::to_owned))
                .ok_or_else(|| {
                    GailError::invalid_config(format!(
                        "Python PEFT trainer requires a Hugging Face base model; set GAIL_TRAIN_HF_BASE_MODEL for Ollama model {ollama_base_model}"
                    ))
                })?;
            return Ok(Some(format!(
                "env -u LD_LIBRARY_PATH {} {} --dataset {} --output {} --algorithm {} --base-model {} --ollama-base-model {}{}",
                shell_escape(python.as_str()),
                shell_escape(python_runner.as_str()),
                shell_escape(&dataset_path.to_string_lossy()),
                shell_escape(&snapshot_dir.to_string_lossy()),
                shell_escape(trainer.algorithm.as_str()),
                shell_escape(hf_base_model.as_str()),
                shell_escape(ollama_base_model),
                resume_adapter
                    .map(|path| format!(
                        " --resume-adapter {}",
                        shell_escape(&path.to_string_lossy())
                    ))
                    .unwrap_or_default(),
            )));
        }
        let runner = std::env::var("GAIL_RUST_QLORA_SFT_BIN")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "gail-qlora-sft".to_string());
        let base_model = std::env::var("GAIL_TRAIN_BASE_MODEL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| trainer.ollama_base_model.clone());
        let (model_module, tokenizer) =
            ensure_torchscript_artifacts(trainer, base_model.as_str(), dataset_path).await?;

        return Ok(Some(format!(
            "{} --dataset {} --output {} --algorithm {} --base-model {} --model-module {} --tokenizer {} --timeout-seconds {}{}",
            shell_escape(runner.as_str()),
            shell_escape(&dataset_path.to_string_lossy()),
            shell_escape(&snapshot_dir.to_string_lossy()),
            shell_escape(trainer.algorithm.as_str()),
            shell_escape(base_model.as_str()),
            shell_escape(&model_module.to_string_lossy()),
            shell_escape(&tokenizer.to_string_lossy()),
            trainer.command_timeout_seconds.max(1),
            resume_adapter
                .map(|path| format!(
                    " --resume-adapter {}",
                    shell_escape(&path.to_string_lossy())
                ))
                .unwrap_or_default(),
        )));
    }

    Ok(None)
}

async fn ensure_torchscript_artifacts(
    trainer: &TrainerConfig,
    base_model: &str,
    dataset_path: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let artifact_mode = training_artifact_mode();
    let explicit_model_module = std::env::var("GAIL_TCH_MODEL_MODULE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let explicit_tokenizer = std::env::var("GAIL_TCH_TOKENIZER")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let has_explicit_overrides = explicit_model_module.is_some() || explicit_tokenizer.is_some();

    let model_module =
        explicit_model_module.unwrap_or_else(|| default_model_module_path(trainer, base_model));
    let tokenizer =
        explicit_tokenizer.unwrap_or_else(|| default_tokenizer_path(trainer, base_model));
    if model_module.exists() && tokenizer.exists() {
        return Ok((model_module, tokenizer));
    }
    if has_explicit_overrides {
        return Err(GailError::invalid_config(format!(
            "TorchScript model module/tokenizer not found (model_module={}, tokenizer={}). Verify GAIL_TCH_MODEL_MODULE and GAIL_TCH_TOKENIZER.",
            model_module.display(),
            tokenizer.display()
        )));
    }

    if matches!(artifact_mode, TrainingArtifactMode::Production) {
        return Err(GailError::invalid_config(format!(
            "TorchScript artifacts are required for production training and were not found \
            (model_module={}, tokenizer={}). Provide explicit artifacts or set \
            GAIL_TRAIN_ARTIFACT_MODE=development_fixture for synthetic bootstrap only.",
            model_module.display(),
            tokenizer.display()
        )));
    }

    bootstrap_torchscript_artifacts(trainer, base_model, dataset_path, &model_module, &tokenizer)
        .await?;
    if model_module.exists() && tokenizer.exists() {
        return Ok((model_module, tokenizer));
    }
    Err(GailError::invalid_config(format!(
        "TorchScript bootstrap completed without required artifacts (model_module={}, tokenizer={})",
        model_module.display(),
        tokenizer.display()
    )))
}

async fn bootstrap_torchscript_artifacts(
    trainer: &TrainerConfig,
    base_model: &str,
    dataset_path: &Path,
    model_module: &Path,
    tokenizer: &Path,
) -> Result<()> {
    if let Some(parent) = model_module.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            GailError::invalid_config(format!(
                "failed to create TorchScript model directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    if let Some(parent) = tokenizer.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            GailError::invalid_config(format!(
                "failed to create tokenizer directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    let bootstrap_python = bootstrap_python_binary();
    let bootstrap_script_path = PathBuf::from("/tmp/gail_torchscript_bootstrap.py");
    let timeout_seconds = bootstrap_timeout_seconds();
    let hidden_size = bootstrap_env_usize("GAIL_TCH_BOOTSTRAP_HIDDEN_SIZE", 192, 64, 2048);
    let lora_rank = bootstrap_env_usize("GAIL_TCH_BOOTSTRAP_LORA_RANK", 16, 1, 512);
    let vocab_size = bootstrap_env_usize("GAIL_TCH_BOOTSTRAP_VOCAB_SIZE", 8_192, 256, 65_536);
    let hf_model_hint = bootstrap_hf_model_hint(base_model);
    tracing::info!(
        algorithm = %trainer.algorithm,
        base_model = %base_model,
        model_module = %model_module.display(),
        tokenizer = %tokenizer.display(),
        python = %bootstrap_python,
        timeout_seconds,
        hidden_size,
        lora_rank,
        vocab_size,
        hf_model_hint = hf_model_hint.as_deref().unwrap_or(""),
        "TorchScript artifacts missing; bootstrapping development fixture module/tokenizer"
    );
    fs::write(&bootstrap_script_path, TORCHSCRIPT_BOOTSTRAP_PYTHON)
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "failed to write TorchScript bootstrap script {}: {error}",
                bootstrap_script_path.display()
            ))
        })?;

    let started = tokio::time::Instant::now();
    let mut command = Command::new(bootstrap_python.as_str());
    command
        .arg(&bootstrap_script_path)
        .arg("--base-model")
        .arg(base_model)
        .arg("--dataset")
        .arg(dataset_path)
        .arg("--model-module")
        .arg(model_module)
        .arg("--tokenizer")
        .arg(tokenizer)
        .arg("--hidden-size")
        .arg(hidden_size.to_string())
        .arg("--lora-rank")
        .arg(lora_rank.to_string())
        .arg("--vocab-size")
        .arg(vocab_size.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(hf_model_hint) = hf_model_hint.as_deref() {
        command.arg("--hf-model").arg(hf_model_hint);
    }
    let mut child = command.spawn().map_err(|error| {
        GailError::invalid_config(format!(
            "failed to spawn TorchScript bootstrap command: {error}"
        ))
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        GailError::invalid_config("failed to capture TorchScript bootstrap stdout".to_string())
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        GailError::invalid_config("failed to capture TorchScript bootstrap stderr".to_string())
    })?;
    let stdout_task = tokio::spawn(stream_child_output("torchscript.bootstrap.stdout", stdout));
    let stderr_task = tokio::spawn(stream_child_output("torchscript.bootstrap.stderr", stderr));

    let status =
        match tokio::time::timeout(Duration::from_secs(timeout_seconds), child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(error)) => {
                return Err(GailError::invalid_config(format!(
                    "TorchScript bootstrap command failed to execute: {error}"
                )));
            }
            Err(_) => {
                let _ = child.kill().await;
                return Err(GailError::invalid_config(format!(
                    "TorchScript bootstrap timed out after {timeout_seconds}s"
                )));
            }
        };
    let stdout = stdout_task.await.map_err(|error| {
        GailError::invalid_config(format!(
            "TorchScript bootstrap stdout reader failed: {error}"
        ))
    })?;
    let stderr = stderr_task.await.map_err(|error| {
        GailError::invalid_config(format!(
            "TorchScript bootstrap stderr reader failed: {error}"
        ))
    })?;
    if !status.success() {
        let exit_code = status.code().unwrap_or(-1);
        return Err(GailError::invalid_config(format!(
            "TorchScript bootstrap failed with status {exit_code}: {}",
            truncate_chars(&stderr, 1_200)
        )));
    }
    tracing::info!(
        runtime_seconds = started.elapsed().as_secs_f64(),
        model_module = %model_module.display(),
        tokenizer = %tokenizer.display(),
        stdout_tail = %truncate_chars(&stdout, 400),
        "TorchScript bootstrap completed"
    );
    Ok(())
}

fn bootstrap_python_binary() -> String {
    std::env::var("GAIL_TCH_BOOTSTRAP_PYTHON")
        .ok()
        .or_else(|| std::env::var("GAIL_PYTHON").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

fn bootstrap_timeout_seconds() -> u64 {
    std::env::var("GAIL_TCH_BOOTSTRAP_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(900)
        .max(30)
}

fn bootstrap_env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

fn bootstrap_hf_model_hint(base_model: &str) -> Option<String> {
    std::env::var("GAIL_TCH_BOOTSTRAP_HF_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| mapped_hf_model(base_model).map(ToOwned::to_owned))
}

fn mapped_hf_model(base_model: &str) -> Option<&'static str> {
    match base_model.trim().to_ascii_lowercase().as_str() {
        "qwen2.5-coder:0.5b" => Some("Qwen/Qwen2.5-Coder-0.5B"),
        "qwen2.5-coder:1.5b" => Some("Qwen/Qwen2.5-Coder-1.5B"),
        "qwen2.5-coder:3b" => Some("Qwen/Qwen2.5-Coder-3B"),
        "qwen2.5-coder:7b" => Some("Qwen/Qwen2.5-Coder-7B"),
        "qwen2.5:0.5b" => Some("Qwen/Qwen2.5-0.5B"),
        "qwen2.5:1.5b" => Some("Qwen/Qwen2.5-1.5B"),
        "qwen2.5:3b" => Some("Qwen/Qwen2.5-3B"),
        "qwen2.5:7b" => Some("Qwen/Qwen2.5-7B"),
        "qwen3.5:0.8b" => Some("Qwen/Qwen3.5-0.8B"),
        "qwen3.5:2b" => Some("Qwen/Qwen3.5-2B"),
        "qwen3.5:4b" => Some("Qwen/Qwen3.5-4B"),
        _ => None,
    }
}

const TORCHSCRIPT_BOOTSTRAP_PYTHON: &str = r#"
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def clean_text(value):
    return " ".join(str(value or "").split())


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap TorchScript trainer artifacts")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-module", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--hf-model", default="")
    return parser.parse_args()


def read_dataset_texts(dataset_path):
    texts = []
    path = Path(dataset_path)
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            messages = row.get("messages") or []
            if not isinstance(messages, list):
                continue
            chunks = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = clean_text(message.get("role", "user")).lower() or "user"
                content = clean_text(message.get("content", ""))
                if content:
                    chunks.append(f"<|{role}|> {content}")
            rendered = " ".join(chunks).strip()
            if rendered:
                texts.append(rendered)
    return texts


def tokenizer_candidates(base_model, explicit_hf_model):
    candidates = []
    if explicit_hf_model:
        candidates.append(explicit_hf_model.strip())
    raw = (base_model or "").strip()
    mapped = {
        "qwen2.5-coder:0.5b": "Qwen/Qwen2.5-Coder-0.5B",
        "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B",
        "qwen2.5-coder:3b": "Qwen/Qwen2.5-Coder-3B",
        "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B",
        "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B",
        "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
        "qwen2.5:3b": "Qwen/Qwen2.5-3B",
        "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    }.get(raw.lower())
    if mapped:
        candidates.append(mapped)
    if "/" in raw:
        candidates.append(raw)
    ordered = []
    seen = set()
    for candidate in candidates:
        value = candidate.strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def ensure_tokenizer(tokenizer_path, dataset_path, target_vocab_size, base_model, explicit_hf_model):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

    tokenizer_path = Path(tokenizer_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_path.exists():
        existing = Tokenizer.from_file(str(tokenizer_path))
        return max(256, int(existing.get_vocab_size()))

    for candidate in tokenizer_candidates(base_model, explicit_hf_model):
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            if hf_tokenizer.pad_token is None and hf_tokenizer.eos_token is not None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token
            hf_tokenizer.save_pretrained(str(tokenizer_path.parent))
            generated = tokenizer_path.parent / "tokenizer.json"
            if generated.exists():
                if generated.resolve() != tokenizer_path.resolve():
                    shutil.copy2(generated, tokenizer_path)
                size = int(getattr(hf_tokenizer, "vocab_size", 0) or len(hf_tokenizer))
                return max(256, size)
        except Exception as exc:
            print(
                f"torchscript.bootstrap tokenizer candidate failed: {candidate}: {exc}",
                file=sys.stderr,
            )

    texts = read_dataset_texts(dataset_path)
    if not texts:
        texts = ["<|user|> hello <|assistant|> hello"]
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=max(256, int(target_vocab_size)),
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<|system|>", "<|user|>", "<|assistant|>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return max(256, int(tokenizer.get_vocab_size()))


class GailTorchscriptLossModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, lora_rank):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lora_down = nn.Parameter(torch.zeros(hidden_size, lora_rank))
        self.lora_up = nn.Parameter(torch.zeros(lora_rank, hidden_size))
        nn.init.normal_(self.lora_down, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_up)
        self.scale = 1.0 / float(max(1, lora_rank))
        self.embed.weight.requires_grad = False
        self.proj.weight.requires_grad = False
        self.proj.bias.requires_grad = False

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.proj(self.embed(input_ids)))
        delta = torch.matmul(torch.matmul(hidden, self.lora_down), self.lora_up) * self.scale
        logits = torch.matmul(hidden + delta, self.embed.weight.t())
        if logits.size(1) < 2:
            return logits.sum() * 0.0
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


def main():
    args = parse_args()
    model_module = Path(args.model_module)
    tokenizer = Path(args.tokenizer)
    model_module.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.parent.mkdir(parents=True, exist_ok=True)

    vocab_size = ensure_tokenizer(
        tokenizer,
        args.dataset,
        args.vocab_size,
        args.base_model,
        args.hf_model.strip(),
    )
    module = GailTorchscriptLossModule(
        vocab_size=max(256, int(vocab_size)),
        hidden_size=max(64, int(args.hidden_size)),
        lora_rank=max(1, int(args.lora_rank)),
    )
    scripted = torch.jit.script(module)
    scripted.save(str(model_module))

    print(
        json.dumps(
            {
                "model_module": str(model_module),
                "tokenizer": str(tokenizer),
                "vocab_size": int(vocab_size),
                "hidden_size": int(args.hidden_size),
                "lora_rank": int(args.lora_rank),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TorchScript bootstrap failed: {exc}", file=sys.stderr)
        raise
"#;

fn shell_escape(value: &str) -> String {
    if value.chars().all(|ch| {
        ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '=' | '+')
    }) {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

async fn register_snapshot_with_ollama(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    snapshot_dir: &Path,
) -> Result<OllamaRegistrationMode> {
    let tagged_model = format!("{}:{}", trainer.model_prefix, snapshot_id);
    let modelfile_path = snapshot_dir.join("Modelfile");
    let modelfile = if modelfile_path.exists() {
        fs::read_to_string(&modelfile_path).await.map_err(|error| {
            GailError::invalid_config(format!("failed to read Modelfile: {error}"))
        })?
    } else {
        let rendered = format!(
            "FROM {}\nSYSTEM You are the Gail in-house continuously trained model snapshot {}.\n",
            trainer.ollama_base_model, snapshot_id
        );
        fs::write(&modelfile_path, rendered.as_bytes())
            .await
            .map_err(|error| {
                GailError::invalid_config(format!("failed to write Modelfile: {error}"))
            })?;
        rendered
    };
    let mut parsed_modelfile = parse_modelfile(&modelfile);
    validate_registration_artifacts(trainer, &parsed_modelfile)?;
    validate_tokenizer_registration_manifest(snapshot_dir).await?;
    let adapter_directives =
        prepare_ollama_adapter(trainer, snapshot_id, snapshot_dir, &mut parsed_modelfile).await?;
    let create_payload = build_ollama_create_payload_from_modelfile(
        trainer,
        tagged_model.as_str(),
        snapshot_id,
        &render_modelfile_with_adapters(&modelfile, adapter_directives.as_slice()),
    );
    let client = ollama_api_client();
    let mut create_payload = create_payload;
    let mut registration_mode = OllamaRegistrationMode::BaseModel;
    if !adapter_directives.is_empty() {
        let manifest =
            build_ollama_adapter_manifest(snapshot_dir, adapter_directives.as_slice()).await?;
        let adapters = upload_ollama_adapter_blobs(&client, trainer, manifest.as_slice()).await?;
        create_payload["adapters"] = Value::Object(adapters);
        registration_mode = OllamaRegistrationMode::Adapter;
    }
    ollama_api_post(&client, trainer, "create", &create_payload)
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "Ollama API /api/create failed for trained snapshot; the model alias was not changed: {error}"
            ))
        })?;
    ollama_api_post(
        &client,
        trainer,
        "copy",
        &json!({
            "source": format!("{}:{}", trainer.model_prefix, snapshot_id),
            "destination": trainer.model_alias
        }),
    )
    .await?;
    if registration_mode == OllamaRegistrationMode::Adapter {
        publish_llama_cpp_adapter(snapshot_dir, snapshot_id).await?;
    }
    Ok(registration_mode)
}

/// Publish the validated adapter at a stable path consumed by the qc01
/// llama.cpp service.  The rename is atomic on the shared filesystem, so the
/// watcher can never restart llama.cpp against a partially written GGUF.
async fn publish_llama_cpp_adapter(snapshot_dir: &Path, snapshot_id: &str) -> Result<()> {
    let source = snapshot_dir.join("adapter.gguf");
    let metadata = fs::metadata(&source).await.map_err(|error| {
        GailError::invalid_config(format!(
            "validated GGUF adapter is missing for llama.cpp publication: {error}"
        ))
    })?;
    if !metadata.is_file() || metadata.len() < 4 {
        return Err(GailError::invalid_config(
            "validated GGUF adapter is empty; refusing llama.cpp publication",
        ));
    }
    let output_root = snapshot_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| GailError::invalid_config("snapshot path has no training output root"))?;
    let serving_dir = output_root.join("serving");
    fs::create_dir_all(&serving_dir).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to create llama.cpp serving directory: {error}"
        ))
    })?;
    let temporary = serving_dir.join(format!(
        ".adapter.gguf-{snapshot_id}-{}",
        std::process::id()
    ));
    fs::copy(&source, &temporary).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to stage GGUF adapter for llama.cpp: {error}"
        ))
    })?;
    fs::rename(&temporary, serving_dir.join("adapter.gguf"))
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "failed to atomically publish llama.cpp adapter: {error}"
            ))
        })?;
    fs::write(
        serving_dir.join("adapter.snapshot"),
        format!("{snapshot_id}\n"),
    )
    .await
    .map_err(|error| {
        GailError::invalid_config(format!(
            "failed to publish llama.cpp adapter metadata: {error}"
        ))
    })?;
    tracing::info!(snapshot = snapshot_id, path = %serving_dir.display(), "published trained GGUF adapter for llama.cpp");
    Ok(())
}

/// Prepare an adapter in the format accepted by the selected serving runtime.
///
/// Ollama's Safetensors adapter importer only supports a small set of dense
/// architectures.  Qwen3.5 is a hybrid vision/linear-attention architecture;
/// it must be supplied as a GGUF LoRA adapter (or served by a Transformers
/// compatible node).  Keeping conversion behind an explicit command means the
/// worker cannot silently send an incompatible Safetensors artifact and then
/// retry the same request forever.
async fn prepare_ollama_adapter(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    snapshot_dir: &Path,
    parsed: &mut ParsedModelfile,
) -> Result<Vec<String>> {
    if parsed.adapters.is_empty() {
        return Ok(Vec::new());
    }
    if parsed
        .adapters
        .iter()
        .all(|directive| directive.to_ascii_lowercase().ends_with(".gguf"))
    {
        return Ok(parsed.adapters.clone());
    }

    let is_qwen35 = trainer
        .ollama_base_model
        .to_ascii_lowercase()
        .replace('.', "")
        .replace('-', "")
        .replace('_', "")
        .contains("qwen35");
    if is_qwen35 && trainer.ollama_adapter_conversion_command.is_none() {
        return Err(GailError::invalid_config(
            "Qwen3.5 adapters cannot be registered as Safetensors with Ollama; configure trainer.ollama_adapter_conversion_command to produce a GGUF LoRA adapter or use a Transformers/vLLM Gail node",
        ));
    }
    let Some(command_template) = trainer.ollama_adapter_conversion_command.as_deref() else {
        return Ok(parsed.adapters.clone());
    };
    if parsed.adapters.len() != 1 {
        return Err(GailError::invalid_config(
            "adapter conversion currently requires exactly one ADAPTER directive",
        ));
    }
    let directive = parsed.adapters[0].trim();
    let adapter_path = resolve_snapshot_adapter_path(snapshot_dir, directive).await?;
    let output_path = snapshot_dir.join("adapter.gguf");
    let command = render_adapter_conversion_command(
        command_template,
        &adapter_path,
        &output_path,
        trainer.ollama_base_model.as_str(),
        snapshot_dir,
        snapshot_id,
    );
    let mut child = Command::new("bash");
    child
        .arg("-lc")
        .arg(command.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = child.spawn().map_err(|error| {
        GailError::invalid_config(format!(
            "failed to start Ollama adapter conversion command: {error}"
        ))
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        GailError::invalid_config("failed to capture adapter conversion stdout".to_string())
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        GailError::invalid_config("failed to capture adapter conversion stderr".to_string())
    })?;
    let stdout_task = tokio::spawn(stream_child_output("adapter_conversion.stdout", stdout));
    let stderr_task = tokio::spawn(stream_child_output("adapter_conversion.stderr", stderr));
    let status = match tokio::time::timeout(
        Duration::from_secs(trainer.command_timeout_seconds.max(1)),
        child.wait(),
    )
    .await
    {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            return Err(GailError::invalid_config(format!(
                "Ollama adapter conversion command failed to execute: {error}"
            )));
        }
        Err(_) => {
            let _ = child.kill().await;
            return Err(GailError::invalid_config(format!(
                "Ollama adapter conversion timed out after {}s",
                trainer.command_timeout_seconds
            )));
        }
    };
    let stdout = stdout_task.await.unwrap_or_default();
    let stderr = stderr_task.await.unwrap_or_default();
    if !status.success() {
        return Err(GailError::invalid_config(format!(
            "Ollama adapter conversion exited with {}: {} {}",
            status.code().unwrap_or(-1),
            truncate_chars(&stdout, 1200),
            truncate_chars(&stderr, 1200)
        )));
    }
    let metadata = fs::metadata(&output_path).await.map_err(|error| {
        GailError::invalid_config(format!(
            "adapter conversion completed without output {}: {error}",
            output_path.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(GailError::invalid_config(format!(
            "adapter conversion produced an empty output: {}",
            output_path.display()
        )));
    }
    tracing::info!(
        snapshot = snapshot_id,
        adapter = %adapter_path.display(),
        output = %output_path.display(),
        "converted trained adapter to GGUF for Ollama registration"
    );
    let converted = vec!["./adapter.gguf".to_string()];
    parsed.adapters = converted.clone();
    Ok(converted)
}

async fn resolve_snapshot_adapter_path(snapshot_dir: &Path, directive: &str) -> Result<PathBuf> {
    let root = fs::canonicalize(snapshot_dir).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to resolve snapshot directory {}: {error}",
            snapshot_dir.display()
        ))
    })?;
    let relative = directive.strip_prefix("./").unwrap_or(directive);
    let path = fs::canonicalize(root.join(relative))
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "failed to resolve ADAPTER path {directive}: {error}"
            ))
        })?;
    ensure_path_beneath_snapshot(&root, &path, directive)?;
    Ok(path)
}

fn render_adapter_conversion_command(
    template: &str,
    adapter: &Path,
    output: &Path,
    base_model: &str,
    snapshot: &Path,
    snapshot_id: &str,
) -> String {
    template
        .replace("{adapter}", &shell_escape(&adapter.to_string_lossy()))
        .replace("{output}", &shell_escape(&output.to_string_lossy()))
        .replace("{base_model}", &shell_escape(base_model))
        .replace("{snapshot}", &shell_escape(&snapshot.to_string_lossy()))
        .replace("{snapshot_id}", &shell_escape(snapshot_id))
}

fn render_modelfile_with_adapters(modelfile: &str, adapters: &[String]) -> String {
    let mut next_adapter = 0_usize;
    modelfile
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.to_ascii_uppercase().starts_with("ADAPTER ") && next_adapter < adapters.len()
            {
                let indentation = &line[..line.len() - trimmed.len()];
                let replacement = format!("{indentation}ADAPTER {}", adapters[next_adapter]);
                next_adapter += 1;
                replacement
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaRegistrationMode {
    Adapter,
    BaseModel,
}

fn validate_registration_artifacts(
    trainer: &TrainerConfig,
    modelfile: &ParsedModelfile,
) -> Result<()> {
    if trainer.algorithm.to_ascii_lowercase().contains("lora") && modelfile.adapters.is_empty() {
        return Err(GailError::invalid_config(format!(
            "{} training completed without an ADAPTER artifact; refusing to register an unchanged base model",
            trainer.algorithm
        )));
    }
    Ok(())
}

/// Refuse to promote an adapter whose tokenizer contract was not persisted
/// with the snapshot. Serving must use the same special-token IDs as training.
async fn validate_tokenizer_registration_manifest(snapshot_dir: &Path) -> Result<()> {
    let adapter_dir = snapshot_dir.join("adapter");
    let manifest_path = adapter_dir.join("training_manifest.json");
    let raw = fs::read_to_string(&manifest_path).await.map_err(|error| {
        GailError::invalid_config(format!(
            "tokenizer manifest is missing before Ollama registration ({}): {error}",
            manifest_path.display()
        ))
    })?;
    let manifest: Value = serde_json::from_str(&raw).map_err(|error| {
        GailError::invalid_config(format!(
            "tokenizer manifest is invalid before Ollama registration: {error}"
        ))
    })?;
    let metadata = manifest
        .get("tokenizer_metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            GailError::invalid_config(
                "tokenizer metadata is absent before Ollama registration".to_string(),
            )
        })?;
    for name in ["pad_token_id", "bos_token_id", "eos_token_id"] {
        if !metadata.contains_key(name) {
            return Err(GailError::invalid_config(format!(
                "tokenizer metadata is missing {name} before Ollama registration"
            )));
        }
        if !metadata[name].is_null() && metadata[name].as_u64().is_none() {
            return Err(GailError::invalid_config(format!(
                "tokenizer metadata {name} is not an integer or null"
            )));
        }
    }
    if metadata
        .get("probe")
        .and_then(Value::as_object)
        .and_then(|probe| probe.get("round_trip_ok"))
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(GailError::invalid_config(
            "tokenizer startup probe was not successful; refusing Ollama registration".to_string(),
        ));
    }
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let path = adapter_dir.join(filename);
        if !path.is_file() {
            return Err(GailError::invalid_config(format!(
                "tokenizer artifact {filename} is missing before Ollama registration"
            )));
        }
    }
    tracing::info!(
        path = %manifest_path.display(),
        pad_token_id = ?metadata.get("pad_token_id"),
        bos_token_id = ?metadata.get("bos_token_id"),
        eos_token_id = ?metadata.get("eos_token_id"),
        "validated tokenizer/model contract before Ollama registration"
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OllamaAdapterBlob {
    name: String,
    path: PathBuf,
    digest: String,
}

async fn build_ollama_adapter_manifest(
    snapshot_dir: &Path,
    adapter_directives: &[String],
) -> Result<Vec<OllamaAdapterBlob>> {
    let snapshot_root = fs::canonicalize(snapshot_dir).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to resolve training snapshot directory {}: {error}",
            snapshot_dir.display()
        ))
    })?;
    let mut manifest = Vec::new();
    let mut names = std::collections::HashMap::<String, String>::new();

    for directive in adapter_directives {
        let relative = Path::new(directive);
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(GailError::invalid_config(format!(
                "unsafe ADAPTER path in snapshot Modelfile: {directive}"
            )));
        }
        let adapter_root = fs::canonicalize(snapshot_root.join(relative))
            .await
            .map_err(|error| {
                GailError::invalid_config(format!(
                    "failed to resolve ADAPTER path {directive}: {error}"
                ))
            })?;
        ensure_path_beneath_snapshot(&snapshot_root, &adapter_root, directive)?;

        let mut files = collect_adapter_files(&snapshot_root, &adapter_root).await?;
        if files.is_empty() {
            return Err(GailError::invalid_config(format!(
                "ADAPTER path contains no regular files: {directive}"
            )));
        }
        files.sort_by(|left, right| left.0.cmp(&right.0));
        for (name, path) in files {
            let digest = sha256_file(path.as_path()).await?;
            if let Some(existing) = names.get(&name) {
                if existing != &digest {
                    return Err(GailError::invalid_config(format!(
                        "multiple ADAPTER paths produce conflicting file name {name}"
                    )));
                }
                continue;
            }
            names.insert(name.clone(), digest.clone());
            manifest.push(OllamaAdapterBlob { name, path, digest });
        }
    }
    manifest.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(manifest)
}

async fn collect_adapter_files(
    snapshot_root: &Path,
    adapter_root: &Path,
) -> Result<Vec<(String, PathBuf)>> {
    let metadata = fs::metadata(adapter_root).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to inspect ADAPTER path {}: {error}",
            adapter_root.display()
        ))
    })?;
    if metadata.is_file() {
        let name = adapter_root
            .file_name()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| GailError::invalid_config("ADAPTER file has no valid UTF-8 name"))?;
        return Ok(vec![(name.to_string(), adapter_root.to_path_buf())]);
    }
    if !metadata.is_dir() {
        return Err(GailError::invalid_config(format!(
            "ADAPTER path is not a regular file or directory: {}",
            adapter_root.display()
        )));
    }

    let mut files = Vec::new();
    let mut pending = vec![(adapter_root.to_path_buf(), PathBuf::new())];
    let mut visited = HashSet::new();
    visited.insert(adapter_root.to_path_buf());
    while let Some((directory, relative_directory)) = pending.pop() {
        let mut entries = fs::read_dir(&directory).await.map_err(|error| {
            GailError::invalid_config(format!(
                "failed to read ADAPTER directory {}: {error}",
                directory.display()
            ))
        })?;
        while let Some(entry) = entries.next_entry().await.map_err(|error| {
            GailError::invalid_config(format!(
                "failed to enumerate ADAPTER directory {}: {error}",
                directory.display()
            ))
        })? {
            let relative_path = relative_directory.join(entry.file_name());
            let resolved = fs::canonicalize(entry.path()).await.map_err(|error| {
                GailError::invalid_config(format!(
                    "failed to resolve ADAPTER entry {}: {error}",
                    entry.path().display()
                ))
            })?;
            ensure_path_beneath_snapshot(
                snapshot_root,
                &resolved,
                &relative_path.to_string_lossy(),
            )?;
            let entry_metadata = fs::metadata(&resolved).await.map_err(|error| {
                GailError::invalid_config(format!(
                    "failed to inspect ADAPTER entry {}: {error}",
                    resolved.display()
                ))
            })?;
            if entry_metadata.is_dir() {
                if visited.insert(resolved.clone()) {
                    pending.push((resolved, relative_path));
                }
            } else if entry_metadata.is_file() {
                files.push((ollama_adapter_name(&relative_path)?, resolved));
            }
        }
    }
    Ok(files)
}

fn ensure_path_beneath_snapshot(snapshot_root: &Path, path: &Path, label: &str) -> Result<()> {
    if path.starts_with(snapshot_root) {
        return Ok(());
    }
    Err(GailError::invalid_config(format!(
        "ADAPTER path escapes the training snapshot: {label}"
    )))
}

fn ollama_adapter_name(relative: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(value) => {
                let value = value.to_str().ok_or_else(|| {
                    GailError::invalid_config("ADAPTER file name is not valid UTF-8")
                })?;
                parts.push(value);
            }
            _ => {
                return Err(GailError::invalid_config(format!(
                    "unsafe ADAPTER file name: {}",
                    relative.display()
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(GailError::invalid_config("ADAPTER file name is empty"));
    }
    Ok(parts.join("/"))
}

async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to open ADAPTER file {}: {error}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 128 * 1024];
    loop {
        let read = file.read(buffer.as_mut_slice()).await.map_err(|error| {
            GailError::invalid_config(format!(
                "failed to hash ADAPTER file {}: {error}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

async fn upload_ollama_adapter_blobs(
    client: &Client,
    trainer: &TrainerConfig,
    manifest: &[OllamaAdapterBlob],
) -> Result<Map<String, Value>> {
    let mut adapters = Map::new();
    for blob in manifest {
        ensure_ollama_blob(client, trainer, blob).await?;
        adapters.insert(blob.name.clone(), json!(blob.digest));
    }
    Ok(adapters)
}

async fn ensure_ollama_blob(
    client: &Client,
    trainer: &TrainerConfig,
    blob: &OllamaAdapterBlob,
) -> Result<()> {
    let base_url = ollama_base_url(trainer);
    let url = format!("{base_url}/api/blobs/{}", blob.digest);
    let response = client.head(&url).send().await.map_err(|error| {
        GailError::invalid_config(format!(
            "Ollama API request failed while checking adapter blob {}: {error}",
            blob.name
        ))
    })?;
    if response.status().is_success() {
        return Ok(());
    }
    if response.status() != reqwest::StatusCode::NOT_FOUND {
        return Err(GailError::invalid_config(format!(
            "Ollama adapter blob check failed with HTTP {} for {}",
            response.status().as_u16(),
            blob.name
        )));
    }

    let body = fs::read(&blob.path).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to read ADAPTER file {}: {error}",
            blob.path.display()
        ))
    })?;
    let actual_digest = format!("sha256:{}", hex::encode(Sha256::digest(&body)));
    if actual_digest != blob.digest {
        return Err(GailError::invalid_config(format!(
            "ADAPTER file changed while preparing Ollama blob: {}",
            blob.path.display()
        )));
    }
    let response = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(body)
        .send()
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "Ollama API request failed while uploading adapter blob {}: {error}",
                blob.name
            ))
        })?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let text = response.text().await.unwrap_or_default();
    Err(GailError::invalid_config(format!(
        "Ollama adapter blob upload failed with HTTP {} for {}: {}",
        status.as_u16(),
        blob.name,
        truncate_chars(&text, 600)
    )))
}

async fn rotate_ollama_models(trainer: &TrainerConfig) -> Result<()> {
    let client = ollama_api_client();
    let output = ollama_api_get(&client, trainer, "tags").await?;
    let prefix = format!("{}:", trainer.model_prefix);
    let mut models = output
        .get("models")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entry| {
            entry
                .get("name")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
        })
        .filter(|name| name.starts_with(prefix.as_str()))
        .filter(|name| name != &trainer.model_alias)
        .collect::<Vec<_>>();
    models.sort_by(|a, b| b.cmp(a));
    let remove = models
        .into_iter()
        .skip(trainer.rotate_keep)
        .collect::<Vec<_>>();
    for model in remove {
        if let Err(error) = ollama_api_delete(&client, trainer, model.as_str()).await {
            tracing::warn!(model = %model, error = %error, "failed to delete stale Ollama snapshot model");
        }
    }
    Ok(())
}

async fn write_dataset(entries: &[llm_ledger::LedgerInteraction], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .await?;
    for entry in entries {
        let Some(response) = entry.response_text.as_deref() else {
            continue;
        };
        if response.trim().is_empty() {
            continue;
        }
        let mut messages = Vec::new();
        if let Some(system) = entry.system_prompt.as_deref()
            && !system.trim().is_empty()
        {
            messages.push(json!({
                "role": "system",
                "content": system,
            }));
        }
        messages.push(json!({
            "role": "user",
            "content": entry.prompt_text,
        }));
        messages.push(json!({
            "role": "assistant",
            "content": response,
        }));
        let line = json!({
            "messages": messages,
            "metadata": {
                "request_id": entry.request_id,
                "workflow": entry.workflow,
                "role": entry.role,
                "provider": entry.provider_resolved.clone().or(entry.provider_requested.clone()),
                "model": entry.model_resolved.clone().or(entry.model_requested.clone()),
                "request_category": entry.request_category,
                "status": entry.status,
                "latency_ms": entry.latency_ms,
            }
        });
        let mut rendered = serde_json::to_string(&line)?;
        rendered.push('\n');
        file.write_all(rendered.as_bytes()).await?;
    }
    file.flush().await?;
    Ok(())
}

async fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let mut rendered = serde_json::to_string_pretty(value)?;
    rendered.push('\n');
    fs::write(path, rendered).await?;
    Ok(())
}

fn active_snapshot_id(trainer: &TrainerConfig) -> Result<Option<String>> {
    let path = PathBuf::from(&trainer.output_root).join("active_snapshot.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Ok(None);
    };
    let value: Value = serde_json::from_str(&raw).map_err(|error| {
        GailError::invalid_config(format!("invalid active training snapshot pointer: {error}"))
    })?;
    let snapshot = value
        .get("snapshot_id")
        .and_then(Value::as_str)
        .filter(|value| {
            !value.is_empty()
                && value.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
                })
        })
        .ok_or_else(|| {
            GailError::invalid_config("active training snapshot pointer has an unsafe snapshot_id")
        })?;
    Ok(Some(snapshot.to_string()))
}

fn active_adapter_path(trainer: &TrainerConfig) -> Result<Option<PathBuf>> {
    let Some(snapshot) = active_snapshot_id(trainer)? else {
        return Ok(None);
    };
    let root = PathBuf::from(&trainer.output_root);
    let adapter = root.join("snapshots").join(snapshot).join("adapter");
    let resolved_root = std::fs::canonicalize(&root).map_err(|error| {
        GailError::invalid_config(format!("failed to resolve training output root: {error}"))
    })?;
    let resolved_adapter = std::fs::canonicalize(&adapter).map_err(|error| {
        GailError::invalid_config(format!("active training adapter is unavailable: {error}"))
    })?;
    if !resolved_adapter.starts_with(&resolved_root) || !resolved_adapter.is_dir() {
        return Err(GailError::invalid_config(
            "active training adapter escapes the training output root",
        ));
    }
    Ok(Some(resolved_adapter))
}

async fn publish_active_snapshot(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    snapshot_dir: &Path,
    serving_target: Option<&ServingTargetSelection>,
) -> Result<()> {
    let adapter = snapshot_dir.join("adapter");
    let metadata = fs::metadata(&adapter).await.map_err(|error| {
        GailError::invalid_config(format!("trained snapshot adapter is unavailable: {error}"))
    })?;
    if !metadata.is_dir() {
        return Err(GailError::invalid_config(
            "trained snapshot adapter is not a directory",
        ));
    }
    let output_root = PathBuf::from(&trainer.output_root);
    let pointer = output_root.join("active_snapshot.json");
    let temporary = output_root.join(format!(".active_snapshot.json-{}", std::process::id()));
    let value = json!({
        "version": 1,
        "snapshot_id": snapshot_id,
        "adapter_path": format!("snapshots/{snapshot_id}/adapter"),
        "base_model": trainer.ollama_base_model,
        "model_alias": trainer.model_alias,
        "promoted_at": now_ts(),
        "serving_target": serving_target.map(|selection| json!({
            "host_id": selection.target.host_id,
            "endpoint": selection.target.endpoint,
            "model_alias": selection.target.model_alias,
            "base_model": selection.target.base_model,
            "vram_mb": selection.target.vram_mb,
            "throughput_tokens_per_second": selection.throughput_tokens_per_second,
        })),
    });
    write_json(&temporary, &value).await?;
    fs::rename(&temporary, &pointer).await.map_err(|error| {
        GailError::invalid_config(format!(
            "failed to atomically publish active training snapshot: {error}"
        ))
    })?;
    tracing::info!(snapshot = snapshot_id, pointer = %pointer.display(), "published cumulative Gail training base");
    Ok(())
}

async fn rollback_ollama_alias(
    trainer: &TrainerConfig,
    previous_snapshot: Option<&str>,
) -> Result<()> {
    let Some(previous_snapshot) = previous_snapshot else {
        return Ok(());
    };
    ollama_api_post(
        &ollama_api_client(),
        trainer,
        "copy",
        &json!({
            "source": format!("{}:{previous_snapshot}", trainer.model_prefix),
            "destination": trainer.model_alias,
        }),
    )
    .await
    .map(|_| ())
}

fn render_training_command(
    template: &str,
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
    resume_adapter: Option<&Path>,
) -> String {
    template
        .replace("{snapshot}", snapshot_id)
        .replace("{dataset}", &dataset_path.to_string_lossy())
        .replace("{output}", &snapshot_dir.to_string_lossy())
        .replace("{algorithm}", trainer.algorithm.as_str())
        .replace(
            "{device}",
            if hardware.gpu_count() > 0 {
                "cuda"
            } else {
                "cpu"
            },
        )
        .replace(
            "{cpu_threads}",
            &hardware.preferred_worker_threads().to_string(),
        )
        .replace("{gpu_count}", &hardware.gpu_count().to_string())
        .replace(
            "{resume_adapter}",
            &resume_adapter
                .map(|path| shell_escape(&path.to_string_lossy()))
                .unwrap_or_default(),
        )
}

fn default_model_module_path(trainer: &TrainerConfig, base_model: &str) -> PathBuf {
    let path = PathBuf::from(base_model);
    if path.is_file() || base_model.trim().ends_with(".pt") {
        return path;
    }
    if path.is_dir() {
        return path.join("model_train.pt");
    }
    torchscript_cache_root(trainer, base_model).join("model_train.pt")
}

fn default_tokenizer_path(trainer: &TrainerConfig, base_model: &str) -> PathBuf {
    let path = PathBuf::from(base_model);
    if path.is_dir() {
        return path.join("tokenizer.json");
    }
    if path.is_file() {
        return path
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .unwrap_or_else(|| PathBuf::from("tokenizer.json"));
    }
    torchscript_cache_root(trainer, base_model).join("tokenizer.json")
}

fn torchscript_cache_root(trainer: &TrainerConfig, base_model: &str) -> PathBuf {
    let sanitized = sanitize_path_component(base_model);
    PathBuf::from(trainer.output_root.as_str())
        .join("torchscript")
        .join(sanitized)
}

fn sanitize_path_component(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            rendered.push(ch);
        } else {
            rendered.push('_');
        }
    }
    let trimmed = rendered.trim_matches('_');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit.max(1)).collect()
}

#[derive(Default)]
struct ParsedModelfile {
    from: Option<String>,
    system: Option<String>,
    parameters: Map<String, Value>,
    adapters: Vec<String>,
}

fn build_ollama_create_payload_from_modelfile(
    trainer: &TrainerConfig,
    tagged_model: &str,
    snapshot_id: &str,
    modelfile: &str,
) -> Value {
    let parsed = parse_modelfile(modelfile);
    let from = parsed
        .from
        .unwrap_or_else(|| trainer.ollama_base_model.clone());
    let system = parsed.system.unwrap_or_else(|| {
        format!("You are the Gail in-house continuously trained model snapshot {snapshot_id}.")
    });
    let mut payload = json!({
        "model": tagged_model,
        "from": from,
        "stream": false,
    });
    if !system.trim().is_empty() {
        payload["system"] = json!(system);
    }
    if !parsed.parameters.is_empty() {
        payload["parameters"] = Value::Object(parsed.parameters);
    }
    payload
}

fn parse_modelfile(modelfile: &str) -> ParsedModelfile {
    let mut parsed = ParsedModelfile::default();
    for line in modelfile.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let Some(directive) = parts.next() else {
            continue;
        };
        let rest = parts.next().unwrap_or_default().trim();
        if rest.is_empty() {
            continue;
        }
        if directive.eq_ignore_ascii_case("FROM") {
            parsed.from = Some(rest.to_string());
            continue;
        }
        if directive.eq_ignore_ascii_case("SYSTEM") {
            parsed.system = Some(unquote_modelfile_value(rest));
            continue;
        }
        if directive.eq_ignore_ascii_case("ADAPTER") {
            let adapter = unquote_modelfile_value(rest);
            if !adapter.trim().is_empty() {
                parsed.adapters.push(adapter);
            }
            continue;
        }
        if directive.eq_ignore_ascii_case("PARAMETER") {
            let mut parameter_parts = rest.splitn(2, char::is_whitespace);
            let key = parameter_parts.next().unwrap_or_default().trim();
            let value = parameter_parts.next().unwrap_or_default().trim();
            if key.is_empty() || value.is_empty() {
                continue;
            }
            parsed
                .parameters
                .insert(key.to_string(), parse_modelfile_parameter_value(value));
        }
    }
    parsed
}

fn parse_modelfile_parameter_value(value: &str) -> Value {
    let normalized = unquote_modelfile_value(value);
    let lowered = normalized.to_ascii_lowercase();
    if lowered == "true" {
        return json!(true);
    }
    if lowered == "false" {
        return json!(false);
    }
    if let Ok(parsed) = normalized.parse::<i64>() {
        return json!(parsed);
    }
    if let Ok(parsed) = normalized.parse::<f64>() {
        return json!(parsed);
    }
    json!(normalized)
}

fn unquote_modelfile_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        let first = bytes[0] as char;
        let last = bytes[trimmed.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return trimmed[1..trimmed.len() - 1].to_string();
        }
    }
    trimmed.to_string()
}

fn ollama_api_client() -> Client {
    Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .tcp_keepalive(Duration::from_secs(30))
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn ollama_base_url(trainer: &TrainerConfig) -> String {
    trainer
        .ollama_host
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .or_else(|| std::env::var("GAIL_OLLAMA_BASE_URL").ok())
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "http://localhost:11434".to_string())
}

async fn ollama_api_post(
    client: &Client,
    trainer: &TrainerConfig,
    path: &str,
    payload: &serde_json::Value,
) -> Result<serde_json::Value> {
    let base_url = ollama_base_url(trainer);
    let url = format!("{base_url}/api/{path}");
    let response = client
        .post(url.as_str())
        .json(payload)
        .send()
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "Ollama API request failed for /api/{path}: {error}"
            ))
        })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        GailError::invalid_config(format!("failed to read Ollama API response: {error}"))
    })?;
    let parsed = serde_json::from_str::<serde_json::Value>(text.as_str())
        .unwrap_or_else(|_| json!({ "message": text }));
    if status.is_success() {
        return Ok(parsed);
    }
    Err(GailError::invalid_config(format!(
        "Ollama API /api/{path} failed with HTTP {}: {}",
        status.as_u16(),
        truncate_chars(&parsed.to_string(), 600)
    )))
}

async fn ollama_api_get(
    client: &Client,
    trainer: &TrainerConfig,
    path: &str,
) -> Result<serde_json::Value> {
    let base_url = ollama_base_url(trainer);
    let url = format!("{base_url}/api/{path}");
    let response = client.get(url.as_str()).send().await.map_err(|error| {
        GailError::invalid_config(format!(
            "Ollama API request failed for /api/{path}: {error}"
        ))
    })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        GailError::invalid_config(format!("failed to read Ollama API response: {error}"))
    })?;
    let parsed = serde_json::from_str::<serde_json::Value>(text.as_str())
        .unwrap_or_else(|_| json!({ "message": text }));
    if status.is_success() {
        return Ok(parsed);
    }
    Err(GailError::invalid_config(format!(
        "Ollama API /api/{path} failed with HTTP {}: {}",
        status.as_u16(),
        truncate_chars(&parsed.to_string(), 600)
    )))
}

async fn ollama_api_delete(client: &Client, trainer: &TrainerConfig, model: &str) -> Result<()> {
    let payload = json!({ "model": model });
    let base_url = ollama_base_url(trainer);
    let url = format!("{base_url}/api/delete");
    let response = client
        .delete(url.as_str())
        .json(&payload)
        .send()
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "Ollama API request failed for /api/delete: {error}"
            ))
        })?;
    let status = response.status();
    let text = response.text().await.map_err(|error| {
        GailError::invalid_config(format!("failed to read Ollama API response: {error}"))
    })?;
    let parsed = serde_json::from_str::<serde_json::Value>(text.as_str())
        .unwrap_or_else(|_| json!({ "message": text }));
    if status.is_success() {
        return Ok(());
    }
    let error_message = parsed.to_string();
    if status.as_u16() == 405
        || error_message
            .to_ascii_lowercase()
            .contains("method not allowed")
    {
        ollama_api_post(client, trainer, "delete", &payload).await?;
        return Ok(());
    }
    Err(GailError::invalid_config(format!(
        "Ollama API /api/delete failed with HTTP {}: {}",
        status.as_u16(),
        truncate_chars(&error_message, 600)
    )))
}

fn snapshot_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("{ts}")
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_slurm_result_exit_code_requires_valid_result() {
        assert_eq!(
            terminal_slurm_result_exit_code(r#"{"exit_code":0}"#),
            Some(0)
        );
        assert_eq!(
            terminal_slurm_result_exit_code(r#"{"exit_code":75}"#),
            Some(75)
        );
        assert_eq!(
            terminal_slurm_result_exit_code(r#"{"state":"completed"}"#),
            None
        );
        assert_eq!(terminal_slurm_result_exit_code("not-json"), None);
    }

    #[test]
    fn parse_modelfile_extracts_from_system_and_parameters() {
        let parsed = parse_modelfile(
            r#"
            # comment
            FROM qwen2.5-coder:1.5b
            ADAPTER ./adapter
            PARAMETER temperature 0.2
            PARAMETER num_ctx 4096
            PARAMETER mirostat true
            SYSTEM "hello world"
            "#,
        );
        assert_eq!(parsed.from.as_deref(), Some("qwen2.5-coder:1.5b"));
        assert_eq!(parsed.adapters, vec!["./adapter".to_string()]);
        assert_eq!(parsed.system.as_deref(), Some("hello world"));
        assert_eq!(parsed.parameters.get("temperature"), Some(&json!(0.2)));
        assert_eq!(parsed.parameters.get("num_ctx"), Some(&json!(4096)));
        assert_eq!(parsed.parameters.get("mirostat"), Some(&json!(true)));
    }

    #[test]
    fn build_ollama_create_payload_prefers_modelfile_directives() {
        let trainer = TrainerConfig {
            ollama_base_model: "fallback-model:latest".to_string(),
            ..TrainerConfig::default()
        };
        let payload = build_ollama_create_payload_from_modelfile(
            &trainer,
            "gail-inhouse:test",
            "123",
            "FROM qwen2.5-coder:1.5b\nSYSTEM tuned system\nPARAMETER temperature 0.2\n",
        );
        assert_eq!(payload["model"], json!("gail-inhouse:test"));
        assert_eq!(payload["from"], json!("qwen2.5-coder:1.5b"));
        assert_eq!(payload["system"], json!("tuned system"));
        assert_eq!(payload["parameters"]["temperature"], json!(0.2));
    }

    #[test]
    fn build_ollama_create_payload_uses_defaults_when_modelfile_is_sparse() {
        let trainer = TrainerConfig {
            ollama_base_model: "fallback-model:latest".to_string(),
            ..TrainerConfig::default()
        };
        let payload =
            build_ollama_create_payload_from_modelfile(&trainer, "gail-inhouse:test", "456", "");
        assert_eq!(payload["from"], json!("fallback-model:latest"));
        assert_eq!(
            payload["system"],
            json!("You are the Gail in-house continuously trained model snapshot 456.")
        );
        assert!(payload.get("parameters").is_none());
    }

    #[tokio::test]
    async fn ollama_adapter_manifest_maps_relative_names_and_digests() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let snapshot = temporary.path().join("snapshot");
        let adapter = snapshot.join("adapter");
        std::fs::create_dir_all(adapter.join("nested")).expect("adapter directory");
        std::fs::write(adapter.join("adapter_config.json"), b"{\"rank\":8}\n")
            .expect("adapter config");
        std::fs::write(adapter.join("nested/adapter_model.safetensors"), b"weights")
            .expect("adapter weights");

        let manifest = build_ollama_adapter_manifest(&snapshot, &["./adapter".to_string()])
            .await
            .expect("adapter manifest");
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[0].name, "adapter_config.json");
        assert_eq!(
            manifest[0].digest,
            format!("sha256:{}", hex::encode(Sha256::digest(b"{\"rank\":8}\n")))
        );
        assert_eq!(manifest[1].name, "nested/adapter_model.safetensors");
        assert_eq!(
            manifest[1].digest,
            format!("sha256:{}", hex::encode(Sha256::digest(b"weights")))
        );

        let trainer = TrainerConfig {
            ollama_base_model: "qwen2.5-coder:1.5b".to_string(),
            ..TrainerConfig::default()
        };
        let mut payload = build_ollama_create_payload_from_modelfile(
            &trainer,
            "gail-inhouse:test",
            "789",
            "FROM qwen2.5-coder:1.5b\nADAPTER ./adapter\n",
        );
        payload["adapters"] = Value::Object(
            manifest
                .iter()
                .map(|blob| (blob.name.clone(), json!(blob.digest)))
                .collect(),
        );
        assert_eq!(payload["from"], json!("qwen2.5-coder:1.5b"));
        assert_eq!(
            payload["adapters"]["adapter_config.json"],
            json!(manifest[0].digest)
        );
    }

    #[tokio::test]
    async fn active_training_marker_refresh_preserves_snapshot_start_time() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let trainer = TrainerConfig {
            output_root: temporary.path().display().to_string(),
            ..TrainerConfig::default()
        };
        write_active_training_marker(&trainer, "snapshot-1", &[42], "submitted")
            .await
            .expect("write marker");
        let before = serde_json::from_str::<Value>(
            &std::fs::read_to_string(active_training_marker_path(&trainer)).expect("read marker"),
        )
        .expect("valid marker");
        refresh_active_training_marker(&trainer, "snapshot-1", &[42])
            .await
            .expect("refresh marker");
        let after = serde_json::from_str::<Value>(
            &std::fs::read_to_string(active_training_marker_path(&trainer)).expect("read marker"),
        )
        .expect("valid refreshed marker");
        assert_eq!(after["snapshot_id"], json!("snapshot-1"));
        assert_eq!(after["started_ts"], before["started_ts"]);
        assert!(
            after["heartbeat_ts"].as_f64().unwrap_or(0.0)
                >= before["heartbeat_ts"].as_f64().unwrap_or(0.0)
        );
    }

    #[tokio::test]
    async fn ollama_adapter_manifest_rejects_path_traversal() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let snapshot = temporary.path().join("snapshot");
        std::fs::create_dir_all(&snapshot).expect("snapshot directory");
        std::fs::write(temporary.path().join("outside.safetensors"), b"outside")
            .expect("outside file");

        let error =
            build_ollama_adapter_manifest(&snapshot, &["../outside.safetensors".to_string()])
                .await
                .expect_err("path traversal must fail");
        assert!(error.to_string().contains("unsafe ADAPTER path"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ollama_adapter_manifest_rejects_symlink_escape() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let snapshot = temporary.path().join("snapshot");
        let adapter = snapshot.join("adapter");
        std::fs::create_dir_all(&adapter).expect("adapter directory");
        let outside = temporary.path().join("outside.safetensors");
        std::fs::write(&outside, b"outside").expect("outside file");
        std::os::unix::fs::symlink(&outside, adapter.join("escaped.safetensors"))
            .expect("adapter symlink");

        let error = build_ollama_adapter_manifest(&snapshot, &["adapter".to_string()])
            .await
            .expect_err("symlink escape must fail");
        assert!(error.to_string().contains("escapes the training snapshot"));
    }

    #[test]
    fn lora_registration_requires_an_adapter_artifact() {
        let trainer = TrainerConfig {
            algorithm: "qlora_sft".to_string(),
            ..TrainerConfig::default()
        };
        let missing = ParsedModelfile::default();
        let error = validate_registration_artifacts(&trainer, &missing)
            .expect_err("LoRA without adapter must not promote the base model");
        assert!(error.to_string().contains("without an ADAPTER artifact"));

        let with_adapter = ParsedModelfile {
            adapters: vec!["./adapter".to_string()],
            ..ParsedModelfile::default()
        };
        validate_registration_artifacts(&trainer, &with_adapter)
            .expect("adapter-backed LoRA may be registered");
    }

    #[test]
    fn qwen35_safetensors_registration_requires_conversion() {
        let trainer = TrainerConfig {
            ollama_base_model: "qwen3.5:4b".to_string(),
            ..TrainerConfig::default()
        };
        let mut parsed = ParsedModelfile {
            adapters: vec!["./adapter".to_string()],
            ..ParsedModelfile::default()
        };
        let temporary = tempfile::tempdir().expect("temporary directory");
        let snapshot = temporary.path().join("snapshot");
        std::fs::create_dir_all(snapshot.join("adapter")).expect("adapter directory");
        std::fs::write(
            snapshot.join("adapter/adapter_model.safetensors"),
            b"fixture",
        )
        .expect("adapter weights");
        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let error = runtime
            .block_on(prepare_ollama_adapter(
                &trainer,
                "snapshot",
                snapshot.as_path(),
                &mut parsed,
            ))
            .expect_err("Qwen3.5 Safetensors must not be sent to Ollama");
        assert!(
            error
                .to_string()
                .contains("cannot be registered as Safetensors")
        );
    }

    #[test]
    fn adapter_conversion_command_quotes_paths_and_replaces_placeholders() {
        let command = render_adapter_conversion_command(
            "convert --input {adapter} --output {output} --base {base_model} --id {snapshot_id}",
            Path::new("/tmp/a path/adapter"),
            Path::new("/tmp/a path/adapter.gguf"),
            "Qwen/Qwen3.5-4B-Base",
            Path::new("/tmp/a path"),
            "1786023597",
        );
        assert!(command.contains("'/tmp/a path/adapter'"));
        assert!(command.contains("Qwen/Qwen3.5-4B-Base"));
        assert!(!command.contains('{'));
    }

    #[test]
    fn build_training_execution_plan_cpu_uses_arm_profile() {
        let trainer = TrainerConfig {
            algorithm: "lora_sft".to_string(),
            ..TrainerConfig::default()
        };
        let hardware = HardwareProfile {
            cpu_cores: 46,
            cpu_arch: "aarch64".to_string(),
            cpu_model: Some("Qualcomm Centriq 2400".to_string()),
            total_memory_mb: 64 * 1024,
            available_memory_mb: 48 * 1024,
            gpus: Vec::new(),
        };
        let plan = build_training_execution_plan(&trainer, &hardware);
        assert_eq!(plan.device, "cpu");
        assert_eq!(plan.backend, "cpu_lora");
        assert_eq!(plan.profile, "centriq_cpu_armv8");
        assert_eq!(plan.gpu_count, 0);
        assert!(plan.dynamic_padding);
        assert!(plan.sequence_packing);
    }

    #[test]
    fn build_training_execution_plan_gpu_uses_cuda_qlora_backend() {
        let trainer = TrainerConfig {
            algorithm: "qlora_sft".to_string(),
            ..TrainerConfig::default()
        };
        let hardware = HardwareProfile {
            cpu_cores: 46,
            cpu_arch: "aarch64".to_string(),
            cpu_model: Some("Qualcomm Centriq 2400".to_string()),
            total_memory_mb: 64 * 1024,
            available_memory_mb: 48 * 1024,
            gpus: vec![crate::hardware::GpuDevice {
                index: 0,
                name: "NVIDIA GeForce RTX 3060".to_string(),
                memory_mb: 12_288,
                free_memory_mb: 11_000,
                compute_capability: Some("8.6".to_string()),
            }],
        };
        let plan = build_training_execution_plan(&trainer, &hardware);
        assert_eq!(plan.device, "cuda");
        assert_eq!(plan.backend, "cuda_qlora");
        assert_eq!(plan.profile, "centriq_rtx3060_12gb");
        assert_eq!(plan.gpu_count, 1);
        assert_eq!(plan.gpu_memory_mb, 12_288);
        assert_eq!(plan.gpu_free_memory_mb, 11_000);
    }

    fn target(host_id: &str, vram_mb: u64, throughput: f64) -> ServingTargetSelection {
        ServingTargetSelection {
            target: TrainerServingTarget {
                host_id: host_id.to_string(),
                endpoint: format!("http://{host_id}:18080/v1"),
                model_alias: None,
                base_model: "qwen3.5:4b".to_string(),
                vram_mb,
                enabled: true,
            },
            throughput_tokens_per_second: throughput,
        }
    }

    #[test]
    fn trained_model_target_prefers_largest_vram() {
        let mut targets = vec![target("fast-cpu", 0, 900.0), target("sm00", 16_384, 30.0)];
        rank_serving_targets(&mut targets);
        assert_eq!(targets[0].target.host_id, "sm00");
    }

    #[test]
    fn trained_model_target_uses_throughput_for_equal_vram_and_cpu() {
        let mut targets = vec![target("slow", 0, 4.0), target("fast", 0, 42.0)];
        rank_serving_targets(&mut targets);
        assert_eq!(targets[0].target.host_id, "fast");
    }

    #[test]
    fn trained_model_target_tie_is_deterministic() {
        let mut targets = vec![target("sm01", 12_288, 10.0), target("sm00", 12_288, 10.0)];
        rank_serving_targets(&mut targets);
        assert_eq!(targets[0].target.host_id, "sm00");
    }

    #[test]
    fn trained_model_target_matches_url_metrics_to_logical_host() {
        let mut target = target("sm01", 12_288, 0.0);
        target.target.endpoint = "http://192.168.1.68:18080/v1".to_string();
        assert!(serving_target_metric_matches(
            &target.target,
            Some("http://192.168.1.68:18080/v1"),
            "openai/qwen3.5:9b@192_168_1_68_18080_v1"
        ));
    }
}
