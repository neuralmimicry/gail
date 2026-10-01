//! Durable file-queue worker for native ELM candidate production.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::Digest;
use tokio::{fs, signal, task};

use crate::{config::GailConfig, elm_config::ElmMode};

use super::{
    ElmError, LifecycleStage, ModelRegistry, Result,
    data::DatasetManifest,
    data::{TrainedEvaluation, train_and_evaluate},
    lifecycle::evaluate_qualification,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Trained,
    Evaluating,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElmTrainingJob {
    pub job_id: String,
    pub idempotency_key: String,
    pub manifest_path: PathBuf,
    pub task_id: String,
    pub dataset_revision: String,
    pub split_specification: String,
    pub search_space: String,
    pub compute_budget_seconds: u64,
    pub retry_limit: usize,
    pub state: JobState,
    pub retries: usize,
    pub created_at: u64,
    pub updated_at: u64,
    pub model_id: Option<String>,
    pub error: Option<String>,
}

pub async fn run(config: GailConfig) -> anyhow::Result<()> {
    if config.elm.mode == ElmMode::Off {
        tracing::warn!(
            "ELM trainer role selected while elm.mode=off; exiting without reading a dataset"
        );
        return Ok(());
    }
    let registry_path = config.elm.registry_path.clone();
    let registry = Arc::new(
        task::spawn_blocking(move || {
            ModelRegistry::open(registry_path, config.elm.lifecycle.artifact_max_bytes)
        })
        .await??,
    );
    let jobs_path = config.elm.registry_path.join("jobs");
    fs::create_dir_all(&jobs_path).await?;
    tracing::info!(path = %jobs_path.display(), "native ELM trainer worker started");
    loop {
        let advance_config = config.clone();
        let advance_registry = registry.clone();
        match task::spawn_blocking(move || {
            advance_shadow_models_to_canary(&advance_config, &advance_registry)
        })
        .await
        {
            Ok(Ok(promoted)) if promoted > 0 => {
                tracing::info!(
                    count = promoted,
                    "automatically advanced evidence-qualified ELM models to bounded canary"
                )
            }
            Ok(Ok(_)) => {}
            Ok(Err(error)) => tracing::warn!(error = %error, "ELM canary lifecycle scan failed"),
            Err(error) => tracing::warn!(error = %error, "ELM canary lifecycle task failed"),
        }
        discover_automatic_job(&config, &jobs_path).await;
        match process_next_job(&config, &registry, &jobs_path).await {
            Ok(true) => continue,
            Ok(false) => {}
            Err(error) => tracing::error!(error = %error, "ELM worker job scan failed"),
        }
        tokio::select! {
            _ = signal::ctrl_c() => {
                tracing::info!("ELM trainer worker received shutdown signal");
                break;
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {}
        }
    }
    Ok(())
}

/// Advance only the current, still-qualified shadow champion to a bounded
/// canary after its configured dwell. Canary exposure is capped by both the
/// task rollout and the durable per-model decision budget; this worker never
/// promotes a model directly to the unrestricted active stage.
fn advance_shadow_models_to_canary(config: &GailConfig, registry: &ModelRegistry) -> Result<usize> {
    registry.refresh()?;
    if !config.elm.lifecycle.automatic_promotion {
        return Ok(0);
    }
    let now = unix_seconds();
    let mut promoted = 0;
    for record in registry.records() {
        if record.stage != LifecycleStage::Shadow || record.fixture_only {
            continue;
        }
        let Some(task) = config.elm.tasks.get(&record.task_id) else {
            continue;
        };
        if !task.allowed
            || config.elm.effective_mode(&record.task_id) != ElmMode::Auto
            || task.max_stage < crate::elm_config::ModelStage::Canary
            || task.influence_cap <= 0.0
            || now.saturating_sub(record.updated_at) < config.elm.lifecycle.minimum_dwell_seconds
        {
            continue;
        }
        let Some(policy) = config.elm.promotion_policies.get(&task.promotion_policy) else {
            continue;
        };
        let canary_limit = policy.canary_max_decisions.unwrap_or(0) as u64;
        if !policy.is_complete()
            || task.rollout_fraction > policy.canary_fraction.unwrap_or(0.0)
            || policy.authorised_influence.unwrap_or(0.0) <= 0.0
            || canary_limit == 0
            || registry.canary_decisions(&record.model_id) >= canary_limit
        {
            continue;
        }
        if matches!(
            record.task_id.as_str(),
            "quant_net_edge" | "trading_advisory"
        ) && config.trading.live_execution_enabled
            && !config.elm.trading.allow_live_influence
        {
            continue;
        }
        let Some(champion) = registry.champion(&record.task_id) else {
            continue;
        };
        if champion.metadata.model_id != record.model_id || champion.content_sha256 != record.digest
        {
            continue;
        }
        let report_value = match registry.evaluation_json(&record.model_id) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let mut report: super::data::TrainingEvaluationReport =
            match serde_json::from_value(report_value) {
                Ok(value) => value,
                Err(_) => continue,
            };
        let recorded_hash = report.report_sha256.clone();
        report.report_sha256.clear();
        let Ok(canonical) = serde_json::to_vec(&report) else {
            continue;
        };
        if recorded_hash != hex::encode(sha2::Sha256::digest(canonical)) {
            continue;
        }
        report.report_sha256 = recorded_hash;
        if !report.qualified
            || report.fixture_only
            || report.task_id != record.task_id
            || report.dataset_digest != champion.metadata.dataset_digest
            || !evaluate_qualification(&config.elm, &record.task_id, &champion, &report).qualified
        {
            continue;
        }
        match registry.promote(
            &record.model_id,
            LifecycleStage::Canary,
            registry.revision(),
            "automatically entered bounded canary after verified qualification and shadow dwell",
        ) {
            Ok(_) => promoted += 1,
            Err(ElmError::RevisionConflict) => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(promoted)
}

async fn discover_automatic_job(config: &GailConfig, jobs_path: &Path) {
    if !config.elm.lifecycle.automatic_training || config.elm.mode == ElmMode::Off {
        return;
    }
    let Some(manifest_path) = config.elm.dataset_manifest.clone() else {
        return;
    };
    let max_bytes = config.elm.lifecycle.artifact_max_bytes;
    let manifest =
        task::spawn_blocking(move || DatasetManifest::load(&manifest_path, max_bytes)).await;
    let Ok(Ok(manifest)) = manifest else {
        return;
    };
    if !manifest.permitted
        || config.elm.effective_mode(&manifest.task_id) == ElmMode::Off
        || config
            .elm
            .tasks
            .get(&manifest.task_id)
            .is_none_or(|task| !task.allowed)
    {
        return;
    }
    let Ok(digest) = manifest.digest() else {
        return;
    };
    let key = format!("{}:{digest}", manifest.task_id);
    let filename = format!("auto-{}.json", &digest[..digest.len().min(24)]);
    let path = jobs_path.join(filename);
    if path.exists() {
        return;
    }
    let now = unix_seconds();
    let job = ElmTrainingJob {
        job_id: uuid::Uuid::new_v4().to_string(),
        idempotency_key: key,
        manifest_path: config.elm.dataset_manifest.clone().unwrap_or_default(),
        task_id: manifest.task_id,
        dataset_revision: digest,
        split_specification: "chronological_grouped_60_15_15_10_embargo_v1".into(),
        search_space: "elm_hidden=[32,64,128,256];baseline=ridge_or_logistic;lambda=1e-3".into(),
        compute_budget_seconds: config.elm.budgets.max_training_seconds,
        retry_limit: config.elm.budgets.max_training_retries,
        state: JobState::Queued,
        retries: 0,
        created_at: now,
        updated_at: now,
        model_id: None,
        error: None,
    };
    if let Err(error) = write_job(&path, &job).await {
        tracing::warn!(error = %error, "failed to enqueue automatic ELM job");
    } else {
        tracing::info!(task = %job.task_id, dataset_digest = %job.dataset_revision, "queued automatic ELM challenger job");
    }
}

async fn process_next_job(
    config: &GailConfig,
    registry: &Arc<ModelRegistry>,
    jobs_path: &Path,
) -> Result<bool> {
    let mut entries = fs::read_dir(jobs_path)
        .await
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    let mut paths = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|error| ElmError::Artifact(error.to_string()))?
    {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            paths.push(path);
        }
    }
    paths.sort();
    for path in paths {
        let metadata = fs::metadata(&path)
            .await
            .map_err(|error| ElmError::Artifact(error.to_string()))?;
        if metadata.len() as usize > config.elm.lifecycle.artifact_max_bytes {
            continue;
        }
        let bytes = fs::read(&path)
            .await
            .map_err(|error| ElmError::Artifact(error.to_string()))?;
        let mut job: ElmTrainingJob = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(path = %path.display(), error = %error, "ignoring invalid ELM job file");
                continue;
            }
        };
        if job.state == JobState::Running {
            job.state = JobState::Queued;
            job.error = Some("worker restarted; incomplete job safely re-queued".into());
            write_job(&path, &job).await?;
        }
        if job.state != JobState::Queued {
            continue;
        }
        if config.elm.effective_mode(&job.task_id) == ElmMode::Off
            || config
                .elm
                .tasks
                .get(&job.task_id)
                .is_none_or(|task| !task.allowed)
        {
            job.state = JobState::Cancelled;
            job.error = Some(
                "task is disabled or not permitted by the configured authority envelope".into(),
            );
            job.updated_at = unix_seconds();
            write_job(&path, &job).await?;
            return Ok(true);
        }
        if let Some(existing) = registry.find_dataset(&job.task_id, &job.dataset_revision)
            && matches!(
                existing.stage,
                LifecycleStage::Evaluated
                    | LifecycleStage::Qualified
                    | LifecycleStage::Shadow
                    | LifecycleStage::Canary
                    | LifecycleStage::Active
            )
        {
            job.state = JobState::Completed;
            job.model_id = Some(existing.model_id);
            job.error = None;
            job.updated_at = unix_seconds();
            write_job(&path, &job).await?;
            return Ok(true);
        }
        job.state = JobState::Running;
        job.updated_at = unix_seconds();
        job.error = None;
        write_job(&path, &job).await?;
        let manifest_path = job.manifest_path.clone();
        let budgets = config.elm.budgets.clone();
        let limits = super::data::TrainingLimits {
            max_samples: budgets.max_samples,
            max_features: budgets.max_features,
            max_hidden_units: budgets.max_hidden_units,
            max_memory_mb: budgets.max_training_memory_mb,
            max_training_threads: budgets.max_training_threads,
            max_training_seconds: budgets.max_training_seconds,
            minimum_label_age_seconds: config.elm.data.minimum_label_age_seconds,
        };
        let max_bytes = config.elm.lifecycle.artifact_max_bytes;
        let fit = task::spawn_blocking(move || {
            let manifest = DatasetManifest::load(&manifest_path, max_bytes)?;
            train_and_evaluate(&manifest, unix_seconds() as f64, &limits)
        })
        .await
        .map_err(|error| ElmError::Numerical(format!("training task failed: {error}")))?;
        match fit {
            Ok(result) => {
                job.state = JobState::Trained;
                job.updated_at = unix_seconds();
                write_job(&path, &job).await?;
                job.state = JobState::Evaluating;
                job.updated_at = unix_seconds();
                write_job(&path, &job).await?;
                let current_registry = registry.clone();
                task::spawn_blocking(move || current_registry.refresh())
                    .await
                    .map_err(|error| {
                        ElmError::Artifact(format!("registry refresh task failed: {error}"))
                    })??;
                finish_success(config, registry, &mut job, result, max_bytes)?;
                job.state = JobState::Completed;
                job.updated_at = unix_seconds();
                write_job(&path, &job).await?;
                tracing::info!(job_id = %job.job_id, task = %job.task_id, model_id = ?job.model_id, "ELM candidate training and evaluation completed");
            }
            Err(error) => {
                job.retries = job.retries.saturating_add(1);
                job.state = if job.retries > job.retry_limit {
                    JobState::Failed
                } else {
                    JobState::Queued
                };
                job.error = Some(error.to_string());
                job.updated_at = unix_seconds();
                write_job(&path, &job).await?;
                tracing::warn!(job_id = %job.job_id, retries = job.retries, error = %error, "ELM candidate training failed");
            }
        }
        return Ok(true);
    }
    Ok(false)
}

fn finish_success(
    config: &GailConfig,
    registry: &ModelRegistry,
    job: &mut ElmTrainingJob,
    result: TrainedEvaluation,
    max_bytes: usize,
) -> Result<()> {
    let model_id = result.artifact.metadata.model_id.clone();
    let mut result = result;
    let qualification =
        evaluate_qualification(&config.elm, &job.task_id, &result.artifact, &result.report);
    result.report.qualified = qualification.qualified;
    result.report.qualification_reasons = qualification.reason_codes.clone();
    result.report.report_sha256.clear();
    let canonical = serde_json::to_vec(&result.report)
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    result.report.report_sha256 = hex::encode(sha2::Sha256::digest(canonical));
    let report_bytes = serde_json::to_vec_pretty(&result.report)
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    let record = registry.publish_candidate(result.artifact, registry.revision())?;
    registry.store_evaluation(&model_id, &report_bytes)?;
    registry.mark_evaluated(
        &model_id,
        registry.revision(),
        if qualification.qualified {
            "independent final evaluation passed every configured evidence gate"
        } else {
            "evaluation completed; qualification withheld by evidence gates"
        },
    )?;
    if qualification.qualified && config.elm.lifecycle.automatic_promotion {
        registry.promote(
            &model_id,
            LifecycleStage::Qualified,
            registry.revision(),
            "automatically qualified by the configured evidence policy",
        )?;
        if config.elm.max_stage(&job.task_id) >= crate::elm_config::ModelStage::Shadow {
            registry.promote(
                &model_id,
                LifecycleStage::Shadow,
                registry.revision(),
                "automatically entered the authorised shadow stage",
            )?;
        }
    }
    if job.task_id != record.task_id || report_bytes.len() > max_bytes {
        return Err(ElmError::Artifact(
            "job artifact/report scope validation failed".into(),
        ));
    }
    job.model_id = Some(model_id);
    job.error = if qualification.qualified {
        None
    } else {
        Some(format!(
            "evaluation completed without qualification: {}",
            qualification.reason_codes.join(",")
        ))
    };
    Ok(())
}

async fn write_job(path: &Path, job: &ElmTrainingJob) -> Result<()> {
    let raw =
        serde_json::to_vec_pretty(job).map_err(|error| ElmError::Artifact(error.to_string()))?;
    let parent = path
        .parent()
        .ok_or_else(|| ElmError::Artifact("job path has no parent".into()))?;
    fs::create_dir_all(parent)
        .await
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    fs::write(&temporary, raw)
        .await
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    fs::rename(&temporary, path)
        .await
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_secs())
}

/// Build the canonical job envelope used by operators or a separate submitter.
pub fn new_job(
    manifest_path: PathBuf,
    task_id: String,
    digest: String,
    retry_limit: usize,
    compute_budget_seconds: u64,
) -> ElmTrainingJob {
    let now = unix_seconds();
    ElmTrainingJob {
        job_id: uuid::Uuid::new_v4().to_string(),
        idempotency_key: format!("{task_id}:{digest}"),
        manifest_path,
        task_id,
        dataset_revision: digest,
        split_specification: "chronological_grouped_60_15_15_10_embargo_v1".into(),
        search_space: "elm_hidden=[32,64,128,256];baseline=ridge_or_logistic;lambda=1e-3".into(),
        compute_budget_seconds,
        retry_limit,
        state: JobState::Queued,
        retries: 0,
        created_at: now,
        updated_at: now,
        model_id: None,
        error: None,
    }
}
