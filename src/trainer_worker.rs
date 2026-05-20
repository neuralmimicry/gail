use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use reqwest::Client;
use serde_json::json;
use tokio::{
    fs::{self, OpenOptions},
    io::AsyncWriteExt,
    process::Command,
};

use crate::{
    config::{GailConfig, TrainerConfig},
    errors::{GailError, Result},
    hardware::{HardwareProfile, detect_hardware, log_hardware_profile},
    llm_ledger,
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
    let trainer = config.trainer.clone();
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
    let poll_interval = Duration::from_secs(trainer.poll_interval_seconds);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("trainer worker received shutdown signal");
                break;
            }
            _ = tokio::time::sleep(poll_interval) => {}
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
            for entry in entries {
                let _ = llm_ledger::mark_training_retry(
                    &dsn,
                    entry.id,
                    format!("dataset_write_failed: {error}").as_str(),
                    trainer.max_attempts,
                    trainer.retry_backoff_seconds,
                )
                .await;
            }
            continue;
        }
        let ids = entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
        let train_outcome = run_training_pipeline(
            &trainer,
            &hardware,
            &snapshot_id,
            dataset_path.as_path(),
            snapshot_dir.as_path(),
        )
        .await;
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
                for id in ids {
                    let _ = llm_ledger::mark_training_retry(
                        &dsn,
                        id,
                        error_text.as_str(),
                        trainer.max_attempts,
                        trainer.retry_backoff_seconds,
                    )
                    .await;
                }
            }
        }
    }
    Ok(())
}

struct TrainingOutcome {
    snapshot_tag: String,
    status: String,
}

async fn run_training_pipeline(
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
) -> Result<TrainingOutcome> {
    fs::create_dir_all(snapshot_dir).await.map_err(|error| {
        GailError::invalid_config(format!("failed to create snapshot output path: {error}"))
    })?;
    let mut pipeline_report = json!({
        "snapshot_id": snapshot_id,
        "algorithm": trainer.algorithm,
        "dataset_path": dataset_path.to_string_lossy().to_string(),
        "snapshot_dir": snapshot_dir.to_string_lossy().to_string(),
        "cpu_cores": hardware.cpu_cores,
        "gpu_count": hardware.gpu_count(),
        "gpu_memory_mb": hardware.total_gpu_memory_mb(),
        "started_ts": now_ts(),
    });
    if let Some(command_template) = trainer.command_template.as_deref() {
        let command_line = render_training_command(
            command_template,
            trainer,
            hardware,
            snapshot_id,
            dataset_path,
            snapshot_dir,
        );
        let command_output = execute_training_command(
            command_line.as_str(),
            trainer,
            hardware,
            snapshot_id,
            dataset_path,
            snapshot_dir,
        )
        .await?;
        pipeline_report["training_command"] = json!(command_line);
        pipeline_report["training_stdout"] = json!(command_output.stdout);
        pipeline_report["training_stderr"] = json!(command_output.stderr);
        pipeline_report["training_exit_code"] = json!(command_output.exit_code);
    } else {
        pipeline_report["training_command"] = json!("skipped (trainer.command_template is unset)");
    }
    let mut snapshot_tag = format!("{}:{}", trainer.model_prefix, snapshot_id);
    if trainer.register_with_ollama {
        register_snapshot_with_ollama(trainer, snapshot_id, snapshot_dir).await?;
        rotate_ollama_models(trainer).await?;
        snapshot_tag = trainer.model_alias.clone();
    }
    pipeline_report["snapshot_tag"] = json!(snapshot_tag);
    pipeline_report["finished_ts"] = json!(now_ts());
    write_json(
        snapshot_dir.join("pipeline.json").as_path(),
        &pipeline_report,
    )
    .await?;
    Ok(TrainingOutcome {
        snapshot_tag,
        status: if trainer.command_template.is_some() {
            "trained".to_string()
        } else {
            "snapshotted".to_string()
        },
    })
}

struct CommandOutcome {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

async fn execute_training_command(
    command_line: &str,
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
) -> Result<CommandOutcome> {
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
            hardware.preferred_worker_threads().to_string(),
        )
        .env(
            "GAIL_TRAIN_DEVICE",
            if hardware.gpu_count() > 0 {
                "cuda"
            } else {
                "cpu"
            },
        )
        .env("GAIL_TRAIN_GPU_COUNT", hardware.gpu_count().to_string())
        .env(
            "GAIL_TRAIN_GPU_MEMORY_MB",
            hardware.total_gpu_memory_mb().to_string(),
        );
    let child = command.spawn().map_err(|error| {
        GailError::invalid_config(format!("failed to spawn trainer command: {error}"))
    })?;
    let timeout_duration = Duration::from_secs(trainer.command_timeout_seconds);
    let output = match tokio::time::timeout(timeout_duration, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return Err(GailError::invalid_config(format!(
                "trainer command failed to execute: {error}"
            )));
        }
        Err(_) => {
            return Err(GailError::invalid_config(format!(
                "trainer command timed out after {}s",
                trainer.command_timeout_seconds
            )));
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);
    if !output.status.success() {
        return Err(GailError::invalid_config(format!(
            "trainer command exited with status {exit_code}: {}",
            truncate_chars(&stderr, 600)
        )));
    }
    Ok(CommandOutcome {
        stdout: truncate_chars(&stdout, 8_000),
        stderr: truncate_chars(&stderr, 8_000),
        exit_code,
    })
}

async fn register_snapshot_with_ollama(
    trainer: &TrainerConfig,
    snapshot_id: &str,
    snapshot_dir: &Path,
) -> Result<()> {
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
    let client = ollama_api_client();
    ollama_api_post(
        &client,
        trainer,
        "create",
        &json!({
            "model": tagged_model,
            "modelfile": modelfile,
            "stream": false
        }),
    )
    .await?;
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
    Ok(())
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
        let _ = ollama_api_post(
            &client,
            trainer,
            "delete",
            &json!({
                "model": model
            }),
        )
        .await;
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

fn render_training_command(
    template: &str,
    trainer: &TrainerConfig,
    hardware: &HardwareProfile,
    snapshot_id: &str,
    dataset_path: &Path,
    snapshot_dir: &Path,
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
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit.max(1)).collect()
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

fn snapshot_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("{ts}")
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
