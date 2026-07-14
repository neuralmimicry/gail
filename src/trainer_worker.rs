use std::{
    env,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use reqwest::Client;
use serde::Serialize;
use serde_json::{Map, Value, json};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
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
) -> Result<TrainingOutcome> {
    fs::create_dir_all(snapshot_dir).await.map_err(|error| {
        GailError::invalid_config(format!("failed to create snapshot output path: {error}"))
    })?;
    let execution_plan = build_training_execution_plan(trainer, hardware);
    let artifact_mode = training_artifact_mode();
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
    });
    let training_invocation =
        resolve_training_invocation(trainer, hardware, snapshot_id, dataset_path, snapshot_dir)
            .await?;
    let mut training_executed = false;
    if let Some(command_line) = training_invocation {
        let command_output = execute_training_command(
            command_line.as_str(),
            trainer,
            hardware,
            &execution_plan,
            snapshot_id,
            dataset_path,
            snapshot_dir,
        )
        .await?;
        pipeline_report["training_command"] = json!(command_line);
        pipeline_report["training_stdout_tail"] = json!(command_output.stdout);
        pipeline_report["training_stderr_tail"] = json!(command_output.stderr);
        pipeline_report["training_exit_code"] = json!(command_output.exit_code);
        pipeline_report["training_runtime_seconds"] = json!(command_output.runtime_seconds);
        training_executed = true;
    } else {
        pipeline_report["training_command"] = json!(
            "skipped: trainer command unresolved (unsupported algorithm, command_template unset, or Rust qlora model artifacts missing)"
        );
    }
    let mut snapshot_tag = format!("{}:{}", trainer.model_prefix, snapshot_id);
    if trainer.register_with_ollama && training_executed {
        register_snapshot_with_ollama(trainer, snapshot_id, snapshot_dir).await?;
        rotate_ollama_models(trainer).await?;
        snapshot_tag = trainer.model_alias.clone();
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
    Ok(TrainingOutcome {
        snapshot_tag,
        status: if pipeline_report
            .get("training_exit_code")
            .and_then(|value| value.as_i64())
            .unwrap_or(-1)
            == 0
        {
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
    runtime_seconds: f64,
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
) -> Result<Option<String>> {
    if let Some(command_template) = trainer.command_template.as_deref() {
        return Ok(Some(render_training_command(
            command_template,
            trainer,
            hardware,
            snapshot_id,
            dataset_path,
            snapshot_dir,
        )));
    }

    if matches!(trainer.algorithm.as_str(), "qlora_sft" | "lora_sft") {
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
            "{} --dataset {} --output {} --algorithm {} --base-model {} --model-module {} --tokenizer {} --timeout-seconds {}",
            shell_escape(runner.as_str()),
            shell_escape(&dataset_path.to_string_lossy()),
            shell_escape(&snapshot_dir.to_string_lossy()),
            shell_escape(trainer.algorithm.as_str()),
            shell_escape(base_model.as_str()),
            shell_escape(&model_module.to_string_lossy()),
            shell_escape(&tokenizer.to_string_lossy()),
            trainer.command_timeout_seconds.max(1),
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
    let create_payload = build_ollama_create_payload_from_modelfile(
        trainer,
        tagged_model.as_str(),
        snapshot_id,
        &modelfile,
    );
    let parsed_modelfile = parse_modelfile(&modelfile);
    let requires_modelfile_adapter = !parsed_modelfile.adapters.is_empty();
    let client = ollama_api_client();
    if requires_modelfile_adapter {
        ollama_api_post(
            &client,
            trainer,
            "create",
            &json!({
                "model": tagged_model.as_str(),
                "modelfile": modelfile.as_str(),
                "stream": false
            }),
        )
        .await
        .map_err(|error| {
            GailError::invalid_config(format!(
                "Ollama API /api/create failed for ADAPTER Modelfile payload: {error}"
            ))
        })?;
    } else if let Err(primary_error) =
        ollama_api_post(&client, trainer, "create", &create_payload).await
    {
        let primary_error_text = primary_error.to_string();
        tracing::warn!(
            model = %tagged_model,
            error = %primary_error_text,
            "Ollama create via from-based payload failed; retrying with Modelfile payload"
        );
        if let Err(fallback_error) = ollama_api_post(
            &client,
            trainer,
            "create",
            &json!({
                "model": tagged_model.as_str(),
                "modelfile": modelfile.as_str(),
                "stream": false
            }),
        )
        .await
        {
            return Err(GailError::invalid_config(format!(
                "Ollama API /api/create failed with both payload styles: from-based={primary_error_text}; modelfile-based={fallback_error}"
            )));
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
