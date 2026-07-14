//! Native Rust LoRA/QLoRA-style supervised fine-tuning runner.
//!
//! This trainer is designed for mixed hardware fleets:
//! - ARM CPU-heavy hosts (for example Centriq 2400) where memory bandwidth,
//!   padding minimisation, and thread isolation matter more than GPU kernels.
//! - CUDA hosts with constrained VRAM (for example RTX 3060 12 GB) where
//!   pre-materialising all batches on device causes immediate pressure.
//!
//! Core design constraints:
//! 1. Keep base weights frozen and train only adapter-tagged tensors.
//! 2. Stream/prefetch CPU batches and transfer per-step instead of caching all
//!    padded batches on GPU.
//! 3. Record execution details and metrics in reproducible manifests.
//! 4. Export adapter tensors as safetensors (`adapter_model.safetensors`) with a
//!    matching `adapter_config.json` and `training_manifest.json`.

use std::{
    borrow::Cow,
    collections::HashMap,
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use safetensors::{
    serialize_to_file,
    tensor::{Dtype, View},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tch::{CModule, Cuda, Device, Kind, Tensor, autocast, no_grad};
use tokenizers::Tokenizer;
use tokio::fs;

const SUPPORTED_ALGORITHMS: &[&str] = &["qlora_sft", "lora_sft"];
const DEFAULT_SYSTEM_PROMPT: &str = "You are the Gail in-house continuously trained model. Use prior interaction learning when useful.";

#[derive(Debug, Clone, Serialize)]
struct TrainingConfig {
    dataset: PathBuf,
    output: PathBuf,
    base_model: String,
    model_module: PathBuf,
    tokenizer: PathBuf,
    algorithm: String,
    epochs: f64,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    learning_rate: f64,
    warmup_ratio: f64,
    max_seq_len: usize,
    lora_r: usize,
    lora_alpha: usize,
    lora_dropout: f64,
    weight_decay: f64,
    adam_beta1: f64,
    adam_beta2: f64,
    adam_eps: f64,
    max_grad_norm: f64,
    system_prompt: String,
    timeout_seconds: u64,
    trainable_marker: String,
    forward_mode: ForwardMode,
    seed: u64,
    prefetch_batches: usize,
    dynamic_padding: bool,
    sequence_packing: bool,
    loss_scale_initial: f64,
    loss_scale_growth_factor: f64,
    loss_scale_backoff_factor: f64,
    loss_scale_growth_interval: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ForwardMode {
    Loss,
    Logits,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ComputeDtype {
    Fp32,
    Fp16,
    Bf16,
}

impl ComputeDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TrainingBackendKind {
    CpuLora,
    CudaLora,
    CudaQlora,
}

impl TrainingBackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::CpuLora => "cpu_lora",
            Self::CudaLora => "cuda_lora",
            Self::CudaQlora => "cuda_qlora",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct TrainingExecutionPlan {
    profile: String,
    backend: TrainingBackendKind,
    device_label: String,
    device_index: Option<usize>,
    gpu_count: usize,
    cpu_intraop_threads: usize,
    cpu_interop_threads: usize,
    tokenizer_threads: usize,
    prefetch_batches: usize,
    compute_dtype: ComputeDtype,
    quantisation_backend: String,
    mixed_precision: bool,
    activation_checkpointing: bool,
    dynamic_padding: bool,
    sequence_packing: bool,
    micro_batch_size: usize,
    gradient_accumulation_steps: usize,
    max_sequence_length: usize,
}

#[derive(Debug, Deserialize)]
struct DatasetRow {
    messages: Option<Vec<ChatMessage>>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Debug, Clone)]
struct TokenisedSample {
    ids: Vec<i64>,
}

#[derive(Debug, Clone)]
struct BatchPlan {
    sample_indices: Vec<usize>,
}

#[derive(Debug)]
struct CpuBatch {
    input_ids: Tensor,
    labels: Tensor,
    sample_count: usize,
    sequence_length: usize,
    non_padding_tokens: usize,
    padding_tokens: usize,
    packed_sequences: usize,
}

#[derive(Debug)]
struct TrainableParam {
    name: String,
    tensor: Tensor,
    m: Tensor,
    v: Tensor,
}

#[derive(Debug, Serialize, Clone)]
struct TrainingRunResult {
    metrics: TrainingMetrics,
    executed_algorithm: String,
    exported_adapter_tensors: usize,
    final_loss_scale: Option<f64>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct TrainingMetrics {
    samples: usize,
    batches_per_epoch: usize,
    total_micro_steps: usize,
    total_optimizer_steps: usize,
    trainable_parameters: usize,
    trainable_tensors: Vec<String>,
    total_tokens: u64,
    non_padding_tokens: u64,
    padding_tokens: u64,
    packed_sequences: u64,
    final_loss: Option<f64>,
    final_grad_norm: Option<f64>,
    skipped_optimizer_steps: usize,
    clipped_optimizer_steps: usize,
    data_wait_seconds: f64,
    host_to_device_seconds: f64,
    forward_seconds: f64,
    backward_seconds: f64,
    optimizer_seconds: f64,
    runtime_seconds: f64,
    tokens_per_second: f64,
    non_padding_tokens_per_second: f64,
    padding_percentage: f64,
}

#[derive(Debug, Clone)]
struct BatchBuildSettings {
    max_seq_len: usize,
    pad_id: i64,
    separator_id: i64,
    dynamic_padding: bool,
    sequence_packing: bool,
}

#[derive(Debug)]
struct BatchPrefetcher {
    receiver: mpsc::Receiver<anyhow::Result<CpuBatch>>,
    producer: Option<thread::JoinHandle<()>>,
}

impl BatchPrefetcher {
    fn start(
        samples: Arc<Vec<TokenisedSample>>,
        schedule: Vec<BatchPlan>,
        settings: BatchBuildSettings,
        prefetch_batches: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(prefetch_batches.max(1));
        let producer = thread::Builder::new()
            .name("gail-batch-prefetch".to_string())
            .spawn(move || {
                for plan in schedule {
                    let batch = build_cpu_batch(samples.as_slice(), &plan, &settings);
                    if sender.send(batch).is_err() {
                        break;
                    }
                }
            })
            .ok();
        Self { receiver, producer }
    }

    fn next_batch(&self) -> anyhow::Result<Option<CpuBatch>> {
        match self.receiver.recv() {
            Ok(result) => result.map(Some),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for BatchPrefetcher {
    fn drop(&mut self) {
        if let Some(producer) = self.producer.take() {
            let _ = producer.join();
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct DynamicLossScaler {
    enabled: bool,
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    growth_tracker: usize,
    overflow_steps: usize,
}

impl DynamicLossScaler {
    fn new(cfg: &TrainingConfig, enabled: bool) -> Self {
        Self {
            enabled,
            scale: cfg.loss_scale_initial.max(1.0),
            growth_factor: cfg.loss_scale_growth_factor.max(1.01),
            backoff_factor: cfg.loss_scale_backoff_factor.clamp(0.1, 0.99),
            growth_interval: cfg.loss_scale_growth_interval.max(1),
            growth_tracker: 0,
            overflow_steps: 0,
        }
    }

    fn scale_loss(&self, loss: &Tensor, accumulation_steps: usize) -> Tensor {
        let accumulation = accumulation_steps.max(1) as f64;
        if self.enabled {
            (loss / accumulation) * self.scale
        } else {
            loss / accumulation
        }
    }

    fn unscale_gradients(&self, params: &mut [TrainableParam]) {
        if !self.enabled || self.scale <= 0.0 {
            return;
        }
        let inv = 1.0 / self.scale;
        for param in params {
            let mut grad = param.tensor.grad();
            if grad.defined() {
                let _ = grad.g_mul_scalar_(inv);
            }
        }
    }

    fn update(&mut self, found_inf: bool) {
        if !self.enabled {
            return;
        }
        if found_inf {
            self.overflow_steps = self.overflow_steps.saturating_add(1);
            self.growth_tracker = 0;
            self.scale = (self.scale * self.backoff_factor).max(1.0);
            return;
        }
        self.growth_tracker = self.growth_tracker.saturating_add(1);
        if self.growth_tracker >= self.growth_interval {
            self.growth_tracker = 0;
            self.scale = (self.scale * self.growth_factor).min(2_f64.powi(24));
        }
    }
}

#[derive(Debug)]
struct OwnedSafeTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedSafeTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(self.data.as_slice())
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct ClipOutcome {
    grad_norm: Option<f64>,
    clipped: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let code = match async_main().await {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Training failed: {error}");
            1
        }
    };
    std::process::exit(code);
}

async fn async_main() -> anyhow::Result<()> {
    let cfg = parse_args(env::args_os().skip(1))?;
    if !SUPPORTED_ALGORITHMS.contains(&cfg.algorithm.as_str()) {
        anyhow::bail!(
            "unsupported algorithm '{}'; supported: {}",
            cfg.algorithm,
            SUPPORTED_ALGORITHMS.join(", ")
        );
    }

    fs::create_dir_all(&cfg.output).await?;
    let started = std::time::Instant::now();
    let plan = build_execution_plan(&cfg);
    apply_cpu_thread_limits(&plan);

    let texts = load_training_texts(&cfg.dataset).await?;
    if texts.is_empty() {
        anyhow::bail!("dataset is empty after message parsing");
    }

    let rendered_dataset = cfg.output.join("dataset.rendered.jsonl");
    write_rendered_dataset(&rendered_dataset, &texts).await?;

    let cfg_for_training = cfg.clone();
    let plan_for_training = plan.clone();
    let run_result = tokio::task::spawn_blocking(move || {
        train_with_tch(&cfg_for_training, &plan_for_training, &texts)
    })
    .await??;

    let adapter_dir = cfg.output.join("adapter");
    fs::create_dir_all(&adapter_dir).await?;
    write_adapter_config(&cfg, &adapter_dir).await?;
    write_training_manifest(&cfg, &plan, &run_result, &adapter_dir).await?;

    let modelfile = cfg.output.join("Modelfile");
    write_modelfile(&cfg, &modelfile).await?;

    let report = json!({
        "algorithm_requested": cfg.algorithm,
        "algorithm_executed": run_result.executed_algorithm,
        "backend": plan.backend.as_str(),
        "base_model": cfg.base_model,
        "model_module": cfg.model_module.to_string_lossy(),
        "tokenizer": cfg.tokenizer.to_string_lossy(),
        "device": plan.device_label,
        "device_index": plan.device_index,
        "gpu_count": plan.gpu_count,
        "cpu_intraop_threads": plan.cpu_intraop_threads,
        "cpu_interop_threads": plan.cpu_interop_threads,
        "tokenizer_threads": plan.tokenizer_threads,
        "compute_dtype": plan.compute_dtype.as_str(),
        "quantisation_backend": plan.quantisation_backend,
        "execution_profile": plan.profile,
        "metrics": run_result.metrics,
        "exported_adapter_tensors": run_result.exported_adapter_tensors,
        "final_loss_scale": run_result.final_loss_scale,
        "rendered_dataset": rendered_dataset.canonicalize().unwrap_or(rendered_dataset).to_string_lossy(),
        "adapter_dir": adapter_dir.canonicalize().unwrap_or(adapter_dir).to_string_lossy(),
        "modelfile": modelfile.canonicalize().unwrap_or(modelfile).to_string_lossy(),
        "started_ts": now_ts() - started.elapsed().as_secs_f64(),
        "finished_ts": now_ts(),
        "runtime_seconds": started.elapsed().as_secs_f64(),
    });

    let report_path = cfg.output.join("training_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)? + "\n").await?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn train_with_tch(
    cfg: &TrainingConfig,
    plan: &TrainingExecutionPlan,
    texts: &[String],
) -> anyhow::Result<TrainingRunResult> {
    let device = tch_device(plan);
    let tokenizer = Tokenizer::from_file(&cfg.tokenizer)
        .map_err(|error| anyhow::anyhow!("failed to load tokenizer: {error}"))?;
    let pad_id = tokenizer
        .get_padding()
        .map(|padding| padding.pad_id as i64)
        .or_else(|| token_to_id(&tokenizer, "<|endoftext|>"))
        .or_else(|| token_to_id(&tokenizer, "</s>"))
        .or_else(|| token_to_id(&tokenizer, "<|eot_id|>"))
        .unwrap_or(0);
    let separator_id = token_to_id(&tokenizer, "</s>")
        .or_else(|| token_to_id(&tokenizer, "<|eot_id|>"))
        .unwrap_or(pad_id);

    let samples = tokenise_parallel(&tokenizer, texts, cfg.max_seq_len)?;
    if samples.is_empty() {
        anyhow::bail!("no trainable tokenised samples were produced");
    }
    let sample_count = samples.len();

    let batches_per_epoch = compute_batches_per_epoch(samples.len(), cfg.batch_size);
    let total_micro_steps =
        ((cfg.epochs.max(0.01) * batches_per_epoch as f64).ceil() as usize).max(1);
    let accumulation = cfg.gradient_accumulation_steps.max(1);
    let total_optimizer_steps = total_micro_steps.div_ceil(accumulation);
    let warmup_steps = ((cfg.warmup_ratio.max(0.0) * total_optimizer_steps as f64).round()
        as usize)
        .min(total_optimizer_steps);

    let batch_schedule = build_batch_schedule(
        samples.as_slice(),
        cfg.batch_size,
        total_micro_steps,
        cfg.seed,
    );
    if batch_schedule.is_empty() {
        anyhow::bail!("batch scheduler produced zero batches");
    }

    let mut module = CModule::load_on_device(&cfg.model_module, device)
        .map_err(|error| anyhow::anyhow!("failed to load TorchScript module: {error}"))?;
    module.set_train();

    let mut params = collect_trainable_parameters(&module, cfg)?;
    if params.is_empty() {
        anyhow::bail!(
            "no trainable tensors found. Expected parameter names containing '{}'; \
             set GAIL_TCH_TRAIN_ALL=1 only if this is intentional",
            cfg.trainable_marker
        );
    }
    let tensor_names = params.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
    let trainable_parameters = params
        .iter()
        .map(|param| param.tensor.numel())
        .sum::<usize>();

    let executed_algorithm =
        if cfg.algorithm == "qlora_sft" && plan.quantisation_backend.eq_ignore_ascii_case("none") {
            eprintln!(
                "warning: qlora_sft requested but quantisation backend is unavailable; \
                 executing as lora_sft"
            );
            "lora_sft".to_string()
        } else {
            cfg.algorithm.clone()
        };

    let batch_settings = BatchBuildSettings {
        max_seq_len: cfg.max_seq_len.max(8),
        pad_id,
        separator_id,
        dynamic_padding: cfg.dynamic_padding,
        sequence_packing: cfg.sequence_packing,
    };
    let prefetcher = BatchPrefetcher::start(
        Arc::new(samples),
        batch_schedule,
        batch_settings,
        cfg.prefetch_batches.max(plan.prefetch_batches),
    );

    let mut scaler = DynamicLossScaler::new(cfg, plan.mixed_precision);
    let mut final_loss = None;
    let mut optimiser_step = 0_usize;
    let mut metrics = TrainingMetrics {
        samples: sample_count,
        batches_per_epoch,
        total_micro_steps,
        total_optimizer_steps,
        trainable_parameters,
        trainable_tensors: tensor_names,
        ..TrainingMetrics::default()
    };
    let started = std::time::Instant::now();

    zero_param_grads(&mut params);
    for micro_step in 0..total_micro_steps {
        let wait_started = std::time::Instant::now();
        let Some(cpu_batch) = prefetcher.next_batch()? else {
            anyhow::bail!("batch prefetcher ended before all steps were consumed");
        };
        metrics.data_wait_seconds += wait_started.elapsed().as_secs_f64();
        metrics.total_tokens = metrics
            .total_tokens
            .saturating_add((cpu_batch.sample_count * cpu_batch.sequence_length) as u64);
        metrics.non_padding_tokens = metrics
            .non_padding_tokens
            .saturating_add(cpu_batch.non_padding_tokens as u64);
        metrics.padding_tokens = metrics
            .padding_tokens
            .saturating_add(cpu_batch.padding_tokens as u64);
        metrics.packed_sequences = metrics
            .packed_sequences
            .saturating_add(cpu_batch.packed_sequences as u64);

        let h2d_started = std::time::Instant::now();
        let input_ids = cpu_batch.input_ids.to_device(device);
        let labels = cpu_batch.labels.to_device(device);
        metrics.host_to_device_seconds += h2d_started.elapsed().as_secs_f64();

        let forward_started = std::time::Instant::now();
        let loss = autocast(plan.mixed_precision, || -> anyhow::Result<Tensor> {
            match cfg.forward_mode {
                ForwardMode::Loss => module
                    .forward_ts(&[input_ids.shallow_clone(), labels.shallow_clone()])
                    .map_err(|error| anyhow::anyhow!("forward(loss) failed: {error}")),
                ForwardMode::Logits => {
                    let logits = module
                        .forward_ts(&[input_ids.shallow_clone()])
                        .map_err(|error| anyhow::anyhow!("forward(logits) failed: {error}"))?;
                    causal_lm_loss(logits, labels.shallow_clone())
                }
            }
        })?;
        metrics.forward_seconds += forward_started.elapsed().as_secs_f64();

        let loss = loss.to_kind(Kind::Float);
        let scaled_loss = scaler.scale_loss(&loss, accumulation);
        let backward_started = std::time::Instant::now();
        scaled_loss.backward();
        metrics.backward_seconds += backward_started.elapsed().as_secs_f64();

        let should_step =
            (micro_step + 1) % accumulation == 0 || micro_step + 1 == total_micro_steps;
        if should_step {
            scaler.unscale_gradients(&mut params);
            let gradients_finite = gradients_are_finite(&params)?;
            if gradients_finite {
                let clip = clip_gradient_norm(&mut params, cfg.max_grad_norm)?;
                if clip.clipped {
                    metrics.clipped_optimizer_steps =
                        metrics.clipped_optimizer_steps.saturating_add(1);
                }
                metrics.final_grad_norm = clip.grad_norm;
                optimiser_step = optimiser_step.saturating_add(1);
                let lr = scheduled_lr(cfg.learning_rate, optimiser_step, warmup_steps);
                let optimiser_started = std::time::Instant::now();
                adamw_step(&mut params, optimiser_step, lr, cfg)?;
                metrics.optimizer_seconds += optimiser_started.elapsed().as_secs_f64();
                scaler.update(false);
            } else {
                metrics.skipped_optimizer_steps = metrics.skipped_optimizer_steps.saturating_add(1);
                scaler.update(true);
                eprintln!(
                    "native_tch_train step={}/{} detected non-finite gradients; skipped optimiser step",
                    micro_step + 1,
                    total_micro_steps
                );
            }
            zero_param_grads(&mut params);
        }

        let should_log =
            micro_step == 0 || (micro_step + 1) % 10 == 0 || micro_step + 1 == total_micro_steps;
        if should_log {
            final_loss = Some(loss.double_value(&[]));
            eprintln!(
                "native_tch_train step={}/{} optimiser_step={} loss={:.6} scale={:.1} device={} trainable_params={} non_padding_tokens={}",
                micro_step + 1,
                total_micro_steps,
                optimiser_step,
                final_loss.unwrap_or(f64::NAN),
                scaler.scale,
                plan.device_label,
                trainable_parameters,
                metrics.non_padding_tokens
            );
        }
    }

    let runtime_seconds = started.elapsed().as_secs_f64().max(1e-9);
    metrics.final_loss = final_loss;
    metrics.runtime_seconds = runtime_seconds;
    metrics.tokens_per_second = metrics.total_tokens as f64 / runtime_seconds;
    metrics.non_padding_tokens_per_second = metrics.non_padding_tokens as f64 / runtime_seconds;
    metrics.padding_percentage = if metrics.total_tokens == 0 {
        0.0
    } else {
        (metrics.padding_tokens as f64 / metrics.total_tokens as f64) * 100.0
    };

    let adapter_dir = cfg.output.join("adapter");
    std::fs::create_dir_all(&adapter_dir)?;
    let adapter_model = adapter_dir.join("adapter_model.safetensors");
    let exported_adapter_tensors = export_adapter_safetensors(&params, &adapter_model)?;
    if env_bool("GAIL_TCH_EXPORT_DEBUG_MODULE", false) {
        let debug_module = adapter_dir.join("adapter_debug_module.pt");
        module
            .save(&debug_module)
            .map_err(|error| anyhow::anyhow!("failed to save debug TorchScript module: {error}"))?;
    }

    Ok(TrainingRunResult {
        metrics,
        executed_algorithm,
        exported_adapter_tensors,
        final_loss_scale: if scaler.enabled {
            Some(scaler.scale)
        } else {
            None
        },
    })
}

fn collect_trainable_parameters(
    module: &CModule,
    cfg: &TrainingConfig,
) -> anyhow::Result<Vec<TrainableParam>> {
    let train_all = env_bool("GAIL_TCH_TRAIN_ALL", false);
    let marker = cfg.trainable_marker.to_ascii_lowercase();
    let named = module
        .named_parameters()
        .map_err(|error| anyhow::anyhow!("failed to inspect model parameters: {error}"))?;
    let mut params = Vec::new();
    for (name, tensor) in named {
        let lower = name.to_ascii_lowercase();
        let trainable = train_all || lower.contains(marker.as_str());
        if trainable {
            let tensor = tensor.set_requires_grad(true);
            let shape = tensor.size();
            let device = tensor.device();
            let m = Tensor::zeros(shape.as_slice(), (Kind::Float, device));
            let v = Tensor::zeros(shape.as_slice(), (Kind::Float, device));
            params.push(TrainableParam { name, tensor, m, v });
        } else {
            let _ = tensor.set_requires_grad(false);
        }
    }
    Ok(params)
}

fn gradients_are_finite(params: &[TrainableParam]) -> anyhow::Result<bool> {
    for param in params {
        let grad = param.tensor.grad();
        if !grad.defined() {
            continue;
        }
        let finite = grad.isfinite().all().to_kind(Kind::Int64).int64_value(&[]) == 1;
        if !finite {
            return Ok(false);
        }
    }
    Ok(true)
}

fn clip_gradient_norm(params: &mut [TrainableParam], max_norm: f64) -> anyhow::Result<ClipOutcome> {
    if max_norm <= 0.0 {
        return Ok(ClipOutcome {
            grad_norm: None,
            clipped: false,
        });
    }
    let mut total_norm_sq = 0.0_f64;
    for param in params.iter() {
        let grad = param.tensor.grad();
        if !grad.defined() {
            continue;
        }
        let contribution = grad
            .to_kind(Kind::Float)
            .square()
            .sum(Kind::Float)
            .double_value(&[]);
        total_norm_sq += contribution;
    }
    let grad_norm = total_norm_sq.sqrt();
    if !grad_norm.is_finite() || grad_norm <= max_norm {
        return Ok(ClipOutcome {
            grad_norm: Some(grad_norm),
            clipped: false,
        });
    }

    let scale = max_norm / (grad_norm + 1e-6);
    no_grad(|| {
        for param in params {
            let mut grad = param.tensor.grad();
            if grad.defined() {
                let _ = grad.g_mul_scalar_(scale);
            }
        }
    });
    Ok(ClipOutcome {
        grad_norm: Some(grad_norm),
        clipped: true,
    })
}

fn adamw_step(
    params: &mut [TrainableParam],
    step: usize,
    lr: f64,
    cfg: &TrainingConfig,
) -> anyhow::Result<()> {
    let beta1 = cfg.adam_beta1;
    let beta2 = cfg.adam_beta2;
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);
    no_grad(|| -> anyhow::Result<()> {
        for param in params.iter_mut() {
            let grad = param.tensor.grad();
            if !grad.defined() {
                continue;
            }
            let grad_fp32 = grad.to_kind(Kind::Float);
            param.m = &param.m * beta1 + &grad_fp32 * (1.0 - beta1);
            param.v = &param.v * beta2 + grad_fp32.square() * (1.0 - beta2);
            let m_hat = &param.m / bias_correction1;
            let v_hat = &param.v / bias_correction2;
            let mut update = &m_hat / (v_hat.sqrt() + cfg.adam_eps);
            if cfg.weight_decay > 0.0 {
                update = update + param.tensor.to_kind(Kind::Float) * cfg.weight_decay;
            }
            let delta = (update * lr).to_kind(param.tensor.kind());
            param.tensor.f_sub_(&delta)?;
        }
        Ok(())
    })
}

fn zero_param_grads(params: &mut [TrainableParam]) {
    for param in params {
        let mut grad = param.tensor.grad();
        if grad.defined() {
            let _ = grad.zero_();
        }
    }
}

fn export_adapter_safetensors(
    params: &[TrainableParam],
    output_path: &Path,
) -> anyhow::Result<usize> {
    let mut tensors = Vec::<(String, OwnedSafeTensor)>::new();
    for param in params {
        let exported = param
            .tensor
            .detach()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .contiguous();
        let shape = exported
            .size()
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let numel = exported.numel();
        let mut values = vec![0f32; numel];
        exported.copy_data(values.as_mut_slice(), numel);
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push((
            param.name.clone(),
            OwnedSafeTensor {
                dtype: Dtype::F32,
                shape,
                data,
            },
        ));
    }
    let metadata = Some(HashMap::from([
        ("format".to_string(), "gail-lora-adapter".to_string()),
        ("tensor_dtype".to_string(), "f32".to_string()),
    ]));
    serialize_to_file(
        tensors.iter().map(|(name, tensor)| (name.as_str(), tensor)),
        &metadata,
        output_path,
    )
    .map_err(|error| anyhow::anyhow!("failed to write safetensors adapter: {error}"))?;
    Ok(tensors.len())
}

fn causal_lm_loss(logits: Tensor, labels: Tensor) -> anyhow::Result<Tensor> {
    let sizes = logits.size();
    if sizes.len() != 3 {
        anyhow::bail!("expected logits shape [batch, seq, vocab], got {:?}", sizes);
    }
    let seq_len = sizes[1];
    let vocab = sizes[2];
    if seq_len < 2 {
        anyhow::bail!("sequence length must be at least 2 for causal LM loss");
    }
    let shift_logits = logits.narrow(1, 0, seq_len - 1).reshape([-1, vocab]);
    let shift_labels = labels.narrow(1, 1, seq_len - 1).reshape([-1]);
    Ok(shift_logits.cross_entropy_for_logits(&shift_labels))
}

fn tokenise_parallel(
    tokenizer: &Tokenizer,
    texts: &[String],
    max_seq_len: usize,
) -> anyhow::Result<Vec<TokenisedSample>> {
    let max_len = max_seq_len.max(8);
    let encoded = texts
        .par_iter()
        .map(|text| {
            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| anyhow::anyhow!("tokenisation failed: {error}"))?;
            let ids = encoding
                .get_ids()
                .iter()
                .copied()
                .map(i64::from)
                .take(max_len)
                .collect::<Vec<_>>();
            Ok::<_, anyhow::Error>(TokenisedSample { ids })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(encoded
        .into_iter()
        .filter(|sample| sample.ids.len() > 1)
        .collect())
}

fn compute_batches_per_epoch(sample_count: usize, batch_size: usize) -> usize {
    sample_count.div_ceil(batch_size.max(1)).max(1)
}

fn build_batch_schedule(
    samples: &[TokenisedSample],
    batch_size: usize,
    total_micro_steps: usize,
    seed: u64,
) -> Vec<BatchPlan> {
    if samples.is_empty() || total_micro_steps == 0 {
        return Vec::new();
    }
    let mut schedule = Vec::with_capacity(total_micro_steps);
    let mut epoch = 0usize;
    while schedule.len() < total_micro_steps {
        let epoch_plans = build_epoch_batch_plans(samples, batch_size, seed, epoch);
        if epoch_plans.is_empty() {
            break;
        }
        for plan in epoch_plans {
            schedule.push(plan);
            if schedule.len() == total_micro_steps {
                break;
            }
        }
        epoch = epoch.saturating_add(1);
    }
    schedule
}

fn build_epoch_batch_plans(
    samples: &[TokenisedSample],
    batch_size: usize,
    seed: u64,
    epoch: usize,
) -> Vec<BatchPlan> {
    let batch_size = batch_size.max(1);
    let mut indices = (0..samples.len())
        .filter(|index| samples[*index].ids.len() > 1)
        .collect::<Vec<_>>();
    indices.sort_by_key(|index| samples[*index].ids.len());
    if indices.is_empty() {
        return Vec::new();
    }

    let bucket_size = (batch_size * 8).max(8);
    let mut buckets = indices
        .chunks(bucket_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    let epoch_seed = seed ^ ((epoch as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = StdRng::seed_from_u64(epoch_seed);
    for bucket in &mut buckets {
        bucket.shuffle(&mut rng);
    }
    buckets.shuffle(&mut rng);

    let flattened = buckets.into_iter().flatten().collect::<Vec<_>>();
    flattened
        .chunks(batch_size)
        .map(|chunk| BatchPlan {
            sample_indices: chunk.to_vec(),
        })
        .collect()
}

fn build_cpu_batch(
    samples: &[TokenisedSample],
    plan: &BatchPlan,
    settings: &BatchBuildSettings,
) -> anyhow::Result<CpuBatch> {
    let original_sequences = plan.sample_indices.len();
    let mut sequences = plan
        .sample_indices
        .iter()
        .filter_map(|index| samples.get(*index).map(|sample| sample.ids.clone()))
        .filter(|ids| ids.len() > 1)
        .collect::<Vec<_>>();
    if sequences.is_empty() {
        anyhow::bail!("empty batch plan after index filtering");
    }
    if settings.sequence_packing {
        sequences = pack_short_sequences(sequences, settings.max_seq_len, settings.separator_id);
    }
    if sequences.is_empty() {
        anyhow::bail!("sequence packing dropped all rows");
    }
    let batch_sequences = sequences.len();
    let packed_sequence_count = original_sequences.saturating_sub(batch_sequences);

    let dynamic_len = sequences.iter().map(Vec::len).max().unwrap_or(2).max(2);
    let seq_len = if settings.dynamic_padding {
        dynamic_len.min(settings.max_seq_len.max(2))
    } else {
        settings.max_seq_len.max(2)
    };

    let mut input_rows = Vec::with_capacity(sequences.len() * seq_len);
    let mut label_rows = Vec::with_capacity(sequences.len() * seq_len);
    let mut non_padding_tokens = 0usize;
    let mut padding_tokens = 0usize;
    for mut ids in sequences {
        ids.truncate(seq_len);
        let valid = ids.len();
        ids.resize(seq_len, settings.pad_id);
        let mut labels = ids.clone();
        for label in labels.iter_mut().skip(valid) {
            *label = -100;
        }
        non_padding_tokens = non_padding_tokens.saturating_add(valid);
        padding_tokens = padding_tokens.saturating_add(seq_len.saturating_sub(valid));
        input_rows.extend_from_slice(&ids);
        label_rows.extend_from_slice(&labels);
    }

    let bsz = batch_sequences.max(1) as i64;
    let input_ids = Tensor::from_slice(&input_rows)
        .to_kind(Kind::Int64)
        .reshape([bsz, seq_len as i64]);
    let labels = Tensor::from_slice(&label_rows)
        .to_kind(Kind::Int64)
        .reshape([bsz, seq_len as i64]);
    Ok(CpuBatch {
        input_ids,
        labels,
        sample_count: bsz as usize,
        sequence_length: seq_len,
        non_padding_tokens,
        padding_tokens,
        packed_sequences: packed_sequence_count,
    })
}

fn pack_short_sequences(
    sequences: Vec<Vec<i64>>,
    max_seq_len: usize,
    separator_id: i64,
) -> Vec<Vec<i64>> {
    if sequences.len() <= 1 {
        return sequences;
    }
    let mut packed = Vec::<Vec<i64>>::new();
    let mut current = Vec::<i64>::new();
    for mut sequence in sequences {
        if sequence.is_empty() {
            continue;
        }
        if current.is_empty() {
            current = sequence;
            continue;
        }
        if current.len() + 1 + sequence.len() <= max_seq_len {
            current.push(separator_id);
            current.append(&mut sequence);
        } else {
            packed.push(current);
            current = sequence;
        }
    }
    if !current.is_empty() {
        packed.push(current);
    }
    packed
}

async fn load_training_texts(dataset_path: &Path) -> anyhow::Result<Vec<String>> {
    let body = fs::read_to_string(dataset_path).await?;
    let mut texts = Vec::new();
    for (line_no, line) in body.lines().enumerate() {
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }
        let row: DatasetRow = serde_json::from_str(raw)
            .map_err(|error| anyhow::anyhow!("invalid JSONL at line {}: {error}", line_no + 1))?;
        let Some(messages) = row.messages else {
            continue;
        };
        let rendered = manual_chat_template(&messages);
        if !rendered.trim().is_empty() {
            texts.push(rendered);
        }
    }
    Ok(texts)
}

fn manual_chat_template(messages: &[ChatMessage]) -> String {
    let mut rendered = String::new();
    for message in messages {
        let role = message
            .role
            .as_deref()
            .unwrap_or("user")
            .trim()
            .to_ascii_lowercase();
        let content = message.content.as_deref().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str("<|");
        rendered.push_str(if role.is_empty() {
            "user"
        } else {
            role.as_str()
        });
        rendered.push_str("|>\n");
        rendered.push_str(content);
    }
    rendered.trim().to_string()
}

async fn write_rendered_dataset(path: &Path, texts: &[String]) -> anyhow::Result<()> {
    let mut body = String::new();
    for text in texts {
        body.push_str(&serde_json::to_string(&json!({ "text": text }))?);
        body.push('\n');
    }
    fs::write(path, body).await?;
    Ok(())
}

async fn write_adapter_config(cfg: &TrainingConfig, adapter_dir: &Path) -> anyhow::Result<()> {
    let adapter_config = json!({
        "base_model_name_or_path": cfg.base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "fan_in_fan_out": false,
        "bias": "none",
        "target_modules_marker": cfg.trainable_marker,
    });
    fs::write(
        adapter_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_config)? + "\n",
    )
    .await?;
    Ok(())
}

async fn write_training_manifest(
    cfg: &TrainingConfig,
    plan: &TrainingExecutionPlan,
    run_result: &TrainingRunResult,
    adapter_dir: &Path,
) -> anyhow::Result<()> {
    let manifest = json!({
        "format": "gail-lora-training-manifest",
        "algorithm_requested": cfg.algorithm,
        "algorithm_executed": run_result.executed_algorithm,
        "backend": plan.backend.as_str(),
        "profile": plan.profile,
        "device": plan.device_label,
        "compute_dtype": plan.compute_dtype.as_str(),
        "quantisation_backend": plan.quantisation_backend,
        "activation_checkpointing": plan.activation_checkpointing,
        "dynamic_padding": plan.dynamic_padding,
        "sequence_packing": plan.sequence_packing,
        "micro_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "max_sequence_length": plan.max_sequence_length,
        "base_model": cfg.base_model,
        "model_module": cfg.model_module.to_string_lossy(),
        "tokenizer": cfg.tokenizer.to_string_lossy(),
        "adapter_model": "adapter_model.safetensors",
        "adapter_config": "adapter_config.json",
        "metrics": run_result.metrics,
        "final_loss_scale": run_result.final_loss_scale,
        "exported_adapter_tensors": run_result.exported_adapter_tensors,
    });
    fs::write(
        adapter_dir.join("training_manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .await?;
    Ok(())
}

async fn write_modelfile(cfg: &TrainingConfig, modelfile: &Path) -> anyhow::Result<()> {
    let rendered = format!(
        "FROM {}\nADAPTER ./adapter\nPARAMETER temperature 0.2\nSYSTEM {}\n",
        cfg.base_model,
        cfg.system_prompt.replace('\n', " ")
    );
    fs::write(modelfile, rendered).await?;
    Ok(())
}

fn parse_args<I>(args: I) -> anyhow::Result<TrainingConfig>
where
    I: IntoIterator<Item = OsString>,
{
    let args = args.into_iter().collect::<Vec<_>>();
    let mut dataset: Option<PathBuf> = env::var_os("GAIL_TRAIN_DATASET_PATH").map(PathBuf::from);
    let mut output: Option<PathBuf> = env::var_os("GAIL_TRAIN_OUTPUT_DIR").map(PathBuf::from);
    let mut base_model = env_or("GAIL_TRAIN_BASE_MODEL", "qwen2.5-coder:1.5b");
    let mut model_module = env::var_os("GAIL_TCH_MODEL_MODULE").map(PathBuf::from);
    let mut tokenizer = env::var_os("GAIL_TCH_TOKENIZER").map(PathBuf::from);
    let mut algorithm = env_or("GAIL_TRAIN_ALGORITHM", "qlora_sft");
    let mut epochs = parse_env("GAIL_TRAIN_EPOCHS", 1.0_f64);
    let mut batch_size = parse_env("GAIL_TRAIN_BATCH_SIZE", 1_usize);
    let mut gradient_accumulation_steps =
        parse_env("GAIL_TRAIN_GRADIENT_ACCUMULATION_STEPS", 8_usize);
    let mut learning_rate = parse_env("GAIL_TRAIN_LEARNING_RATE", 2e-4_f64);
    let mut warmup_ratio = parse_env("GAIL_TRAIN_WARMUP_RATIO", 0.03_f64);
    let mut max_seq_len = parse_env("GAIL_TRAIN_MAX_SEQ_LEN", 2048_usize);
    let mut lora_r = parse_env("GAIL_TRAIN_LORA_R", 32_usize);
    let mut lora_alpha = parse_env("GAIL_TRAIN_LORA_ALPHA", 64_usize);
    let mut lora_dropout = parse_env("GAIL_TRAIN_LORA_DROPOUT", 0.05_f64);
    let mut weight_decay = parse_env("GAIL_TRAIN_WEIGHT_DECAY", 0.0_f64);
    let mut adam_beta1 = parse_env("GAIL_TRAIN_ADAM_BETA1", 0.9_f64);
    let mut adam_beta2 = parse_env("GAIL_TRAIN_ADAM_BETA2", 0.999_f64);
    let mut adam_eps = parse_env("GAIL_TRAIN_ADAM_EPS", 1e-8_f64);
    let mut max_grad_norm = parse_env("GAIL_TRAIN_MAX_GRAD_NORM", 1.0_f64);
    let mut system_prompt = env_or("GAIL_TRAIN_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT);
    let mut timeout_seconds = parse_env("GAIL_TRAIN_COMMAND_TIMEOUT_SECONDS", 86_400_u64);
    let mut trainable_marker = env_or("GAIL_TCH_TRAINABLE_MARKER", "lora");
    let mut forward_mode = parse_forward_mode(&env_or("GAIL_TCH_FORWARD_MODE", "loss"))?;
    let mut seed = parse_env("GAIL_TRAIN_SEED", 42_u64);
    let mut prefetch_batches = parse_env("GAIL_TRAIN_PREFETCH_BATCHES", 2_usize);
    let mut dynamic_padding = env_bool("GAIL_TRAIN_DYNAMIC_PADDING", true);
    let mut sequence_packing = env_bool("GAIL_TRAIN_SEQUENCE_PACKING", true);
    let mut loss_scale_initial = parse_env("GAIL_TRAIN_LOSS_SCALE", 1024.0_f64);
    let mut loss_scale_growth_factor = parse_env("GAIL_TRAIN_LOSS_SCALE_GROWTH_FACTOR", 2.0_f64);
    let mut loss_scale_backoff_factor = parse_env("GAIL_TRAIN_LOSS_SCALE_BACKOFF_FACTOR", 0.5_f64);
    let mut loss_scale_growth_interval =
        parse_env("GAIL_TRAIN_LOSS_SCALE_GROWTH_INTERVAL", 200_usize);

    let mut index = 0;
    while index < args.len() {
        let flag = args[index].to_string_lossy().to_string();
        if flag == "--help" || flag == "-h" {
            anyhow::bail!("{}", help_text());
        }
        let next = |index: &mut usize, flag: &str| -> anyhow::Result<String> {
            *index += 1;
            args.get(*index)
                .map(|value| value.to_string_lossy().to_string())
                .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--dataset" => dataset = Some(PathBuf::from(next(&mut index, &flag)?)),
            "--output" => output = Some(PathBuf::from(next(&mut index, &flag)?)),
            "--base-model" => base_model = next(&mut index, &flag)?,
            "--model-module" => model_module = Some(PathBuf::from(next(&mut index, &flag)?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(next(&mut index, &flag)?)),
            "--algorithm" => algorithm = next(&mut index, &flag)?,
            "--epochs" => epochs = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--batch-size" => batch_size = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--gradient-accumulation-steps" => {
                gradient_accumulation_steps = parse_value(&next(&mut index, &flag)?, &flag)?
            }
            "--learning-rate" => learning_rate = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--warmup-ratio" => warmup_ratio = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--max-seq-len" => max_seq_len = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--lora-r" => lora_r = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--lora-alpha" => lora_alpha = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--lora-dropout" => lora_dropout = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--weight-decay" => weight_decay = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--max-grad-norm" => max_grad_norm = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--system-prompt" => system_prompt = next(&mut index, &flag)?,
            "--timeout-seconds" => timeout_seconds = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--trainable-marker" => trainable_marker = next(&mut index, &flag)?,
            "--forward-mode" => forward_mode = parse_forward_mode(&next(&mut index, &flag)?)?,
            "--seed" => seed = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--prefetch-batches" => {
                prefetch_batches = parse_value(&next(&mut index, &flag)?, &flag)?
            }
            "--dynamic-padding" => {
                dynamic_padding = parse_bool_value(&next(&mut index, &flag)?, &flag)?
            }
            "--disable-dynamic-padding" => dynamic_padding = false,
            "--sequence-packing" => {
                sequence_packing = parse_bool_value(&next(&mut index, &flag)?, &flag)?
            }
            "--disable-sequence-packing" => sequence_packing = false,
            "--loss-scale" => loss_scale_initial = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--loss-scale-growth-factor" => {
                loss_scale_growth_factor = parse_value(&next(&mut index, &flag)?, &flag)?
            }
            "--loss-scale-backoff-factor" => {
                loss_scale_backoff_factor = parse_value(&next(&mut index, &flag)?, &flag)?
            }
            "--loss-scale-growth-interval" => {
                loss_scale_growth_interval = parse_value(&next(&mut index, &flag)?, &flag)?
            }
            other => anyhow::bail!("unknown argument: {other}\n\n{}", help_text()),
        }
        index += 1;
    }

    let dataset = dataset.ok_or_else(|| anyhow::anyhow!("--dataset is required"))?;
    let output = output.ok_or_else(|| anyhow::anyhow!("--output is required"))?;
    let model_module = model_module.unwrap_or_else(|| default_model_module(&base_model));
    let tokenizer = tokenizer.unwrap_or_else(|| default_tokenizer(&base_model));
    if !model_module.exists() {
        anyhow::bail!(
            "TorchScript model module not found: {}. Set --model-module or GAIL_TCH_MODEL_MODULE.",
            model_module.display()
        );
    }
    if !tokenizer.exists() {
        anyhow::bail!(
            "tokenizer.json not found: {}. Set --tokenizer or GAIL_TCH_TOKENIZER.",
            tokenizer.display()
        );
    }

    Ok(TrainingConfig {
        dataset,
        output,
        base_model,
        model_module,
        tokenizer,
        algorithm,
        epochs: epochs.max(0.01),
        batch_size: batch_size.max(1),
        gradient_accumulation_steps: gradient_accumulation_steps.max(1),
        learning_rate: learning_rate.max(1e-8),
        warmup_ratio: warmup_ratio.clamp(0.0, 1.0),
        max_seq_len: max_seq_len.max(8),
        lora_r: lora_r.max(1),
        lora_alpha: lora_alpha.max(1),
        lora_dropout: lora_dropout.clamp(0.0, 1.0),
        weight_decay: weight_decay.max(0.0),
        adam_beta1: adam_beta1.clamp(0.0, 0.999_999),
        adam_beta2: adam_beta2.clamp(0.0, 0.999_999),
        adam_eps: adam_eps.max(1e-12),
        max_grad_norm: max_grad_norm.max(0.0),
        system_prompt,
        timeout_seconds: timeout_seconds.max(30),
        trainable_marker,
        forward_mode,
        seed,
        prefetch_batches: prefetch_batches.max(1),
        dynamic_padding,
        sequence_packing,
        loss_scale_initial: loss_scale_initial.max(1.0),
        loss_scale_growth_factor: loss_scale_growth_factor.max(1.01),
        loss_scale_backoff_factor: loss_scale_backoff_factor.clamp(0.1, 0.99),
        loss_scale_growth_interval: loss_scale_growth_interval.max(1),
    })
}

fn help_text() -> &'static str {
    "Usage: gail-qlora-sft --dataset <path> --output <dir> \\\n       --model-module <torchscript.pt> --tokenizer <tokenizer.json> \\\n       [--algorithm qlora_sft|lora_sft] [--forward-mode loss|logits] \\\n       [--dynamic-padding true|false] [--sequence-packing true|false]"
}

fn default_model_module(base_model: &str) -> PathBuf {
    let path = PathBuf::from(base_model);
    if path.is_file() {
        return path;
    }
    path.join("model_train.pt")
}

fn default_tokenizer(base_model: &str) -> PathBuf {
    let path = PathBuf::from(base_model);
    if path.is_dir() {
        return path.join("tokenizer.json");
    }
    PathBuf::from("tokenizer.json")
}

fn parse_forward_mode(value: &str) -> anyhow::Result<ForwardMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "loss" => Ok(ForwardMode::Loss),
        "logits" => Ok(ForwardMode::Logits),
        other => anyhow::bail!("unsupported forward mode '{other}'; expected loss or logits"),
    }
}

fn parse_compute_dtype(value: &str) -> ComputeDtype {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp16" | "float16" | "half" => ComputeDtype::Fp16,
        "bf16" | "bfloat16" => ComputeDtype::Bf16,
        _ => ComputeDtype::Fp32,
    }
}

fn build_execution_plan(cfg: &TrainingConfig) -> TrainingExecutionPlan {
    let cpu_intraop_threads = parse_env(
        "GAIL_TRAIN_CPU_INTRAOP_THREADS",
        parse_env("GAIL_TRAIN_CPU_THREADS", num_cpus_fallback()),
    )
    .max(1);
    let cpu_interop_threads = parse_env("GAIL_TRAIN_CPU_INTEROP_THREADS", 1_usize).max(1);
    let tokenizer_threads = parse_env(
        "GAIL_TRAIN_TOKENIZER_THREADS",
        (cpu_intraop_threads / 3).max(1),
    )
    .max(1);
    let gpu_count = Cuda::device_count().max(0) as usize;
    let requested_device = env_or(
        "GAIL_TRAIN_DEVICE",
        if gpu_count > 0 { "cuda" } else { "cpu" },
    );
    let use_gpu = (requested_device.eq_ignore_ascii_case("cuda")
        || requested_device.eq_ignore_ascii_case("gpu"))
        && Cuda::is_available()
        && gpu_count > 0;
    let device_index = if use_gpu { Some(0) } else { None };
    let device_label = device_index
        .map(|index| format!("cuda:{index}"))
        .unwrap_or_else(|| "cpu".to_string());
    let compute_dtype = parse_compute_dtype(&env_or(
        "GAIL_TRAIN_COMPUTE_DTYPE",
        if use_gpu { "fp16" } else { "fp32" },
    ));
    let quantisation_backend = if env_bool("GAIL_TCH_BASE_PREQUANTISED", false) {
        "prequantised_base".to_string()
    } else {
        "none".to_string()
    };
    let backend = if use_gpu && cfg.algorithm.eq_ignore_ascii_case("qlora_sft") {
        TrainingBackendKind::CudaQlora
    } else if use_gpu {
        TrainingBackendKind::CudaLora
    } else {
        TrainingBackendKind::CpuLora
    };
    let profile = env_or(
        "GAIL_TRAIN_EXECUTION_PROFILE",
        if std::env::consts::ARCH.eq_ignore_ascii_case("aarch64") && use_gpu {
            "centriq_rtx3060_12gb"
        } else if std::env::consts::ARCH.eq_ignore_ascii_case("aarch64") {
            "centriq_cpu_armv8"
        } else if use_gpu {
            "generic_cuda"
        } else {
            "generic_cpu"
        },
    );
    let mixed_precision = use_gpu && !matches!(compute_dtype, ComputeDtype::Fp32);
    TrainingExecutionPlan {
        profile,
        backend,
        device_label,
        device_index,
        gpu_count,
        cpu_intraop_threads,
        cpu_interop_threads,
        tokenizer_threads,
        prefetch_batches: cfg.prefetch_batches.max(1),
        compute_dtype,
        quantisation_backend,
        mixed_precision,
        activation_checkpointing: env_bool("GAIL_TRAIN_ACTIVATION_CHECKPOINTING", false),
        dynamic_padding: cfg.dynamic_padding,
        sequence_packing: cfg.sequence_packing,
        micro_batch_size: cfg.batch_size,
        gradient_accumulation_steps: cfg.gradient_accumulation_steps,
        max_sequence_length: cfg.max_seq_len,
    }
}

fn tch_device(plan: &TrainingExecutionPlan) -> Device {
    plan.device_index.map(Device::Cuda).unwrap_or(Device::Cpu)
}

fn apply_cpu_thread_limits(plan: &TrainingExecutionPlan) {
    let intra = plan.cpu_intraop_threads.max(1) as i32;
    let inter = plan.cpu_interop_threads.max(1) as i32;
    tch::set_num_threads(intra);
    tch::set_num_interop_threads(inter);

    // Configure the global Rayon pool once for dataset tokenisation/preprocessing work.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(plan.tokenizer_threads.max(1))
        .build_global();
}

fn scheduled_lr(base_lr: f64, step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps > 0 && step <= warmup_steps {
        base_lr * (step as f64 / warmup_steps as f64)
    } else {
        base_lr
    }
}

fn token_to_id(tokenizer: &Tokenizer, token: &str) -> Option<i64> {
    tokenizer.token_to_id(token).map(i64::from)
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
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

fn parse_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_value<T>(value: &str, flag: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid value for {flag}: {error}"))
}

fn parse_bool_value(value: &str, flag: &str) -> anyhow::Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("invalid boolean value for {flag}: {value}"),
    }
}

fn num_cpus_fallback() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs_f64()
}
