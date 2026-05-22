//! Gail native Rust LoRA/QLoRA-style supervised fine-tuning runner.
//!
//! This binary replaces `qlora_sft.py` with a Rust-native training backend built
//! on `tch`/libtorch.  It is intentionally designed for Gail's trainer worker:
//! it consumes the same JSONL ChatML dataset shape, performs CPU-parallel
//! tokenisation/batching, trains on CUDA when available, streams progress, and
//! writes a Gail/Ollama-compatible snapshot directory.
//!
//! Practical model contract
//! ------------------------
//! Rust cannot call Hugging Face `AutoModelForCausalLM`/PEFT/BitsAndBytes
//! directly.  The production-native path is therefore to export or package a
//! TorchScript/libtorch training module that already contains:
//!   * the frozen base causal-LM weights;
//!   * LoRA trainable parameters named with `lora` by convention;
//!   * optionally a pre-quantised base if `--algorithm qlora_sft` is requested.
//!
//! The module may expose either:
//!   1. `forward(input_ids, labels) -> scalar_loss`  (`GAIL_TCH_FORWARD_MODE=loss`), or
//!   2. `forward(input_ids) -> logits`               (`GAIL_TCH_FORWARD_MODE=logits`).
//!
//! Only tensors whose names contain the configured trainable marker, default
//! `lora`, are updated.  This keeps base-model weights frozen, matching LoRA
//! fine-tuning semantics.  Set `GAIL_TCH_TRAIN_ALL=1` only for development.
//!
//! Recommended Cargo dependencies:
//!   anyhow = "1"
//!   serde = { version = "1", features = ["derive"] }
//!   serde_json = "1"
//!   tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
//!   tch = "0.17"
//!   rayon = "1"
//!   tokio = { version = "1", features = ["full"] }

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tch::{CModule, Cuda, Device, Kind, Tensor, no_grad};
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
    system_prompt: String,
    timeout_seconds: u64,
    trainable_marker: String,
    forward_mode: ForwardMode,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ForwardMode {
    Loss,
    Logits,
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

#[derive(Debug, Clone, Serialize)]
struct HardwarePlan {
    device_label: String,
    device_index: Option<usize>,
    gpu_count: usize,
    cpu_threads: usize,
    compute_dtype: String,
    quantisation: String,
}

#[derive(Debug, Clone)]
struct TokenisedSample {
    ids: Vec<i64>,
}

#[derive(Debug)]
struct Batch {
    input_ids: Tensor,
    labels: Tensor,
}

#[derive(Debug)]
struct TrainableParam {
    name: String,
    tensor: Tensor,
    m: Tensor,
    v: Tensor,
}

#[derive(Debug, Serialize)]
struct TrainingMetrics {
    samples: usize,
    batches: usize,
    trainable_parameters: usize,
    trainable_tensors: Vec<String>,
    total_steps: usize,
    final_loss: Option<f64>,
    runtime_seconds: f64,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let code = match unsafe { async_main() }.await {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Training failed: {error}");
            1
        }
    };
    std::process::exit(code);
}

async unsafe fn async_main() -> anyhow::Result<()> {
    let cfg = parse_args(env::args_os().skip(1))?;
    if !SUPPORTED_ALGORITHMS.contains(&cfg.algorithm.as_str()) {
        anyhow::bail!(
            "unsupported algorithm '{}'; supported: {}",
            cfg.algorithm,
            SUPPORTED_ALGORITHMS.join(", ")
        );
    }
    if cfg.algorithm == "qlora_sft" && !env_bool("GAIL_TCH_BASE_PREQUANTISED", false) {
        eprintln!(
            "warning: qlora_sft requested, but GAIL_TCH_BASE_PREQUANTISED is not true; \
             running native LoRA training against the supplied libtorch module"
        );
    }

    fs::create_dir_all(&cfg.output).await?;
    let started = std::time::Instant::now();
    let plan = hardware_plan();
    apply_cpu_thread_limits(plan.cpu_threads);

    let texts = load_training_texts(&cfg.dataset).await?;
    if texts.is_empty() {
        anyhow::bail!("dataset is empty after message parsing");
    }

    let rendered_dataset = cfg.output.join("dataset.rendered.jsonl");
    write_rendered_dataset(&rendered_dataset, &texts).await?;

    let cfg_for_training = cfg.clone();
    let plan_for_training = plan.clone();
    let metrics = tokio::task::spawn_blocking(move || {
        train_with_tch(&cfg_for_training, &plan_for_training, &texts)
    })
    .await??;

    let adapter_dir = cfg.output.join("adapter");
    fs::create_dir_all(&adapter_dir).await?;
    write_adapter_manifest(&cfg, &plan, &metrics, &adapter_dir).await?;

    let modelfile = cfg.output.join("Modelfile");
    write_modelfile(&cfg, &adapter_dir, &modelfile).await?;

    let report = json!({
        "algorithm": cfg.algorithm,
        "backend": "tch_libtorch_native",
        "base_model": cfg.base_model,
        "model_module": cfg.model_module.to_string_lossy(),
        "tokenizer": cfg.tokenizer.to_string_lossy(),
        "device": plan.device_label,
        "device_index": plan.device_index,
        "gpu_count": plan.gpu_count,
        "cpu_threads": plan.cpu_threads,
        "compute_dtype": plan.compute_dtype,
        "quantisation": plan.quantisation,
        "samples": metrics.samples,
        "batches": metrics.batches,
        "trainable_parameters": metrics.trainable_parameters,
        "trainable_tensors": metrics.trainable_tensors,
        "total_steps": metrics.total_steps,
        "training_loss": metrics.final_loss,
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
    plan: &HardwarePlan,
    texts: &[String],
) -> anyhow::Result<TrainingMetrics> {
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

    let samples = tokenise_parallel(&tokenizer, texts, cfg.max_seq_len)?;
    let batches = build_batches(&samples, cfg.batch_size, cfg.max_seq_len, pad_id, device)?;
    if batches.is_empty() {
        anyhow::bail!("no trainable batches were produced from the dataset");
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

    let total_micro_steps = ((cfg.epochs.max(0.0) * batches.len() as f64).ceil() as usize).max(1);
    let warmup_steps =
        ((cfg.warmup_ratio.max(0.0) * total_micro_steps as f64).round() as usize).max(1);
    let accumulation = cfg.gradient_accumulation_steps.max(1);
    let mut final_loss = None;
    let mut optimiser_step = 0_usize;
    let started = std::time::Instant::now();

    zero_param_grads(&mut params);
    for micro_step in 0..total_micro_steps {
        let batch = &batches[micro_step % batches.len()];
        let loss = match cfg.forward_mode {
            ForwardMode::Loss => module
                .forward_ts(&[
                    batch.input_ids.shallow_clone(),
                    batch.labels.shallow_clone(),
                ])
                .map_err(|error| anyhow::anyhow!("forward(loss) failed: {error}"))?,
            ForwardMode::Logits => {
                let logits = module
                    .forward_ts(&[batch.input_ids.shallow_clone()])
                    .map_err(|error| anyhow::anyhow!("forward(logits) failed: {error}"))?;
                causal_lm_loss(logits, batch.labels.shallow_clone())?
            }
        };
        let scaled_loss = &loss / accumulation as f64;
        scaled_loss.backward();
        final_loss = Some(loss.double_value(&[]));

        if (micro_step + 1) % accumulation == 0 || micro_step + 1 == total_micro_steps {
            optimiser_step += 1;
            let lr = scheduled_lr(cfg.learning_rate, optimiser_step, warmup_steps);
            adamw_step(&mut params, optimiser_step, lr, cfg)?;
            zero_param_grads(&mut params);
        }

        if micro_step == 0 || (micro_step + 1) % 10 == 0 || micro_step + 1 == total_micro_steps {
            eprintln!(
                "native_tch_train step={}/{} optimiser_step={} loss={:.6} device={} trainable_params={}",
                micro_step + 1,
                total_micro_steps,
                optimiser_step,
                final_loss.unwrap_or(f64::NAN),
                plan.device_label,
                trainable_parameters
            );
        }
    }

    let adapter_dir = cfg.output.join("adapter");
    std::fs::create_dir_all(&adapter_dir)?;
    let adapter_module = adapter_dir.join("adapter.pt");
    module
        .save(&adapter_module)
        .map_err(|error| anyhow::anyhow!("failed to save trained TorchScript module: {error}"))?;

    Ok(TrainingMetrics {
        samples: samples.len(),
        batches: batches.len(),
        trainable_parameters,
        trainable_tensors: tensor_names,
        total_steps: optimiser_step,
        final_loss,
        runtime_seconds: started.elapsed().as_secs_f64(),
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
            let m = Tensor::zeros_like(&tensor);
            let v = Tensor::zeros_like(&tensor);
            params.push(TrainableParam { name, tensor, m, v });
        } else {
            let _ = tensor.set_requires_grad(false);
        }
    }
    Ok(params)
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
            param.m = &param.m * beta1 + &grad * (1.0 - beta1);
            param.v = &param.v * beta2 + grad.square() * (1.0 - beta2);
            let m_hat = &param.m / bias_correction1;
            let v_hat = &param.v / bias_correction2;
            let mut update = &m_hat / (v_hat.sqrt() + cfg.adam_eps);
            if cfg.weight_decay > 0.0 {
                update = update + &param.tensor * cfg.weight_decay;
            }
            let delta = update * lr;
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

fn build_batches(
    samples: &[TokenisedSample],
    batch_size: usize,
    max_seq_len: usize,
    pad_id: i64,
    device: Device,
) -> anyhow::Result<Vec<Batch>> {
    let batch_size = batch_size.max(1);
    let seq_len = max_seq_len.max(8);
    let mut batches = Vec::new();
    for chunk in samples.chunks(batch_size) {
        let mut input_rows = Vec::with_capacity(chunk.len() * seq_len);
        let mut label_rows = Vec::with_capacity(chunk.len() * seq_len);
        for sample in chunk {
            let mut ids = sample.ids.clone();
            ids.truncate(seq_len);
            let valid = ids.len();
            ids.resize(seq_len, pad_id);
            let mut labels = ids.clone();
            for label in labels.iter_mut().skip(valid) {
                *label = -100;
            }
            input_rows.extend_from_slice(&ids);
            label_rows.extend_from_slice(&labels);
        }
        let bsz = chunk.len() as i64;
        let input_ids = Tensor::from_slice(&input_rows)
            .to_kind(Kind::Int64)
            .reshape([bsz, seq_len as i64])
            .to_device(device);
        let labels = Tensor::from_slice(&label_rows)
            .to_kind(Kind::Int64)
            .reshape([bsz, seq_len as i64])
            .to_device(device);
        batches.push(Batch { input_ids, labels });
    }
    Ok(batches)
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

async fn write_adapter_manifest(
    cfg: &TrainingConfig,
    plan: &HardwarePlan,
    metrics: &TrainingMetrics,
    adapter_dir: &Path,
) -> anyhow::Result<()> {
    let manifest = json!({
        "format": "gail-tch-lora-adapter",
        "backend": "tch_libtorch_native",
        "algorithm": cfg.algorithm,
        "base_model": cfg.base_model,
        "model_module": cfg.model_module.to_string_lossy(),
        "trainable_marker": cfg.trainable_marker,
        "device": plan.device_label,
        "quantisation": plan.quantisation,
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout
        },
        "metrics": metrics,
        "note": "adapter.pt is a TorchScript/libtorch artefact. Convert/export to the target serving adapter format before using a runtime that expects PEFT safetensors."
    });
    fs::write(
        adapter_dir.join("adapter_manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .await?;
    Ok(())
}

async fn write_modelfile(
    cfg: &TrainingConfig,
    adapter_dir: &Path,
    modelfile: &Path,
) -> anyhow::Result<()> {
    let rendered = format!(
        "FROM {}\n# Native Rust tch/libtorch training artefact: {}\n# Convert adapter.pt to your serving adapter format if Ollama requires PEFT safetensors.\nPARAMETER temperature 0.2\nSYSTEM {}\n",
        cfg.base_model,
        adapter_dir.join("adapter.pt").display(),
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
    let mut system_prompt = env_or("GAIL_TRAIN_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT);
    let mut timeout_seconds = parse_env("GAIL_TRAIN_COMMAND_TIMEOUT_SECONDS", 86_400_u64);
    let mut trainable_marker = env_or("GAIL_TCH_TRAINABLE_MARKER", "lora");
    let mut forward_mode = parse_forward_mode(&env_or("GAIL_TCH_FORWARD_MODE", "loss"))?;

    let mut index = 0;
    while index < args.len() {
        let flag = args[index].to_string_lossy().to_string();
        if flag == "--help" || flag == "-h" {
            anyhow::bail!("{}", help_text());
        }
        let mut next = |index: &mut usize, flag: &str| -> anyhow::Result<String> {
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
            "--system-prompt" => system_prompt = next(&mut index, &flag)?,
            "--timeout-seconds" => timeout_seconds = parse_value(&next(&mut index, &flag)?, &flag)?,
            "--trainable-marker" => trainable_marker = next(&mut index, &flag)?,
            "--forward-mode" => forward_mode = parse_forward_mode(&next(&mut index, &flag)?)?,
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
        epochs,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
        warmup_ratio,
        max_seq_len,
        lora_r,
        lora_alpha,
        lora_dropout,
        weight_decay,
        adam_beta1,
        adam_beta2,
        adam_eps,
        system_prompt,
        timeout_seconds,
        trainable_marker,
        forward_mode,
    })
}

fn help_text() -> &'static str {
    "Usage: gail-qlora-sft --dataset <path> --output <dir> \\\n       --model-module <torchscript.pt> --tokenizer <tokenizer.json> \\\n       [--algorithm qlora_sft|lora_sft] [--forward-mode loss|logits]"
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

fn hardware_plan() -> HardwarePlan {
    let cpu_threads = parse_env("GAIL_TRAIN_CPU_THREADS", num_cpus_fallback()).max(1);
    let gpu_count = Cuda::device_count().max(0) as usize;
    let use_gpu = env::var("GAIL_TRAIN_DEVICE")
        .map(|value| value.eq_ignore_ascii_case("cuda") || value.eq_ignore_ascii_case("gpu"))
        .unwrap_or(gpu_count > 0);
    let device_index = if use_gpu && Cuda::is_available() && gpu_count > 0 {
        Some(0)
    } else {
        None
    };
    let device_label = device_index
        .map(|index| format!("cuda:{index}"))
        .unwrap_or_else(|| "cpu".to_string());
    let compute_dtype = if device_index.is_some() {
        env_or(
            "GAIL_TCH_COMPUTE_DTYPE",
            "float16_or_bfloat16_module_defined",
        )
    } else {
        "float32".to_string()
    };
    let quantisation = if env_bool("GAIL_TCH_BASE_PREQUANTISED", false) {
        "prequantised_base".to_string()
    } else {
        "none".to_string()
    };
    HardwarePlan {
        device_label,
        device_index,
        gpu_count,
        cpu_threads,
        compute_dtype,
        quantisation,
    }
}

fn tch_device(plan: &HardwarePlan) -> Device {
    plan.device_index.map(Device::Cuda).unwrap_or(Device::Cpu)
}

unsafe fn apply_cpu_thread_limits(cpu_threads: usize) {
    let threads = cpu_threads.max(1) as i32;
    tch::set_num_threads(threads);
    tch::set_num_interop_threads((threads / 2).max(1));
    env::set_var("RAYON_NUM_THREADS", cpu_threads.to_string());
    env::set_var("OMP_NUM_THREADS", cpu_threads.to_string());
    env::set_var("MKL_NUM_THREADS", cpu_threads.to_string());
    env::set_var("OPENBLAS_NUM_THREADS", cpu_threads.to_string());
}

fn scheduled_lr(base_lr: f64, step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps == 0 || step >= warmup_steps {
        base_lr
    } else {
        base_lr * (step as f64 / warmup_steps as f64).max(1.0 / warmup_steps as f64)
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
