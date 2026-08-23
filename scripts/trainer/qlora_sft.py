#!/usr/bin/env python3
"""
QLoRA supervised fine-tuning runner for Gail trainer snapshots.

Input dataset format:
  JSONL where each row includes a `messages` array with ChatML-like
  `{"role": "...", "content": "..."}` entries.

Outputs:
  - <output>/adapter/         (LoRA adapter weights/tokenizer)
  - <output>/Modelfile        (Ollama model definition that attaches the adapter)
  - <output>/training_report.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


def _missing_dependency_error(name: str, install_hint: str) -> RuntimeError:
    return RuntimeError(
        f"Missing Python dependency '{name}'. Install with: {install_hint}"
    )


try:
    import torch
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error("torch", "pip install torch") from exc

try:
    from datasets import Dataset
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error("datasets", "pip install datasets") from exc

try:
    from peft import LoraConfig, PeftModel
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error("peft", "pip install peft") from exc

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error(
        "transformers", "pip install transformers bitsandbytes accelerate"
    ) from exc

try:
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except ImportError:  # Older TRL releases use TrainingArguments directly.
        SFTConfig = None
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error("trl", "pip install trl") from exc


SUPPORTED_ALGORITHMS = {"qlora_sft", "lora_sft"}


@dataclass
class TrainingConfig:
    dataset: str
    output: str
    base_model: str
    algorithm: str
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    system_prompt: str
    report_to: str
    ollama_base_model: str
    save_steps: int
    save_total_limit: int
    resume_adapter: str


def slurm_context() -> tuple[int, int, str]:
    """Return the rank context when launched by the Gail Slurm dispatcher."""
    rank = int(os.getenv("SLURM_PROCID", "0"))
    world_size = int(os.getenv("SLURM_NTASKS", "1"))
    job_id = os.getenv("SLURM_JOB_ID", "local")
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise RuntimeError(
            f"invalid Slurm rank context: rank={rank}, world_size={world_size}"
        )
    return rank, world_size, job_id


def distributed_rank_output(output_root: Path, rank: int, world_size: int) -> Path:
    """Use snapshot-stable paths so a new Slurm allocation can resume."""
    if world_size == 1:
        return output_root
    return output_root / ".distributed" / f"rank-{rank:05d}"


def latest_valid_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir() or not (path / "trainer_state.json").is_file():
            continue
        if not (path / "optimizer.pt").is_file() or not (path / "scheduler.pt").is_file():
            continue
        if not (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        ):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


TOKENIZER_PROBE_TEXT = "Gail tokenizer probe: BTC/USDT 123."


def resolve_tokenizer_metadata(tokenizer) -> dict:
    """Resolve and verify tokenizer IDs before model loading/training."""
    ids = {}
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        ids[name] = int(value) if value is not None else None
    encoded = tokenizer(TOKENIZER_PROBE_TEXT, add_special_tokens=True)
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    if not input_ids:
        raise RuntimeError("tokenizer startup probe produced no input IDs")
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    if not str(decoded).strip() or TOKENIZER_PROBE_TEXT.split()[0] not in str(decoded):
        raise RuntimeError(
            "tokenizer startup probe failed round-trip: "
            f"decoded={decoded!r}"
        )
    return {
        "pad_token_id": ids["pad_token_id"],
        "bos_token_id": ids["bos_token_id"],
        "eos_token_id": ids["eos_token_id"],
        "probe": {
            "text": TOKENIZER_PROBE_TEXT,
            "token_count": len(input_ids),
            "decoded": str(decoded),
            "round_trip_ok": True,
        },
    }


def align_model_special_tokens(model, metadata: dict) -> None:
    """Make model and generation configs use the persisted tokenizer IDs."""
    for config in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if config is None:
            continue
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = metadata.get(name)
            if value is not None:
                setattr(config, name, int(value))


class TrainingProgressCallback(TrainerCallback):
    """Publish a scrape-friendly progress snapshot while a task is running."""

    def __init__(self, path: Path, snapshot_id: str, started_ts: float, rank: int, world_size: int, job_id: str):
        self.path = path
        self.snapshot_id = snapshot_id
        self.started_ts = started_ts
        self.rank = rank
        self.world_size = world_size
        self.job_id = job_id

    def _write(self, state, status: str = "running") -> None:
        now = time.time()
        completed = max(0, int(getattr(state, "global_step", 0)))
        total = max(completed, int(getattr(state, "max_steps", 0)))
        elapsed = max(0.0, now - self.started_ts)
        ratio = min(1.0, completed / total) if total else 0.0
        eta = (elapsed / completed * (total - completed)) if completed and total else 0.0
        payload = {
            "snapshot_id": self.snapshot_id,
            "status": status,
            "backend": "slurm" if self.world_size > 1 else "local",
            "slurm_job_id": self.job_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "completed_steps": completed,
            "total_steps": total,
            "progress_ratio": ratio,
            "progress_per_hour": completed / elapsed * 3600.0 if elapsed else 0.0,
            "eta_seconds": max(0.0, eta),
            "elapsed_seconds": elapsed,
            "started_ts": self.started_ts,
            "updated_ts": now,
        }
        temporary = self.path.with_suffix(f".tmp-{os.getpid()}")
        write_json(temporary, payload)
        temporary.replace(self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(state)

    def on_log(self, args, state, control, **kwargs):
        self._write(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, "completed")


def aggregate_distributed_snapshot(
    cfg: TrainingConfig,
    output_root: Path,
    rank_root: Path,
    rank: int,
    world_size: int,
    job_id: str,
    sample_counts: list[int],
) -> None:
    """Federated-average PEFT adapters after every Slurm rank completes.

    Each rank trains only its deterministic slice of the snapshot.  Rank zero
    then combines the genuine PEFT tensors by sample count, preserving the
    adapter names required by the llama.cpp converter.
    """
    timeout = max(60, int(os.getenv("GAIL_TRAIN_DISTRIBUTED_TIMEOUT_SECONDS", "82800")))
    deadline = time.monotonic() + timeout
    rank_roots = [
        output_root / ".distributed" / f"rank-{index:05d}"
        for index in range(world_size)
    ]
    while True:
        missing = [root for root in rank_roots if not (root / "_SUCCESS").is_file()]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for distributed PEFT ranks: "
                + ", ".join(str(root) for root in missing)
            )
        time.sleep(2)

    from safetensors.torch import load_file, save_file

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise RuntimeError("distributed PEFT snapshot has no training samples")
    averaged = {}
    expected_names = None
    for index, root in enumerate(rank_roots):
        tensors = load_file(str(root / "adapter" / "adapter_model.safetensors"), device="cpu")
        names = set(tensors)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError(f"rank {index} exported a different PEFT tensor set")
        weight = sample_counts[index] / total_samples
        for name, tensor in tensors.items():
            value = tensor.float() * weight
            averaged[name] = value if name not in averaged else averaged[name] + value

    adapter_dir = output_root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(averaged, str(adapter_dir / "adapter_model.safetensors"))
    for filename in (
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_metadata.json",
        "training_manifest.json",
    ):
        source = rank_roots[0] / "adapter" / filename
        if source.is_file():
            shutil.copy2(source, adapter_dir / filename)

    shutil.copy2(rank_roots[0] / "Modelfile", output_root / "Modelfile")
    reports = []
    for root in rank_roots:
        report_path = root / "training_report.json"
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    report = {
        "algorithm": cfg.algorithm,
        "base_model": cfg.base_model,
        "ollama_base_model": cfg.ollama_base_model,
        "backend": "slurm_distributed_peft",
        "distributed": {
            "world_size": world_size,
            "slurm_job_id": job_id,
            "strategy": "deterministic_sharded_federated_average",
            "sample_weights": sample_counts,
            "total_samples": total_samples,
        },
        "rank_reports": reports,
        "adapter_dir": str(adapter_dir.resolve()),
    }
    write_json(output_root / "training_report.json", report)
    (output_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a Gail QLoRA adapter snapshot")
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset path")
    parser.add_argument("--output", required=True, help="Snapshot output directory")
    parser.add_argument(
        "--base-model",
        default=os.getenv("GAIL_TRAIN_BASE_MODEL", "qwen2.5-coder:1.5b"),
        help="Base HF model ID used for training",
    )
    parser.add_argument(
        "--algorithm",
        default=os.getenv("GAIL_TRAIN_ALGORITHM", "qlora_sft"),
        choices=sorted(SUPPORTED_ALGORITHMS),
        help="Training algorithm",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are the Gail in-house continuously trained model. "
            "Use prior interaction learning when useful."
        ),
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--ollama-base-model",
        default=os.getenv("GAIL_TRAIN_OLLAMA_BASE_MODEL", ""),
        help="Ollama model tag used in the generated Modelfile (the training base may be a HF ID)",
    )
    parser.add_argument(
        "--save-steps", type=int,
        default=int(os.getenv("GAIL_TRAIN_SAVE_STEPS", "100")),
        help="Periodic durable checkpoint interval in optimizer steps",
    )
    parser.add_argument(
        "--save-total-limit", type=int,
        default=int(os.getenv("GAIL_TRAIN_SAVE_TOTAL_LIMIT", "3")),
        help="Number of periodic checkpoints to retain per rank",
    )
    parser.add_argument(
        "--resume-adapter",
        default=os.getenv("GAIL_TRAIN_RESUME_ADAPTER", ""),
        help="Previously promoted PEFT adapter directory to continue training",
    )
    args = parser.parse_args(argv)
    return TrainingConfig(
        dataset=args.dataset,
        output=args.output,
        base_model=args.base_model,
        algorithm=args.algorithm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        system_prompt=args.system_prompt,
        report_to=args.report_to,
        ollama_base_model=args.ollama_base_model,
        save_steps=max(1, args.save_steps),
        save_total_limit=max(1, args.save_total_limit),
        resume_adapter=args.resume_adapter.strip(),
    )


def _manual_chat_template(messages: Iterable[dict]) -> str:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"<|{role}|>\n{content}")
    return "\n".join(lines).strip()


def load_training_texts(dataset_path: Path, tokenizer) -> List[str]:
    texts: List[str] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            messages = row.get("messages") or []
            if not isinstance(messages, list) or not messages:
                continue
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    rendered = _manual_chat_template(messages)
            else:
                rendered = _manual_chat_template(messages)
            rendered = str(rendered).strip()
            if rendered:
                texts.append(rendered)
    return texts


def infer_lora_targets(model) -> List[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    discovered = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred:
            discovered.add(leaf)
    if discovered:
        return sorted(discovered)
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def train(cfg: TrainingConfig) -> None:
    rank, world_size, job_id = slurm_context()
    dataset_path = Path(cfg.dataset)
    output_root = Path(cfg.output)
    rank_root = distributed_rank_output(output_root, rank, world_size)
    rank_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output_root / "_SUCCESS").unlink(missing_ok=True)
    started_ts = time.time()
    progress_path = (output_root if rank == 0 else rank_root) / "progress.json"
    adapter_dir = output_root / "adapter"
    adapter_dir = rank_root / "adapter"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # The qc0x arm64 hosts do not expose CPU BF16 instructions. Qwen
        # checkpoints default to BF16, which makes Torch repeatedly enter a
        # slow unsupported-BF16 fallback in every linear layer. Use the
        # native float32 BLAS path and honour the Slurm core allocation.
        torch.set_num_threads(max(1, int(os.getenv("GAIL_TRAIN_CPU_INTRAOP_THREADS", "1"))))
        torch.set_num_interop_threads(max(1, int(os.getenv("GAIL_TRAIN_CPU_INTEROP_THREADS", "1"))))
    quant_config = None
    if cfg.algorithm == "qlora_sft" and device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_metadata = resolve_tokenizer_metadata(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else None,
        dtype=torch.float32 if device == "cpu" else None,
    )
    align_model_special_tokens(model, tokenizer_metadata)
    if device != "cuda":
        model = model.to(device)

    resume_adapter = Path(cfg.resume_adapter) if cfg.resume_adapter else None
    if resume_adapter is not None:
        if not resume_adapter.is_dir():
            raise RuntimeError(f"promoted resume adapter does not exist: {resume_adapter}")
        model = PeftModel.from_pretrained(
            model,
            str(resume_adapter),
            is_trainable=True,
        )

    texts = load_training_texts(dataset_path, tokenizer)
    if world_size > 1:
        texts = texts[rank::world_size]
    if not texts:
        raise RuntimeError(f"rank {rank} has no training samples after deterministic sharding")
    train_dataset = Dataset.from_dict({"text": texts})

    target_modules = infer_lora_targets(model)
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    training_kwargs = dict(
        output_dir=str(rank_root / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=device == "cuda",
    )
    training_parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in training_parameters:
        training_kwargs["warmup_ratio"] = cfg.warmup_ratio
    else:
        # Transformers 5 removed warmup_ratio. The tiny CPU smoke runs used
        # by Gail should not invent a step count, so retain the safe zero-step
        # default when only warmup_steps is available.
        training_kwargs["warmup_steps"] = 0

    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if SFTConfig is not None and "max_length" in inspect.signature(SFTConfig).parameters:
        training_kwargs["dataset_text_field"] = "text"
        training_kwargs["max_length"] = cfg.max_seq_len
        training_args = SFTConfig(**training_kwargs)
    else:
        training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )
    if resume_adapter is None:
        trainer_kwargs["peft_config"] = peft_config
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_parameters:
        trainer_kwargs["max_seq_length"] = cfg.max_seq_len
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.add_callback(
        TrainingProgressCallback(progress_path, output_root.name, started_ts, rank, world_size, job_id)
    )
    checkpoint = latest_valid_checkpoint(rank_root / "checkpoints")
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    trained_model = trainer.model
    if isinstance(trained_model, PeftModel):
        trained_model.save_pretrained(str(adapter_dir))
    else:
        # Keep behaviour explicit even if TRL internals change.
        raise RuntimeError("Expected a PEFT LoRA model but received a non-PEFT model")
    tokenizer.save_pretrained(str(adapter_dir))
    # Some tokenizer implementations omit auxiliary files when they are
    # loaded from a minimal local artifact. Keep the snapshot contract
    # explicit so promotion validation is deterministic.
    if not (adapter_dir / "tokenizer_config.json").is_file():
        write_json(adapter_dir / "tokenizer_config.json", tokenizer.init_kwargs)
    if not (adapter_dir / "special_tokens_map.json").is_file():
        write_json(
            adapter_dir / "special_tokens_map.json",
            {
                "pad_token": tokenizer.pad_token,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token,
            },
        )
    write_json(
        adapter_dir / "tokenizer_metadata.json",
        tokenizer_metadata,
    )

    effective_backend = "cuda_qlora" if device == "cuda" and cfg.algorithm == "qlora_sft" else (
        "cuda_lora" if device == "cuda" else "cpu_lora"
    )
    cpu_fallback = device != "cuda" and cfg.algorithm == "qlora_sft"
    print(
        "GAIL_TRAINING_BACKEND "
        f"requested_algorithm={cfg.algorithm} effective_backend={effective_backend} "
        f"device={device} quantisation={'4bit_nf4' if quant_config is not None else 'none'} "
        f"pin_memory={device == 'cuda'} cpu_fallback={cpu_fallback}",
        file=sys.stderr,
    )

    modelfile = rank_root / "Modelfile"
    ollama_base_model = cfg.ollama_base_model.strip() or cfg.base_model
    # Modelfile paths are interpreted by Gail's promotion worker and must be
    # relative to the snapshot.  An absolute path is both unsafe and rejected
    # by the adapter manifest builder.
    modelfile.write_text(
        (
            f"FROM {ollama_base_model}\n"
            "ADAPTER ./adapter\n"
            "PARAMETER temperature 0.2\n"
            f"SYSTEM {cfg.system_prompt}\n"
        ),
        encoding="utf-8",
    )

    report = {
        "algorithm": cfg.algorithm,
        "base_model": cfg.base_model,
        "ollama_base_model": ollama_base_model,
        "device": device,
        "backend": effective_backend,
        "effective_algorithm": "qlora_sft" if effective_backend == "cuda_qlora" else "lora_sft",
        "quantisation_backend": "4bit_nf4" if quant_config is not None else "none",
        "pin_memory": device == "cuda",
        "cpu_fallback": cpu_fallback,
        "tokenizer_metadata": tokenizer_metadata,
        "samples": len(texts),
        "target_modules": target_modules,
        "training_loss": float(train_result.training_loss)
        if train_result.training_loss is not None
        else None,
        "adapter_dir": str(adapter_dir.resolve()),
        "modelfile": str(modelfile.resolve()),
        "resume_adapter": str(resume_adapter.resolve()) if resume_adapter else None,
        "cumulative_training": resume_adapter is not None,
    }
    write_json(
        adapter_dir / "training_manifest.json",
        {
            "format": "gail-lora-training-manifest",
            "algorithm_requested": cfg.algorithm,
            "algorithm_executed": report["effective_algorithm"],
            "backend": effective_backend,
            "device": device,
            "cpu_fallback": cpu_fallback,
            "quantisation_backend": report["quantisation_backend"],
            "pin_memory": report["pin_memory"],
            "base_model": cfg.base_model,
            "tokenizer": "tokenizer.json",
            "tokenizer_metadata": tokenizer_metadata,
            "tokenizer_files": [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            "adapter_model": "adapter_model.safetensors",
            "adapter_config": "adapter_config.json",
            "cumulative_training": resume_adapter is not None,
        },
    )
    report["slurm"] = {"rank": rank, "world_size": world_size, "job_id": job_id}
    write_json(rank_root / "training_report.json", report)
    if world_size == 1:
        (rank_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        return
    (rank_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    if rank == 0:
        sample_counts = []
        for index in range(world_size):
            peer_report = rank_root.parent / f"rank-{index:05d}" / "training_report.json"
            deadline = time.monotonic() + max(60, int(os.getenv("GAIL_TRAIN_DISTRIBUTED_TIMEOUT_SECONDS", "82800")))
            while not peer_report.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank report: {peer_report}")
                time.sleep(2)
            sample_counts.append(int(json.loads(peer_report.read_text(encoding="utf-8"))["samples"]))
        aggregate_distributed_snapshot(
            cfg, output_root, rank_root, rank, world_size, job_id, sample_counts
        )
    print(json.dumps(report))


def main(argv: Sequence[str]) -> int:
    cfg = parse_args(argv)
    if not Path(cfg.dataset).exists():
        print(f"Dataset not found: {cfg.dataset}", file=sys.stderr)
        return 2
    try:
        train(cfg)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
