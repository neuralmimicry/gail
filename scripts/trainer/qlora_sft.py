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


def distributed_rank_output(output_root: Path, rank: int, world_size: int, job_id: str) -> Path:
    if world_size == 1:
        return output_root
    return output_root / ".distributed" / job_id / f"rank-{rank:05d}"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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
        output_root / ".distributed" / job_id / f"rank-{index:05d}"
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
    for filename in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
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
    rank_root = distributed_rank_output(output_root, rank, world_size, job_id)
    rank_root.mkdir(parents=True, exist_ok=True)
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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float32 if device == "cpu" else None,
    )
    if device != "cuda":
        model = model.to(device)

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
        save_strategy="epoch",
        report_to=cfg.report_to,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
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
        peft_config=peft_config,
    )
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
    train_result = trainer.train()

    trained_model = trainer.model
    if isinstance(trained_model, PeftModel):
        trained_model.save_pretrained(str(adapter_dir))
    else:
        # Keep behaviour explicit even if TRL internals change.
        raise RuntimeError("Expected a PEFT LoRA model but received a non-PEFT model")
    tokenizer.save_pretrained(str(adapter_dir))

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
        "samples": len(texts),
        "target_modules": target_modules,
        "training_loss": float(train_result.training_loss)
        if train_result.training_loss is not None
        else None,
        "adapter_dir": str(adapter_dir.resolve()),
        "modelfile": str(modelfile.resolve()),
    }
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
        aggregate_distributed_snapshot(cfg, output_root, rank_root, rank, world_size, job_id, sample_counts)
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
