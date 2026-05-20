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
import json
import os
import sys
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
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover
    raise _missing_dependency_error(
        "transformers", "pip install transformers bitsandbytes accelerate"
    ) from exc

try:
    from trl import SFTTrainer
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
    dataset_path = Path(cfg.dataset)
    output_root = Path(cfg.output)
    adapter_dir = output_root / "adapter"
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    )
    if device != "cuda":
        model = model.to(device)

    texts = load_training_texts(dataset_path, tokenizer)
    if not texts:
        raise RuntimeError("Dataset is empty after message parsing")
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

    training_args = TrainingArguments(
        output_dir=str(output_root / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        report_to=cfg.report_to,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_len,
        args=training_args,
        peft_config=peft_config,
    )
    train_result = trainer.train()

    trained_model = trainer.model
    if isinstance(trained_model, PeftModel):
        trained_model.save_pretrained(str(adapter_dir))
    else:
        # Keep behaviour explicit even if TRL internals change.
        raise RuntimeError("Expected a PEFT LoRA model but received a non-PEFT model")
    tokenizer.save_pretrained(str(adapter_dir))

    modelfile = output_root / "Modelfile"
    modelfile.write_text(
        (
            f"FROM {cfg.base_model}\n"
            f"ADAPTER {adapter_dir.resolve()}\n"
            "PARAMETER temperature 0.2\n"
            f"SYSTEM {cfg.system_prompt}\n"
        ),
        encoding="utf-8",
    )

    report = {
        "algorithm": cfg.algorithm,
        "base_model": cfg.base_model,
        "device": device,
        "samples": len(texts),
        "target_modules": target_modules,
        "training_loss": float(train_result.training_loss)
        if train_result.training_loss is not None
        else None,
        "adapter_dir": str(adapter_dir.resolve()),
        "modelfile": str(modelfile.resolve()),
    }
    (output_root / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
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
