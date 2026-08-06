#!/usr/bin/env python3
"""Convert a genuine Hugging Face PEFT LoRA adapter to a GGUF LoRA.

The llama.cpp converter is shipped in the Gail image at a pinned revision.
This small wrapper owns the production contract: it rejects the generic
TorchScript fixture emitted by the legacy trainer, resolves the HF base model
configuration, and verifies the output before Ollama is called.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import NoReturn


CONVERTER_ROOT = Path(os.environ.get("GAIL_LLAMA_CPP_CONVERTER_ROOT", "/opt/gail-llama-converter"))
CONVERTER = CONVERTER_ROOT / "convert_lora_to_gguf.py"


def python_converter_environment() -> dict[str, str]:
    """Return an environment suitable for the wheel-based Python converter.

    The Gail runtime exports ``LD_LIBRARY_PATH=/opt/libtorch/lib`` for the
    native Rust/tch service.  The training/conversion venv has its own PyTorch
    wheel, however, and loading Gail's libtorch beside that wheel can produce
    C++ ABI mismatches (for example in ``torch/lib/libshm.so``).  Keep the
    native process environment unchanged and remove only this inherited path
    for the Python subprocess.
    """

    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--base-model-id", default=os.environ.get("GAIL_GGUF_BASE_MODEL_ID", ""))
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--outtype", default="auto", choices=("auto", "f16", "bf16", "f32", "q8_0"))
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    print(f"gail GGUF conversion: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_adapter(adapter: Path) -> None:
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        fail("expected adapter_config.json and adapter_model.safetensors")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid adapter_config.json: {exc}")
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        fail("only PEFT LoRA adapters are supported")
    # Inspect metadata without loading the tensors into RAM.  Safetensors
    # headers are sufficient to distinguish a real PEFT adapter from the old
    # generic lora_down/lora_up TorchScript fixture.
    try:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            names = list(handle.keys())
    except Exception as exc:  # pragma: no cover - dependency/runtime detail
        fail(f"unable to inspect Safetensors adapter: {exc}")
    has_a = any(".lora_A." in name or ".lora_embedding_A" in name for name in names)
    has_b = any(".lora_B." in name or ".lora_embedding_B" in name for name in names)
    if not (has_a and has_b):
        fail("adapter has no paired PEFT lora_A/lora_B tensors; refusing synthetic or incompatible artifacts")


def main() -> int:
    args = parse_args()
    adapter = args.adapter.resolve()
    if not adapter.is_dir():
        fail(f"adapter directory does not exist: {adapter}")
    validate_adapter(adapter)
    if not CONVERTER.is_file():
        fail(f"pinned llama.cpp converter is not installed: {CONVERTER}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(CONVERTER),
        "--outfile",
        str(args.output),
        "--outtype",
        args.outtype,
        "--trust-remote-code",
    ]
    base = Path(args.base_model).expanduser() if args.base_model else None
    if base and base.is_dir():
        command.extend(("--base", str(base.resolve())))
    else:
        base_model_id = args.base_model_id
        if not base_model_id:
            # PEFT normally records the HF ID in adapter_config.json.  The
            # explicit flag is preferred for Ollama tags such as qwen3.5:4b.
            config = json.loads((adapter / "adapter_config.json").read_text())
            base_model_id = str(config.get("base_model_name_or_path", ""))
        if not base_model_id or ":" in base_model_id and "/" not in base_model_id:
            fail("a Hugging Face base model ID/path is required; set GAIL_GGUF_BASE_MODEL_ID")
        command.extend(("--base-model-id", base_model_id))
    command.append(str(adapter))
    print(f"converting LoRA snapshot {args.snapshot or adapter.name} with pinned llama.cpp", flush=True)
    completed = subprocess.run(
        command,
        check=False,
        env=python_converter_environment(),
    )
    if completed.returncode != 0:
        return completed.returncode
    with args.output.open("rb") as output:
        if output.read(4) != b"GGUF":
            fail(f"converter produced a non-GGUF output: {args.output}")
    print(f"wrote validated GGUF LoRA: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
