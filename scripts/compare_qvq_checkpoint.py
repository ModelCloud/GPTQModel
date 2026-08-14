#!/usr/bin/env python3
"""Compare one QVQ checkpoint with its dense model on held-out text rows.

The evaluator captures final logits and selected decoder-layer outputs from the
same non-padding tokens.  Run one process per GPU/checkpoint; every result is
self-contained and can therefore be compared without retaining a dense model
or vocabulary-sized logits between sweep arms.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel

if __package__:
    from scripts.analyze_gptq_low_bit_grid import capture_forward, load_nm_evaluation_batch, tensor_metrics
else:
    from analyze_gptq_low_bit_grid import capture_forward, load_nm_evaluation_batch, tensor_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("neuralmagic/calibration"))
    parser.add_argument("--row-offset", type=int, default=128)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_dense(path: Path, device: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16,
        device_map={"": device},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 1:
        raise ValueError("layer count must be positive")
    if args.row_offset < 0 or args.rows < 1 or args.max_length < 1:
        raise ValueError("evaluation row offset must be nonnegative and sizes must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, evaluation = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.row_offset,
        rows=args.rows,
        max_length=args.max_length,
    )
    encoded = {name: value.to(args.device) for name, value in encoded.items()}

    dense_load_started = time.perf_counter()
    dense = _load_dense(args.dense_model, args.device)
    dense_load_seconds = time.perf_counter() - dense_load_started
    dense_forward_started = time.perf_counter()
    dense_logits, _, dense_outputs = capture_forward(
        dense,
        encoded,
        {},
        capture_inputs=False,
        layer_count=args.layers,
    )
    torch.cuda.synchronize(args.device)
    dense_forward_seconds = time.perf_counter() - dense_forward_started
    del dense
    gc.collect()
    torch.cuda.empty_cache()

    quantized_load_started = time.perf_counter()
    quantized = GPTQModel.load(
        str(args.checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    quantized_load_seconds = time.perf_counter() - quantized_load_started
    quantized_forward_started = time.perf_counter()
    quantized_logits, _, quantized_outputs = capture_forward(
        quantized.model,
        encoded,
        {},
        capture_inputs=False,
        layer_count=args.layers,
    )
    torch.cuda.synchronize(args.device)
    quantized_forward_seconds = time.perf_counter() - quantized_forward_started

    report = {
        "dense_model": str(args.dense_model),
        "checkpoint": str(args.checkpoint),
        "commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip(),
        "python": platform.python_version(),
        "python_gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "torch": torch.__version__,
        "dtype": str(torch.float16),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "evaluation": evaluation,
        "seconds": {
            "dense_load": dense_load_seconds,
            "dense_forward": dense_forward_seconds,
            "quantized_load": quantized_load_seconds,
            "quantized_forward": quantized_forward_seconds,
        },
        "logits": tensor_metrics(dense_logits, quantized_logits, normalize_distribution=False),
        "layers": {
            f"layer.{index}": tensor_metrics(
                dense_outputs[f"layer.{index}.hidden"],
                quantized_outputs[f"layer.{index}.hidden"],
                normalize_distribution=True,
            )
            for index in range(args.layers)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
