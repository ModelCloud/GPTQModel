#!/usr/bin/env python3
"""Compare one QVQ checkpoint with its dense model on held-out text rows.

The evaluator captures final logits and selected decoder-layer outputs from the
same non-padding tokens.  Run one process per GPU/checkpoint; every result is
self-contained and can therefore be compared without retaining a dense model
or vocabulary-sized logits between sweep arms.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel

if __package__:
    from scripts.analyze_gptq_low_bit_grid import (
        capture_forward,
        load_nm_evaluation_batch,
        tensor_metrics,
    )
    from scripts.compare_qvq_codecs_llama_qkvo import (
        _independent_greedy_divergence_metrics,
    )
else:
    from analyze_gptq_low_bit_grid import (
        capture_forward,
        load_nm_evaluation_batch,
        tensor_metrics,
    )
    from compare_qvq_codecs_llama_qkvo import _independent_greedy_divergence_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("neuralmagic/calibration"))
    parser.add_argument("--row-offset", type=int, default=128)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument(
        "--full-rows",
        action="store_true",
        help="Preserve each selected evaluation row at its full tokenized length.",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--include-topn", action="store_true")
    parser.add_argument("--divergence-rows", type=int, default=300)
    parser.add_argument("--divergence-tokens", type=int, default=32)
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
    if args.row_offset < 0 or args.rows < 1 or args.max_length < 1 or args.eval_batch_size < 1:
        raise ValueError("evaluation row offset must be nonnegative and sizes must be positive")
    if args.divergence_rows < 1 or args.divergence_tokens < 1:
        raise ValueError("divergence rows and token horizon must be positive")
    if args.eval_batch_size != 1:
        raise ValueError("snapshot divergence validation requires --eval-batch-size 1")
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
        max_length=None if args.full_rows else args.max_length,
    )
    dense_load_started = time.perf_counter()
    dense = _load_dense(args.dense_model, args.device)
    dense_load_seconds = time.perf_counter() - dense_load_started
    quantized_load_started = time.perf_counter()
    quantized = GPTQModel.load(
        str(args.checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    quantized_load_seconds = time.perf_counter() - quantized_load_started

    from scripts.compare_qvq_codecs_llama_qkvo import _WeightedMetricAccumulator

    logit_accumulator = _WeightedMetricAccumulator()
    layer_accumulators = {
        f"layer.{index}": _WeightedMetricAccumulator() for index in range(args.layers)
    }
    divergence_values = []
    dense_forward_seconds = 0.0
    quantized_forward_seconds = 0.0
    row_count = int(encoded["input_ids"].shape[0])
    for row_index in range(row_count):
        row = {
            name: value[row_index : row_index + 1].to(args.device, non_blocking=True)
            for name, value in encoded.items()
        }
        started = time.perf_counter()
        dense_row_logits, _, dense_row_outputs = capture_forward(
            dense, row, {}, capture_inputs=False, layer_count=args.layers
        )
        torch.cuda.synchronize(args.device)
        dense_forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        quantized_row_logits, _, quantized_row_outputs = capture_forward(
            quantized.model, row, {}, capture_inputs=False, layer_count=args.layers
        )
        torch.cuda.synchronize(args.device)
        quantized_forward_seconds += time.perf_counter() - started
        token_count = dense_row_logits.shape[0]
        logit_accumulator.add(
            tensor_metrics(dense_row_logits, quantized_row_logits, normalize_distribution=False),
            rows=token_count,
        )
        for name, accumulator in layer_accumulators.items():
            accumulator.add(
                tensor_metrics(
                    dense_row_outputs[f"{name}.hidden"],
                    quantized_row_outputs[f"{name}.hidden"],
                    normalize_distribution=True,
                ),
                rows=token_count,
            )
        if row_index < args.divergence_rows:
            metric = _independent_greedy_divergence_metrics(
                dense,
                quantized,
                row,
                token_count=args.divergence_tokens,
            )
            divergence_values.append({name: float(value.item()) for name, value in metric.items()})
        del row, dense_row_logits, quantized_row_logits, dense_row_outputs, quantized_row_outputs
        if row_index == 0 or (row_index + 1) % 16 == 0 or row_index + 1 == row_count:
            print(
                f"snapshot eval {row_index + 1}/{row_count} rows "
                f"tokens={logit_accumulator.weight} divergence={len(divergence_values)}",
                flush=True,
            )

    divergence = (
        {
            name: sum(value[name] for value in divergence_values) / len(divergence_values)
            for name in divergence_values[0]
        }
        if divergence_values
        else {}
    )
    divergence.update(
        {
            "requested_sequences": min(args.divergence_rows, row_count),
            "valid_sequences": len(divergence_values),
            "token_horizon": args.divergence_tokens,
            "protocol": "independent_greedy_rollout",
        }
    )
    logits_metrics = logit_accumulator.result()
    if not args.include_topn:
        for key in (
            "top5_overlap",
            "top5_exact_agreement",
            "dense_top1_in_quantized_top5",
            "quantized_top1_in_dense_top5",
            "top10_overlap",
            "top10_exact_agreement",
            "dense_top1_in_quantized_top10",
            "quantized_top1_in_dense_top10",
        ):
            logits_metrics.pop(key, None)

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
        "logits": logits_metrics,
        "divergence_300": divergence,
        "layers": {name: accumulator.result() for name, accumulator in layer_accumulators.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
