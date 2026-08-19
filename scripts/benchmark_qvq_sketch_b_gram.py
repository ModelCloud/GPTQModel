#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact YAQA Sketch-B Gram strategies on real Llama activations."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ModelCloud/Llama3.2-1B-Instruct")
    parser.add_argument("--dataset", default="neuralmagic/calibration")
    parser.add_argument("--dataset-config", default="LLM")
    parser.add_argument("--row-start", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--strategies", nargs="+", default=("batched", "flattened", "token_space"))
    parser.add_argument("--roles", nargs="+", default=("q_proj", "k_proj", "v_proj", "o_proj"))
    parser.add_argument("--results", type=Path, required=True)
    return parser


def _sync() -> None:
    torch.mps.synchronize()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.double() - expected.double())
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def main() -> None:
    args = _parser().parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Sketch-B Apple benchmark requires MPS")
    if args.rows < 1:
        raise ValueError("rows must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset(args.dataset, name=args.dataset_config, split="train")
    texts = [dataset[index]["text"] for index in range(args.row_start, args.row_start + args.rows)]
    texts.sort(key=lambda text: len(tokenizer(text, truncation=False)["input_ids"]), reverse=True)
    batches = []
    for start in range(0, len(texts), args.batch_size):
        encoded = tokenizer(
            texts[start : start + args.batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batches.append({name: value for name, value in encoded.items() if torch.is_tensor(value)})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("mps").eval()
    layers = tuple(model.model.layers)
    first_layer = layers[0]
    modules = {
        role: getattr(first_layer.self_attn if hasattr(first_layer.self_attn, role) else first_layer.mlp, role)
        for role in args.roles
    }
    results = []
    baseline = None
    for arm in args.strategies:
        projection_rank = int(arm.removeprefix("projected_")) if arm.startswith("projected_") else None
        gram_strategy = "projected" if projection_rank is not None else ("batched" if arm == "batched_mps" else arm)
        accumulator_device = torch.device("cpu") if arm == "batched" else torch.device("mps")
        _sync()
        started = time.perf_counter()
        input_factors, output_factors, stats = capture_yaqa_sketch_b(
            model,
            batches,
            modules,
            device=torch.device("mps"),
            seed=20260819,
            minimum_sequences=args.rows,
            checkpoint_modules=layers,
            accumulator_device=accumulator_device,
            gram_strategy=gram_strategy,
            gram_projection_rank=projection_rank,
        )
        _sync()
        seconds = time.perf_counter() - started
        factors = {f"input:{name}": factor for name, factor in input_factors.items()}
        factors.update({f"output:{name}": factor for name, factor in output_factors.items()})
        if baseline is None:
            baseline = {name: factor.clone() for name, factor in factors.items()}
        drifts = {name: _relative_l2(factor, baseline[name]) for name, factor in factors.items()}
        record = {
            "strategy": gram_strategy,
            "arm": arm,
            "accumulator_device": accumulator_device.type,
            "seconds": seconds,
            "speedup_vs_batched": results[0]["seconds"] / seconds if results else 1.0,
            "maximum_factor_relative_l2": max(drifts.values()),
            "mean_factor_relative_l2": sum(drifts.values()) / len(drifts),
            "capture_stats": stats,
            "within_critical_math_tolerance": max(drifts.values()) <= 1e-6,
        }
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del input_factors, output_factors, factors
        gc.collect()
        torch.mps.empty_cache()

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "rows": args.rows,
        "row_start": args.row_start,
        "valid_tokens": sum(int(batch["attention_mask"].sum().item()) for batch in batches),
        "batch_size": args.batch_size,
        "roles": args.roles,
        "dtype": "float16",
        "device": "mps",
        "critical_math_tolerance": 1e-6,
        "results": results,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
