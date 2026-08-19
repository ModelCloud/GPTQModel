#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark YAQA Sketch-B factor-pass planning on a real Llama model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b

ROLE_GROUPS = {
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all-linear": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ModelCloud/Llama3.2-1B-Instruct")
    parser.add_argument("--dataset", default="neuralmagic/calibration")
    parser.add_argument("--dataset-config", default="LLM")
    parser.add_argument("--row-start", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument(
        "--module-scope", choices=tuple(ROLE_GROUPS), default="all-linear"
    )
    parser.add_argument("--factor-budget-gib", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--results", type=Path, required=True)
    return parser


def _sync() -> None:
    torch.mps.synchronize()


def _factor_digest(factor: torch.Tensor) -> str:
    if factor.device.type != "cpu":
        factor = factor.cpu()
    return hashlib.sha256(factor.contiguous().numpy().tobytes()).hexdigest()


def _target_modules(model, layers: tuple[torch.nn.Module, ...], roles: tuple[str, ...]):
    targets = {}
    for layer_index, layer in enumerate(layers):
        for role in roles:
            owner = layer.self_attn if hasattr(layer.self_attn, role) else layer.mlp
            module = getattr(owner, role)
            targets[f"model.layers.{layer_index}.{owner._get_name()}.{role}"] = module
    return targets


def main() -> None:
    args = _parser().parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Sketch-B pass-planning benchmark requires MPS")
    if args.rows < 1 or args.batch_size < 1 or args.layers < 1:
        raise ValueError("rows, batch-size, and layers must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset(args.dataset, name=args.dataset_config, split="train")
    texts = [
        dataset[index]["text"]
        for index in range(args.row_start, args.row_start + args.rows)
    ]
    texts.sort(
        key=lambda text: len(tokenizer(text, truncation=False)["input_ids"]),
        reverse=True,
    )
    batches = []
    for start in range(0, len(texts), args.batch_size):
        encoded = tokenizer(
            texts[start : start + args.batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batches.append(
            {name: value for name, value in encoded.items() if torch.is_tensor(value)}
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .to("mps")
        .eval()
    )
    layers = tuple(model.model.layers[: args.layers])
    if len(layers) != args.layers:
        raise ValueError(f"requested {args.layers} layers but model has {len(layers)}")
    targets = _target_modules(model, layers, ROLE_GROUPS[args.module_scope])
    budget_bytes = int(args.factor_budget_gib * 1024**3)
    chunks = QVQProcessor._yaqa_target_chunks(
        targets,
        list(layers),
        budget_bytes,
        packed_symmetric=True,
    )

    pass_records = []
    factor_digests = {}
    capture_started = time.perf_counter()
    for pass_index, chunk in enumerate(chunks, start=1):
        print(
            f"Sketch-B pass {pass_index}/{len(chunks)}: targets={len(chunk)} budget_gib={args.factor_budget_gib:g}",
            flush=True,
        )

        def progress(stats: dict, current_pass: int = pass_index) -> None:
            completed = stats["completed_batches"]
            if (
                completed == 1
                or completed == stats["total_batches"]
                or completed % 4 == 0
            ):
                print(
                    f"pass={current_pass}/{len(chunks)} batches={completed}/{stats['total_batches']} "
                    f"rows={stats['completed_sequences']}/{args.rows} valid_tokens={stats['valid_tokens']}",
                    flush=True,
                )

        _sync()
        started = time.perf_counter()
        input_factors, output_factors, stats = capture_yaqa_sketch_b(
            model,
            batches,
            chunk,
            device=torch.device("mps"),
            seed=args.seed,
            minimum_sequences=args.rows,
            checkpoint_modules=layers,
            progress_callback=progress,
        )
        _sync()
        seconds = time.perf_counter() - started
        for name, factor in input_factors.items():
            factor_digests[f"input:{name}"] = _factor_digest(factor)
        for name, factor in output_factors.items():
            factor_digests[f"output:{name}"] = _factor_digest(factor)
        pass_records.append(
            {
                "pass": pass_index,
                "targets": len(chunk),
                "seconds": seconds,
                "factor_storage_bytes": stats["factor_storage_bytes"],
                "valid_tokens": stats["valid_output_samples"],
                "capture_stats": stats,
            }
        )
        print(json.dumps(pass_records[-1], sort_keys=True), flush=True)
        del input_factors, output_factors
        gc.collect()
        torch.mps.empty_cache()

    capture_seconds = time.perf_counter() - capture_started
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "row_start": args.row_start,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "layers": args.layers,
        "module_scope": args.module_scope,
        "target_count": len(targets),
        "factor_budget_gib": args.factor_budget_gib,
        "factor_passes": len(chunks),
        "capture_seconds": capture_seconds,
        "pass_seconds": sum(record["seconds"] for record in pass_records),
        "valid_tokens": sum(
            int(batch["attention_mask"].sum().item()) for batch in batches
        ),
        "factor_digests": factor_digests,
        "passes": pass_records,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "factor_digests"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
