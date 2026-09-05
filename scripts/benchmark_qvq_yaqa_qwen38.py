#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark YAQA Fisher/Sketch-B collection on real Qwen3.8-27B geometry."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_idle_preflight import (
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/monster/data/model/Qwen3.5-27B")
    parser.add_argument(
        "--model-label",
        default="Qwen/Qwen3.8-27B@1d4bf0f2 (matched local Qwen3.5-27B geometry proxy)",
    )
    parser.add_argument("--dataset", default="/monster/data/model/dataset/nm-calibration/llm.parquet")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--layer-count", type=int, default=0, help="Limit --all-targets to this many layers")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=("gate_proj",),
        choices=("gate_proj", "up_proj", "down_proj"),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=("exact", "projected_64", "projected_128"),
        help="exact, projected_<rank>, or streaming_<rank>",
    )
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--accumulator-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--quant-block-size", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_idle_preflight_args(parser)
    return parser


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strategy(arm: str) -> tuple[str, int | None]:
    if arm == "exact":
        return "batched", None
    if arm.startswith("projected_"):
        try:
            rank = int(arm.removeprefix("projected_"))
        except ValueError as error:
            raise ValueError(f"invalid projected arm: {arm!r}") from error
        if rank < 1:
            raise ValueError("projection ranks must be positive")
        return "projected", rank
    if arm.startswith("streaming_"):
        try:
            rank = int(arm.removeprefix("streaming_"))
        except ValueError as error:
            raise ValueError(f"invalid streaming arm: {arm!r}") from error
        if rank < 1:
            raise ValueError("projection ranks must be positive")
        return "streaming_projected", rank
    raise ValueError(f"unknown arm: {arm!r}")


def _relative_l2(actual, expected) -> float:
    import torch

    numerator = torch.linalg.vector_norm(actual.double() - expected.double())
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def _load_batches(args, tokenizer):
    import pyarrow.parquet as pq

    table = pq.read_table(args.dataset, columns=["text"]).slice(args.row_start, args.rows)
    texts = [row["text"] for row in table.to_pylist()]
    if len(texts) != args.rows:
        raise ValueError(f"requested {args.rows} rows but read {len(texts)}")
    texts.sort(key=lambda text: len(tokenizer(text, truncation=False)["input_ids"]), reverse=True)
    batches = []
    for start in range(0, len(texts), args.batch_size):
        encoded = tokenizer(
            texts[start : start + args.batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.sequence_length,
        )
        batches.append({name: value for name, value in encoded.items() if name in {"input_ids", "attention_mask"}})
    return batches


def _hardware(torch, preflight) -> dict:
    properties = torch.cuda.get_device_properties(0)
    return {
        "physical_id": preflight.physical_id,
        "pci_bus_id": preflight.pci_bus_id,
        "uuid": preflight.uuid,
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def main() -> None:
    args = _parser().parse_args()
    if min(args.rows, args.batch_size, args.sequence_length) < 1:
        raise ValueError("rows, batch-size, and sequence-length must be positive")
    if args.layer_count < 0:
        raise ValueError("layer-count must be nonnegative")
    if args.quant_block_size < 0 or args.quant_block_size % 16:
        raise ValueError("quant-block-size must be zero or a positive multiple of 16")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("benchmark arms must be unique")
    for arm in args.arms:
        _strategy(arm)
    if any(_strategy(arm)[0] == "batched" for arm in args.arms) and _strategy(args.arms[0])[0] != "batched":
        raise ValueError("the exact benchmark arm must be first when present")

    preflight = bootstrap_gpu_idle_preflight()
    if preflight is None:
        raise RuntimeError("formal Qwen YAQA timing requires the GPU idle preflight")

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.quantization.qvq_yaqa import YaqaGramSketch, capture_yaqa_sketch_b

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("this formal benchmark requires exactly one visible NVIDIA H200")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    batches = _load_batches(args, tokenizer)
    valid_tokens = sum(int(batch["attention_mask"].sum().item()) for batch in batches)

    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    layers = tuple(model.model.language_model.layers)
    if not 0 <= args.layer < len(layers):
        raise ValueError(f"layer {args.layer} is outside the {len(layers)}-layer model")
    layer = layers[args.layer]
    if args.all_targets:
        target_roles = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.startswith("model.language_model.layers.")
            and (
                args.layer_count == 0
                or args.layer <= int(name.split(".")[3]) < args.layer + args.layer_count
            )
            and name.rsplit(".", 1)[-1] in target_roles
        }
    else:
        modules = {
            f"model.language_model.layers.{args.layer}.mlp.{role}": getattr(layer.mlp, role)
            for role in args.roles
        }
    checkpoint_modules = () if args.no_activation_checkpointing else layers
    recheck_gpu_exclusivity(preflight)

    exact_factors = None
    exact_quantized = None
    exact_quant_objective = None
    records = []
    accumulator_device = None if args.accumulator_device == "auto" else torch.device(args.accumulator_device)
    for arm in args.arms:
        strategy, projection_rank = _strategy(arm)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        input_factors, output_factors, stats = capture_yaqa_sketch_b(
            model,
            batches,
            modules,
            device=device,
            seed=args.seed,
            minimum_sequences=args.rows,
            first_decoder_layer=layers[0],
            checkpoint_modules=checkpoint_modules,
            accumulator_device=accumulator_device,
            gram_strategy=strategy,
            gram_projection_rank=projection_rank,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        factors = {}
        if strategy == "batched" or exact_factors is not None:
            factors = {
                f"input:{name}": factor.materialize(device=device).cpu()
                if isinstance(factor, YaqaGramSketch)
                else factor
                for name, factor in input_factors.items()
            }
            factors.update(
                {
                    f"output:{name}": factor.materialize(device=device).cpu()
                    if isinstance(factor, YaqaGramSketch)
                    else factor
                    for name, factor in output_factors.items()
                }
            )
        comparison_requested = len(args.arms) > 1
        if strategy == "batched" and comparison_requested:
            exact_factors = {name: factor.clone() for name, factor in factors.items()}
            drifts = {name: 0.0 for name in factors}
        elif strategy == "batched":
            drifts = {name: 0.0 for name in factors}
        elif exact_factors is not None:
            drifts = {name: _relative_l2(factor, exact_factors[name]) for name, factor in factors.items()}
        else:
            drifts = None
        quant_metrics = None
        if args.quant_block_size:
            if not factors:
                raise ValueError("quant-block-size requires an exact arm before projected arms")
            module_name, target_module = next(iter(modules.items()))
            block_size = args.quant_block_size
            if min(target_module.in_features, target_module.out_features) < block_size:
                raise ValueError("quant-block-size exceeds the first target module geometry")
            block_weight = target_module.weight[:block_size, :block_size].detach().to(torch.float32)
            block_input = factors[f"input:{module_name}"][:block_size, :block_size].to(device)
            block_output = factors[f"output:{module_name}"][:block_size, :block_size].to(device)
            quantized = quantize_qvq_linear(
                block_weight,
                block_input,
                bits=3.5,
                output_hessian=block_output,
                seed=args.seed,
                input_hadamard=True,
                output_hadamard=False,
                damp_percent=0.1,
                rounding="yaqa",
                yaqa_v2b2_family_mode="fixed_block_ldlq",
                vector_size=2,
                v2b2_p32=True,
                bank_count=2,
            )
            reconstructed = quantized.weight.to(torch.float32)
            if strategy == "batched":
                exact_quantized = reconstructed.clone()
                exact_quant_objective = (block_weight, block_input, block_output)
            assert exact_quant_objective is not None
            objective_weight, objective_input, objective_output = exact_quant_objective
            error = reconstructed - objective_weight
            exact_proxy = torch.einsum(
                "oi,oj,jk,ki->",
                error,
                objective_output,
                error,
                objective_input,
            )
            quant_metrics = {
                "block_size": block_size,
                "exact_objective_proxy": float(exact_proxy.item()),
                "proxy_ratio_vs_exact_factors": (
                    1.0
                    if strategy == "batched"
                    else float(exact_proxy.item()) / records[0]["quant_metrics"]["exact_objective_proxy"]
                ),
                "weight_relative_l2_vs_exact_factors": (
                    0.0 if exact_quantized is reconstructed else _relative_l2(reconstructed.cpu(), exact_quantized.cpu())
                ),
            }
        record = {
            "arm": arm,
            "strategy": strategy,
            "projection_rank": projection_rank,
            "seconds": seconds,
            "tokens_per_second": valid_tokens / seconds,
            "speedup_vs_exact": (
                next((record["seconds"] for record in records if record["strategy"] == "batched"), None)
                / seconds
                if any(record["strategy"] == "batched" for record in records)
                else None
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "maximum_factor_relative_l2": None if drifts is None else max(drifts.values()),
            "mean_factor_relative_l2": None if drifts is None else sum(drifts.values()) / len(drifts),
            "factor_relative_l2": drifts,
            "quant_metrics": quant_metrics,
            "capture": stats,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del input_factors, output_factors, factors

    payload = {
        "schema": "qvq.yaqa.qwen38.benchmark.v1",
        "revision": _git_revision(),
        "model": args.model_label,
        "model_path": args.model,
        "dataset": args.dataset,
        "row_start": args.row_start,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "valid_tokens": valid_tokens,
        "layer": args.layer,
        "roles": args.roles,
        "all_targets": args.all_targets,
        "layer_count": args.layer_count,
        "target_count": len(modules),
        "activation_checkpointing": not args.no_activation_checkpointing,
        "accumulator_device": args.accumulator_device,
        "dtype": "bfloat16",
        "load_seconds": load_seconds,
        "hardware": _hardware(torch, preflight),
        "idle_preflight": preflight.as_dict(),
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": records}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
