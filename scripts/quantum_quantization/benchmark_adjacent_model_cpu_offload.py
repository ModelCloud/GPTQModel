#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare whole-module Adjacent candidate generation on CUDA and parallel CPU offload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent_model import AdjacentModelConfig, apply_adjacent_model_hybrid  # noqa: E402
from gptqmodel.utils.adjacent_exact import prewarm_adjacent_exact_cuda  # noqa: E402
from scripts.quantum_quantization.benchmark_adjacent_model_cpu_gpu import (  # noqa: E402
    MODULE_TYPES,
    ModuleType,
    hardware_metadata,
    make_hessian,
    median_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/monster/data/model/Qwen3-8B"))
    parser.add_argument("--module", choices=tuple(module.name for module in MODULE_TYPES), default="q_proj")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=898)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--cpu-row-chunk-size", type=int, default=512)
    parser.add_argument("--gpu-row-chunk-size", type=int, default=2048)
    parser.add_argument("--cpu-workers", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    return parser.parse_args()


def load_full_weight(model: Path, module: ModuleType) -> Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][module.tensor_name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(module.tensor_name)
    if list(weight.shape) != [module.rows, module.columns]:
        raise ValueError(
            f"{module.tensor_name} has shape {list(weight.shape)}, expected {[module.rows, module.columns]}."
        )
    return weight.contiguous()


def make_problem(
    *,
    model: Path,
    module: ModuleType,
    bits: int,
    group_size: int,
    seed: int,
) -> dict[str, Any]:
    if module.columns % group_size:
        raise ValueError("This production-shape benchmark requires columns divisible by group size.")
    device = torch.device("cuda")
    weight = load_full_weight(model, module).to(device)
    group_count = module.columns // group_size
    grouped_weight = weight.to(torch.float32).view(module.rows, group_count, group_size)
    maxq = (1 << bits) - 1
    scales = 2.0 * grouped_weight.abs().amax(dim=2).clamp_min_(1e-8) / maxq
    zeros = torch.full_like(scales, (maxq + 1) / 2)
    per_column_scales = scales.repeat_interleave(group_size, dim=1)
    per_column_zeros = zeros.repeat_interleave(group_size, dim=1)
    classic = (
        torch.round(weight.to(torch.float32) / per_column_scales + per_column_zeros)
        .clamp_(0, maxq)
        .sub_(per_column_zeros)
        .mul_(per_column_scales)
        .to(weight.dtype)
    )

    group_hessian = make_hessian(group_size, seed).to(device=device, dtype=torch.float32)
    hessian = torch.zeros((module.columns, module.columns), device=device, dtype=torch.float32)
    for group_start in range(0, module.columns, group_size):
        group_stop = group_start + group_size
        hessian[group_start:group_stop, group_start:group_stop] = group_hessian
    return {
        "module_name": module.name,
        "weight": weight,
        "hessian": hessian,
        "classic_quantized": classic,
        "scale_parts": list(scales.split(1, dim=1)),
        "zero_parts": list(zeros.split(1, dim=1)),
        "bits": bits,
        "group_size": group_size,
    }


def run_once(
    problem: dict[str, Any],
    config: AdjacentModelConfig,
) -> tuple[Tensor, dict[str, Any], float, int]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    started = time.perf_counter()
    hybrid, stats = apply_adjacent_model_hybrid(**problem, config=config)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    peak_allocated_delta = torch.cuda.max_memory_allocated() - start_allocated
    hybrid_cpu = hybrid.cpu()
    del hybrid
    return hybrid_cpu, stats, wall_seconds, peak_allocated_delta


def configs(args: argparse.Namespace) -> tuple[AdjacentModelConfig, AdjacentModelConfig]:
    common = {
        "coordinate_starts": ("nearest", "zero", "one", "linear"),
        "max_coordinate_flips": 32,
        "coordinate_rebase_interval": 8,
        "objective_row_chunk_size": 256,
        "native_refinements_per_module": 0,
    }
    return (
        AdjacentModelConfig(
            **common,
            executor="cuda",
            row_chunk_size=args.gpu_row_chunk_size,
        ),
        AdjacentModelConfig(
            **common,
            executor="cpu",
            cpu_row_chunk_size=args.cpu_row_chunk_size,
            cpu_workers=args.cpu_workers,
        ),
    )


def main() -> None:
    args = parse_args()
    if min(
        args.cpu_row_chunk_size,
        args.gpu_row_chunk_size,
        args.cpu_workers,
        args.timing_repeats,
    ) < 1:
        raise ValueError("Chunk sizes, worker count, and timing repeats must be positive.")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative.")
    if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled():
        raise RuntimeError("Run with PYTHON_GIL=0 so CPU candidate workers can execute concurrently.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    module = next(module for module in MODULE_TYPES if module.name == args.module)
    problem = make_problem(
        model=args.model,
        module=module,
        bits=args.bits,
        group_size=args.group_size,
        seed=args.seed,
    )
    cuda_config, cpu_config = configs(args)
    prewarm_adjacent_exact_cuda()

    for _ in range(args.warmups):
        run_once(problem, cuda_config)
        run_once(problem, cpu_config)

    records: dict[str, list[Any]] = {
        "cuda_wall_seconds": [],
        "cuda_candidate_seconds": [],
        "cuda_peak_allocated_delta_bytes": [],
        "cpu_wall_seconds": [],
        "cpu_candidate_seconds": [],
        "cpu_peak_allocated_delta_bytes": [],
    }
    reference_outputs: dict[str, Tensor] = {}
    reference_stats: dict[str, dict[str, Any]] = {}
    for repeat in range(args.timing_repeats):
        order = (("cuda", cuda_config), ("cpu", cpu_config))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, config in order:
            output, stats, wall, peak = run_once(problem, config)
            records[f"{name}_wall_seconds"].append(wall)
            records[f"{name}_candidate_seconds"].append(stats["candidate_phase_wall_seconds"])
            records[f"{name}_peak_allocated_delta_bytes"].append(peak)
            reference_outputs.setdefault(name, output)
            reference_stats.setdefault(name, stats)
            torch.testing.assert_close(output, reference_outputs[name], rtol=0.0, atol=0.0)

    cuda_stats = reference_stats["cuda"]
    cpu_stats = reference_stats["cpu"]
    torch.testing.assert_close(reference_outputs["cpu"], reference_outputs["cuda"], rtol=0.0, atol=0.0)
    exact_stat_keys = (
        "coordinate_converged_row_groups",
        "coordinate_capped_row_groups",
        "coordinate_total_flips",
        "coordinate_max_flips",
        "hybrid_selected_rows",
    )
    if any(cpu_stats[key] != cuda_stats[key] for key in exact_stat_keys):
        raise AssertionError("CPU and CUDA whole-module candidate statistics differ.")

    cuda_candidate = statistics.median(records["cuda_candidate_seconds"])
    cpu_candidate = statistics.median(records["cpu_candidate_seconds"])
    cuda_wall = statistics.median(records["cuda_wall_seconds"])
    cpu_wall = statistics.median(records["cpu_wall_seconds"])
    payload = {
        "schema": "gptqmodel-adjacent-model-cpu-offload-v1",
        "command": sys.argv,
        "model": str(args.model),
        "module": module.name,
        "shape": [module.rows, module.columns],
        "seed": args.seed,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "hessian": "deterministic synthetic block-diagonal SPD; full matrix used by objective replay",
        "native_refinements_per_module": 0,
        "cpu_workers": args.cpu_workers,
        "cpu_row_chunk_size": args.cpu_row_chunk_size,
        "gpu_row_chunk_size": args.gpu_row_chunk_size,
        "warmups": args.warmups,
        "timing_repeats": args.timing_repeats,
        "records": records,
        "summaries": {
            "cuda_wall_seconds": median_summary(records["cuda_wall_seconds"]),
            "cuda_candidate_seconds": median_summary(records["cuda_candidate_seconds"]),
            "cpu_wall_seconds": median_summary(records["cpu_wall_seconds"]),
            "cpu_candidate_seconds": median_summary(records["cpu_candidate_seconds"]),
            "cuda_peak_allocated_delta_bytes": max(records["cuda_peak_allocated_delta_bytes"]),
            "cpu_peak_allocated_delta_bytes": max(records["cpu_peak_allocated_delta_bytes"]),
            "candidate_speedup_cpu_over_cuda": cuda_candidate / cpu_candidate,
            "whole_module_speedup_cpu_over_cuda": cuda_wall / cpu_wall,
        },
        "correctness": {
            "hybrid_exact_match": True,
            "exact_stat_keys": list(exact_stat_keys),
            "local_group_hessian_error_abs_diff": abs(
                cpu_stats["local_group_hessian_error"] - cuda_stats["local_group_hessian_error"]
            ),
            "classic_full_hessian_error_abs_diff": abs(
                cpu_stats["classic_full_hessian_error"] - cuda_stats["classic_full_hessian_error"]
            ),
            "adjacent_full_hessian_error_abs_diff": abs(
                cpu_stats["adjacent_full_hessian_error"] - cuda_stats["adjacent_full_hessian_error"]
            ),
        },
        "hardware": hardware_metadata(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"{module.name}: candidate CUDA={cuda_candidate:.3f}s CPU={cpu_candidate:.3f}s "
        f"speedup={cuda_candidate / cpu_candidate:.3f}x"
    )
    print(
        f"{module.name}: whole CUDA={cuda_wall:.3f}s CPU={cpu_wall:.3f}s "
        f"speedup={cuda_wall / cpu_wall:.3f}x"
    )
    print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
