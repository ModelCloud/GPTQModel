#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Time the Adjacent model hot path on CPU and GPU for real Qwen projection shapes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import (  # noqa: E402
    adjacent_branch_bound_cuda,
    build_adjacent_rounding_qubo,
)
from gptqmodel.quantization.adjacent_model import (  # noqa: E402
    AdjacentModelConfig,
    _adjacent_group_candidate,
)
from gptqmodel.utils.adjacent_exact import prewarm_adjacent_exact_cuda  # noqa: E402
from gptqmodel.utils.cpp import resolved_cuda_arch_flags  # noqa: E402


@dataclass(frozen=True)
class ModuleType:
    name: str
    tensor_name: str
    rows: int
    columns: int


MODULE_TYPES = (
    ModuleType("q_proj", "model.layers.0.self_attn.q_proj.weight", 4096, 4096),
    ModuleType("k_proj", "model.layers.0.self_attn.k_proj.weight", 1024, 4096),
    ModuleType("v_proj", "model.layers.0.self_attn.v_proj.weight", 1024, 4096),
    ModuleType("o_proj", "model.layers.0.self_attn.o_proj.weight", 4096, 4096),
    ModuleType("gate_proj", "model.layers.0.mlp.gate_proj.weight", 12288, 4096),
    ModuleType("up_proj", "model.layers.0.mlp.up_proj.weight", 12288, 4096),
    ModuleType("down_proj", "model.layers.0.mlp.down_proj.weight", 4096, 12288),
)


@dataclass(frozen=True)
class CandidateInputs:
    weight: Tensor
    hessian: Tensor
    scale: Tensor
    zero: Tensor

    def to(self, device: torch.device) -> CandidateInputs:
        return CandidateInputs(
            weight=self.weight.to(device),
            hessian=self.hessian.to(device),
            scale=self.scale.to(device),
            zero=self.zero.to(device),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/monster/data/model/Qwen3-8B"))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=898)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--row-chunk-size", type=int, help="Compatibility override that sets both CPU and GPU chunks.")
    parser.add_argument("--cpu-row-chunk-size", type=int, default=512)
    parser.add_argument("--gpu-row-chunk-size", type=int, default=2048)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--native-timing-repeats", type=int, default=3)
    parser.add_argument("--skip-cpu-one", action="store_true")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=tuple(module.name for module in MODULE_TYPES),
        default=[module.name for module in MODULE_TYPES],
    )
    return parser.parse_args()


def median_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def load_weight_slice(model: Path, module: ModuleType, row_chunk_size: int, group_size: int) -> Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][module.tensor_name]
    rows = min(module.rows, row_chunk_size)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(module.tensor_name)
        shape = tensor_slice.get_shape()
        if shape != [module.rows, module.columns]:
            raise ValueError(f"{module.tensor_name} has shape {shape}, expected {[module.rows, module.columns]}.")
        return tensor_slice[:rows, :group_size].contiguous()


def make_hessian(group_size: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(512, max(16, group_size // 4), generator=generator, dtype=torch.float64)
    mixing = torch.randn(latent.shape[1], group_size, generator=generator, dtype=torch.float64)
    activations = latent @ mixing + 0.25 * torch.randn(
        512,
        group_size,
        generator=generator,
        dtype=torch.float64,
    )
    activations /= activations.square().mean(dim=0).sqrt().clamp_min_(1e-8)
    hessian = activations.mT @ activations / activations.shape[0]
    damping = 0.05 * hessian.diagonal().mean()
    hessian.diagonal().add_(damping)
    return hessian


def make_inputs(
    model: Path,
    module: ModuleType,
    *,
    bits: int,
    group_size: int,
    row_chunk_size: int,
    seed: int,
) -> CandidateInputs:
    weight = load_weight_slice(model, module, row_chunk_size, group_size)
    work_weight = weight.to(torch.float32)
    absolute_max = work_weight.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8)
    maxq = (1 << bits) - 1
    scale = (2.0 * absolute_max / maxq).to(torch.float32)
    zero = torch.full_like(scale, (maxq + 1) / 2)
    return CandidateInputs(
        weight=weight,
        hessian=make_hessian(group_size, seed),
        scale=scale,
        zero=zero,
    )


def candidate_call(inputs: CandidateInputs, config: AdjacentModelConfig, bits: int) -> tuple[Tensor, ...]:
    return _adjacent_group_candidate(
        weight=inputs.weight,
        hessian=inputs.hessian,
        scale=inputs.scale,
        zero=inputs.zero,
        bits=bits,
        config=config,
    )


def consume_candidate(result: tuple[Tensor, ...]) -> tuple[float, int, int]:
    candidate, costs, converged, flips = result
    checksum = float(costs.sum().item()) + float(candidate[0, 0].item())
    return checksum, int(converged.sum().item()), int(flips.sum().item())


def run_sequential_tasks(
    inputs: CandidateInputs,
    config: AdjacentModelConfig,
    bits: int,
    task_count: int,
) -> tuple[float, int, int]:
    checksum = 0.0
    converged = 0
    flips = 0
    for _ in range(task_count):
        task_checksum, task_converged, task_flips = consume_candidate(candidate_call(inputs, config, bits))
        checksum += task_checksum
        converged += task_converged
        flips += task_flips
    return checksum, converged, flips


def run_parallel_tasks(
    executor: ThreadPoolExecutor,
    inputs: CandidateInputs,
    config: AdjacentModelConfig,
    bits: int,
    task_count: int,
) -> tuple[float, int, int]:
    def one_task(_index: int) -> tuple[float, int, int]:
        return consume_candidate(candidate_call(inputs, config, bits))

    results = executor.map(one_task, range(task_count))
    checksum = 0.0
    converged = 0
    flips = 0
    for task_checksum, task_converged, task_flips in results:
        checksum += task_checksum
        converged += task_converged
        flips += task_flips
    return checksum, converged, flips


def time_cpu(function) -> tuple[tuple[float, int, int], float]:
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def time_gpu(function) -> tuple[tuple[float, int, int], float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    result = function()
    end_event.record()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started, start_event.elapsed_time(end_event) / 1e3


def check_cpu_gpu(
    cpu_inputs: CandidateInputs,
    gpu_inputs: CandidateInputs,
    config: AdjacentModelConfig,
    bits: int,
) -> dict[str, Any]:
    cpu_candidate, cpu_costs, cpu_converged, cpu_flips = candidate_call(cpu_inputs, config, bits)
    gpu_candidate, gpu_costs, gpu_converged, gpu_flips = candidate_call(gpu_inputs, config, bits)
    gpu_candidate = gpu_candidate.cpu()
    gpu_costs = gpu_costs.cpu()
    gpu_converged = gpu_converged.cpu()
    gpu_flips = gpu_flips.cpu()
    return {
        "candidate_max_abs_diff": float((cpu_candidate - gpu_candidate).abs().max().item()),
        "cost_max_abs_diff": float((cpu_costs - gpu_costs).abs().max().item()),
        "cost_max_rel_diff": float(
            ((cpu_costs - gpu_costs).abs() / cpu_costs.abs().clamp_min(1e-15)).max().item()
        ),
        "converged_exact_match": bool(torch.equal(cpu_converged, gpu_converged)),
        "flips_exact_match": bool(torch.equal(cpu_flips, gpu_flips)),
    }


def benchmark_native(
    gpu_inputs: CandidateInputs,
    *,
    bits: int,
    repeats: int,
) -> dict[str, Any]:
    problem = build_adjacent_rounding_qubo(
        gpu_inputs.weight[0],
        gpu_inputs.hessian,
        scale=gpu_inputs.scale[0, 0],
        zero=gpu_inputs.zero[0, 0],
        bits=bits,
    )

    def solve():
        return adjacent_branch_bound_cuda(
            problem,
            split_depth=min(6, problem.active_decisions),
            max_nodes_per_worker=500,
            certificate_tolerance=1e-12,
            require_optimal=False,
        )

    solve()
    wall_samples = []
    cuda_samples = []
    results = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        started = time.perf_counter()
        result = solve()
        end_event.record()
        torch.cuda.synchronize()
        wall_samples.append(time.perf_counter() - started)
        cuda_samples.append(start_event.elapsed_time(end_event) / 1e3)
        results.append(result)
    return {
        "cpu": None,
        "cpu_unavailable_reason": (
            "The classical CPU reference enumerates at most 20 decisions; this real group has "
            f"{problem.active_decisions} active decisions."
        ),
        "active_decisions": problem.active_decisions,
        "gpu_wall_seconds": median_summary(wall_samples),
        "gpu_cuda_seconds": median_summary(cuda_samples),
        "optimal_samples": [result.optimal for result in results],
        "nodes_visited_samples": [result.nodes_visited for result in results],
        "cost_samples": [result.cost for result in results],
    }


def hardware_metadata() -> dict[str, Any]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    visible_uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    physical = next((line for line in smi if visible_uuid in line), None)
    return {
        "physical_gpu": physical,
        "cuda_visible_devices": visible_uuid,
        "torch_device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "python_gil_enabled": bool(sys._is_gil_enabled()) if hasattr(sys, "_is_gil_enabled") else None,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_arch_flags": resolved_cuda_arch_flags(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }


def benchmark_module(
    module: ModuleType,
    cpu_inputs: CandidateInputs,
    gpu_inputs: CandidateInputs,
    correctness_cpu_inputs: CandidateInputs,
    *,
    config: AdjacentModelConfig,
    bits: int,
    group_size: int,
    cpu_row_chunk_size: int,
    gpu_row_chunk_size: int,
    timing_repeats: int,
    native_timing_repeats: int,
    executor: ThreadPoolExecutor,
    cpu_workers: int,
    time_cpu_one: bool,
) -> dict[str, Any]:
    cpu_row_chunks = math.ceil(module.rows / cpu_row_chunk_size)
    gpu_row_chunks = math.ceil(module.rows / gpu_row_chunk_size)
    groups = math.ceil(module.columns / group_size)
    cpu_task_count = cpu_row_chunks * groups
    gpu_task_count = gpu_row_chunks * groups

    consume_candidate(candidate_call(cpu_inputs, config, bits))
    consume_candidate(candidate_call(gpu_inputs, config, bits))
    torch.cuda.synchronize()

    cpu_one_samples = []
    cpu_parallel_samples = []
    gpu_wall_samples = []
    gpu_cuda_samples = []
    checksums: dict[str, list[tuple[float, int, int]]] = {"cpu_1": [], "cpu_parallel": [], "gpu": []}
    peak_gpu_allocated = 0
    for _ in range(timing_repeats):
        if time_cpu_one:
            result, elapsed = time_cpu(
                lambda: run_sequential_tasks(cpu_inputs, config, bits, cpu_task_count)
            )
            checksums["cpu_1"].append(result)
            cpu_one_samples.append(elapsed)

        result, elapsed = time_cpu(
            lambda: run_parallel_tasks(executor, cpu_inputs, config, bits, cpu_task_count)
        )
        checksums["cpu_parallel"].append(result)
        cpu_parallel_samples.append(elapsed)

        torch.cuda.reset_peak_memory_stats()
        result, wall, cuda = time_gpu(
            lambda: run_sequential_tasks(gpu_inputs, config, bits, gpu_task_count)
        )
        checksums["gpu"].append(result)
        gpu_wall_samples.append(wall)
        gpu_cuda_samples.append(cuda)
        peak_gpu_allocated = max(peak_gpu_allocated, torch.cuda.max_memory_allocated())

    cpu_one = statistics.median(cpu_one_samples) if cpu_one_samples else None
    cpu_parallel = statistics.median(cpu_parallel_samples)
    gpu_wall = statistics.median(gpu_wall_samples)
    return {
        "module_type": module.name,
        "tensor_name": module.tensor_name,
        "shape": [module.rows, module.columns],
        "cpu_prototype_shape": list(cpu_inputs.weight.shape),
        "gpu_prototype_shape": list(gpu_inputs.weight.shape),
        "group_count": groups,
        "cpu_row_chunks": cpu_row_chunks,
        "gpu_row_chunks": gpu_row_chunks,
        "cpu_candidate_task_count": cpu_task_count,
        "gpu_candidate_task_count": gpu_task_count,
        "cpu_1_worker_wall_seconds": median_summary(cpu_one_samples) if cpu_one_samples else None,
        f"cpu_{cpu_workers}_worker_wall_seconds": median_summary(cpu_parallel_samples),
        "gpu_wall_seconds": median_summary(gpu_wall_samples),
        "gpu_cuda_seconds": median_summary(gpu_cuda_samples),
        "speedup_gpu_vs_cpu_1": cpu_one / gpu_wall if cpu_one is not None else None,
        f"speedup_gpu_vs_cpu_{cpu_workers}": cpu_parallel / gpu_wall,
        f"speedup_cpu_{cpu_workers}_vs_cpu_1": (
            cpu_one / cpu_parallel
            if cpu_one is not None
            else None
        ),
        "peak_gpu_allocated_bytes": peak_gpu_allocated,
        "checksums": checksums,
        "correctness": check_cpu_gpu(correctness_cpu_inputs, gpu_inputs, config, bits),
        "native_exact_tail": benchmark_native(
            gpu_inputs,
            bits=bits,
            repeats=native_timing_repeats,
        ),
    }


def print_table(results: list[dict[str, Any]], cpu_workers: int) -> None:
    print(
        f"module     shape          tasks C/G | CPU-1 s CPU-{cpu_workers} s GPU wall s GPU CUDA s | "
        f"GPU/CPU1 GPU/CPU{cpu_workers} CPU{cpu_workers}/CPU1"
    )
    print(
        "---------- -------------- --------- | ------- --------- ---------- ---------- | "
        "-------- -------- -----------"
    )
    for result in results:
        rows, columns = result["shape"]
        cpu_one_summary = result["cpu_1_worker_wall_seconds"]
        cpu_one = cpu_one_summary["median"] if cpu_one_summary is not None else None
        cpu_parallel = result[f"cpu_{cpu_workers}_worker_wall_seconds"]["median"]
        gpu_wall = result["gpu_wall_seconds"]["median"]
        gpu_cuda = result["gpu_cuda_seconds"]["median"]
        cpu_one_text = f"{cpu_one:>7.3f}" if cpu_one is not None else "      -"
        gpu_cpu_one = result["speedup_gpu_vs_cpu_1"]
        gpu_cpu_one_text = f"{gpu_cpu_one:>8.3f}" if gpu_cpu_one is not None else "       -"
        cpu_parallel_cpu_one = result[f"speedup_cpu_{cpu_workers}_vs_cpu_1"]
        cpu_parallel_cpu_one_text = (
            f"{cpu_parallel_cpu_one:>11.3f}"
            if cpu_parallel_cpu_one is not None
            else "          -"
        )
        print(
            f"{result['module_type']:<10} {rows:>5}x{columns:<8} "
            f"{result['cpu_candidate_task_count']:>4}/{result['gpu_candidate_task_count']:<4} | "
            f"{cpu_one_text} {cpu_parallel:>9.3f} {gpu_wall:>10.3f} {gpu_cuda:>10.3f} | "
            f"{gpu_cpu_one_text} "
            f"{result[f'speedup_gpu_vs_cpu_{cpu_workers}']:>8.3f} "
            f"{cpu_parallel_cpu_one_text}"
        )


def main() -> None:
    args = parse_args()
    cpu_row_chunk_size = args.row_chunk_size or args.cpu_row_chunk_size
    gpu_row_chunk_size = args.row_chunk_size or args.gpu_row_chunk_size
    if args.cpu_workers < 1:
        raise ValueError("--cpu-workers must be positive.")
    if cpu_row_chunk_size < 1 or gpu_row_chunk_size < 1:
        raise ValueError("CPU and GPU row chunk sizes must be positive.")
    if args.timing_repeats < 1 or args.native_timing_repeats < 1:
        raise ValueError("Timing repeat counts must be positive.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU.")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(f"This benchmark targets the requested sm_80 GPU, found {properties.major}.{properties.minor}.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    config = AdjacentModelConfig(
        coordinate_starts=("nearest", "zero", "one", "linear"),
        max_coordinate_flips=32,
        coordinate_rebase_interval=8,
        row_chunk_size=gpu_row_chunk_size,
        cpu_row_chunk_size=cpu_row_chunk_size,
        cpu_workers=args.cpu_workers,
        native_refinements_per_module=4,
        native_split_depth=6,
        native_max_nodes_per_worker=500,
    )
    metadata = hardware_metadata()
    prewarm_adjacent_exact_cuda()

    results = []
    with ThreadPoolExecutor(max_workers=args.cpu_workers, thread_name_prefix="adjacent-cpu") as executor:
        selected_modules = [module for module in MODULE_TYPES if module.name in args.modules]
        for index, module in enumerate(selected_modules):
            print(f"preparing {module.name} ({module.rows}x{module.columns})", flush=True)
            cpu_inputs = make_inputs(
                args.model,
                module,
                bits=args.bits,
                group_size=args.group_size,
                row_chunk_size=cpu_row_chunk_size,
                seed=args.seed + 1009 * index,
            )
            correctness_cpu_inputs = make_inputs(
                args.model,
                module,
                bits=args.bits,
                group_size=args.group_size,
                row_chunk_size=gpu_row_chunk_size,
                seed=args.seed + 1009 * index,
            )
            gpu_inputs = correctness_cpu_inputs.to(torch.device("cuda", 0))
            print(f"timing {module.name}", flush=True)
            results.append(
                benchmark_module(
                    module,
                    cpu_inputs,
                    gpu_inputs,
                    correctness_cpu_inputs,
                    config=config,
                    bits=args.bits,
                    group_size=args.group_size,
                    cpu_row_chunk_size=cpu_row_chunk_size,
                    gpu_row_chunk_size=gpu_row_chunk_size,
                    timing_repeats=args.timing_repeats,
                    native_timing_repeats=args.native_timing_repeats,
                    executor=executor,
                    cpu_workers=args.cpu_workers,
                    time_cpu_one=not args.skip_cpu_one,
                )
            )

    payload = {
        "schema": "gptqmodel-adjacent-model-cpu-gpu-v2",
        "model": str(args.model),
        "layer": 0,
        "seed": args.seed,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "algorithm": {
            "phase": "_adjacent_group_candidate plus production-style scalar statistics",
            "coordinate_starts": list(config.coordinate_starts),
            "max_coordinate_flips": config.max_coordinate_flips,
            "coordinate_rebase_interval": config.coordinate_rebase_interval,
            "cpu_row_chunk_size": cpu_row_chunk_size,
            "gpu_row_chunk_size": gpu_row_chunk_size,
            "cpu_outer_workers": args.cpu_workers,
            "cpu_intraop_threads": 1,
            "cpu_one_execution": "skipped; use the recorded baseline" if args.skip_cpu_one else "measured",
            "gpu_execution": "sequential candidate tasks, matching apply_adjacent_model_hybrid",
            "input": (
                "real Qwen3-8B layer-0 weight slices and sym=True affine qparams; deterministic "
                "correlated positive-definite 128x128 Hessian shared identically by CPU and GPU"
            ),
            "timing_repeats": args.timing_repeats,
        },
        "hardware": metadata,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print_table(results, args.cpu_workers)
    print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
