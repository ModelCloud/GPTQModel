#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep CPU chunk size and outer workers for a production-shaped Adjacent module."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent_model import AdjacentModelConfig  # noqa: E402
from scripts.quantum_quantization.benchmark_adjacent_model_cpu_gpu import (  # noqa: E402
    MODULE_TYPES,
    candidate_call,
    consume_candidate,
    make_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/monster/data/model/Qwen3-8B"))
    parser.add_argument("--module", choices=tuple(module.name for module in MODULE_TYPES), default="q_proj")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=898)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--row-chunk-sizes", type=int, nargs="+", default=(256, 512, 1024, 2048))
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--comparison-baseline-seconds",
        type=float,
        help="Optional independently measured baseline used only for the reported speedup.",
    )
    return parser.parse_args()


def summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def run_task(inputs, config: AdjacentModelConfig, bits: int) -> tuple[float, int, int]:
    return consume_candidate(candidate_call(inputs, config, bits))


def run_tasks(
    inputs,
    config: AdjacentModelConfig,
    bits: int,
    task_count: int,
    executor: ThreadPoolExecutor | None,
) -> tuple[float, int, int]:
    if executor is None:
        results = (run_task(inputs, config, bits) for _ in range(task_count))
    else:
        results = executor.map(
            lambda _index: run_task(inputs, config, bits),
            range(task_count),
        )
    checksum = 0.0
    converged = 0
    flips = 0
    for task_checksum, task_converged, task_flips in results:
        checksum += task_checksum
        converged += task_converged
        flips += task_flips
    return checksum, converged, flips


def hardware_metadata() -> dict[str, Any]:
    cpu_model = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.partition(":")[2].strip()
            break
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_gpu = None
    if visible_gpu:
        smi_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
        physical_gpu = next((line for line in smi_rows if visible_gpu in line), None)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "process_cpu_affinity_count": len(affinity) if affinity is not None else None,
        "platform": platform.platform(),
        "python": sys.version,
        "python_gil_enabled": bool(sys._is_gil_enabled()) if hasattr(sys, "_is_gil_enabled") else None,
        "torch": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "cuda_visible_devices": visible_gpu,
        "physical_gpu": physical_gpu,
    }


def main() -> None:
    args = parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive.")
    if any(chunk < 1 for chunk in args.row_chunk_sizes):
        raise ValueError("Row chunk sizes must be positive.")
    if any(worker < 1 for worker in args.workers):
        raise ValueError("Worker counts must be positive.")
    if args.comparison_baseline_seconds is not None and args.comparison_baseline_seconds <= 0:
        raise ValueError("--comparison-baseline-seconds must be positive.")
    if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled():
        raise RuntimeError("Run with PYTHON_GIL=0 so outer Python threads can execute concurrently.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    module = next(module for module in MODULE_TYPES if module.name == args.module)
    config = AdjacentModelConfig(
        coordinate_starts=("nearest", "zero", "one", "linear"),
        max_coordinate_flips=32,
        coordinate_rebase_interval=8,
        native_refinements_per_module=0,
    )
    results = []
    for row_chunk_size in args.row_chunk_sizes:
        inputs = make_inputs(
            args.model,
            module,
            bits=args.bits,
            group_size=args.group_size,
            row_chunk_size=row_chunk_size,
            seed=args.seed,
        )
        group_count = math.ceil(module.columns / args.group_size)
        row_chunks = math.ceil(module.rows / row_chunk_size)
        task_count = group_count * row_chunks
        row_group_count = module.rows * group_count
        reference_checksum = None
        for workers in args.workers:
            executor = (
                ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"adjacent-cpu-{workers}")
                if workers > 1
                else None
            )
            try:
                warmup_tasks = min(task_count, workers)
                run_tasks(inputs, config, args.bits, warmup_tasks, executor)
                samples = []
                checksums = []
                for _ in range(args.timing_repeats):
                    started = time.perf_counter()
                    checksum = run_tasks(inputs, config, args.bits, task_count, executor)
                    samples.append(time.perf_counter() - started)
                    checksums.append(checksum)
            finally:
                if executor is not None:
                    executor.shutdown()

            if any(checksum != checksums[0] for checksum in checksums[1:]):
                raise AssertionError("Repeated CPU runs produced different checksums.")
            if reference_checksum is None:
                reference_checksum = checksums[0]
            elif checksums[0] != reference_checksum:
                raise AssertionError("Worker-count variants produced different checksums.")
            median = statistics.median(samples)
            result = {
                "row_chunk_size": row_chunk_size,
                "workers": workers,
                "task_count": task_count,
                "row_group_count": row_group_count,
                "wall_seconds": summary(samples),
                "candidate_tasks_per_second": task_count / median,
                "row_groups_per_second": row_group_count / median,
                "checksum": checksums[0],
            }
            results.append(result)
            print(
                f"chunk={row_chunk_size:>4} workers={workers:>2} tasks={task_count:>4} "
                f"wall={median:>8.3f}s row-groups/s={row_group_count / median:>10.1f}",
                flush=True,
            )

    best = max(results, key=lambda result: result["row_groups_per_second"])
    in_run_baseline = min(
        (
            result
            for result in results
            if result["row_chunk_size"] == max(args.row_chunk_sizes)
        ),
        key=lambda result: result["workers"],
    )
    payload = {
        "schema": "gptqmodel-adjacent-cpu-scaling-v2",
        "command": sys.argv,
        "model": str(args.model),
        "module": module.name,
        "shape": [module.rows, module.columns],
        "seed": args.seed,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "coordinate_starts": list(config.coordinate_starts),
        "max_coordinate_flips": config.max_coordinate_flips,
        "coordinate_rebase_interval": config.coordinate_rebase_interval,
        "hardware": hardware_metadata(),
        "timing_repeats": args.timing_repeats,
        "results": results,
        "best": best,
        "in_run_baseline": in_run_baseline,
        "best_speedup_over_in_run_baseline": (
            in_run_baseline["wall_seconds"]["median"] / best["wall_seconds"]["median"]
        ),
        "comparison_baseline_seconds": args.comparison_baseline_seconds,
        "best_speedup_over_comparison_baseline": (
            args.comparison_baseline_seconds / best["wall_seconds"]["median"]
            if args.comparison_baseline_seconds is not None
            else None
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    status = (
        f"best chunk={best['row_chunk_size']} workers={best['workers']} "
        f"wall={best['wall_seconds']['median']:.3f}s "
        f"in-run-speedup={payload['best_speedup_over_in_run_baseline']:.3f}x"
    )
    if payload["best_speedup_over_comparison_baseline"] is not None:
        status += f" comparison-speedup={payload['best_speedup_over_comparison_baseline']:.3f}x"
    print(status, flush=True)
    print(f"JSON: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
