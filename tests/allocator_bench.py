#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Stress the CUDA caching allocator to compare PYTORCH_ALLOC_CONF tunings."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Iterable, List

import torch


PROFILES: Dict[str, List[Dict[str, Iterable[int]]]] = {
    "24gb": [
        {"allocate_mb": [2048, 2048, 2048, 1536, 1536, 1280], "release": [1, 4], "sleep_cycles": 80_000},
        {"allocate_mb": [2560, 1792, 1024, 1600], "release": [0, 3], "sleep_cycles": 80_000},
        {"allocate_mb": [1280, 2048, 896, 2304], "release": [], "sleep_cycles": 80_000},
    ],
    "80gb": [
        {"allocate_mb": [4096, 4096, 4096, 3584, 3584, 3072, 3072], "release": [1, 3, 5], "sleep_cycles": 120_000},
        {"allocate_mb": [5120, 4096, 3584, 2560, 2304], "release": [0, 2], "sleep_cycles": 120_000},
        {"allocate_mb": [3584, 4096, 4608, 5120], "release": [], "sleep_cycles": 120_000},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES.keys(), default="24gb")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--final-sleep-cycles", type=int, default=500_000, help="extra sleep cycles after each iteration")
    parser.add_argument("--phase-sleep-scale", type=float, default=1.0, help="scale factor applied to per-phase sleep settings")
    parser.add_argument("--fill", choices=("none", "uniform", "normal"), default="none")
    parser.add_argument("--json", action="store_true", help="emit metrics as JSON")
    return parser.parse_args()


def _dtype_from_string(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unsupported dtype {dtype_name!r}")
    return dtype


def _allocate_tensor(
    size_mb: int,
    dtype: torch.dtype,
    device: torch.device,
    element_size: int,
    fill_mode: str,
) -> torch.Tensor:
    numel = (size_mb * 1024 * 1024) // element_size
    if numel == 0:
        raise ValueError(f"Requested allocation too small: {size_mb} MB")
    tensor = torch.empty((numel,), dtype=dtype, device=device)
    if fill_mode == "uniform":
        tensor.uniform_(-1.0, 1.0)
    elif fill_mode == "normal":
        tensor.normal_(mean=0.0, std=1.0)
    return tensor


def _run_iteration(
    phases: List[Dict[str, Iterable[int]]],
    dtype: torch.dtype,
    device: torch.device,
    element_size: int,
    fill_mode: str,
    phase_sleep_scale: float,
    final_sleep_cycles: int,
) -> None:
    allocations: List[torch.Tensor] = []
    for phase in phases:
        for size_mb in phase["allocate_mb"]:
            allocations.append(_allocate_tensor(size_mb, dtype, device, element_size, fill_mode))
        sleep_cycles = int(phase.get("sleep_cycles", 0) * phase_sleep_scale)
        if sleep_cycles:
            torch.cuda._sleep(sleep_cycles)
        release_indices = phase.get("release", [])
        if release_indices:
            for idx in sorted(release_indices, reverse=True):
                if 0 <= idx < len(allocations):
                    del allocations[idx]
        post_sleep_cycles = int(phase.get("post_sleep_cycles", 0) * phase_sleep_scale)
        if post_sleep_cycles:
            torch.cuda._sleep(post_sleep_cycles)
    if final_sleep_cycles:
        torch.cuda._sleep(final_sleep_cycles)
    allocations.clear()


def main() -> None:
    args = parse_args()
    dtype = _dtype_from_string(args.dtype)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    element_size = torch.tensor([], dtype=dtype).element_size()

    phases = PROFILES[args.profile]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        _run_iteration(
            phases,
            dtype,
            device,
            element_size,
            args.fill,
            args.phase_sleep_scale,
            args.final_sleep_cycles,
        )
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    durations: List[float] = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _run_iteration(
            phases,
            dtype,
            device,
            element_size,
            args.fill,
            args.phase_sleep_scale,
            args.final_sleep_cycles,
        )
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)

    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    metrics = {
        "device": str(device),
        "profile": args.profile,
        "iterations": args.iterations,
        "dtype": args.dtype,
        "fill": args.fill,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": peak_reserved / (1024 ** 3),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gib": peak_allocated / (1024 ** 3),
        "per_iter_seconds": durations,
        "mean_iter_seconds": sum(durations) / len(durations),
        "stdev_iter_seconds": float(torch.tensor(durations).std(unbiased=False)) if len(durations) > 1 else 0.0,
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF", "<unset>"),
    }

    if args.json:
        print(json.dumps(metrics))
    else:
        print("Benchmark metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
