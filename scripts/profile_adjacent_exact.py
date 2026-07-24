#!/usr/bin/env python3
"""Benchmark AdjacentExact CUDA exact and branch-bound kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import AdjacentRoundingQUBO, adjacent_branch_bound_cuda, adjacent_exact_cuda, build_adjacent_rounding_qubo  # noqa: E402


def make_problem(size: int, device: str, bits: int = 4):
    generator = torch.Generator(device=device).manual_seed(42)
    maxq = (1 << bits) - 1
    # Use values that fall strictly between quantization codes so every decision is active.
    x = torch.rand((size,), generator=generator, device=device, dtype=torch.float64) * (maxq - 0.2) + 0.1
    A = torch.randn((size, size), generator=generator, device=device, dtype=torch.float64)
    A = (A + A.t()) * 0.5
    return build_adjacent_rounding_qubo(x, A, scale=1.0, zero=0.0, bits=bits)


def benchmark_exact(size: int, device: str, warps: int = 0, repeats: int = 5):
    problem = make_problem(size, device)
    # warmup
    adjacent_exact_cuda(problem, decompose=False, warps=warps)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        adjacent_exact_cuda(problem, decompose=False, warps=warps)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {"size": size, "warps": warps, "ms_mean": statistics.mean(times), "ms_median": statistics.median(times), "ms_min": min(times), "ms_max": max(times)}


def benchmark_branch_bound(size: int, device: str, repeats: int = 5):
    problem = make_problem(size, device)
    # warmup
    adjacent_branch_bound_cuda(problem)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        adjacent_branch_bound_cuda(problem)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {"size": size, "ms_mean": statistics.mean(times), "ms_median": statistics.median(times), "ms_min": min(times), "ms_max": max(times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    print("Exact kernel", flush=True)
    for size in (20, 24, 26, 28):
        print(f" size={size}", flush=True)
        results.append(benchmark_exact(size, args.device))
        print(json.dumps(results[-1], indent=2), flush=True)

    print("Branch-bound kernel", flush=True)
    for size in (40, 48, 56, 64):
        print(f" size={size}", flush=True)
        results.append(benchmark_branch_bound(size, args.device))
        print(json.dumps(results[-1], indent=2), flush=True)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
