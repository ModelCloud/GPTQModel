#!/usr/bin/env python3
"""Run a single AdjacentExact branch-bound kernel call for ncu profiling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import (  # noqa: E402
    adjacent_branch_bound_cuda,
    build_adjacent_rounding_qubo,
)


def make_problem(size: int, device: str, bits: int = 4):
    generator = torch.Generator(device=device).manual_seed(42)
    maxq = (1 << bits) - 1
    x = torch.rand((size,), generator=generator, device=device, dtype=torch.float64) * (maxq - 0.2) + 0.1
    A = torch.randn((size, size), generator=generator, device=device, dtype=torch.float64)
    A = (A + A.t()) * 0.5
    return build_adjacent_rounding_qubo(x, A, scale=1.0, zero=0.0, bits=bits)


if __name__ == "__main__":
    device = "cuda:0"
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    problem = make_problem(size, device)
    # Warmup to avoid JIT compile inside profiled region.
    adjacent_branch_bound_cuda(problem, split_depth=min(20, size))
    torch.cuda.synchronize()
    adjacent_branch_bound_cuda(problem, split_depth=min(20, size))
    torch.cuda.synchronize()
    print(f"size={size} done", flush=True)
