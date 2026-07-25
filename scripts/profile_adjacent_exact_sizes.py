#!/usr/bin/env python3
"""Benchmark AdjacentExact CUDA exact kernel for active decision sizes near the limit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import adjacent_exact_cuda, build_adjacent_rounding_qubo  # noqa: E402


def main():
    device = "cuda:0"
    bits = 4
    maxq = (1 << bits) - 1
    generator = torch.Generator(device=device).manual_seed(42)
    for size in (28, 30, 32):
        x = torch.rand((size,), generator=generator, device=device, dtype=torch.float64) * (maxq - 0.2) + 0.1
        A = torch.randn((size, size), generator=generator, device=device, dtype=torch.float64)
        A = (A + A.t()) * 0.5
        problem = build_adjacent_rounding_qubo(x, A, scale=1.0, zero=0.0, bits=bits)
        adjacent_exact_cuda(problem, decompose=False, warps=0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = adjacent_exact_cuda(problem, decompose=False, warps=0)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000.0
        print(f"size={size} active={problem.active_decisions} ms={ms:.3f} cost={result.cost:.6f}", flush=True)


if __name__ == "__main__":
    main()
