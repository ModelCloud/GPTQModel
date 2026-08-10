#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the healthy MPS GPTQ inverse-Cholesky lifecycle."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn

import gptqmodel.quantization.gptq as gptq_module
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def timed(call, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.mps.synchronize()
        started = time.perf_counter()
        call()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    return statistics.median(samples)


def benchmark(size: int, repeats: int) -> tuple[float, float]:
    torch.manual_seed(size)
    source = torch.randn(size, size, device="mps")
    hessian = source.T @ source / size + torch.eye(size, device="mps") * 0.1
    quantizer = GPTQ(
        nn.Linear(size, 4, bias=False, device="mps"),
        QuantizeConfig(
            bits=4,
            group_size=128,
            damp_percent=0.05,
            offload_to_disk=False,
        ),
    )

    gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = False
    reference, _ = quantizer._compute_hessian_inverse_uncached(hessian.clone())
    gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = True
    actual, _ = quantizer._compute_hessian_inverse_uncached(hessian.clone())
    torch.mps.synchronize()
    if not torch.equal(actual, reference):
        raise RuntimeError(
            "optimized MPS factorization is not bitwise equal to the canonical result"
        )

    for _ in range(3):
        gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = False
        quantizer._compute_hessian_inverse_uncached(hessian.clone())
        gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = True
        quantizer._compute_hessian_inverse_uncached(hessian.clone())
    torch.mps.synchronize()
    gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = False
    canonical_ms = timed(
        lambda: quantizer._compute_hessian_inverse_uncached(hessian.clone()), repeats
    )
    gptq_module._USE_GPTQ_MPS_FAST_HESSIAN = True
    optimized_ms = timed(
        lambda: quantizer._compute_hessian_inverse_uncached(hessian.clone()), repeats
    )
    return canonical_ms, optimized_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("This benchmark requires an available MPS device.")

    print(f"Device: mps | Torch: {torch.__version__}")
    print("+------+--------------+--------------+---------+")
    print("| Size | Canonical ms | Optimized ms | Speedup |")
    print("+------+--------------+--------------+---------+")
    for size in (128, 512, 1024):
        canonical_ms, optimized_ms = benchmark(size, args.repeats)
        print(
            f"| {size:4d} | {canonical_ms:12.3f} | {optimized_ms:12.3f} | {canonical_ms / optimized_ms:6.2f}x |"
        )
    print("+------+--------------+--------------+---------+")


if __name__ == "__main__":
    main()
