# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired NF12 linear benchmark for short batched-decode and prefill rows."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F


DEFAULT_SHAPES = (
    (2048, 2048),
    (512, 2048),
    (8192, 2048),
    (2048, 8192),
    (4096, 4096),
    (128256, 2048),
)


def _median_us(fn: Callable[[], torch.Tensor], *, warmup: int, trials: int, iterations: int) -> float:
    samples = []
    for _ in range(trials):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / iterations / 1_000)
    return statistics.median(samples)


def _backend():
    if torch.version.hip is not None:
        from gptqmodel.utils import nf12_hip

        if not nf12_hip.prewarm_nf12_hip_extension():
            raise RuntimeError(nf12_hip.last_error() or "NF12 HIP extension unavailable")

        def encode(weight):
            return nf12_hip.encode(weight)

        def linear(x, packed):
            return nf12_hip.linear(x, *packed)

        return encode, linear, torch.cuda.get_device_properties(0).gcnArchName

    import gptqmodel_nf12 as nf12

    nf12.load()
    major, minor = torch.cuda.get_device_capability(0)
    return nf12.encode, nf12.linear, f"sm_{major}{minor}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m", nargs="+", type=int, default=(1, 4, 8, 16))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encode, linear, arch = _backend()
    print(f"DEVICE {torch.cuda.get_device_name(device)} {arch}", flush=True)

    for n, k in DEFAULT_SHAPES:
        generator = torch.Generator(device=device).manual_seed(20261113 + n + k)
        weight = (torch.randn((n, k), device=device, generator=generator) * 0.02).bfloat16()
        packed = encode(weight)
        for m in args.m:
            x = (torch.randn((m, k), device=device, generator=generator) * 0.5).bfloat16()
            dense = F.linear(x, weight)
            actual = linear(x, packed)
            error = (actual.float() - dense.float()).abs()
            if not torch.isfinite(actual).all():
                raise RuntimeError(f"nonfinite NF12 output for N={n} K={k} M={m}")

            dense_us = _median_us(
                partial(F.linear, x, weight),
                warmup=args.warmup,
                trials=args.trials,
                iterations=args.iterations,
            )
            nf12_us = _median_us(
                partial(linear, x, packed),
                warmup=args.warmup,
                trials=args.trials,
                iterations=args.iterations,
            )
            print(
                "RESULT "
                + json.dumps(
                    {
                        "shape": [n, k],
                        "m": m,
                        "dense_us": dense_us,
                        "nf12_us": nf12_us,
                        "speedup": dense_us / nf12_us,
                        "max_abs": error.max().item(),
                        "mean_abs": error.mean().item(),
                    }
                ),
                flush=True,
            )
        torch.cuda.synchronize()
        del weight, packed


if __name__ == "__main__":
    main()
