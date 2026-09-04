#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-bounded Phase-11 low-stage launch on H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

N = 8192


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("scalar", "half2"))
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    args = parser.parse_args()
    if args.warmup <= 0:
        parser.error("warmup must be positive")
    return args


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import qvq_cuda_swiglu_precondition_multiblock

    common._assert_h100(torch)
    generator = torch.Generator(device="cuda").manual_seed(88000 + args.m)
    gate = torch.randn(
        (args.m, N), generator=generator, device="cuda", dtype=torch.float16
    )
    up = torch.randn(
        (args.m, N), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (N,), generator=generator, device="cuda", dtype=torch.float16
    )

    def scalar():
        return qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=True,
            fuse_silu=True,
        )

    def half2():
        return qvq_cuda_swiglu_precondition_multiblock(
            gate,
            up,
            pre_scale,
            half2_high=True,
            fuse_silu=True,
            half2_low=True,
        )

    with torch.inference_mode():
        expected = scalar()
        actual = half2()
        torch.cuda.synchronize()
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            raise RuntimeError("profile controls are not bit-exact")
        call = half2 if args.variant == "half2" else scalar
        for _ in range(args.warmup):
            profiled_output = call()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        profiled_output = call()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    if not torch.equal(profiled_output.view(torch.int16), expected.view(torch.int16)):
        raise RuntimeError("profiled output changed")
    print(f"profiled variant={args.variant} M={args.m} N={N}")


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
