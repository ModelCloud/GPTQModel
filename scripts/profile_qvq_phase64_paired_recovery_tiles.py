#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-bounded Phase-63 or paired recovery launch on H100."""

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
    parser.add_argument("variant", choices=("phase63", "paired"))
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=16)
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

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
    )

    common._assert_h100(torch)
    generator = torch.Generator(device="cuda").manual_seed(164100 + args.m)
    input0 = torch.randn((args.m, N), generator=generator, device="cuda") * 20
    input1 = torch.randn((args.m, N), generator=generator, device="cuda") * 20
    scale0 = torch.randn((N,), generator=generator, device="cuda")
    scale1 = torch.randn((N,), generator=generator, device="cuda")
    bias0 = torch.randn((N,), generator=generator, device="cuda")
    bias1 = torch.randn((N,), generator=generator, device="cuda")
    pre_scale = torch.randn(
        (N,), generator=generator, device="cuda", dtype=torch.float16
    )

    def call(pair_tiles: bool):
        return qvq_cuda_hadamard_pair_swiglu_precondition_multiblock(
            input0,
            input1,
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            bias1=bias1,
            pre_scale=pre_scale,
            pad_to_16=args.m < 16,
            pair_tiles=pair_tiles,
        )

    with torch.inference_mode():
        expected = call(False)
        actual = call(True)
        torch.cuda.synchronize()
        if not torch.equal(actual, expected):
            raise RuntimeError("profile controls are not bit-exact")
        use_pair = args.variant == "paired"
        for _ in range(args.warmup):
            profiled_output = call(use_pair)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        profiled_output = call(use_pair)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    if not torch.equal(profiled_output, expected):
        raise RuntimeError("profiled output changed")
    print(f"profiled variant={args.variant} M={args.m} N={N}")


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
