#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-marked Phase-25 down-recovery launch on physical H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scalar", "multiblock"), required=True)
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=4)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard_ordered_split16_fp32_to_fp16,
    )

    common._assert_h100(torch)
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20262501 + args.m)
    partials = (
        torch.randn(
            (16, 16, 2048),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    post_scale = torch.randn(
        (2048,), generator=generator, device=device, dtype=torch.float32
    )
    bias = torch.randn(
        (2048,), generator=generator, device=device, dtype=torch.float32
    )

    def call(multiblock: bool):
        return qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
            partials,
            post_scale=post_scale,
            bias=bias,
            scale_mode=3,
            logical_rows=args.m,
            multiblock=multiblock,
        )

    scalar = call(False)
    candidate = call(True)
    torch.cuda.synchronize()
    if not torch.equal(scalar.view(torch.int16), candidate.view(torch.int16)):
        raise RuntimeError("Phase-25 profiler input is not bit-exact")
    selected = args.variant == "multiblock"
    for _ in range(20):
        call(selected)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    output = call(selected)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    if not torch.equal(output.view(torch.int16), scalar.view(torch.int16)):
        raise RuntimeError("profiled Phase-25 output changed")
    print(
        f"profiled variant={args.variant} M={args.m} exact=True "
        "device=GPU-f5ea03cf-efa4-9807-7de5-b174957a1348"
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
