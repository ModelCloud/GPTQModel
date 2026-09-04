#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-marked Phase-26 input-Hadamard launch on physical H100."""

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
    parser.add_argument("--variant", choices=("one_block", "multiblock"), required=True)
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=4)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard,
        qvq_cuda_hadamard_input_fp16_padded_multiblock,
    )

    common._assert_h100(torch)
    generator = torch.Generator(device="cuda").manual_seed(20262690 + args.m)
    x = torch.randn(
        (args.m, 2048), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (2048,), generator=generator, device="cuda", dtype=torch.float16
    )

    def one_block():
        return qvq_cuda_hadamard(
            x, pre_scale=pre_scale, scale_mode=2, pad_to_16=True
        )

    def multiblock():
        return qvq_cuda_hadamard_input_fp16_padded_multiblock(
            x, pre_scale=pre_scale
        )

    expected = one_block()
    candidate = multiblock()
    torch.cuda.synchronize()
    if not torch.equal(expected.view(torch.int16), candidate.view(torch.int16)):
        raise RuntimeError("Phase-26 profiler controls are not bit-exact")
    call = multiblock if args.variant == "multiblock" else one_block
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    output = call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    if not torch.equal(output.view(torch.int16), expected.view(torch.int16)):
        raise RuntimeError("profiled Phase-26 output changed")
    print(
        f"profiled variant={args.variant} M={args.m} exact=True "
        "device=GPU-f5ea03cf-efa4-9807-7de5-b174957a1348"
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
