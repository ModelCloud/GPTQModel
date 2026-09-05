#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile one-block or multiblock Mx2048 FP32 output recovery on H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("one_block", "multiblock"))
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard,
        qvq_cuda_hadamard_fp32_to_fp16_multiblock,
    )

    if torch.cuda.get_device_name() != "NVIDIA H100":
        raise RuntimeError("large-M output recovery profiling requires H100")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260909 + args.m)
    input = torch.randn((args.m, 2048), generator=generator, device=device) * 20
    scale = torch.randn((2048,), generator=generator, device=device)
    bias = torch.randn((2048,), generator=generator, device=device)

    def one_block():
        return qvq_cuda_hadamard(
            input,
            post_scale=scale,
            bias=bias,
            scale_mode=3,
            output_fp16=True,
        )

    def multiblock():
        return qvq_cuda_hadamard_fp32_to_fp16_multiblock(
            input, post_scale=scale, bias=bias, scale_mode=3
        )

    with torch.inference_mode():
        expected = one_block()
        candidate = multiblock()
        if not torch.equal(expected.view(torch.int16), candidate.view(torch.int16)):
            raise RuntimeError("output recovery profiler controls are not bit exact")
        call = one_block if args.variant == "one_block" else multiblock
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
        if not torch.equal(output.view(torch.int16), expected.view(torch.int16)):
            raise RuntimeError("profiled output recovery changed")


if __name__ == "__main__":
    _main(_args())
