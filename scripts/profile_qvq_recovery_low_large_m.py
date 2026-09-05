#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile the plain or paired N=8192 recovery-low kernel on Hopper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("plain", "paired"))
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=1)
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock,
        qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
    )

    if torch.cuda.get_device_name() != "NVIDIA H100":
        raise RuntimeError("large-M recovery-low profiling requires the physical H100")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260905 + args.m)
    input0 = torch.randn((args.m, 8192), generator=generator, device=device) * 20
    input1 = torch.randn((args.m, 8192), generator=generator, device=device) * 20
    post_scale0 = torch.ones(8192, device=device)
    post_scale1 = torch.ones(8192, device=device)
    pre_scale = torch.ones(8192, device=device, dtype=torch.float16)

    if args.variant == "plain":

        def call():
            return qvq_cuda_hadamard_pair_fp32_to_fp16_multiblock(
                input0,
                input1,
                post_scale0=post_scale0,
                post_scale1=post_scale1,
                scale_mode=3,
                warp_low=True,
            )

    else:

        def call():
            return qvq_cuda_hadamard_pair_swiglu_precondition_multiblock(
                input0,
                input1,
                post_scale0=post_scale0,
                post_scale1=post_scale1,
                pre_scale=pre_scale,
                scale_mode=3,
                pair_tiles=True,
            )

    with torch.inference_mode():
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    _main(_args())
