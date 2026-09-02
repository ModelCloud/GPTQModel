#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-bounded direct input-padding control on H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

K = 2048


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("separate", "direct"))
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=4)
    args = parser.parse_args()
    if args.warmup <= 0:
        parser.error("warmup must be positive")
    return args


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.utils.qvq_cuda import qvq_cuda_hadamard

    common._assert_h100(torch)
    generator = torch.Generator(device="cuda").manual_seed(18100 + args.m)
    x = torch.randn(
        (args.m, K), generator=generator, device="cuda", dtype=torch.float16
    )
    pre_scale = torch.randn(
        (K,), generator=generator, device="cuda", dtype=torch.float16
    )

    def separate():
        transformed = qvq_cuda_hadamard(x, pre_scale=pre_scale, scale_mode=2)
        padded = torch.zeros((16, K), dtype=transformed.dtype, device="cuda")
        padded[: args.m].copy_(transformed)
        return padded

    def direct():
        return qvq_cuda_hadamard(
            x,
            pre_scale=pre_scale,
            scale_mode=2,
            pad_to_16=True,
        )

    with torch.inference_mode():
        expected = separate()
        actual = direct()
        torch.cuda.synchronize()
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            raise RuntimeError("profile controls are not bit-exact")
        call = direct if args.variant == "direct" else separate
        if args.cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                profiled_output = call()
            torch.cuda.synchronize()
            for _ in range(args.warmup):
                graph.replay()
            replay = graph.replay
        else:
            for _ in range(args.warmup):
                profiled_output = call()
            replay = call
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    if not torch.equal(profiled_output.view(torch.int16), expected.view(torch.int16)):
        raise RuntimeError("profiled output changed")
    print(
        f"profiled variant={args.variant} M={args.m} K={K} "
        f"cuda_graph={args.cuda_graph}"
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
