#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Emit one profiler-bounded production A41/R0 Llama MLP on H100."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_a41_phase5_mlp as mlp_benchmark


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, choices=(2, 2.5, 3, 3.5), default=3)
    parser.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), default=1)
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

    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups

    common._assert_h100(torch)
    device = torch.device("cuda:0")
    x = (
        torch.randn(
            (args.m, mlp_benchmark.HIDDEN),
            generator=torch.Generator(device=device).manual_seed(19000 + args.m),
            device=device,
        )
        * 0.02
    ).half()
    mlp = mlp_benchmark._qvq_mlp(torch, args.bits, device)

    with torch.inference_mode():
        expected = mlp(x)
        counts = install_qvq_hopper_groups(
            mlp, qkv=False, gate_up_activation=True
        )
        if counts != {"gate_up": 1}:
            raise RuntimeError(f"failed to install fused MLP runtime: {counts}")
        actual = mlp(x)
        torch.cuda.synchronize()
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            raise RuntimeError("profile control changed MLP output")

        if args.cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                profiled_output = mlp(x)
            torch.cuda.synchronize()
            for _ in range(args.warmup):
                graph.replay()
            replay = graph.replay
        else:
            for _ in range(args.warmup):
                profiled_output = mlp(x)
            replay = lambda: mlp(x)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    if not torch.equal(
        profiled_output.view(torch.int16), expected.view(torch.int16)
    ):
        raise RuntimeError("profiled MLP output changed")
    print(
        f"profiled W{args.bits:g} M{args.m} production MLP "
        f"cuda_graph={args.cuda_graph}"
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
