#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Capture one bounded Qwen3.8-27B H100 MLP CUDA Graph range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_qwen38_27b_h100_mlp as bench


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, choices=bench.RATES, default=3)
    parser.add_argument("--m", type=int, choices=bench.M_VALUES, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip the dense reference after a matched benchmark already passed it.",
    )
    parser.add_argument(
        "--eager-profile",
        action="store_true",
        help="Profile eager launches for kernel mapping when graph-node tracing is impractical.",
    )
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward
    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    mlp = bench.qwen38_qvq_mlp(torch, args.bits, device)
    if install_qvq_hopper_groups(mlp, qkv=False) != {"gate_up": 1}:
        raise RuntimeError("failed to install Qwen grouped MLP")
    x = (
        torch.randn(
            (args.m, bench.HIDDEN),
            generator=torch.Generator(device=device).manual_seed(53000 + args.m),
            device=device,
        )
        * 0.02
    ).half()

    def call():
        return (mlp(x),)

    with torch.inference_mode():
        actual = call()
        max_abs = None
        if not args.skip_oracle:
            gate = qvq_dense_oracle_forward(mlp.gate_proj, x, device=device).half()
            up = qvq_dense_oracle_forward(mlp.up_proj, x, device=device).half()
            intermediate = torch.nn.functional.silu(gate) * up
            expected = (
                qvq_dense_oracle_forward(mlp.down_proj, intermediate, device=device).half(),
            )
            max_abs = float(
                (actual[0].float() - expected[0].float()).abs().max().item()
            )
            if max_abs > 2e-3:
                raise RuntimeError(f"dense-P32 error gate failed: {max_abs}")
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        if args.eager_profile:
            captured = actual
            replay = call
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call()
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize(device)
            replay = graph.replay
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"qwen38_mlp_w{args.bits:g}_m{args.m}")
        for _ in range(args.replays):
            replay()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    print(
        json.dumps(
            {
                "device": device_info,
                "bits": args.bits,
                "m": args.m,
                "gate_up_mkn": [args.m, bench.HIDDEN, bench.INTERMEDIATE],
                "down_mkn": [args.m, bench.INTERMEDIATE, bench.HIDDEN],
                "graph_replays": args.replays,
                "execution": "eager-mapping" if args.eager_profile else "cuda-graph",
                "max_abs_error": max_abs,
                "captured_shape": list(captured[0].shape),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
