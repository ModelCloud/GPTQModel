#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Capture a bounded CUDA-profiler range for one large-M grouped P32 site."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=float, choices=(2, 2.5, 3, 3.5), default=3)
    parser.add_argument("--group", choices=tuple(common.GROUPS), default="gate_up")
    parser.add_argument("--m", type=int, choices=(32, 64, 128, 256), default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--replays", type=int, default=5)
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
    names, widths, alt_ids = common.GROUPS[args.group]
    shared_su = torch.ones(common.K, device=device, dtype=torch.float32)
    children = tuple(
        common._qvq_child(
            torch,
            name,
            width,
            args.bits,
            alt_id,
            93000 + index,
            device,
            shared_su,
        )
        for index, (name, width, alt_id) in enumerate(
            zip(names, widths, alt_ids, strict=True)
        )
    )
    for child in children:
        child.SV.fill_(0.002)
        child._dtype_cache_clear()
    parent = common._projection_parent(torch, names, children)
    installed = install_qvq_hopper_groups(
        parent, qkv=args.group == "qkv", gate_up=args.group == "gate_up"
    )
    if installed[args.group] != 1:
        raise RuntimeError(f"failed to install grouped runtime: {installed}")
    input = (
        torch.randn(
            (args.m, common.K),
            generator=torch.Generator(device=device).manual_seed(93100 + args.m),
            device=device,
        )
        * 0.02
    ).half()

    def call():
        return common._call_children(parent, names, input)

    with torch.inference_mode():
        expected = tuple(
            qvq_dense_oracle_forward(child, input, device=device) for child in children
        )
        eager = call()
        max_abs = max(
            float((actual.float() - reference.float()).abs().max().item())
            for actual, reference in zip(eager, expected, strict=True)
        )
        if max_abs > 2e-3:
            raise RuntimeError(f"dense P32 accuracy failed: max_abs={max_abs}")
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(
            f"qvq_large_m_{args.group}_w{args.bits:g}_m{args.m}"
        )
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
    print(
        json.dumps(
            {
                "device": device_info,
                "bits": args.bits,
                "group": args.group,
                "mkn": [args.m, common.K, sum(widths)],
                "child_n": widths,
                "graph_replays": args.replays,
                "max_abs_error": max_abs,
                "captured_shapes": [list(output.shape) for output in captured],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
