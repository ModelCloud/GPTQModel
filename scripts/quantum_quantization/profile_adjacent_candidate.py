#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Capture one warmed Adjacent candidate task inside a CUDA profiler range."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent_model import AdjacentModelConfig  # noqa: E402
from scripts.quantum_quantization.benchmark_adjacent_model_cpu_gpu import (  # noqa: E402
    MODULE_TYPES,
    candidate_call,
    consume_candidate,
    make_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/monster/data/model/Qwen3-8B"))
    parser.add_argument("--module", choices=tuple(module.name for module in MODULE_TYPES), default="q_proj")
    parser.add_argument("--seed", type=int, default=898)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", type=int, default=128, choices=(32, 64, 128))
    parser.add_argument("--row-chunk-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active-calls", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU.")

    module = next(module for module in MODULE_TYPES if module.name == args.module)
    config = AdjacentModelConfig(
        coordinate_starts=("nearest", "zero", "one", "linear"),
        max_coordinate_flips=32,
        coordinate_rebase_interval=8,
        row_chunk_size=args.row_chunk_size,
        native_refinements_per_module=4,
        native_split_depth=6,
        native_max_nodes_per_worker=500,
    )
    inputs = make_inputs(
        args.model,
        module,
        bits=args.bits,
        group_size=args.group_size,
        row_chunk_size=args.row_chunk_size,
        seed=args.seed,
    ).to(torch.device("cuda", 0))

    for _ in range(args.warmup):
        consume_candidate(candidate_call(inputs, config, args.bits))
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"adjacent_candidate:{args.module}")
    checksums = [
        consume_candidate(candidate_call(inputs, config, args.bits))
        for _ in range(args.active_calls)
    ]
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(
        f"module={args.module} shape={tuple(inputs.weight.shape)} "
        f"active_calls={args.active_calls} checksums={checksums}",
        flush=True,
    )


if __name__ == "__main__":
    main()
