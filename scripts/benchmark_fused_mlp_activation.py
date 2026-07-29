#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused gate/up activation MLP forward against the unfused path.

Run with the target GPU selected, e.g.:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_mlp_activation.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


def _make_tritonv2_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 128,
) -> TritonV2Linear:
    maxq = 2**bits - 1
    zero_point = maxq // 2
    m = TritonV2Linear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    W = torch.randn(out_features, in_features) * 0.3
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = W
    num_groups = math.ceil(in_features / group_size)
    scales = torch.rand(out_features, num_groups) * 0.4 + 0.2
    zeros = torch.full((out_features, num_groups), zero_point, dtype=torch.int32)
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
    m.pack_block(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    return m.cuda().eval()


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = _make_tritonv2_linear(hidden, intermediate)
        self.up_proj = _make_tritonv2_linear(hidden, intermediate)
        self.down_proj = _make_tritonv2_linear(intermediate, hidden)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _measure(fn: callable, x: torch.Tensor, warmup: int, iters: int) -> Tuple[float, torch.Tensor]:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn(x)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / iters
    out = fn(x)
    return elapsed_ms, out


def _idle_preflight(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    time.sleep(0.5)
    torch.cuda.synchronize(device)


def _run(
    hidden: int,
    intermediate: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    model = MLP(hidden, intermediate).to(device).eval()
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = model(x)

    ms_sep, _ = _measure(model, x, warmup, iters)

    fused_count = install_fused_gate_up(model)
    assert fused_count == 1

    ms_fused, actual = _measure(model, x, warmup, iters)
    max_diff = (expected - actual).abs().max().item()

    tokens = batch * seq_len
    return {
        "hidden": hidden,
        "intermediate": intermediate,
        "batch": batch,
        "seq_len": seq_len,
        "ms_sep": ms_sep,
        "ms_fused": ms_fused,
        "speedup": ms_sep / ms_fused if ms_fused > 0 else float("inf"),
        "throughput_sep": tokens / (ms_sep / 1000.0),
        "throughput_fused": tokens / (ms_fused / 1000.0),
        "max_diff": max_diff,
    }


def _print_env() -> None:
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Compute capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    import triton
    print("Triton:", triton.__version__)
    print()


def _print_table(rows: List[dict]) -> None:
    header = f"{'hidden':>6} {'intermediate':>12} {'batch':>6} {'seq':>4} {'ms_sep':>10} {'ms_fused':>10} {'speedup':>8} {'tok/s_sep':>12} {'tok/s_fused':>14} {'max_diff':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['hidden']:>6} {r['intermediate']:>12} {r['batch']:>6} {r['seq_len']:>4} "
            f"{r['ms_sep']:>10.3f} {r['ms_fused']:>10.3f} {r['speedup']:>8.3f} "
            f"{r['throughput_sep']:>12.1f} {r['throughput_fused']:>14.1f} {r['max_diff']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused gate/up activation MLP")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _idle_preflight(device)
    _print_env()

    shapes = [
        ("Laguna-S-2.1", 3072, 12288),
        ("Qwen3.5-27B", 5120, 17408),
        ("Llama-like", 4096, 11008),
    ]
    decode_batches = [1, 2, 4, 8, 16, 32]
    prefill = [(1, 128), (1, 1024), (1, 4096), (16, 128)]

    all_rows: List[dict] = []
    for name, hidden, intermediate in shapes:
        print(f"\n=== {name}: hidden={hidden}, intermediate={intermediate} ===")
        rows: List[dict] = []
        for batch in decode_batches:
            r = _run(hidden, intermediate, batch, 1, device, args.warmup, args.iters)
            r["model"] = name
            rows.append(r)
            all_rows.append(r)
        for batch, seq_len in prefill:
            r = _run(hidden, intermediate, batch, seq_len, device, args.warmup, args.iters)
            r["model"] = name
            rows.append(r)
            all_rows.append(r)
        _print_table(rows)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nWrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
