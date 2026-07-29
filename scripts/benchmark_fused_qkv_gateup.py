#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark fused QKV and gate/up GPTQ projections against separate calls.

Run with the target GPU selected, e.g.:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_fused_qkv_gateup.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from gptqmodel.nn_modules.fused_quant_linear import (
    install_fused_gate_up,
    install_fused_qkv,
)
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


class Attn(nn.Module):
    def __init__(self, hidden: int, q_out: int, kv_out: int):
        super().__init__()
        self.q_proj = _make_tritonv2_linear(hidden, q_out)
        self.k_proj = _make_tritonv2_linear(hidden, kv_out)
        self.v_proj = _make_tritonv2_linear(hidden, kv_out)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = _make_tritonv2_linear(hidden, intermediate)
        self.up_proj = _make_tritonv2_linear(hidden, intermediate)


def _measure(fn: callable, x: torch.Tensor, warmup: int, iters: int) -> Tuple[float, torch.Tensor]:
    # Warmup
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


def _run_qkv(
    hidden: int,
    q_out: int,
    kv_out: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    model = Attn(hidden, q_out, kv_out).to(device).eval()
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=torch.bfloat16)

    def separate(x):
        return model.q_proj(x), model.k_proj(x), model.v_proj(x)

    ms_sep, _ = _measure(separate, x, warmup, iters)

    with torch.inference_mode():
        sep = torch.cat(separate(x), dim=-1)

    fused_count = install_fused_qkv(model)
    assert fused_count == 1

    def fused(x):
        return model.q_proj(x), model.k_proj(x), model.v_proj(x)

    ms_fused, _ = _measure(fused, x, warmup, iters)

    with torch.inference_mode():
        fus = torch.cat(fused(x), dim=-1)
    max_diff = (sep - fus).abs().max().item()

    tokens = batch * seq_len
    return {
        "op": "qkv",
        "hidden": hidden,
        "out": (q_out, kv_out, kv_out),
        "batch": batch,
        "seq_len": seq_len,
        "ms_sep": ms_sep,
        "ms_fused": ms_fused,
        "speedup": ms_sep / ms_fused if ms_fused > 0 else float("inf"),
        "throughput_sep": tokens / (ms_sep / 1000.0),
        "throughput_fused": tokens / (ms_fused / 1000.0),
        "max_diff": max_diff,
    }


def _run_gate_up(
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

    def separate(x):
        gate = model.gate_proj(x)
        up = model.up_proj(x)
        return torch.nn.functional.silu(gate) * up

    ms_sep, _ = _measure(separate, x, warmup, iters)

    with torch.inference_mode():
        out_sep = separate(x)

    fused_count = install_fused_gate_up(model)
    assert fused_count == 1

    def fused(x):
        gate = model.gate_proj(x)
        up = model.up_proj(x)
        return torch.nn.functional.silu(gate) * up

    ms_fused, _ = _measure(fused, x, warmup, iters)

    with torch.inference_mode():
        out_fus = fused(x)
    max_diff = (out_sep - out_fus).abs().max().item()

    tokens = batch * seq_len
    return {
        "op": "gate_up",
        "hidden": hidden,
        "out": (intermediate, intermediate),
        "batch": batch,
        "seq_len": seq_len,
        "ms_sep": ms_sep,
        "ms_fused": ms_fused,
        "speedup": ms_sep / ms_fused if ms_fused > 0 else float("inf"),
        "throughput_sep": tokens / (ms_sep / 1000.0),
        "throughput_fused": tokens / (ms_fused / 1000.0),
        "max_diff": max_diff,
    }


def _idle_preflight(device: torch.device) -> None:
    torch.cuda.synchronize(device)
    time.sleep(0.5)
    torch.cuda.synchronize(device)


def _print_env() -> None:
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Compute capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    import triton
    print("Triton:", triton.__version__)
    print()


def _print_table(rows: List[dict]) -> None:
    header = f"{'op':>8} {'hidden':>6} {'out':>18} {'batch':>6} {'seq':>4} {'ms_sep':>10} {'ms_fused':>10} {'speedup':>8} {'tok/s_sep':>12} {'tok/s_fused':>14} {'max_diff':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        out_str = str(r["out"])
        print(
            f"{r['op']:>8} {r['hidden']:>6} {out_str:>18} {r['batch']:>6} {r['seq_len']:>4} "
            f"{r['ms_sep']:>10.3f} {r['ms_fused']:>10.3f} {r['speedup']:>8.3f} "
            f"{r['throughput_sep']:>12.1f} {r['throughput_fused']:>14.1f} {r['max_diff']:>10.4f}"
        )


_ALL_SHAPES = [
    ("Laguna-S-2.1", 3072, (6144, 1024, 1024), 12288),
    ("Qwen3.5-27B", 5120, (6144, 1024, 1024), 17408),
    ("Llama-like", 4096, (4096, 1024, 1024), 11008),
]


def _parse_models(arg: str) -> List[str]:
    return [m.strip() for m in arg.split(",") if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=None,
        help="Comma-separated list of model names to benchmark, e.g. 'Laguna-S-2.1,Qwen3.5-27B'",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to write row data")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    _idle_preflight(device)
    _print_env()

    # Kimi-K3 uses KDA/MLA and latent-space MoE; skip standard QKV/gate-up in this benchmark.
    batch_seqs = [
        # decode batch sweep
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (16, 1),
        (32, 1),
        # prefill lengths
        (1, 128),
        (1, 1024),
        (1, 4096),
        (16, 128),
    ]

    shapes = _ALL_SHAPES
    if args.models:
        name_set = set(args.models)
        shapes = [s for s in shapes if s[0] in name_set]
        missing = name_set - {s[0] for s in shapes}
        if missing:
            print(f"Unknown model names: {missing}", file=sys.stderr)
            sys.exit(1)
        if not shapes:
            print("No matching models to benchmark", file=sys.stderr)
            sys.exit(1)

    rows: List[dict] = []
    for name, hidden, qkv_out, intermediate in shapes:
        q_out, k_out, v_out = qkv_out
        for batch, seq_len in batch_seqs:
            print(f"Benchmarking {name} QKV batch={batch} seq={seq_len} ...")
            rows.append(_run_qkv(hidden, q_out, k_out, batch, seq_len, device, args.warmup, args.iters))
            print(f"Benchmarking {name} gate/up batch={batch} seq={seq_len} ...")
            rows.append(_run_gate_up(hidden, intermediate, batch, seq_len, device, args.warmup, args.iters))

    print()
    _print_table(rows)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
