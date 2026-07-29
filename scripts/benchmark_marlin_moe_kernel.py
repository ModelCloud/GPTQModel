#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the batched/offset Marlin MoE mega-kernel.

Targets Laguna-S-2.1, Qwen3.5-27B, Llama-like and Kimi-K3 proxy MoE shapes.
Compares the per-expert Marlin loop, the dense grouped_mm path, and the new
``marlin_moe`` mega-kernel.  All printed times are CUDA-event latencies in ms
and include warm-up iterations.

The ``--gpu`` flag is parsed before any CUDA context is created.
"""

from __future__ import annotations

import argparse
import os
import sys
from unittest.mock import patch

# Parse --gpu before importing torch so CUDA_VISIBLE_DEVICES takes effect.
_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
        break
    if arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

# Make the repository root take precedence over any installed editable egg so
# local changes are benchmarked, not an unrelated checkout.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.utils import moe_dispatch as _moe_dispatch  # noqa: E402
from gptqmodel.utils.moe_dispatch import (  # noqa: E402
    GROUPED_DISPATCH_FLAG,
    linear_loop_experts_forward,
)


MODEL_SHAPES = {
    "laguna-s-2.1": (3072, 12288),
    "qwen3.5-27b": (5120, 17408),
    "llama-like": (4096, 11008),
    "kimi-k3": (1792, 1792),
}


def _make_marlin_linear(
    in_features: int,
    out_features: int,
    group_size: int = 128,
) -> MarlinLinear:
    m = MarlinLinear(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
        dtype=torch.bfloat16,
    )
    device = torch.device("cuda")
    m.qweight.data = torch.randint(
        0, 2**31, m.qweight.shape, dtype=torch.int32, device=device
    )
    m.scales.data = torch.rand(m.scales.shape, dtype=torch.bfloat16, device=device) * 0.4 + 0.2
    m.qzeros.data = torch.randint(
        0, 2**31, m.qzeros.shape, dtype=torch.int32, device=device
    )
    m.g_idx.data = torch.arange(in_features, dtype=torch.int32, device=device) // group_size
    m = m.to(device)
    m.post_init()
    return m.eval()


class _Expert(nn.Module):
    pass


class _Experts(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int,
        group_size: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        for i in range(num_experts):
            e = _Expert()
            e.gate_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size)
            e.up_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size)
            e.down_proj = _make_marlin_linear(intermediate_dim, hidden_dim, group_size)
            self.add_module(str(i), e)
        self.act_fn = F.silu


def _make_inputs(
    hidden_dim: int,
    batch: int,
    seq: int,
    top_k: int,
    num_experts: int,
    dtype: torch.dtype,
):
    total = batch * seq
    hidden_states = torch.randn(batch, seq, hidden_dim, device="cuda", dtype=dtype)
    topk_idx = torch.randint(0, num_experts, (total, top_k), device="cuda")
    topk_w = torch.rand(total, top_k, device="cuda", dtype=dtype)
    topk_w = topk_w / topk_w.sum(dim=1, keepdim=True)
    return hidden_states, topk_idx, topk_w


def _time_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    repeats: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def _benchmark(
    experts: nn.Module,
    hidden_dim: int,
    batch: int,
    seq: int,
    top_k: int,
    repeats: int,
) -> tuple[float, float]:
    hidden_states, topk_idx, topk_w = _make_inputs(
        hidden_dim, batch, seq, top_k, experts.num_experts, torch.bfloat16
    )
    ms = _time_forward(experts, hidden_states, topk_idx, topk_w, repeats)
    total_tokens = batch * seq * top_k
    tokens_per_sec = total_tokens / (ms / 1000.0)
    return ms, tokens_per_sec


def _run_backend(
    experts: nn.Module,
    hidden_dim: int,
    cases: list,
    backend: str,
    repeats: int,
) -> list[tuple[float, float]]:
    # Always reset cached dispatch decisions so each backend is actually exercised.
    experts._grouped_mm_ok = False
    experts._grouped_mm_failed = False
    for attr in ("_moe_dispatch_backend", "_marlin_moe_workspace"):
        if hasattr(experts, attr):
            delattr(experts, attr)

    if backend == "per_expert":
        # No grouped dispatch flag: falls through to the defuser per-expert loop.
        if hasattr(experts, GROUPED_DISPATCH_FLAG):
            delattr(experts, GROUPED_DISPATCH_FLAG)
    elif backend == "grouped_mm":
        setattr(experts, GROUPED_DISPATCH_FLAG, True)
        # Force the legacy dense grouped_mm path for Marlin by disabling the mega-kernel.
        patcher = patch.object(
            _moe_dispatch,
            "_batched_marlin_moe_supported",
            return_value=False,
        )
        patcher.start()
    else:  # marlin_moe
        setattr(experts, GROUPED_DISPATCH_FLAG, True)
        patcher = None

    results = []
    try:
        for mode, batch, seq in cases:
            reps = max(repeats, 10 if (batch * seq) <= 4 else 5)
            if mode == "prefill":
                reps = max(repeats // 5, 3)
            ms, tokps = _benchmark(experts, hidden_dim, batch, seq, experts.top_k, reps)
            results.append((ms, tokps))
    finally:
        if backend == "grouped_mm":
            patcher.stop()  # type: ignore[union-attr]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="laguna-s-2.1", choices=list(MODEL_SHAPES.keys()))
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--backends",
        type=str,
        default="per_expert,grouped_mm,marlin_moe",
        help="Comma-separated list from: per_expert, grouped_mm, marlin_moe",
    )
    args = parser.parse_args()

    hidden_dim, intermediate_dim = MODEL_SHAPES[args.model]
    top_k = args.top_k
    print(
        f"Benchmarking {args.model} (Marlin MoE): hidden={hidden_dim}, "
        f"intermediate={intermediate_dim}, experts={args.num_experts}, top_k={top_k}"
    )
    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    decode_batches = [1, 2, 4, 8, 16, 32]
    prefill_shapes = [(1, 128), (1, 1024), (1, 4096), (16, 128)]
    cases = [("decode", b, 1) for b in decode_batches] + [
        ("prefill", b, s) for b, s in prefill_shapes
    ]

    experts = _Experts(
        hidden_dim, intermediate_dim, args.num_experts, args.group_size
    ).cuda()
    experts.top_k = top_k
    fused = install_fused_gate_up(experts)
    print(f"\nFused {fused} expert gate/up groups")

    backends = [b.strip() for b in args.backends.split(",")]
    all_results: dict[str, list[tuple[float, float]]] = {}
    for backend in backends:
        print(f"\n--- backend: {backend} ---")
        results = _run_backend(experts, hidden_dim, cases, backend, args.repeats)
        all_results[backend] = results
        for (mode, batch, seq), (ms, tokps) in zip(cases, results):
            print(f"{mode:<8} batch={batch:2d} seq={seq:4d}  ms={ms:.3f}  tok/s={tokps:,.1f}")

    print("\n=== Summary table (ms / tok/s / speedup vs per_expert) ===")
    header = f"{'mode':<8} {'batch':>5} {'seq':>5}"
    for backend in backends:
        header += f" {backend:>12}_ms {backend:>14}_tok/s"
    if "per_expert" in backends:
        for backend in backends:
            if backend == "per_expert":
                continue
            header += f" {backend:>10}x"
    print(header)

    baseline = all_results.get("per_expert")
    for i, (mode, batch, seq) in enumerate(cases):
        row = f"{mode:<8} {batch:>5} {seq:>5}"
        for backend in backends:
            ms, tokps = all_results[backend][i]
            row += f" {ms:>12.3f} {tokps:>14,.1f}"
        if baseline is not None:
            for backend in backends:
                if backend == "per_expert":
                    continue
                ms_base, _ = baseline[i]
                ms_backend, _ = all_results[backend][i]
                row += f" {ms_base / ms_backend:>10.3f}"
        print(row)


if __name__ == "__main__":
    main()
