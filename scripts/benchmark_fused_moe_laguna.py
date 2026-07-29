#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fused gate/up MoE expert sub-group fusion.

Targets Laguna-S-2.1, Qwen3.5-27B and Llama-like MoE shapes. Compares the
grouped GEMM dispatch (`linear_loop_experts_forward`) before and after
fusing each expert's gate_proj/up_proj into a single quantized group.

The `--gpu` flag must be applied before any CUDA context is created. This is
handled at the top of the script by parsing ``sys.argv`` for ``--gpu`` and
setting ``CUDA_VISIBLE_DEVICES`` before the torch/gptqmodel imports below.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Parse --gpu before importing torch so CUDA_VISIBLE_DEVICES takes effect.
# `--gpu N` or `--gpu=N` is supported.
_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
        break
    if arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear  # noqa: E402
from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward  # noqa: E402


# Target hidden/intermediate sizes for the three reference models.
MODEL_SHAPES = {
    "laguna-s-2.1": (3072, 12288),
    "qwen3.5-27b": (5120, 17408),
    "llama-like": (4096, 11008),
}


def _make_tritonv2_linear(in_features: int, out_features: int, bits: int = 4, group_size: int = 128) -> TritonV2Linear:
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


class _Expert(nn.Module):
    pass


class _Experts(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        for i in range(num_experts):
            e = _Expert()
            e.gate_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim)
            e.up_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim)
            e.down_proj = _make_tritonv2_linear(intermediate_dim, hidden_dim)
            self.add_module(str(i), e)
        self.act_fn = F.silu


def _make_inputs(hidden_dim: int, batch: int, seq: int, top_k: int, num_experts: int, dtype: torch.dtype):
    total = batch * seq
    hidden_states = torch.randn(batch, seq, hidden_dim, device="cuda", dtype=dtype)
    topk_idx = torch.randint(0, num_experts, (total, top_k), device="cuda")
    topk_w = torch.rand(total, top_k, device="cuda", dtype=dtype)
    topk_w = topk_w / topk_w.sum(dim=1, keepdim=True)
    return hidden_states, topk_idx, topk_w


def _time_forward(experts: nn.Module, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(3):
        _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def _benchmark(experts: nn.Module, hidden_dim: int, batch: int, seq: int, top_k: int, repeats: int) -> tuple[float, float]:
    hidden_states, topk_idx, topk_w = _make_inputs(hidden_dim, batch, seq, top_k, experts.num_experts, torch.bfloat16)
    ms = _time_forward(experts, hidden_states, topk_idx, topk_w, repeats)
    total_tokens = batch * seq * top_k
    tokens_per_sec = total_tokens / (ms / 1000.0)
    return ms, tokens_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="laguna-s-2.1", choices=list(MODEL_SHAPES.keys()))
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    hidden_dim, intermediate_dim = MODEL_SHAPES[args.model]
    print(f"Benchmarking {args.model}: hidden={hidden_dim}, intermediate={intermediate_dim}, experts={args.num_experts}, top_k={args.top_k}")
    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    decode_batches = [1, 2, 4, 8, 16, 32]
    prefill_shapes = [(1, 128), (1, 1024), (1, 4096), (16, 128)]

    cases = [("decode", b, 1) for b in decode_batches] + [("prefill", b, s) for b, s in prefill_shapes]

    experts = _Experts(hidden_dim, intermediate_dim, args.num_experts).cuda()
    setattr(experts, GROUPED_DISPATCH_FLAG, True)

    before = []
    print("\n--- unfused grouped dispatch ---")
    for mode, batch, seq in cases:
        repeats = max(args.repeats, 10 if (batch * seq) <= 4 else 5)
        if mode == "prefill":
            repeats = max(args.repeats // 5, 3)
        ms, tokps = _benchmark(experts, hidden_dim, batch, seq, args.top_k, repeats)
        before.append((ms, tokps))
        print(f"{mode:<8} batch={batch:2d} seq={seq:4d}  ms={ms:.3f}  tok/s={tokps:,.1f}")

    fused = install_fused_gate_up(experts)
    print(f"\nFused {fused} expert gate/up groups")

    after = []
    print("\n--- fused gate/up grouped dispatch ---")
    for mode, batch, seq in cases:
        repeats = max(args.repeats, 10 if (batch * seq) <= 4 else 5)
        if mode == "prefill":
            repeats = max(args.repeats // 5, 3)
        ms, tokps = _benchmark(experts, hidden_dim, batch, seq, args.top_k, repeats)
        after.append((ms, tokps))
        print(f"{mode:<8} batch={batch:2d} seq={seq:4d}  ms={ms:.3f}  tok/s={tokps:,.1f}")

    print("\n=== Summary table ===")
    print(f"{'mode':<8} {'batch':>5} {'seq':>5} {'ms_unfused':>10} {'tok/s_unfused':>14} {'ms_fused':>10} {'tok/s_fused':>12} {'speedup':>8}")
    for (mode, batch, seq), (ms_u, tok_u), (ms_f, tok_f) in zip(cases, before, after):
        print(f"{mode:<8} {batch:>5} {seq:>5} {ms_u:>10.3f} {tok_u:>14,.1f} {ms_f:>10.3f} {tok_f:>12,.1f} {ms_u / ms_f:>8.3f}")


if __name__ == "__main__":
    main()
