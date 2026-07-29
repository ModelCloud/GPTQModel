#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile MoE dispatch on Kimi-K3-like proxy shapes.

Compares:
- dense `nn.Linear` experts with grouped_mm dispatch (current best TritonV2/dense path)
- Marlin-packed experts with per-expert kernel launches (current Marlin path)
- Marlin-packed experts with fused gate/up and per-expert launches

The goal is to quantify how much of the MoE forward time is Python/kernel launch
overhead vs actual GEMM work, so we can decide whether a batched/offset Marlin
kernel is justified.
"""

from __future__ import annotations

import argparse
import os
import sys

# Parse --gpu before any CUDA context is created.
_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
        break
    if arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _gated_experts_fixture(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
) -> nn.Module:
    from defuser.modeling.moe_experts_interface import _ExpertContainer

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        container.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=dtype)
        container.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=dtype)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.cuda()


def _tritonv2_experts_fixture(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    bits: int = 4,
    group_size: int = 128,
) -> nn.Module:
    from defuser.modeling.moe_experts_interface import _ExpertContainer

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim, bits, group_size)
        container.up_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim, bits, group_size)
        container.down_proj = _make_tritonv2_linear(intermediate_dim, hidden_dim, bits, group_size)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.cuda()


def _marlin_experts_fixture(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    bits: int = 4,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    from defuser.modeling.moe_experts_interface import _ExpertContainer

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_marlin_linear(hidden_dim, intermediate_dim, bits, group_size, dtype)
        container.up_proj = _make_marlin_linear(hidden_dim, intermediate_dim, bits, group_size, dtype)
        container.down_proj = _make_marlin_linear(intermediate_dim, hidden_dim, bits, group_size, dtype)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.cuda()


def _make_tritonv2_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 128,
):
    import math

    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

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


def _make_marlin_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    m = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
        dtype=dtype,
    )
    device = torch.device("cuda")
    m.qweight.data = torch.randint(
        0, 2**31, m.qweight.shape, dtype=torch.int32, device=device
    )
    m.scales.data = torch.rand(m.scales.shape, dtype=dtype, device=device) * 0.4 + 0.2
    m.qzeros.data = torch.randint(
        0, 2**31, m.qzeros.shape, dtype=torch.int32, device=device
    )
    m.g_idx.data = torch.arange(in_features, dtype=torch.int32, device=device) // group_size
    m = m.to(device)
    m.post_init()
    return m.eval()


def _run_experts(
    module: nn.Module,
    hs: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
    repeats: int,
    warmup: int,
) -> tuple[list[float], torch.Tensor]:
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    setattr(module, GROUPED_DISPATCH_FLAG, True)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.inference_mode():
        for _ in range(repeats):
            start.record()
            out = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return times, out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--intermediate", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=1)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Proxy Kimi-K3 MoE: hidden={args.hidden}, intermediate={args.intermediate}, "
          f"num_experts={args.num_experts}, top_k={args.top_k}, batch={args.batch}, seq={args.seq}")

    total_tokens = args.batch * args.seq
    hs = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
    topk_idx = torch.randint(0, args.num_experts, (total_tokens, args.top_k), device=device)
    topk_w = torch.rand(total_tokens, args.top_k, device=device, dtype=dtype)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # Dense baseline (grouped_mm)
    dense_mod = _gated_experts_fixture(args.num_experts, args.hidden, args.intermediate, dtype)
    dense_times, _ = _run_experts(dense_mod, hs, topk_idx, topk_w, args.repeats, args.warmup)

    # TritonV2 packed baseline (grouped_mm with dequant)
    triton_mod = _tritonv2_experts_fixture(args.num_experts, args.hidden, args.intermediate)
    triton_times, _ = _run_experts(triton_mod, hs, topk_idx, topk_w, args.repeats, args.warmup)

    # Marlin packed per-expert launch
    marlin_mod = _marlin_experts_fixture(args.num_experts, args.hidden, args.intermediate, dtype=dtype)
    marlin_times, _ = _run_experts(marlin_mod, hs, topk_idx, topk_w, args.repeats, args.warmup)

    # Marlin with fused gate/up per-expert launch
    from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up
    fused_count = install_fused_gate_up(marlin_mod)
    fused_times, _ = _run_experts(marlin_mod, hs, topk_idx, topk_w, args.repeats, args.warmup)

    print("\n=== Latency (ms) ===")
    print(f"dense nn.Linear:      {_mean(dense_times):.3f} (min {min(dense_times):.3f}, max {max(dense_times):.3f})")
    print(f"TritonV2 packed:      {_mean(triton_times):.3f} (min {min(triton_times):.3f}, max {max(triton_times):.3f})")
    print(f"Marlin per-expert:    {_mean(marlin_times):.3f} (min {min(marlin_times):.3f}, max {max(marlin_times):.3f})")
    print(f"Marlin fused gate/up: {_mean(fused_times):.3f} (min {min(fused_times):.3f}, max {max(fused_times):.3f})")
    print(f"fused gate/up groups installed: {fused_count}")

    print("\n=== Relative to dense nn.Linear ===")
    dense_mean = _mean(dense_times)
    print(f"TritonV2 packed:      {_mean(triton_times) / dense_mean:.2f}x")
    print(f"Marlin per-expert:    {_mean(marlin_times) / dense_mean:.2f}x")
    print(f"Marlin fused gate/up: {_mean(fused_times) / dense_mean:.2f}x")

    print("\nActive experts:", torch.unique(topk_idx).numel())


if __name__ == "__main__":
    main()
