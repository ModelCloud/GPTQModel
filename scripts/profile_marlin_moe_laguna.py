#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile the synthetic Laguna MoE dispatch backends."""

from __future__ import annotations

import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.utils.moe_dispatch import (  # noqa: E402
    GROUPED_DISPATCH_FLAG,
    linear_loop_experts_forward,
)


def _make_marlin_linear(in_features: int, out_features: int, group_size: int = 64) -> MarlinLinear:
    m = MarlinLinear(
        in_features=in_features,
        out_features=out_features,
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    device = torch.device("cuda")
    m.qweight.data = torch.randint(0, 2**31, m.qweight.shape, dtype=torch.int32, device=device)
    m.scales.data = torch.rand(m.scales.shape, dtype=torch.bfloat16, device=device) * 0.4 + 0.2
    m.qzeros.data = torch.randint(0, 2**31, m.qzeros.shape, dtype=torch.int32, device=device)
    m.g_idx.data = torch.arange(in_features, dtype=torch.int32, device=device) // group_size
    m = m.to(device)
    m.post_init()
    return m.eval()


class _Expert(nn.Module):
    pass


class _Experts(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int, group_size: int = 64):
        super().__init__()
        self.num_experts = num_experts
        for i in range(num_experts):
            e = _Expert()
            e.gate_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size)
            e.up_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size)
            e.down_proj = _make_marlin_linear(intermediate_dim, hidden_dim, group_size)
            self.add_module(str(i), e)
        self.act_fn = F.silu

    @property
    def top_k(self) -> int:
        return 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--backend", type=str, default="marlin", choices=["marlin", "marlin_moe", "grouped_mm"])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    hidden_dim, intermediate_dim = 3072, 1024
    experts = _Experts(hidden_dim, intermediate_dim, args.num_experts, args.group_size).cuda()
    experts._grouped_mm_ok = False
    experts._grouped_mm_failed = False
    for attr in ("_moe_dispatch_backend", "_marlin_moe_workspace"):
        if hasattr(experts, attr):
            delattr(experts, attr)
    setattr(experts, GROUPED_DISPATCH_FLAG, True)

    if args.backend == "grouped_mm":
        from unittest.mock import patch
        import gptqmodel.utils.moe_dispatch as _moe_dispatch
        patch.object(_moe_dispatch, "_grouped_mm_available", return_value=True).start()
        patch.object(_moe_dispatch, "_batched_marlin_moe_supported", return_value=False).start()
    elif args.backend == "marlin_moe":
        from unittest.mock import patch
        import gptqmodel.utils.moe_dispatch as _moe_dispatch
        patch.object(_moe_dispatch, "_grouped_mm_available", return_value=False).start()

    total = args.batch * args.seq
    hidden_states = torch.randn(args.batch, args.seq, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, args.num_experts, (total, args.top_k), device="cuda")
    topk_w = torch.rand(total, args.top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=1, keepdim=True)

    for _ in range(10):
        _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    torch.cuda.synchronize()

    backend = getattr(experts, "_moe_dispatch_backend", None)
    print(f"Dispatch backend resolved to: {backend}")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.repeats):
            _ = linear_loop_experts_forward(experts, hidden_states, topk_idx, topk_w)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace("/tmp/profile_marlin_moe_laguna.json")
    print("Trace written to /tmp/profile_marlin_moe_laguna.json")


if __name__ == "__main__":
    main()
