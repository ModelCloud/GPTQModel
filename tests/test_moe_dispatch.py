# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from defuser.modeling.moe_experts_interface import _ExpertContainer
from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up, install_fused_qkv
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


def _make_tritonv2_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 128,
) -> TritonV2Linear:
    """Build a packed TritonV2Linear with random quantized weights for testing."""
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
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> MarlinLinear:
    """Build a MarlinLinear with random packed weights and call post_init."""
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


def _gated_experts_fixture(
    num_experts: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build a fixture with split gate/up/down projections."""
    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=True, dtype=dtype)
        container.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=True, dtype=dtype)
        container.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=True, dtype=dtype)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.to(device)


def test_grouped_moe_dispatch_runs_on_cpu() -> None:
    """The dispatch must fall back to the per-expert loop on CPU."""
    from gptqmodel.utils.moe_dispatch import linear_loop_experts_forward

    module = _gated_experts_fixture(num_experts=2, hidden_dim=16, intermediate_dim=8, dtype=torch.float32)
    hs = torch.randn(3, 16, dtype=torch.float32)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, 0]])
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
    out = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    assert out.shape == (3, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_grouped_moe_dispatch_matches_per_expert_loop() -> None:
    """The grouped GEMM fast path must produce the same output as the per-expert loop."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = _gated_experts_fixture(num_experts=4, hidden_dim=16, intermediate_dim=8, device="cuda")
    setattr(module, GROUPED_DISPATCH_FLAG, True)
    hs = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda")
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16)

    grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_grouped_moe_dispatch_square_dims() -> None:
    """Square expert projections must not be silently transposed by the grouped path."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = _gated_experts_fixture(num_experts=4, hidden_dim=16, intermediate_dim=16, device="cuda")
    setattr(module, GROUPED_DISPATCH_FLAG, True)
    hs = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda")
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16)

    grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)


class _FakeQuantLinear(nn.Module):
    """Fake quantized linear that stores weight in [in_features, out_features] layout."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16))

    def dequantize_weight(self) -> torch.Tensor:
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class _FakeQuantLinearWithAdapter(_FakeQuantLinear):
    """Fake quantized linear whose forward adds an adapter term to the output."""

    def __init__(self, in_features: int, out_features: int, add: float = 0.5) -> None:
        super().__init__(in_features, out_features)
        self.adapter = lambda x, out: out + add  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.adapter(x, out)


def _fake_quant_gated_experts_fixture(
    num_experts: int = 4,
    hidden_dim: int = 16,
    intermediate_dim: int = 16,
    device: str | torch.device = "cpu",
) -> nn.Module:
    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _FakeQuantLinear(hidden_dim, intermediate_dim)
        container.up_proj = _FakeQuantLinear(hidden_dim, intermediate_dim)
        container.down_proj = _FakeQuantLinear(intermediate_dim, hidden_dim)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_grouped_moe_dispatch_fake_quant_square_orientation() -> None:
    """The orientation probe must correctly handle dequantize_weight that returns [in, out]."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = _fake_quant_gated_experts_fixture(num_experts=4, hidden_dim=16, intermediate_dim=16, device="cuda")
    setattr(module, GROUPED_DISPATCH_FLAG, True)
    hs = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda")
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16)

    grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_grouped_moe_dispatch_adapter_falls_back_to_per_expert() -> None:
    """Experts with an active adapter must not be silently replaced by raw dequant weights."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = nn.Module()
    module.num_experts = 2
    for i in range(2):
        container = _ExpertContainer()
        container.gate_proj = _FakeQuantLinearWithAdapter(16, 8)
        container.up_proj = _FakeQuantLinear(16, 8)
        container.down_proj = _FakeQuantLinear(8, 16)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    module = module.cuda()
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    hs = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, 0]], device="cuda")
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16)

    grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
    loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_register_linear_loop_experts_overrides_mapping() -> None:
    """register_linear_loop_experts installs the GPTQModel dispatch and respects the per-module flag."""
    from defuser.modeling.moe_experts_interface import register_linear_loop_experts as defuser_register
    from gptqmodel.utils.moe_dispatch import (
        GROUPED_DISPATCH_FLAG,
        linear_loop_experts_forward,
        register_linear_loop_experts,
    )

    # Defuser registers the per-expert fallback first.
    assert defuser_register() is True
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    original = ALL_EXPERTS_FUNCTIONS._global_mapping["linear_loop"]
    assert original is not linear_loop_experts_forward

    # GPTQModel overrides it with the grouped dispatch.
    assert register_linear_loop_experts() is True
    assert ALL_EXPERTS_FUNCTIONS._global_mapping["linear_loop"] is linear_loop_experts_forward

    hs = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, 0]], device="cuda")
    topk_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], device="cuda", dtype=torch.bfloat16)

    # Without the per-module flag the dispatch must fall back to the defuser loop.
    unflagged = _gated_experts_fixture(num_experts=2, hidden_dim=16, intermediate_dim=8, device="cuda")
    fallback = linear_loop_experts_forward(unflagged, hs, topk_idx, topk_w)
    expected = original(unflagged, hs, topk_idx, topk_w)
    torch.testing.assert_close(fallback, expected, atol=5e-3, rtol=1e-2)

    # With the flag set the grouped path is used.
    flagged = _gated_experts_fixture(num_experts=2, hidden_dim=16, intermediate_dim=8, device="cuda")
    setattr(flagged, GROUPED_DISPATCH_FLAG, True)
    grouped = linear_loop_experts_forward(flagged, hs, topk_idx, topk_w)
    loop = original(flagged, hs, topk_idx, topk_w)
    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)

    # Restore to avoid side effects in later tests.
    ALL_EXPERTS_FUNCTIONS._global_mapping["linear_loop"] = original


def _tritonv2_experts_fixture(
    num_experts: int = 4,
    hidden_dim: int = 512,
    intermediate_dim: int = 1024,
) -> nn.Module:
    """Build an MoE fixture where each expert uses packed TritonV2Linear layers."""
    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim)
        container.up_proj = _make_tritonv2_linear(hidden_dim, intermediate_dim)
        container.down_proj = _make_tritonv2_linear(intermediate_dim, hidden_dim)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    return module.cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
@pytest.mark.parametrize("num_experts,top_k", [(4, 2), (8, 4)])
def test_grouped_moe_dispatch_fused_gateup_matches_unfused(
    num_experts: int,
    top_k: int,
) -> None:
    """Fusing gate/up inside MoE experts must not change the grouped GEMM output."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = _tritonv2_experts_fixture(num_experts=num_experts, hidden_dim=512, intermediate_dim=1024)
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    hs = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (2 * 16, top_k), device="cuda")
    topk_w = torch.rand(2 * 16, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        expected = linear_loop_experts_forward(module, hs, topk_idx, topk_w)

    assert install_fused_gate_up(module) == num_experts

    with torch.inference_mode():
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
        loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, expected, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(loop, expected, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin dispatch requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin requires Ampere or newer",
)
@pytest.mark.parametrize("num_experts,top_k", [(4, 2), (8, 4)])
def test_grouped_moe_dispatch_marlin_fused_gateup_matches_unfused(
    num_experts: int,
    top_k: int,
) -> None:
    """Marlin-packed MoE experts with fused gate/up must match the per-expert loop."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        # dims must be multiples of 32 (Marlin tile requirement) and of 64 for gate/up fusion
        container.gate_proj = _make_marlin_linear(128, 64, group_size=64)
        container.up_proj = _make_marlin_linear(128, 64, group_size=64)
        container.down_proj = _make_marlin_linear(64, 128, group_size=64)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    module = module.cuda()

    hs = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (2 * 16, top_k), device="cuda")
    topk_w = torch.rand(2 * 16, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # The defuser per-expert loop is the reference: it executes the same packed
    # Marlin kernels as the fused-gate/up per-expert loop.
    with torch.inference_mode():
        expected = per_expert_forward(module, hs, topk_idx, topk_w)

    assert install_fused_gate_up(module) == num_experts
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    with torch.inference_mode():
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
        loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(loop, expected, atol=2.0, rtol=0.05)
    torch.testing.assert_close(grouped, expected, atol=2.0, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_grouped_moe_dispatch_kimi_k3_proxy() -> None:
    """Kimi-K3-like proxy shape: many experts, small latent dim, top-k routing."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    num_experts = 64
    hidden_dim = 1792
    intermediate_dim = 1792
    batch = 16
    seq = 1
    top_k = 8

    module = _tritonv2_experts_fixture(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    total_tokens = batch * seq
    hs = torch.randn(batch, seq, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (total_tokens, top_k), device="cuda")
    topk_w = torch.rand(total_tokens, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)
        loop = per_expert_forward(module, hs, topk_idx, topk_w)

    torch.testing.assert_close(grouped, loop, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm requires CUDA")
def test_fused_qkv_and_moe_gateup_coexist() -> None:
    """QKV fusion and MoE gate/up fusion must both install and not break each other."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(64, 32)
            self.k_proj = _make_tritonv2_linear(64, 32)
            self.v_proj = _make_tritonv2_linear(64, 32)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()
            self.moe = nn.Module()
            self.moe.num_experts = 4
            for i in range(4):
                container = _ExpertContainer()
                container.gate_proj = _make_tritonv2_linear(64, 32)
                container.up_proj = _make_tritonv2_linear(64, 32)
                container.down_proj = _make_tritonv2_linear(32, 64)
                self.moe.add_module(str(i), container)
            self.moe.act_fn = F.silu

    layer = _Layer().cuda()
    setattr(layer.moe, GROUPED_DISPATCH_FLAG, True)

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        q = layer.self_attn.q_proj(x)
        k = layer.self_attn.k_proj(x)
        v = layer.self_attn.v_proj(x)
    expected_qkv = torch.cat([q, k, v], dim=-1)

    hs = torch.randn(3, 16, 64, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, 4, (3 * 16, 2), device="cuda")
    topk_w = torch.rand(3 * 16, 2, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    expected_moe = linear_loop_experts_forward(layer.moe, hs, topk_idx, topk_w)

    assert install_fused_qkv(layer.self_attn) == 1
    assert install_fused_gate_up(layer.moe) == 4

    with torch.inference_mode():
        qf = layer.self_attn.q_proj(x)
        kf = layer.self_attn.k_proj(x)
        vf = layer.self_attn.v_proj(x)
    fused_qkv = torch.cat([qf, kf, vf], dim=-1)
    torch.testing.assert_close(fused_qkv, expected_qkv, atol=2.0, rtol=0.05)

    grouped_moe = linear_loop_experts_forward(layer.moe, hs, topk_idx, topk_w)
    loop_moe = per_expert_forward(layer.moe, hs, topk_idx, topk_w)
    torch.testing.assert_close(grouped_moe, expected_moe, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(loop_moe, expected_moe, atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin MoE requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin requires Ampere or newer",
)
@pytest.mark.parametrize("num_experts,top_k", [(4, 2), (8, 4)])
def test_grouped_moe_dispatch_marlin_moe_non_fused(
    num_experts: int,
    top_k: int,
    monkeypatch,
) -> None:
    """Batched/offset Marlin MoE mega-kernel matches per-expert loop without fused gate/up."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    monkeypatch.setenv("GPTQMODEL_MARLIN_MOE_BACKEND", "marlin_moe")

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_marlin_linear(128, 64, group_size=64)
        container.up_proj = _make_marlin_linear(128, 64, group_size=64)
        container.down_proj = _make_marlin_linear(64, 128, group_size=64)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    module = module.cuda()
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    hs = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (3, top_k), device="cuda")
    topk_w = torch.rand(3, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        expected = per_expert_forward(module, hs, topk_idx, topk_w)
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)

    assert getattr(module, "_moe_dispatch_backend", None) == "marlin_moe"
    torch.testing.assert_close(grouped, expected, atol=2.0, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin MoE requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin requires Ampere or newer",
)
def test_grouped_moe_dispatch_marlin_moe_laguna_like(monkeypatch) -> None:
    """Batched Marlin MoE must be accurate on Laguna-S-2.1-like fused gate/up shapes."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.nn_modules.fused_quant_linear import install_fused_gate_up
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    num_experts = 16
    hidden_dim = 1792
    intermediate_dim = 1792
    top_k = 6

    monkeypatch.setenv("GPTQMODEL_MARLIN_MOE_BACKEND", "marlin_moe")

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size=128)
        container.up_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size=128)
        container.down_proj = _make_marlin_linear(intermediate_dim, hidden_dim, group_size=128)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    module = module.cuda()

    assert install_fused_gate_up(module) == num_experts
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    hs = torch.randn(4, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (4, top_k), device="cuda")
    topk_w = torch.rand(4, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        expected = per_expert_forward(module, hs, topk_idx, topk_w)
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)

    assert getattr(module, "_moe_dispatch_backend", None) == "marlin_moe"
    torch.testing.assert_close(grouped, expected, atol=2.0, rtol=0.05)


def test_moe_scatter_cluster_output_respects_pair_id_order() -> None:
    """_moe_scatter_cluster_output must write each pair's result to its global pair id."""
    from gptqmodel.utils.moe_dispatch import _moe_scatter_cluster_output

    num_pairs = 12
    hidden = 4
    # Each row encodes its pair id in the first column.
    cluster_out = torch.arange(num_pairs, dtype=torch.float32).view(-1, 1).expand(-1, hidden)
    # Non-monotonic token-expert ids: the argsort below will permute pair ids.
    topk_ids = torch.tensor(
        [[0, 2], [3, 1], [0, 3], [2, 1], [1, 0], [3, 2]],
        dtype=torch.int64,
    )
    flat = topk_ids.flatten()
    sorted_indices = torch.argsort(flat, stable=True)
    sorted_token_ids = torch.full((16,), num_pairs, dtype=torch.int32)
    sorted_token_ids[:num_pairs] = sorted_indices
    in_cluster = torch.tensor([True, True, True, True], dtype=torch.bool)
    global_out = torch.full((num_pairs, hidden), -1.0)

    _moe_scatter_cluster_output(cluster_out, sorted_token_ids, in_cluster, topk_ids, global_out)

    expected = cluster_out[:num_pairs]
    torch.testing.assert_close(global_out, expected)


def test_active_cluster_builder_maps_sparse_global_expert_ids_to_compact_targets() -> None:
    """Shape clustering must not index an active-only target list with global expert ids."""
    from gptqmodel.utils.moe_dispatch import _build_active_clusters_for_proj

    weight_type = SimpleNamespace(id=4)
    active_global_ids = [3, 7]
    active_targets = [
        SimpleNamespace(
            qweight=torch.full((1, 2), 3, dtype=torch.int32),
            scales=torch.full((1, 1), 0.3),
            weight_type=weight_type,
            padded_in_features=16,
            padded_out_features=32,
            is_k_full=True,
        ),
        SimpleNamespace(
            qweight=torch.full((2, 2), 7, dtype=torch.int32),
            scales=torch.full((2, 1), 0.7),
            weight_type=weight_type,
            padded_in_features=16,
            padded_out_features=32,
            is_k_full=True,
        ),
    ]

    clusters = _build_active_clusters_for_proj(
        active_targets,
        active_global_ids,
        num_experts=8,
        device=torch.device("cpu"),
    )

    assert len(clusters) == 2
    for cluster, global_id, target in zip(clusters, active_global_ids, active_targets):
        assert cluster["global_ids"].tolist() == [global_id]
        assert cluster["in_cluster"].nonzero().flatten().tolist() == [global_id]
        assert cluster["membership"][global_id].item() == 0
        torch.testing.assert_close(cluster["qweight"][0], target.qweight)
        torch.testing.assert_close(cluster["scales"][0], target.scales)


@pytest.mark.parametrize("invalid_expert_id", [-1, 2])
def test_marlin_one_token_invalid_expert_ids_match_fallback(invalid_expert_id: int) -> None:
    """The Marlin decode path must not route sentinel ids to the first or last expert."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import _marlin_experts_project_one_token

    module = _gated_experts_fixture(num_experts=2, hidden_dim=4, intermediate_dim=3, dtype=torch.float32)
    # Avoid selecting the CUDA-only fused SiLU helper in this CPU semantic test.
    module.act_fn = torch.tanh
    hidden_states = torch.randn(1, 4)
    topk_idx = torch.tensor([[invalid_expert_id, 0]])
    topk_w = torch.tensor([[0.75, 0.25]])

    expected = per_expert_forward(module, hidden_states, topk_idx, topk_w)
    actual = _marlin_experts_project_one_token(module, hidden_states, topk_idx, topk_w)

    torch.testing.assert_close(actual, expected)


def test_grouped_moe_invalid_expert_ids_match_fallback(monkeypatch) -> None:
    """Grouped dispatch must preserve fallback semantics for both low and high sentinel ids."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import _grouped_mm_dequant_experts_forward

    module = _gated_experts_fixture(num_experts=2, hidden_dim=4, intermediate_dim=3, dtype=torch.float32)
    module.act_fn = torch.tanh
    module._moe_dispatch_backend = "grouped_mm"

    def grouped_mm(x: torch.Tensor, weights: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        outputs = []
        start = 0
        for weight, end in zip(weights, offs.tolist()):
            outputs.append(x[start:end] @ weight)
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(torch.nn.functional, "grouped_mm", grouped_mm, raising=False)
    hidden_states = torch.randn(2, 4)
    topk_idx = torch.tensor([[-1, 0], [1, 2]])
    topk_w = torch.tensor([[0.75, 0.25], [0.4, 0.6]])

    expected = per_expert_forward(module, hidden_states, topk_idx, topk_w)
    actual = _grouped_mm_dequant_experts_forward(module, hidden_states, topk_idx, topk_w)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("hidden_states", "topk_idx", "topk_w"),
    [
        (torch.empty(0, 4), torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2)),
        (torch.empty(2, 0, 4), torch.empty(2, 0, 2, dtype=torch.long), torch.empty(2, 0, 2)),
        (torch.randn(3, 4), torch.empty(3, 0, dtype=torch.long), torch.empty(3, 0)),
    ],
)
def test_batched_marlin_moe_empty_routes_return_zero_output(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_w: torch.Tensor,
) -> None:
    """Empty token and empty top-k batches must match the generic MoE fallback shape contract."""
    from gptqmodel.utils.moe_dispatch import _batched_marlin_moe_forward

    module = SimpleNamespace(num_experts=2)
    actual = _batched_marlin_moe_forward(module, hidden_states, topk_idx, topk_w)

    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(hidden_states))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin dispatch requires CUDA")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin requires Ampere or newer",
)
def test_marlin_experts_project_one_token() -> None:
    """The M=1 Marlin decode fast path must match the per-expert loop."""
    from defuser.modeling.moe_experts_interface import linear_loop_experts_forward as per_expert_forward
    from gptqmodel.utils.moe_dispatch import GROUPED_DISPATCH_FLAG, linear_loop_experts_forward

    num_experts = 4
    hidden_dim = 128
    intermediate_dim = 64
    top_k = 2

    module = nn.Module()
    module.num_experts = num_experts
    for i in range(num_experts):
        container = _ExpertContainer()
        container.gate_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size=64)
        container.up_proj = _make_marlin_linear(hidden_dim, intermediate_dim, group_size=64)
        container.down_proj = _make_marlin_linear(intermediate_dim, hidden_dim, group_size=64)
        module.add_module(str(i), container)
    module.act_fn = F.silu
    module = module.cuda()

    assert install_fused_gate_up(module) == num_experts
    setattr(module, GROUPED_DISPATCH_FLAG, True)

    hs = torch.randn(1, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_idx = torch.randint(0, num_experts, (1, top_k), device="cuda")
    topk_w = torch.rand(1, top_k, device="cuda", dtype=torch.bfloat16)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        expected = per_expert_forward(module, hs, topk_idx, topk_w)
        grouped = linear_loop_experts_forward(module, hs, topk_idx, topk_w)

    assert getattr(module, "_moe_dispatch_backend", None) == "marlin"
    torch.testing.assert_close(grouped, expected, atol=2.0, rtol=0.05)
