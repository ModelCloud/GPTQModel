# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from defuser.modeling.moe_experts_interface import _ExpertContainer


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
