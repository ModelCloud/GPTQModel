# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""QQQ_TORCH must not overflow an FP32 intermediate for a finite result (#5608).

The torch forward applied the per-token activation scale and the per-channel
weight scale one after the other in FP32. A BF16 identity projection whose
exact result is ``2**124`` first produced ``2**128``: representable in neither
FP32 nor the output dtype, so the visible result was ``inf`` even though the
small channel scale would have brought it back. The two scales are now applied
through an overflow-safe product that keeps the historical rounding whenever
the historical intermediate is finite.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import scaled_accumulation
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear


IN_FEATURES = 128
OUT_FEATURES = 128
EXTREME = 2.0**124


def _identity_projection() -> tuple[torch.nn.Linear, QQQTorchLinear]:
    """The issue fixture: one exact input/output pair with channel scale 1/16."""

    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[:, 0] = 1
    module = QQQTorchLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
    )
    module.pack(linear, torch.ones(IN_FEATURES, 1))
    return linear, module


def _extreme_input(device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, IN_FEATURES, dtype=torch.bfloat16, device=device)
    x[:, 0] = EXTREME
    return x


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward_keeps_the_finite_identity_projection(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    target = torch.device(device)
    linear, module = _identity_projection()
    linear = linear.to(target)
    module = module.to(target)
    x = _extreme_input(target)
    with torch.no_grad():
        expected = linear(x)
        actual = module(x)
    assert torch.isfinite(actual).all()
    assert actual[0, 0].item() == EXTREME
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_scaled_accumulation_preserves_finite_rounding():
    """Every representable element keeps the historical bit pattern."""

    torch.manual_seed(5608)
    accum = torch.randint(-(2**20), 2**20, (9, 17)).float()
    s1 = torch.rand(9, 1) * 1e-3 + 1e-6
    s_channel = torch.rand(1, 17) * 1e-3 + 1e-6
    historical = (accum * s1) * s_channel
    assert torch.isfinite(historical).all()
    assert torch.equal(scaled_accumulation(accum, s1, s_channel), historical)


def test_scaled_accumulation_recovers_an_overflowing_intermediate():
    accum = torch.tensor([[2032.0]])  # 127 * 16
    s1 = torch.tensor([[2.0**124 / 127]])
    s_channel = torch.tensor([[1.0 / 16.0]])
    assert ((accum * s1) * s_channel).isinf().all()
    recovered = scaled_accumulation(accum, s1, s_channel)
    assert recovered.item() == EXTREME


def test_scaled_accumulation_keeps_the_historical_order_when_reversing_overflows():
    """The fix must not blindly reverse: this pair is only finite in order."""

    accum = torch.tensor([[2.0**20]])
    s1 = torch.tensor([[2.0**-100]])
    s_channel = torch.tensor([[2.0**120]])
    historical = (accum * s1) * s_channel
    assert historical.isinf().item() is False
    assert ((accum * s_channel) * s1).isinf().all()
    assert torch.equal(scaled_accumulation(accum, s1, s_channel), historical)


def test_scaled_accumulation_reports_a_genuine_overflow():
    """When both associations overflow, the exact product is not FP32."""

    accum = torch.tensor([[1.0]])
    s1 = torch.tensor([[2.0**100]])
    s_channel = torch.tensor([[2.0**100]])
    assert scaled_accumulation(accum, s1, s_channel).isinf().all()
