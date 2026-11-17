# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4472: MACaQ balance pre-scaling must stay finite for one-output Linears.

``torch.std(dim=0)`` is the unbiased (``correction=1``) estimate, so a weight
with a single output row divides by zero and produces NaN.  ``clamp_min`` does
not repair NaN, so the NaN pre-scale reached the packed E8M0 exponent (0xFF)
and the decoded weights.  The fix uses the population standard deviation
(exactly zero for one sample) plus the documented positive floor, and rejects
non-finite statistics before publishing artifacts.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization import macaq


def _reference_pre_scale(weight: torch.Tensor, alpha: float, n_tokens: float) -> torch.Tensor:
    """Independent Torch oracle for the documented D1 pre-scale policy."""
    weight = weight.float()
    if weight.shape[0] == 1:
        w_std = weight.std(dim=0, correction=0)
    else:
        w_std = weight.std(dim=0)
    w_std = w_std.clamp_min(1e-8)
    act_rms = (torch.full((weight.shape[1],), 1.0 / n_tokens)).sqrt()
    return (w_std.pow(alpha) / act_rms.pow(1.0 - alpha)).clamp(1e-4, 1e4)


def _assert_finite_artifacts(result: macaq.MacaqQuantResult, width: int) -> None:
    assert result.pre_scale is not None
    assert torch.isfinite(result.pre_scale).all()

    scale_bytes = result.W_scales.view(torch.uint8)
    assert scale_bytes.numel() == result.out_features * (width // macaq.MXFP4_BLOCK)
    # E8M0 0xFF is the format's NaN code, not a usable exponent.
    assert (scale_bytes != 255).all()

    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, width)
    assert decoded.shape == (result.out_features, width)
    assert torch.isfinite(decoded).all()
    assert torch.isfinite(result.dequantized_weight).all()


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("alpha", [0.5, 0.75])
def test_single_output_balance_pre_scale_is_finite(width: int, alpha: float):
    weight = torch.arange(1, width + 1, dtype=torch.float32).reshape(1, width) / width
    hessian = torch.eye(width)

    pre_scale = macaq.balance_pre_scale(weight, hessian, width, alpha=alpha)
    assert torch.isfinite(pre_scale).all()
    torch.testing.assert_close(
        pre_scale, _reference_pre_scale(weight, alpha, width), rtol=1e-6, atol=0.0
    )

    result = macaq.macaq_quantize_weight(
        weight,
        hessian,
        width,
        do_flip=False,
        pre_mode="balance",
        pre_alpha=alpha,
        return_dequantized=True,
    )
    _assert_finite_artifacts(result, width)


def test_multi_output_balance_control_is_finite():
    width = 32
    weight = torch.arange(1, 2 * width + 1, dtype=torch.float32).reshape(2, width) / width
    hessian = torch.eye(width)

    pre_scale = macaq.balance_pre_scale(weight, hessian, width, alpha=0.5)
    torch.testing.assert_close(
        pre_scale, _reference_pre_scale(weight, 0.5, width), rtol=1e-6, atol=0.0
    )

    result = macaq.macaq_quantize_weight(
        weight,
        hessian,
        width,
        do_flip=False,
        pre_mode="balance",
        pre_alpha=0.5,
        return_dequantized=True,
    )
    _assert_finite_artifacts(result, width)


def test_zero_variance_columns_use_the_positive_floor():
    width = 32
    weight = torch.ones(4, width, dtype=torch.float32)
    hessian = torch.eye(width)

    pre_scale = macaq.balance_pre_scale(weight, hessian, width, alpha=0.5)
    torch.testing.assert_close(
        pre_scale, _reference_pre_scale(weight, 0.5, width), rtol=1e-6, atol=0.0
    )
    assert torch.all(pre_scale > 0)

    result = macaq.macaq_quantize_weight(
        weight,
        hessian,
        width,
        do_flip=False,
        pre_mode="balance",
        pre_alpha=0.5,
        return_dequantized=True,
    )
    _assert_finite_artifacts(result, width)
