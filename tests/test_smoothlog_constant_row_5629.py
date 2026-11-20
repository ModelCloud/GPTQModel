# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""SmoothLog must not erase a constant row for a valid small ``mu`` (#5629).

The log-space clip is ``log1p(mu * |w|) / log1p(mu)`` and its inverse was
written as ``(exp(t * log1p(mu)) - 1) / mu``. For a constant row every
percentile is the row magnitude, so the inverse must return that magnitude for
any positive ``mu``. With ``mu = 1e-8`` the FP32 ``exp`` rounds to exactly 1,
the subtraction yields 0, and the row is clamped to zero: GGUF then packs an
all-zero projection. ``expm1`` keeps the small log-space threshold.
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.quantization.config import Fallback, SmoothLog
from gptqmodel.quantization.fallback_smooth import smooth_block


MU = 1e-8


def _log_reference_threshold(block: torch.Tensor, percentile: float, mu: float) -> torch.Tensor:
    """The documented inverse, evaluated in float64 as a precision oracle."""

    log_mu = math.log1p(mu)
    log_vals = torch.log1p(block.double().abs() * mu) / log_mu
    threshold = torch.quantile(log_vals, percentile / 100.0, dim=1, keepdim=True)
    return (torch.expm1(threshold * log_mu) / mu).float()


@pytest.mark.parametrize("magnitude", [1.0, -1.0])
def test_smooth_block_keeps_a_constant_row_for_small_mu(magnitude):
    block = torch.full((2, 256), magnitude)
    fallback = Fallback(smooth=SmoothLog(percentile=99.0, mu=MU))
    smoothed, scale = smooth_block(block, fallback, group_size=256)
    assert scale is None
    assert torch.equal(smoothed, block)


def test_smooth_block_does_not_collapse_a_varied_row_for_small_mu():
    # A 256-wide block clears SmoothLog's default 128 group-size threshold.
    block = torch.cat(
        [torch.full((1, 128), 1.0), torch.linspace(0.0, 1.0, 128).reshape(1, 128)],
        dim=1,
    )
    fallback = Fallback(smooth=SmoothLog(percentile=50.0, mu=MU))
    smoothed, _ = smooth_block(block, fallback, group_size=256)
    threshold = _log_reference_threshold(block, 50.0, MU)
    assert threshold.item() >= block[block > 0].min().item()
    torch.testing.assert_close(
        smoothed,
        torch.clamp(block, -threshold, threshold),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize("percentile", [25.0, 50.0, 75.0])
def test_smooth_block_uses_the_documented_inverse_for_normal_mu(percentile):
    block = torch.linspace(-4.0, 4.0, 128).reshape(1, 128)
    fallback = Fallback(smooth=SmoothLog(percentile=percentile, mu=8.0))
    smoothed, _ = smooth_block(block, fallback, group_size=128)
    threshold = _log_reference_threshold(block, percentile, 8.0)
    torch.testing.assert_close(
        smoothed,
        torch.clamp(block, -threshold, threshold),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gguf_packs_a_constant_unit_projection_with_small_mu():
    source = torch.nn.Linear(256, 1, bias=False)
    source.weight.data.fill_(1)
    x = torch.zeros(1, 256)
    x[0, 0] = 32768.0
    module = GGUFTorchLinear(
        bits="q8_k",
        in_features=256,
        out_features=1,
        group_size=-1,
        sym=True,
        desc_act=False,
    )
    module.pack_original(source, None, None, smooth=SmoothLog(mu=MU))
    with torch.no_grad():
        assert module(x).item() == 32768.0
