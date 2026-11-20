# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Symmetric fallback quantization must keep the bounds each strategy selects.

#5602: ``_affine_quantize`` remeasured the raw block under ``sym=True``, so
STDCLIP (and MEAN/MEDIAN) silently discarded the chosen ``lo, hi`` and fitted
the outlier it was asked to clip. A symmetric checkpoint stores the fixed zero
point ``(maxq + 1) // 2``, so the fix keeps that midpoint but fits the smallest
zero-centered interval that contains the *selected* range:
``[-max(|lo|, |hi|), max(|lo|, |hi|)]``.

The references below re-derive each strategy's bounds and grid independently,
so they do not merely compare the implementation with itself.
"""

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    GPTQConfig,
    QQQConfig,
)
from gptqmodel.quantization.fallback_smooth import symmetric_range_from_bounds
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


BITS = 4
MAXQ = 2**BITS - 1
ZERO = (MAXQ + 1) // 2
SIGMA = 3.0
STRATEGIES = (
    FallbackStrategy.MIDPOINT,
    FallbackStrategy.MEAN,
    FallbackStrategy.MEDIAN,
    FallbackStrategy.STDCLIP,
)
TASKS = {
    "gptq": (GPTQ, GPTQConfig),
    "qqq": (QQQ, QQQConfig),
}


def _outlier_weight() -> torch.Tensor:
    """One 64 outlier among 127 unit weights (the issue's fixture)."""

    weight = torch.ones(1, 128, dtype=torch.float64)
    weight[0, -1] = 64.0
    return weight


def _selected_bounds(
    weight: torch.Tensor, strategy: FallbackStrategy
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy is FallbackStrategy.MIDPOINT:
        return weight.amin(dim=1, keepdim=True), weight.amax(dim=1, keepdim=True)
    if strategy is FallbackStrategy.MEAN:
        center = weight.mean(dim=1, keepdim=True)
        dev = (weight - center).abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        return center - dev, center + dev
    if strategy is FallbackStrategy.MEDIAN:
        center = weight.median(dim=1, keepdim=True).values
        dev = (weight - center).abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        return center - dev, center + dev
    if strategy is FallbackStrategy.STDCLIP:
        mean = weight.mean(dim=1, keepdim=True)
        std = weight.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-8)
        return mean - SIGMA * std, mean + SIGMA * std
    raise AssertionError(f"unexpected strategy: {strategy}")


def _reference_grid(
    weight: torch.Tensor, strategy: FallbackStrategy
) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = _selected_bounds(weight, strategy)
    bound = torch.maximum(lo.abs(), hi.abs())
    scale = (bound + bound) / MAXQ
    codes = (weight / scale + ZERO).round().clamp(0, MAXQ)
    return (codes - ZERO) * scale, scale


def _run(
    task_name: str, weight: torch.Tensor, strategy: FallbackStrategy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_cls, config_cls = TASKS[task_name]
    module = nn.Linear(128, 1, bias=False)
    module.weight.data.copy_(weight.float())
    task = task_cls(
        module,
        config_cls(
            bits=BITS,
            group_size=128,
            sym=True,
            fallback=Fallback(strategy=strategy.value, threshold=10),
        ),
    )
    if task_cls is QQQ:
        task.quantizer.configure(BITS, perchannel=True, groupsize=128, sym=True)
        quantized, scale, zero = task._fallback_quantize(strategy)[:3]
    else:
        quantized, scale, zero = task._fallback_quantize(strategy, 128)[:3]
    return quantized.double(), scale.double(), zero.double()


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize("task_name", list(TASKS))
def test_symmetric_fallback_fits_the_selected_bounds(task_name, strategy):
    weight = _outlier_weight()
    quantized, scale, zero = _run(task_name, weight, strategy)
    expected, expected_scale = _reference_grid(weight, strategy)

    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(zero, torch.full_like(zero, ZERO), rtol=0, atol=0)
    torch.testing.assert_close(quantized, expected, rtol=1e-6, atol=1e-6)
    assert torch.all(scale > 0)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_symmetric_stdclip_clips_the_outlier(task_name):
    weight = _outlier_weight()
    quantized, scale, zero = _run(task_name, weight, FallbackStrategy.STDCLIP)
    expected, expected_scale = _reference_grid(weight, FallbackStrategy.STDCLIP)
    raw_scale = 2.0 * weight.abs().amax() / MAXQ

    # The outlier no longer sets the grid...
    assert scale.item() == pytest.approx(expected_scale.item(), rel=1e-6)
    assert scale.item() < 0.5 * raw_scale.item()
    # ...and saturates at the top code instead of being represented as 64.
    assert quantized[0, -1].item() == pytest.approx((MAXQ - ZERO) * scale.item(), rel=1e-6)
    assert quantized[0, -1].item() < 0.5 * weight[0, -1].item()
    assert zero.item() == ZERO


@pytest.mark.parametrize("task_name", list(TASKS))
def test_symmetric_midpoint_keeps_the_full_block_range(task_name):
    weight = _outlier_weight()
    quantized, scale, zero = _run(task_name, weight, FallbackStrategy.MIDPOINT)

    expected_scale = 2.0 * weight.abs().amax() / MAXQ
    assert scale.item() == pytest.approx(expected_scale.item(), rel=1e-6)
    assert zero.item() == ZERO
    # The largest weight still lands on the top code.
    assert quantized[0, -1].item() == pytest.approx((MAXQ - ZERO) * scale.item(), rel=1e-6)


@pytest.mark.parametrize("strategy", [FallbackStrategy.MIDPOINT, FallbackStrategy.STDCLIP])
@pytest.mark.parametrize("task_name", list(TASKS))
def test_symmetric_fallback_all_zero_group_stays_finite(task_name, strategy):
    weight = torch.zeros(1, 128, dtype=torch.float64)
    quantized, scale, zero = _run(task_name, weight, strategy)

    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0)
    assert torch.all(zero == ZERO)
    assert torch.equal(quantized, torch.zeros_like(quantized))


def test_symmetric_range_from_bounds_centers_the_selection():
    lo = torch.tensor([[-2.0, 0.0, -0.0]], dtype=torch.float32)
    hi = torch.tensor([[18.0, 0.0, 0.0]], dtype=torch.float32)
    centered_lo, centered_hi = symmetric_range_from_bounds(lo, hi)

    torch.testing.assert_close(centered_lo, torch.tensor([[-18.0, -1.0, -1.0]]))
    torch.testing.assert_close(centered_hi, torch.tensor([[18.0, 1.0, 1.0]]))
    assert centered_lo.dtype == lo.dtype
