# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5651: headroom calibration must not erase a constant signal.

``NVFP4ActivationHeadroom`` bins one log2 maximum per 16-value block into a
fixed histogram over ``[-40, 40]``.  Blocks outside that domain used to be
folded into the edge bins, and percentile extraction returned those bins'
interior midpoints, so a constant block of ``2**-80`` calibrated to
``2**-39.9`` and the headroom global scale drove the block scale to zero.
The tails now keep their exact count/min/max, and an in-range percentile is
clamped to the magnitudes the blocks actually took.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.activation_floatx import (
    NVFP4ActivationHeadroom,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)


RHO = 16384.0


def _collect(x: torch.Tensor, **kwargs) -> NVFP4ActivationHeadroom:
    collector = NVFP4ActivationHeadroom(**kwargs)
    collector.collect(x)
    return collector


def _blocks(*groups: tuple[int, float]) -> torch.Tensor:
    """One 16-value block per row, each holding the requested peak magnitude."""
    return torch.cat([torch.full((count, 16), value) for count, value in groups])


def _decode_ratio(value: float) -> float:
    x = torch.full((64, 16), value)
    scale = nvfp4_global_scale(_collect(x).compute_amax(), recipe="nvidia_headroom")
    return nvfp4_block_qdq(x, scale, recipe="nvidia_headroom")[0, 0].item() / value


@pytest.mark.parametrize(
    "value", [1.0, 2.0**-80, 2.0**80], ids=["unit", "underflow", "overflow"]
)
def test_issue_repro_constant_blocks_round_trip(value: float):
    """The issue's script: a constant signal must survive headroom calibration."""

    assert float(_collect(torch.full((64, 16), value)).compute_amax()) == pytest.approx(
        RHO * value, rel=1e-6
    )
    assert _decode_ratio(value) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    "value",
    [2.0**-40, 2.0**-39.9, 2.0**39.9, 2.0**40],
    ids=["below-domain", "first-bin", "last-bin", "above-domain"],
)
def test_constants_at_the_histogram_domain_edges(value: float):
    """Bin midpoints at either edge must not move a single-valued signal."""

    assert float(_collect(torch.full((64, 16), value)).compute_amax()) == pytest.approx(
        RHO * value, rel=1e-6
    )


def test_constant_non_power_of_two_keeps_its_value():
    """A percentile estimate may not exceed the largest block actually seen."""

    assert float(_collect(torch.full((64, 16), 3.0)).compute_amax()) == pytest.approx(
        RHO * 3.0, rel=1e-6
    )


def test_upper_percentile_resolves_the_overflow_tail():
    """A dense tail above the domain reports the tail magnitude, not 2**40."""

    x = _blocks((20_000, 2.0**30), (5_000, 2.0**50))

    assert float(_collect(x).compute_amax()) == pytest.approx(2.0**50, rel=1e-3)


def test_anchor_percentile_resolves_the_underflow_tail():
    """The 1% anchor must come from the observed tail, not the bin midpoint."""

    x = _blocks((5_000, 2.0**-80), (20_000, 2.0**-70))
    anchor = float(_collect(x).compute_amax()) / RHO

    assert 2.0**-80 <= anchor <= 2.0**-70


def test_literal_upper_percentile_keeps_its_headroom_for_overflow_blocks():
    """``upper_percentile=100`` still applies rho to the literal maximum."""

    x = torch.full((64, 16), 2.0**80)

    assert float(_collect(x, upper_percentile=100.0).compute_amax()) == pytest.approx(
        RHO * 2.0**80, rel=1e-6
    )


def test_zero_blocks_do_not_drag_the_tail_statistics():
    """An all-zero block contributes no magnitude and must not move the tail."""

    x = _blocks((64, 0.0), (64, 2.0**-80))

    assert float(_collect(x).compute_amax()) == pytest.approx(RHO * 2.0**-80, rel=1e-6)
