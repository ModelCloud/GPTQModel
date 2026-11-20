# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Relative reconstruction error must not hide a lost signal (#5635).

MACaQ, VPTQ and RCO divide an FP64 error energy by an FP64 reference energy.
They used to floor that denominator to an absolute constant, so globally
rescaling the reference (and with it the reconstruction) changed the reported
score, and a reconstruction that dropped the whole signal reported a near-zero
relative error instead of 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.quantization.macaq import Macaq
from gptqmodel.quantization.rco_formats import (
    RCOFormatError,
    relative_reconstruction_error,
)
from gptqmodel.quantization.vptq import VPTQ


def _macaq(weight, dequantized, hessian):
    return Macaq.hessian_weighted_error(
        weight_original=weight, dequantized=dequantized, hessian=hessian
    )


def _vptq(weight, quantized, hessian):
    return VPTQ.hessian_weighted_error(
        weight_original=weight, quantized=quantized, hessian=hessian
    )


def _rco(weight, dequantized, hessian):
    return relative_reconstruction_error(weight, dequantized, hessian)


WEIGHTED_METRICS = (
    pytest.param(_macaq, id="macaq"),
    pytest.param(_vptq, id="vptq"),
    pytest.param(_rco, id="rco"),
)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
@pytest.mark.parametrize("exponent", (0, -24, -60, -120))
def test_total_signal_loss_is_one_at_every_scale(metric, exponent):
    """A zero reconstruction of a nonzero reference is exactly 1, not near 0."""

    weight = torch.full((1, 32), 2.0**exponent)
    assert metric(weight, torch.zeros_like(weight), torch.eye(32)) == 1.0


@pytest.mark.parametrize("exponent", (0, -24, -60, -120))
def test_rco_without_a_hessian_also_reports_total_loss_as_one(exponent):
    weight = torch.full((1, 32), 2.0**exponent)
    assert relative_reconstruction_error(weight, torch.zeros_like(weight)) == 1.0


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
@pytest.mark.parametrize("weight_scale", (1e-30, 1e-12, 1.0, 1e12))
@pytest.mark.parametrize("hessian_scale", (1e-8, 1.0, 1e8))
def test_relative_error_is_invariant_to_independent_w_and_h_rescaling(
    metric, weight_scale, hessian_scale
):
    """The ratio only depends on the relative error, never on absolute scale."""

    weight = torch.full((4, 8), 3.0) * weight_scale
    hessian = torch.eye(8) * hessian_scale
    assert metric(weight, torch.zeros_like(weight), hessian) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("metric", WEIGHTED_METRICS)
def test_zero_reference_with_a_zero_reconstruction_scores_zero(metric):
    """Genuinely zero energy: ``0 / 0`` is a perfect fit of a zero signal."""

    zero = torch.zeros(4, 8)
    assert metric(zero, zero, torch.eye(8)) == 0.0


@pytest.mark.parametrize(
    "metric", (pytest.param(_macaq, id="macaq"), pytest.param(_vptq, id="vptq"))
)
def test_zero_reference_with_a_nonzero_reconstruction_is_infinite(metric):
    """Genuinely zero energy: any invented signal is total loss, never 0."""

    assert math.isinf(metric(torch.zeros(4, 8), torch.ones(4, 8), torch.eye(8)))


def test_rco_rejects_a_nonzero_reconstruction_of_a_zero_reference():
    """RCO requires finite scores, so it rejects the total-loss case."""

    with pytest.raises(RCOFormatError):
        relative_reconstruction_error(torch.zeros(4, 8), torch.ones(4, 8), torch.eye(8))


@pytest.mark.parametrize("with_hessian", (False, True))
def test_positive_reference_still_matches_the_fp64_oracle(with_hessian):
    """Removing the floor must not change the ordinary positive-energy ratio."""

    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    dequantized = reference + 0.01 * torch.randn(
        8, 16, generator=generator, dtype=torch.float64
    )
    hessian = torch.randn(16, 16, generator=generator, dtype=torch.float64)
    hessian = hessian @ hessian.mT

    error = dequantized - reference
    if with_hessian:
        expected = ((error @ hessian) * error).sum() / ((reference @ hessian) * reference).sum()
    else:
        expected = error.pow(2).sum() / reference.pow(2).sum()

    got = relative_reconstruction_error(
        reference, dequantized, hessian if with_hessian else None
    )
    assert got == pytest.approx(float(expected), rel=1e-12)
