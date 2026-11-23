# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Row/column RMS smoothing must survive a representable RMS (#6014).

``SmoothRowCol`` and ``SmoothSoftNorm`` computed the RMS as
``block.pow(2).mean(dim).sqrt()``. The square overflows the working dtype well
before the RMS does, so a lane whose RMS is finite produced ``inf``; the
normalized block became ``0`` and the restore ``0 * inf`` became ``nan``.
For FP32 a constant row of ``2**62`` already overflowed the plain ``mean``
reduction, while the true RMS stays representable through ``2**120``.

The regressions below pin the helper, both smoother consumers of it, and the
real ``RTN.quantize`` path. The independent oracle is an FP64
``torch.linalg.vector_norm`` over the same block, never the helper's own
arithmetic. A representable-block control proves the ordinary arithmetic is
unchanged.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    RTNConfig,
    SmoothRowCol,
    SmoothSoftNorm,
)
from gptqmodel.quantization.fallback_smooth import _safe_rms, smooth_block
from gptqmodel.quantization.rtn import RTN


# ``2**62`` already overflows the plain square-then-mean reduction in FP32
# while its RMS is finite; ``2**120`` overflows both the square and the sum.
OVERFLOW_EXPONENTS = (62, 70, 120)


def _constant_row(exponent: int, *, rows: int = 1, columns: int = 128) -> torch.Tensor:
    return torch.full((rows, columns), float(2.0**exponent), dtype=torch.float32)


def _fp64_rms(block: torch.Tensor, dim: int) -> torch.Tensor:
    """Independent FP64 oracle for the root-mean-square over ``dim``.

    ``vector_norm`` scales internally and does not form the overflowing
    square, so the oracle stays finite wherever the mathematical RMS is.
    """

    width = block.shape[dim]
    norm = torch.linalg.vector_norm(block.double(), ord=2, dim=dim, keepdim=True)
    return norm / width**0.5


@pytest.mark.parametrize("exponent", OVERFLOW_EXPONENTS)
@pytest.mark.parametrize("dim", [0, 1])
def test_safe_rms_matches_the_fp64_oracle_past_the_square_overflow(dim, exponent):
    block = _constant_row(exponent)
    expected = _fp64_rms(block, dim)

    actual = _safe_rms(block, dim)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("exponent", OVERFLOW_EXPONENTS)
@pytest.mark.parametrize("axis", ["row", "col"])
def test_smooth_rowcol_restores_a_finite_block_past_the_square_overflow(axis, exponent):
    block = _constant_row(exponent)
    fallback = Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothRowCol(axis=axis))

    smoothed, scale_factor = smooth_block(block, fallback, group_size=block.shape[1])

    # The column axis fits one RMS per column and then collapses those to a
    # single restoring factor, so the oracle reduces the same way.
    expected = _fp64_rms(block, dim=1 if axis == "row" else 0)
    if axis == "col":
        expected = expected.mean().view(1, 1)
    assert torch.isfinite(scale_factor).all()
    torch.testing.assert_close(scale_factor.double(), expected, rtol=1e-6, atol=0)

    # Every element equals the row RMS, so the normalized block is exactly one
    # and the restore must return the original finite magnitude.
    assert torch.isfinite(smoothed).all()
    restored = smoothed * scale_factor
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored.double(), block.double(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("exponent", OVERFLOW_EXPONENTS)
def test_smooth_softnorm_keeps_a_zero_mean_row_finite_past_the_square_overflow(exponent):
    """``SmoothSoftNorm`` subtracts the row mean before its deviation RMS."""

    magnitude = float(2.0**exponent)
    block = torch.zeros(1, 128, dtype=torch.float32)
    block[0, ::2] = magnitude
    block[0, 1::2] = -magnitude
    fallback = Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothSoftNorm(k=3.0))

    smoothed, scale_factor = smooth_block(block, fallback, group_size=block.shape[1])

    assert scale_factor is None
    assert torch.isfinite(smoothed).all()
    # A zero-mean +-magnitude row is exactly one standard deviation per
    # element, so the clamped SoftNorm round trip preserves the magnitude.
    torch.testing.assert_close(smoothed.double(), block.double(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("exponent", (62, 70))
@pytest.mark.parametrize("axis", ["row", "col"])
def test_rtn_smooth_rowcol_survives_the_square_overflow(axis, exponent):
    """The real fallback entry point must emit finite weights and scales."""

    linear = nn.Linear(128, 8, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(float(2.0**exponent))

    qcfg = RTNConfig(
        bits=4,
        group_size=128,
        sym=True,
        smooth=SmoothRowCol(axis=axis),
        offload_to_disk=False,
    )
    quantized, scale, _zero, _g_idx, *_ = RTN(linear, qcfg).quantize()

    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    # The grid must cover the source magnitude, not a collapsed zero block.
    assert quantized.abs().max().item() > 0
    # A symmetric 4-bit grid reconstructs at most ``16/15 * amax`` and, after
    # the code clamp, at least ``amax`` in magnitude; both are finite here.
    assert quantized.abs().max().item() < float(2.0**exponent) * 1.5


@pytest.mark.parametrize("axis", ["row", "col"])
def test_smooth_rowcol_is_unchanged_when_the_plain_square_is_representable(axis):
    """Control: the ordinary path keeps its exact previous arithmetic."""

    torch.manual_seed(0)
    block = torch.randn(64, 128) * 3
    fallback = Fallback(strategy=FallbackStrategy.RTN, smooth=SmoothRowCol(axis=axis))

    _smoothed, scale_factor = smooth_block(block, fallback, group_size=block.shape[1])

    dim = 1 if axis == "row" else 0
    expected = block.pow(2).mean(dim=dim, keepdim=True).sqrt().clamp(min=1e-8)
    if axis == "col":
        expected = expected.mean().view(1, 1)
    assert torch.equal(scale_factor, expected)


def test_safe_rms_preserves_all_zero_and_nonfinite_inputs():
    assert torch.equal(_safe_rms(torch.zeros(2, 8), 1), torch.zeros(2, 1))

    mixed = torch.zeros(3, 8)
    mixed[0] = 1.0
    mixed[1] = -2.0
    assert _safe_rms(mixed, 1).flatten().tolist() == [1.0, 2.0, 0.0]

    # A non-finite element must still propagate rather than be repaired.
    for nonfinite in (float("inf"), float("-inf")):
        block = torch.ones(1, 8)
        block[0, 3] = nonfinite
        assert _safe_rms(block, 1).item() == float("inf")

    nan_block = torch.ones(1, 8)
    nan_block[0, 3] = float("nan")
    assert torch.isnan(_safe_rms(nan_block, 1)).all()
