# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""MACaQ's exponent screen must not score candidates in an absolute-magnitude frame.

``screen_block_exponents`` picked the H-weighted per-block E8M0 exponent by
summing ``H_jj * (w - deq) ** 2`` in the block's own units.  A block rescaled
by ``2**k`` scales that sum by ``2**(2k)``, which FP32 cannot hold across the
whole E8M0/FP32 range: small blocks underflowed every candidate to ``0`` and
large ones overflowed to ``+inf``.  Both cases left the first configured
offset, so the selected exponent depended on the block's magnitude rather than
on its error distribution, and the result changed again once the offset was
frozen and written to the packed payload.

The reference below enumerates the three exponents from the OCP definition and
selects the nearest E2M1 level from an explicit value table in FP64.  It is
written independently of ``screen_block_exponents`` and never consumes the
implementation's outputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.quantization import macaq


pytestmark = [pytest.mark.cpu]

BLOCK = 32
# The E2M1 element set, written out here so the oracle does not share the
# production codebook object.
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fixture(k: int) -> torch.Tensor:
    """One 32-wide row: a unit peak plus 31 BF16 ``0.13`` values, scaled by ``2**k``."""

    value = torch.tensor(0.13, dtype=torch.bfloat16).item()
    row = torch.full((1, BLOCK), value, dtype=torch.float32)
    row[0, 0] = 1.0
    return (row * (2.0**k)).to(torch.bfloat16)


def _ocp_exponent(block: torch.Tensor, offset: int) -> int:
    """``floor(log2(amax)) - 2 + offset`` clamped to E8M0, exactly as OCP defines it."""

    peak = float(block.abs().max().item())
    if peak == 0.0:
        return -127 + offset
    exponent = math.floor(math.log2(peak)) - 2 + offset
    return max(macaq.E8M0_MIN_EXPONENT, min(macaq.E8M0_MAX_EXPONENT, exponent))


def _nearest_level(scaled: float) -> tuple[float, float]:
    """Nearest E2M1 magnitude and its squared distance, in FP64."""

    best = min(E2M1_LEVELS, key=lambda level: (abs(scaled) - level) ** 2)
    return best, (abs(scaled) - best) ** 2


def _oracle_codes(weight: torch.Tensor, exponent: int) -> list[float]:
    """Dequantized row the OCP definition produces for a fixed exponent."""

    scale = math.ldexp(1.0, exponent)
    values = []
    for value in weight[0].tolist():
        magnitude, _ = _nearest_level(value / scale)
        values.append(math.copysign(magnitude * scale, value) if value != 0.0 else 0.0)
    return values


def _oracle_screen(weight: torch.Tensor, hessian_diagonal: torch.Tensor, offsets=(1, 0, -1)):
    """Exact FP64 screen over the same candidate exponents.

    Returns ``(percent_error, best_offset, best_log2, per_offset_errors)`` for
    one row.  Errors are divided by ``2**(2k)`` with ``2**k`` the block peak's
    power of two, which is the common factor every candidate carries.
    """

    row = weight[0].to(torch.float64)
    hdiag = hessian_diagonal.to(torch.float64)
    peak = float(row.abs().max().item())
    scale_power = math.floor(math.log2(peak)) if peak > 0.0 else 0.0
    # One common divisor for every candidate: the block's own magnitude.
    normalizer = math.ldexp(1.0, 2 * int(scale_power))

    errors = {}
    for offset in offsets:
        exponent = _ocp_exponent(row, offset)
        scale = math.ldexp(1.0, exponent)
        total = 0.0
        for column, value in enumerate(row.tolist()):
            _, residual_sq = _nearest_level(value / scale)
            total += float(hdiag[column].item()) * residual_sq
        errors[offset] = total / normalizer

    ordered = list(offsets)
    best_offset = ordered[0]
    for offset in ordered[1:]:
        # Strict ``<`` keeps the earliest offset on a tie, matching the screen.
        if errors[offset] < errors[best_offset]:
            best_offset = offset
    return errors, best_offset, _ocp_exponent(row, best_offset)


def _run(weight: torch.Tensor, hessian_diagonal: torch.Tensor, offsets=(1, 0, -1)):
    return macaq.screen_block_exponents(weight, hessian_diagonal, BLOCK, offsets)


@pytest.mark.parametrize(
    "power",
    [0, -80, 80],
    ids=["control", "tiny-finite", "huge-finite"],
)
def test_screen_is_invariant_under_a_power_of_two_rescale(power):
    """The reported fixture: the control is unchanged and both extremes are fixed."""

    weight = _fixture(power)
    hessian_diagonal = torch.ones(BLOCK)

    _, log2_scale, _ = _run(weight, hessian_diagonal)
    errors, oracle_offset, oracle_exponent = _oracle_screen(weight, hessian_diagonal)

    # The candidate errors themselves are finite and ordered, not all 0/inf.
    assert all(math.isfinite(value) for value in errors.values()), errors
    assert int(log2_scale[0, 0].item()) == oracle_exponent
    selected_offset = oracle_exponent - (_ocp_exponent(weight[0], 0))
    assert selected_offset == oracle_offset


def test_screen_control_matches_the_oracle_offset():
    """Without rescaling, the screen's decision equals the FP64 argmin."""

    weight = _fixture(0)
    hessian_diagonal = torch.ones(BLOCK)

    _, log2_scale, _ = _run(weight, hessian_diagonal)
    errors, oracle_offset, oracle_exponent = _oracle_screen(weight, hessian_diagonal)

    assert oracle_offset == 0
    assert int(log2_scale[0, 0].item()) == oracle_exponent == -2
    # The control's own ordering: offset 0 beats +-1 by the reported margins.
    assert errors[0] < errors[1]
    assert errors[0] < errors[-1]


def test_quantized_payload_translates_with_the_rescale():
    """The packed bytes and the entry point's dequantized weight agree after rescaling."""

    decoded = {}
    for power in (0, -80, 80):
        result = macaq.macaq_quantize_weight(
            _fixture(power),
            torch.eye(BLOCK),
            BLOCK,
            rotation="none",
            pre_mode="none",
            do_flip=False,
            ridge=0.0,
            exp_screen=True,
            return_dequantized=True,
        )
        payload = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, BLOCK)
        # The packed payload is the source of truth, not an internal temporary.
        assert torch.equal(payload, result.dequantized_weight)
        decoded[power] = (payload, int(result.W_scales.view(torch.uint8)[0, 0].item()) - 127)

    # A uniform ``2**k`` rescale must move the chosen exponent by ``k`` and keep
    # the dequantized values at ``2**k`` times the control's.
    base_values, base_exp = decoded[0]
    for power in (-80, 80):
        values, exponent = decoded[power]
        assert exponent == base_exp + power, (power, exponent, base_exp)
        expected = base_values * (2.0**power)
        assert torch.equal(values, expected), power


@pytest.mark.parametrize("power", [0, -80, 80])
def test_screen_codes_are_consistent_across_the_rescale(power):
    """The packed codes are unchanged by the rescale; only the exponent moves."""

    control_index, control_log2, _ = _run(_fixture(0), torch.ones(BLOCK))
    index, log2_scale, _ = _run(_fixture(power), torch.ones(BLOCK))

    assert torch.equal(index, control_index)
    assert float(log2_scale[0, 0].item()) == float(control_log2[0, 0].item()) + power


def test_zero_block_keeps_its_floor_scale():
    """An exact-zero block is unchanged: it must not be normalized into a new code."""

    weight = torch.zeros(1, BLOCK)

    index, log2_scale, _ = _run(weight, torch.ones(BLOCK))
    decoded = macaq.dequantize_mxfp4(
        *macaq.pack_mxfp4_from_signed_index(index, log2_scale, BLOCK), BLOCK
    )

    assert torch.count_nonzero(decoded) == 0
    assert float(log2_scale[0, 0].item()) == -41.0


def test_nonuniform_positive_diagonal_is_respected():
    """A diagonal Hessian that is not a multiple of the identity still scores per column."""

    weight = _fixture(0)
    hdiag = torch.tensor(
        [1e6 if column % 2 == 0 else 1e-6 for column in range(BLOCK)], dtype=torch.float32
    )

    _, log2_scale, _ = _run(weight, hdiag)
    errors, _, oracle_exponent = _oracle_screen(weight, hdiag)

    assert all(math.isfinite(value) for value in errors.values())
    assert int(log2_scale[0, 0].item()) == oracle_exponent


def test_hessian_diagonal_magnitude_does_not_change_the_choice():
    """A uniform Hessian scale is one positive factor and cannot move the argmin."""

    weight = _fixture(0)
    exponents = set()
    for magnitude in (1e-30, 1.0, 1e30):
        _, log2_scale, _ = _run(weight, torch.full((BLOCK,), magnitude))
        exponents.add(int(log2_scale[0, 0].item()))

    assert exponents == {-2}, exponents


def test_dominant_diagonal_entry_does_not_erase_the_rest_of_the_block():
    """A per-block Hessian shift would flush the tiny entry and pick the wrong level.

    The huge column's own residual is exactly zero under the offset-0 candidate,
    so the block is decided by the ``1e-38`` column.  Dividing the diagonal by
    a per-block power of two derived from the block maximum (a tempting way to
    recenter it) puts that column at ``0`` and the first offset wins again.
    """

    weight = _fixture(0)
    hdiag = torch.zeros(BLOCK)
    hdiag[0] = 1e38
    hdiag[1] = 1e-38

    _, log2_scale, _ = _run(weight, hdiag)
    errors, _, oracle_exponent = _oracle_screen(weight, hdiag)

    assert all(math.isfinite(value) for value in errors.values())
    assert errors[0] < errors[1]
    assert int(log2_scale[0, 0].item()) == oracle_exponent == -2


@pytest.mark.parametrize("power", [-127, -126, -125])
def test_clamped_bottom_of_the_scale_range_stays_consistent(power):
    """Blocks at the bottom of the E8M0 range keep the decision the oracle makes.

    Below ``2**-125`` every candidate clamps to the same byte, so the tie is
    legitimate; what must not happen is a different exponent winning because
    every residual square underflowed to zero.
    """

    weight = _fixture(power)
    assert torch.isfinite(weight.float()).all()

    index, log2_scale, _ = _run(weight, torch.ones(BLOCK))
    _, _, oracle_exponent = _oracle_screen(weight, torch.ones(BLOCK))

    assert int(log2_scale[0, 0].item()) == oracle_exponent
    decoded = macaq.dequantize_mxfp4(
        *macaq.pack_mxfp4_from_signed_index(index, log2_scale, BLOCK), BLOCK
    )
    expected = torch.tensor(_oracle_codes(weight, oracle_exponent), dtype=decoded.dtype)
    assert torch.equal(decoded[0], expected)


@pytest.mark.parametrize(
    "diagonal",
    [
        torch.zeros(BLOCK),
        torch.cat([torch.tensor([float("inf")]), torch.ones(BLOCK - 1)]),
        torch.cat([torch.tensor([float("nan")]), torch.ones(BLOCK - 1)]),
        # Wider than FP32's own normal range: no shared factor can preserve both ends.
        torch.cat([torch.tensor([1e38]), torch.full((BLOCK - 1,), 1e-38)]),
    ],
    ids=["all-zero", "inf", "nan", "over-wide"],
)
def test_degenerate_diagonals_keep_a_finite_score(diagonal):
    """Degenerate Hessians must not turn the screen into a first-offset default."""

    weight = _fixture(0)

    index, log2_scale, _ = _run(weight, diagonal)

    assert torch.isfinite(log2_scale).all()
    assert int(index.min()) >= 0 and int(index.max()) <= 14
    # The all-zero diagonal makes every candidate score exactly 0, so the first
    # offset legitimately wins; the others must still pick the FP64 argmin.
    if diagonal.abs().sum() > 0 and torch.isfinite(diagonal).all():
        _, _, oracle_exponent = _oracle_screen(weight, diagonal)
        assert int(log2_scale[0, 0].item()) == oracle_exponent


def test_wide_but_representable_diagonal_recenters_without_flushing():
    """A diagonal spanning ``1e-30``..``1e30`` is inside FP32 and must stay exact."""

    weight = _fixture(0)
    hdiag = torch.logspace(-30, 30, BLOCK)

    _, log2_scale, _ = _run(weight, hdiag)
    # One shared power of two preserves every ratio, so the oracle's weighted
    # argmin is reached.
    _, oracle_offset, oracle_exponent = _oracle_screen(weight, hdiag)
    assert int(log2_scale[0, 0].item()) == oracle_exponent
    assert oracle_offset == 0
