# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The NVFP4 least-squares refinement must be invariant to an exact power-of-two rescale.

``evaluate`` already ranks candidates in per-block power-of-two coordinates, so
an activation and its global scale that are both multiplied by ``2**e`` stay
finite and keep the same block SSE.  ``refine`` still solved the least-squares
problem in absolute FP32: its numerator ``sum(blocks * values)`` overflows to
infinity for activations that are finite only because the global scale absorbs
their magnitude, the resulting ``inverse`` saturates to the E4M3 upper bound,
that candidate loses the score comparison, and the search silently falls back
to the worse seed.  The public QDQ output stays finite, so a plain
``isfinite`` check cannot see the regression.

The fixtures below are a fixed 16-element FP32 block at global scale ``0.001``,
and the identical problem with both tensors multiplied by ``2**124``.  A FP64
oracle that enumerates the 126 positive finite E4M3 block scales and rounds
each code assignment independently to E2M1 supplies the expected optimum.
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.quantization.activation_floatx import (
    _nvfp4_mse_quantize_blocks,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)


pytestmark = [pytest.mark.cpu]

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GLOBAL_SCALE = 0.001
_RESCALE = 2.0**124

# A fixed block whose Four-Over-Six seeds are the E4M3 values 208 (maximum 4)
# and 144 (maximum 6).  The 144 seed has the lower SSE, and the least-squares
# refinement recovers it from the 208 assignment at the small magnitude.
_BLOCK = (
    0.4472350478172302, 0.2393692433834076, -0.028780341148376465,
    0.43915855884552, -0.6220099329948425, -0.3339952528476715,
    -0.0813106968998909, 0.802413821220398, 0.5239754319190979,
    -0.25585582852363586, -0.0012009433703497052, 0.22150030732154846,
    -0.8143987655639648, 0.027795936912298203, 0.6136912107467651,
    -0.5106378793716431,
)


def _e4m3_value(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _e2m1_nearest(x: torch.Tensor) -> torch.Tensor:
    """Independent nearest-even rounding to the E2M1 magnitude grid."""
    magnitude = x.abs().clamp(max=_E2M1_VALUES[-1])
    grid = torch.tensor(_E2M1_VALUES, dtype=x.dtype, device=x.device)
    distance = (grid - magnitude[..., None]).abs()
    closest = distance == distance.min(dim=-1, keepdim=True).values
    even_code = closest & (torch.arange(len(_E2M1_VALUES), device=x.device) % 2 == 0)
    use_even = even_code.any(dim=-1, keepdim=True)
    chosen = torch.where(use_even, even_code, closest)
    values = grid[chosen.to(torch.int64).argmax(dim=-1)]
    return values.copysign(x)


def _oracle(blocks: torch.Tensor, global_scale: float) -> tuple[float, float]:
    """Return the best local E4M3 scale and its normalized FP64 SSE."""

    blocks64 = blocks.double().reshape(-1, 16)
    denominator = blocks64.square().sum()
    best = None
    for code in range(1, 127):
        local = _e4m3_value(code)
        scale = local * global_scale
        if not math.isfinite(scale) or scale <= 0:
            continue
        values = _e2m1_nearest(blocks64 / scale).double()
        error = ((values * scale - blocks64).square().sum() / denominator).item()
        if best is None or error < best[0]:
            best = (error, local)
    assert best is not None
    return best[1], best[0]


def _block(scale: float = 1.0) -> torch.Tensor:
    return (torch.tensor(_BLOCK, dtype=torch.float32) * scale).reshape(1, 16)


def _quantize(blocks: torch.Tensor, global_scale: float, recipe: str = "least_squares"):
    local, values = _nvfp4_mse_quantize_blocks(
        blocks.reshape(1, 1, 16), torch.tensor([global_scale], dtype=torch.float32), recipe
    )
    return float(local.reshape(-1).float().item()), values


def _normalized_sse(values: torch.Tensor, local: float, blocks: torch.Tensor,
                    global_scale: float) -> float:
    reconstructed = values.double() * local * global_scale
    return (
        (reconstructed - blocks.double().reshape(1, 1, 16)).square().sum()
        / blocks.double().square().sum()
    ).item()


def test_refine_recovers_the_same_optimum_after_power_of_two_rescale():
    """Both magnitudes must select the oracle's E4M3 scale and SSE."""

    small = _block()
    large = _block(_RESCALE)
    assert bool(torch.isfinite(large).all())
    assert bool(torch.isfinite(torch.tensor(_GLOBAL_SCALE * _RESCALE)).all())

    oracle_scale, oracle_error = _oracle(small, _GLOBAL_SCALE)
    assert oracle_scale == 144.0

    for blocks, global_scale in ((small, _GLOBAL_SCALE),
                                 (large, _GLOBAL_SCALE * _RESCALE)):
        local, values = _quantize(blocks, global_scale)
        assert local == oracle_scale
        assert _normalized_sse(values, local, blocks, global_scale) == pytest.approx(
            oracle_error, rel=1e-12, abs=1e-12
        )


def test_four_six_control_selects_the_worse_seed_at_both_magnitudes():
    """The seed-only recipe is the negative control: it never refines."""

    for blocks, global_scale in ((_block(), _GLOBAL_SCALE),
                                 (_block(_RESCALE), _GLOBAL_SCALE * _RESCALE)):
        local, values = _quantize(blocks, global_scale, "four_six")
        assert local == 208.0
        assert _normalized_sse(values, local, blocks, global_scale) > 0.008


def test_refine_scale_ratio_stays_finite_for_the_large_magnitude():
    """The defect trigger: the absolute least-squares numerator overflows."""

    blocks = _block(_RESCALE).reshape(1, 1, 16)
    # The E2M1 assignment of the recovered optimum, independent of the search.
    values = _e2m1_nearest(blocks.double() / (144.0 * (_GLOBAL_SCALE * _RESCALE))).float()
    absolute_numerator = (blocks * values).sum()
    assert not bool(torch.isfinite(absolute_numerator).all())

    local, _ = _quantize(_block(_RESCALE), _GLOBAL_SCALE * _RESCALE)
    assert local == 144.0


@pytest.mark.parametrize("recipe", ["least_squares", "least_squares_grid"])
def test_public_qdq_is_scale_invariant_and_finite(recipe):
    """The public QDQ entry point must agree across the exact rescale."""

    small = nvfp4_block_qdq(_block(), torch.tensor(_GLOBAL_SCALE), recipe)
    large = nvfp4_block_qdq(_block(_RESCALE), torch.tensor(_GLOBAL_SCALE * _RESCALE), recipe)

    assert bool(torch.isfinite(small).all())
    assert bool(torch.isfinite(large).all())
    # ``2**124`` is an exact FP32 power of two, so the rescaled public output
    # must be the identical reconstruction scaled by exactly that factor.
    torch.testing.assert_close(
        torch.ldexp(large, torch.tensor(-124, dtype=torch.int32)), small, rtol=0, atol=0
    )


def test_zero_and_small_blocks_keep_their_scale():
    """A zero block keeps the ``1.0`` convention; a small block stays encodable."""

    for blocks, global_scale in ((torch.zeros(1, 16), _GLOBAL_SCALE),
                                 (torch.full((1, 16), 2.0**-140), 2.0**-149)):
        local, values = _quantize(blocks, global_scale)
        assert math.isfinite(local) and local > 0
        if blocks.count_nonzero() == 0:
            assert local == 1.0
            assert values.count_nonzero() == 0


def test_mixed_sign_block_matches_the_oracle():
    """Negative and positive values must round through the same refinement."""

    blocks = _block()
    assert (blocks > 0).any() and (blocks < 0).any()
    oracle_scale, oracle_error = _oracle(blocks, _GLOBAL_SCALE)
    local, values = _quantize(blocks, _GLOBAL_SCALE)
    assert local == oracle_scale
    assert _normalized_sse(values, local, blocks, _GLOBAL_SCALE) == pytest.approx(
        oracle_error, rel=1e-12, abs=1e-12
    )


def test_large_magnitude_block_global_scale_is_accepted():
    """The reported global scale for the rescaled block is finite and positive."""

    blocks = _block(_RESCALE)
    global_scale = nvfp4_global_scale(blocks.abs().amax())
    assert bool(torch.isfinite(global_scale).all()) and float(global_scale) > 0


def test_packing_consumer_matches_the_reference_entry_point():
    """The native input packing path must share the fixed helper."""

    from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import nvfp4_input

    x = _block().repeat(1, 8)
    results = {}
    for label, blocks, global_scale in (("small", x, _GLOBAL_SCALE),
                                        ("large", x * _RESCALE, _GLOBAL_SCALE * _RESCALE)):
        packed, local = nvfp4_input(blocks, torch.tensor(global_scale))
        assert packed.dtype == torch.float4_e2m1fn_x2
        assert packed.shape == (1, 64)
        # Every packed block comes from the oracle's optimum, so the large
        # magnitude cannot silently fall back to the worse seed.
        assert bool((local.float().reshape(-1) == 144.0).all())
        reference = nvfp4_block_qdq(blocks, torch.tensor(global_scale))
        results[label] = (packed.view(torch.uint8).clone(), reference)

    # The packed E2M1 codes and the reconstructed values must both be the exact
    # power-of-two rescale of the small-magnitude result.
    torch.testing.assert_close(results["large"][1], results["small"][1] * _RESCALE,
                               rtol=0, atol=0)
