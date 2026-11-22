# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""An NVFP4 block above the global-scale floor must keep a usable scale (#5652).

``nvfp4_global_scale`` divides a positive activation maximum by ``448 * E2M1
maximum``.  For a block of ``2**-140`` that quotient rounds to zero in FP32
even though the block is exactly encodable (global ``2**-149``, E4M3 local
``128``, E2M1 code ``4``), so the reference QDQ rejected the scale as
non-positive.  The quotient is floored at the least positive FP32 subnormal; a
true zero block keeps the ``1.0`` convention.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.activation_floatx import (
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)


FP32_MIN_SUBNORMAL = 2.0**-149
SMALL_BLOCK = 2.0**-140
RECIPES = (
    "least_squares",
    "nvidia",
    "four_six",
    "least_squares_headroom",
    "least_squares_grid",
    "nvidia_headroom",
)


def _block(value: float = SMALL_BLOCK) -> torch.Tensor:
    return torch.full((1, 16), value, dtype=torch.float32)


def test_underflowing_default_scale_is_the_least_positive_fp32():
    scale = nvfp4_global_scale(_block().amax())
    assert scale.dtype == torch.float32
    assert scale.item() > 0
    assert scale.item() == FP32_MIN_SUBNORMAL


def test_underflowing_scale_reconstructs_the_block_bit_for_bit():
    x = _block()
    scale = nvfp4_global_scale(x.amax())
    assert torch.equal(nvfp4_block_qdq(x, scale), x)


def test_the_smallest_positive_fp32_scale_is_the_reported_control():
    x = _block()
    assert torch.equal(nvfp4_block_qdq(x, FP32_MIN_SUBNORMAL), x)


def test_zero_block_keeps_the_unit_scale_convention():
    value = torch.zeros(4, dtype=torch.float32)
    scale = nvfp4_global_scale(value)
    assert torch.equal(scale, torch.ones(4, dtype=torch.float32))
    x = torch.zeros(1, 16, dtype=torch.float32)
    assert torch.equal(nvfp4_block_qdq(x, scale[0]), x)


def test_only_the_underflowing_element_is_floored():
    value = torch.tensor([0.0, SMALL_BLOCK, 1.0], dtype=torch.float32)
    scale = nvfp4_global_scale(value)
    assert scale[0].item() == 1.0
    assert scale[1].item() == FP32_MIN_SUBNORMAL
    assert scale[2].item() == float(torch.tensor(1.0 / (FP8_E4M3_MAX * 4.0), dtype=torch.float32))


@pytest.mark.parametrize("grid_dtype", [torch.float16, torch.bfloat16])
def test_source_grid_rounding_keeps_the_floored_scale(grid_dtype):
    # Both grids round the subnormal to zero; the FP32 scale must survive.
    scale = nvfp4_global_scale(_block().amax(), grid_dtype=grid_dtype)
    assert scale.item() == FP32_MIN_SUBNORMAL


def test_float_conversion_keeps_the_scale_usable():
    # Calibration stores the scale as a Python float before replay.
    scale = float(nvfp4_global_scale(_block().amax()))
    assert scale > 0
    assert torch.equal(nvfp4_block_qdq(_block(), scale), _block())


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_returns_a_positive_consumable_scale(recipe):
    x = _block()
    scale = nvfp4_global_scale(x.amax(), recipe=recipe)
    assert scale.item() == FP32_MIN_SUBNORMAL
    assert torch.isfinite(nvfp4_block_qdq(x, scale, recipe)).all()


@pytest.mark.parametrize("denominator", [4.0, FP4_E2M1_MAX])
def test_the_positive_quotient_underflows_in_fp32(denominator):
    # Both the headroom-reserving and the NVIDIA denominators round to zero.
    quotient = torch.tensor(SMALL_BLOCK, dtype=torch.float32) / torch.tensor(
        FP8_E4M3_MAX * denominator, dtype=torch.float32
    )
    assert quotient.item() == 0.0
