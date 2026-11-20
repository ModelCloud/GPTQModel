# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5627: MXFP4 must not flush representable subnormal blocks.

``mxfp4_rtn_signed_index`` used to clamp every block peak to ``1e-12`` before
taking ``floor(log2(amax))``.  A constant block of ``2**-50`` is exactly
representable as E2M1 value 4 times the E8M0 scale ``2**-52`` (scale byte 75),
but the floor moved the scale to ``2**-42`` and rounded every code to zero.
The floor now applies only to an exact-zero block; a nonzero peak is clamped to
E8M0's actual finite exponent range before the codes are computed.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization import macaq
from gptqmodel.utils import macaq_triton


BLOCK = 32


def _quantize_constant(value: float, *, exp_screen: bool = False):
    weight = torch.full((1, BLOCK), value)
    return macaq.macaq_quantize_weight(
        weight,
        torch.eye(BLOCK),
        BLOCK,
        rotation="none",
        pre_mode="none",
        exp_screen=exp_screen,
        do_flip=False,
    )


def _scale_bytes(result) -> list[int]:
    return result.W_scales.view(torch.uint8).reshape(-1).tolist()


def test_issue_repro_weight_scale_and_value():
    """The issue's first repro: byte 85 / decoded 0 must become byte 75 / 2**-50."""

    result = _quantize_constant(2.0**-50)

    assert _scale_bytes(result) == [75]
    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, BLOCK)
    assert torch.equal(decoded, torch.full((1, BLOCK), 2.0**-50))


def test_issue_repro_w4a4_forward():
    """The issue's second repro: both operands are exact, so the dot is 1."""

    module = MacaqLinear(
        bits=4,
        group_size=BLOCK,
        in_features=BLOCK,
        out_features=1,
        had_block=BLOCK,
        rotation="none",
        pre_mode="none",
        activation_precision="w4a4",
        dtype=torch.bfloat16,
    )
    module.W_packed.fill_(0x22)  # magnitude 1, sign 0 -> 1 * 2**50
    module.W_scales.fill_(177)  # E8M0 byte for 2**50

    x = torch.zeros(1, BLOCK, dtype=torch.bfloat16)
    x[:, 0] = 2.0**-50
    assert module(x).item() == 1.0


def test_activation_quantizer_keeps_tiny_block():
    """The W4A4 activation quantizer used the same floor as the weight path."""

    x = torch.zeros(1, BLOCK)
    x[0, 0] = 2.0**-50
    assert torch.equal(macaq.macaq_quantize_activations(x), x)


@pytest.mark.parametrize(
    "value",
    [2.0**-39, 2.0**-40, 2.0**-50, 2.0**-100, 2.0**-126, 2.0**-127, 2.0**-128],
    ids=[
        "above-old-floor",
        "just-below-old-floor",
        "issue-repro",
        "tiny",
        "min-normal",
        "min-e8m0-scale",
        "min-mxfp4-value",
    ],
)
def test_representable_small_blocks_round_trip(value: float):
    """Every power of two down to the smallest MXFP4 value must survive exactly."""

    result = _quantize_constant(value)
    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, BLOCK)
    assert torch.equal(decoded, torch.full((1, BLOCK), value))
    assert torch.count_nonzero(decoded) == BLOCK


def test_screened_exponent_path_keeps_tiny_block():
    """The default exponent screen must preserve the same block."""

    result = _quantize_constant(2.0**-50, exp_screen=True)
    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, BLOCK)
    assert torch.equal(decoded, torch.full((1, BLOCK), 2.0**-50))


def test_all_zero_block_is_unchanged():
    """An exact-zero block keeps the old zero-preserving scale and stays zero."""

    result = _quantize_constant(0.0)

    assert _scale_bytes(result) == [85]
    assert torch.equal(result.W_packed, torch.zeros_like(result.W_packed))
    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, BLOCK)
    assert torch.count_nonzero(decoded) == 0


def test_zero_and_tiny_blocks_are_independent():
    """The floor is per block: a zero block must not hide its tiny neighbour."""

    weight = torch.zeros(1, 2 * BLOCK)
    weight[0, :BLOCK] = 2.0**-50
    result = macaq.macaq_quantize_weight(
        weight,
        torch.eye(2 * BLOCK),
        2 * BLOCK,
        rotation="none",
        pre_mode="none",
        exp_screen=False,
        do_flip=False,
    )

    assert _scale_bytes(result) == [75, 85]
    decoded = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, 2 * BLOCK)
    assert torch.equal(decoded[0, :BLOCK], torch.full((BLOCK,), 2.0**-50))
    assert torch.count_nonzero(decoded[0, BLOCK:]) == 0


@pytest.mark.skipif(
    not macaq_triton.triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
def test_triton_activation_quantizers_keep_tiny_block():
    """The accelerated activation kernels had the same nonzero-block floor."""

    x = torch.zeros(2, BLOCK, device="cuda")
    x[0, 0] = 2.0**-50

    packed, scales = macaq_triton.quantize_mxfp4(x)
    assert scales.view(torch.uint8).reshape(-1).tolist() == [75, 58]
    decoded = macaq_triton.dequantize_mxfp4(packed, scales, BLOCK)
    assert decoded[0, 0].item() == 2.0**-50
    assert torch.count_nonzero(decoded[1]) == 0

    identity = torch.eye(BLOCK, device="cuda", dtype=torch.float32)
    fused_packed, fused_scales = macaq_triton.rot_scale_quantize_mxfp4(x, None, identity)
    assert fused_scales.view(torch.uint8).reshape(-1).tolist() == [75, 58]
    fused_decoded = macaq_triton.dequantize_mxfp4(fused_packed, fused_scales, BLOCK)
    assert fused_decoded[0, 0].item() == 2.0**-50
    assert torch.count_nonzero(fused_decoded[1]) == 0
