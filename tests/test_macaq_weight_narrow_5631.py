# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Finite MXFP4 weights must not be narrowed to the activation dtype (#5631).

``_forward_torch`` used to decode the packed weight at the activation dtype
before the FP32 reduction.  A weight whose decoded magnitude exceeds that
dtype's finite range became ``inf`` inside the decoder, so the zero lanes of
the dot contributed ``0 * inf = NaN`` even though the stored value is finite
and the exact output is representable in the activation dtype.

The Torch fallback now decodes in FP32 whenever the packed scales cannot be
represented in the activation dtype, widens the dot's activation operand, and
rounds only the final output.  The narrow path is untouched when the packed
range fits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.macaq as macaq_module
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization.macaq import macaq_quantize_weight


IN_FEATURES = 32
OUT_FEATURES = 1
# 4 * 2**14: finite FP32 (and BF16), but above FP16's 65504.
OVERFLOWING_WEIGHT = 65536.0
# 2**-28: FP32-exact, but it rounds to zero in FP16.
UNDERFLOWING_WEIGHT = 2.0**-28


def _pack(weight: float, dtype: torch.dtype) -> tuple[MacaqLinear, nn.Linear]:
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    linear.weight.data.fill_(weight)
    result = macaq_quantize_weight(
        linear.weight.detach(),
        torch.eye(IN_FEATURES),
        IN_FEATURES,
        rotation="none",
        pre_mode="none",
        do_flip=False,
        exp_screen=False,
    )
    layer = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rotation="none",
        pre_mode="none",
        activation_precision="w4a16",
        dtype=dtype,
        bias=False,
    )
    layer.pack(linear, macaq_result=result)
    layer.eval()
    return layer, linear


def _input(value: float, dtype: torch.dtype) -> torch.Tensor:
    x = torch.zeros(1, IN_FEATURES, dtype=dtype)
    x[0, 0] = value
    return x


def test_fp16_forward_keeps_a_finite_overflowing_weight_finite():
    layer, _ = _pack(OVERFLOWING_WEIGHT, torch.float16)

    # The checkpoint itself is exact in FP32; only the FP16 decode overflows.
    assert layer.dequantize(dtype=torch.float32)[0, 0].item() == OVERFLOWING_WEIGHT
    assert layer.dequantize(dtype=torch.float16)[0, 0].item() == float("inf")

    with torch.inference_mode():
        out = layer(_input(0.25, torch.float16))

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    # 0.25 * 65536 = 16384, exactly representable in FP16.
    assert out[0, 0].item() == 16384.0
    assert layer._packed_range_fits_dtype(torch.float16) is False


def test_fp16_forward_matches_the_fp32_reference():
    layer, linear = _pack(OVERFLOWING_WEIGHT, torch.float16)
    x = _input(0.25, torch.float16)

    with torch.inference_mode():
        out = layer(x)
        reference = linear(x.float()).to(torch.float16)

    assert torch.equal(out, reference)


def test_wide_decode_widens_both_dot_operands():
    layer, _ = _pack(OVERFLOWING_WEIGHT, torch.float16)
    seen = []
    original = macaq_module.linear_fp32_reduce

    def tracked(x, weight, bias=None):
        seen.append((x.dtype, weight.dtype))
        return original(x, weight, bias)

    macaq_module.linear_fp32_reduce = tracked
    try:
        with torch.inference_mode():
            layer(_input(0.25, torch.float16))
    finally:
        macaq_module.linear_fp32_reduce = original

    assert seen == [(torch.float32, torch.float32)]


def test_in_range_weight_still_decodes_at_the_activation_dtype():
    layer, _ = _pack(1.0, torch.float16)
    seen = []
    original = macaq_module.linear_fp32_reduce

    def tracked(x, weight, bias=None):
        seen.append((x.dtype, weight.dtype))
        return original(x, weight, bias)

    macaq_module.linear_fp32_reduce = tracked
    try:
        with torch.inference_mode():
            layer(_input(2.0, torch.float16))
    finally:
        macaq_module.linear_fp32_reduce = original

    assert layer._packed_range_fits_dtype(torch.float16) is True
    assert seen == [(torch.float16, torch.float16)]


def test_bf16_keeps_the_narrow_decode_for_an_fp16_overflowing_weight():
    # Every FP32-finite MXFP4 magnitude is also finite in BF16, so only the
    # FP16 answer may select the wide decode.
    layer, _ = _pack(OVERFLOWING_WEIGHT, torch.bfloat16)

    assert layer._packed_range_fits_dtype(torch.bfloat16) is True
    assert layer._packed_range_fits_by_dtype == {
        torch.float16: False,
        torch.bfloat16: True,
    }


def test_fp16_underflowing_weight_decodes_wide():
    layer, linear = _pack(UNDERFLOWING_WEIGHT, torch.float16)

    assert layer.dequantize(dtype=torch.float32)[0, 0].item() == UNDERFLOWING_WEIGHT
    assert layer.dequantize(dtype=torch.float16)[0, 0].item() == 0.0

    x = _input(65504.0, torch.float16)
    with torch.inference_mode():
        out = layer(x)
        reference = linear(x.float()).to(torch.float16)

    assert torch.isfinite(out).all()
    assert out[0, 0].item() > 0
    assert torch.equal(out, reference)
    assert layer._packed_range_fits_dtype(torch.float16) is False


def test_state_dict_load_refreshes_the_decode_range():
    layer, _ = _pack(OVERFLOWING_WEIGHT, torch.float16)
    restored = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rotation="none",
        pre_mode="none",
        activation_precision="w4a16",
        dtype=torch.float16,
        bias=False,
    )
    restored.load_state_dict(layer.state_dict())
    restored.eval()

    assert restored._packed_range_fits_dtype(torch.float16) is False
    with torch.inference_mode():
        out = restored(_input(0.25, torch.float16))

    assert torch.isfinite(out).all()
    assert out[0, 0].item() == 16384.0
