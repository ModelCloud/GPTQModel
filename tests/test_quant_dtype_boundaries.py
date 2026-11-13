# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Quantization must stay exact or bounded at dtype min/max.

This pins the boundary behavior that a scan for silent overflow/underflow
turned up. Each case here is something that can go wrong quietly: a packed code
spilling into a neighbour, a zero point truncated to the wrong width, a
dequantized value leaving its representable range, or an activation quantizer
turning a finite input into NaN. These cases run on CPU; the Marlin backend
comparisons are in the GPU test suite.
"""

import pytest
import torch

from gptqmodel.quantization.activation_floatx import (
    fp8_token_qdq,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq, pack_awq_columns, unpack_awq
from gptqmodel.quantization.quantizer import quantize
from gptqmodel.utils.nf12_codec import MAX_EXP_BASE, decode, encode


AWQ_BITS = (2, 3, 4, 8)


@pytest.mark.parametrize("bits", AWQ_BITS)
def test_packed_codes_round_trip_at_both_ends(bits):
    maxq = 2**bits - 1
    codes = torch.zeros(4, 64, dtype=torch.int32)
    codes[:, 0] = maxq
    codes[:, 1] = 0
    codes[:, 2] = maxq // 2
    codes[:, 3] = 1
    codes[:, 9] = maxq
    codes[:, 10] = maxq  # A 3-bit code straddles the first word boundary.
    codes[:, 11] = 1
    codes[:, 31] = maxq
    codes[:, 32] = 1
    codes[:, 63] = maxq  # Also check the last code in a partial 3-bit group.

    if bits == 3:
        packed, _ = pack_awq(codes, torch.zeros_like(codes), bits)
    else:
        packed = pack_awq_columns(codes, bits)
    back, _ = unpack_awq(packed, None, bits, count=codes.shape[1])
    assert torch.equal(back[:, : codes.shape[1]].to(torch.int32), codes)


def test_eight_bit_zero_point_keeps_its_full_width():
    """A 4-bit-width mask here would truncate 255 to 15 - the #3516 shape."""
    zeros = torch.full((32, 32), 255, dtype=torch.int32)
    packed = pack_awq_columns(zeros, 8)
    back, _ = unpack_awq(packed, None, 8, count=32)
    assert torch.equal(back.to(torch.int32), zeros)


@pytest.mark.parametrize("bits", (4, 8))
@pytest.mark.parametrize("zero_point", (0, 1, 7, 15, 127, 255))
def test_asymmetric_dequant_stays_inside_its_range(bits, zero_point):
    maxq = 2**bits - 1
    if zero_point > maxq:
        pytest.skip("zero point does not fit this width")
    scale_value = 0.1
    scale = torch.tensor([[scale_value]])
    zero = torch.tensor([[float(zero_point)]])
    x = torch.tensor([[-1e3, -1.0, 0.0, 1.0, 1e3]])

    deq = quantize(x, scale, zero, maxq, False)
    low, high = -zero_point * scale_value, (maxq - zero_point) * scale_value
    # float32 rounding in the dequant product lands within an ulp of the ends.
    tolerance = 1e-5 * max(abs(low), abs(high), 1.0)
    assert torch.isfinite(deq).all()
    assert float(deq.min()) >= low - tolerance
    assert float(deq.max()) <= high + tolerance


@pytest.mark.parametrize("bits", (4, 8))
def test_symmetric_dequant_error_is_half_a_scale(bits):
    maxq = 2**bits - 1
    scale = torch.tensor([[1.0 / maxq]])
    x = torch.tensor([[-1.0, 0.0, 1.0]])

    deq = quantize(x, scale, torch.zeros_like(scale), maxq, True)
    assert (deq - x).abs().max() <= 0.5 * scale + 1e-9


@pytest.mark.parametrize("bits", (4, 8))
def test_zero_scale_and_zero_weight_stay_finite(bits):
    maxq = 2**bits - 1
    deq = quantize(torch.zeros(1, 4), torch.zeros(1, 1), torch.zeros(1, 1), maxq, False)
    assert torch.isfinite(deq).all()
    assert float(deq.abs().max()) == 0.0


@pytest.mark.parametrize("value", (0.0, 1e-30, 1.0, 1e20))
def test_fp8_activation_quantizer_stays_finite(value):
    out = fp8_token_qdq(torch.full((2, 32), value, dtype=torch.float32))
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("value", (float("inf"), float("nan")))
def test_fp8_activation_quantizer_rejects_non_finite_input(value):
    x = torch.zeros(1, 32, dtype=torch.float32)
    x[0, 0] = value
    with pytest.raises(ValueError):
        fp8_token_qdq(x)


@pytest.mark.parametrize("value", (0.0, 1e-30, 1.0, 1e20))
def test_nvfp4_activation_quantizer_stays_finite(value):
    x = torch.full((2, 32), value, dtype=torch.float32)
    out = nvfp4_block_qdq(x, nvfp4_global_scale(x.abs().amax()))
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("value", (float("inf"), float("nan")))
def test_nvfp4_global_scale_rejects_non_finite_amax(value):
    with pytest.raises(ValueError):
        nvfp4_global_scale(torch.tensor(value))


# BF16 extremes: max finite, min normal, denormal, zero, and both infinities.
BF16_EXTREMES = (
    ("max", 3.3895313892515355e38),
    ("min_normal", 1.1754943508222875e-38),
    ("denormal", 9.183549615799121e-41),
    ("zero", 0.0),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
)


@pytest.mark.parametrize("label,value", BF16_EXTREMES)
def test_nf12_round_trip_is_lossless_at_bf16_extremes(label, value):
    """NF12-AD preserves BF16 bits, including signed infinity patterns."""
    w = torch.full((16, 64), value, dtype=torch.bfloat16)
    packed, escapes, base = encode(w)
    assert 0 <= base <= MAX_EXP_BASE
    back = decode(packed, escapes, base, w.shape[0], w.shape[1])
    assert torch.equal(back.view(torch.int16), w.view(torch.int16))
