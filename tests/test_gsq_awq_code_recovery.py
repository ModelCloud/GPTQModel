# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GSQ AWQ code recovery must name the code the real packer stores.

The AWQ packers build the affine offset from the source scales in FP32, divide
by the checkpoint-stored scale in FP32, round, and saturate. GSQ scores hard
candidates through the same reconstruction, so any drift in precision, operand
order or saturation changes which weights the optimizer evaluates.
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import affine_codes, refine_affine_scalar


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 128
BITS = 4

PACKERS = (
    ("awq_gemm", AwqGEMMLinear),
    ("awq_gemv", AwqGEMVLinear),
    ("awq_gemv_fast", AwqGEMVFastLinear),
)


def _bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits = BITS
    module.in_features = IN_FEATURES
    module.out_features = OUT_FEATURES
    module.group_size = GROUP_SIZE
    return module


def _pack(cls, weight, scales, zeros):
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    module = _bare(cls)
    module.pack(linear, scales.clone(), zeros.clone())
    return module.qweight


def _oracle(weight, scales, zeros, groups, scale_dtype):
    """Independent FP64 code oracle from the values the packer reads."""
    stored = scales.to(scale_dtype).double()[:, groups]
    offset = (zeros.double() * scales.double())[:, groups]
    return ((weight.double() + offset) / stored).round().clamp(0, (1 << BITS) - 1)


def _decode(codes, scales, zeros, groups, scale_dtype, packing):
    """Decode with the checkpoint storage the packer writes."""
    stored = scales.to(scale_dtype).float()
    if packing == "awq_gemv_fast":
        offset = -(stored * zeros.float()).half().float()
        return stored[:, groups] * codes + offset[:, groups]
    return stored[:, groups] * (codes - zeros.float()[:, groups])


def _tensors(weight_value, scale_value, zero_value, scale_dtype):
    weight = torch.full((OUT_FEATURES, IN_FEATURES), weight_value, dtype=torch.float16)
    scales = torch.full((OUT_FEATURES, 1), scale_value, dtype=scale_dtype)
    zeros = torch.full((OUT_FEATURES, 1), zero_value, dtype=torch.int32)
    groups = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    return weight, scales, zeros, groups


_HALF_STEP_SCALE = 0.0999755859375
_HALF_STEP_TIE = torch.tensor(0.04998779296875, dtype=torch.float16)
_HALF_STEP_BELOW = torch.nextafter(
    _HALF_STEP_TIE, torch.full((), float("-inf"), dtype=torch.float16)
)

CASES = (
    # One FP16 ULP above the mathematical half step: not a rounding tie.
    ("reported_half_step_above", 0.050018310546875, _HALF_STEP_SCALE, 8),
    # Exact half step: the packer and the FP64 oracle both round to even.
    ("half_step_tie", _HALF_STEP_TIE.item(), _HALF_STEP_SCALE, 8),
    ("half_step_below", _HALF_STEP_BELOW.item(), _HALF_STEP_SCALE, 8),
    # Finite FP16 operands whose FP32 sum leaves the FP16 range.
    ("finite_add_overflow", 30000.0, 5000.0, 8),
    # Saturation endpoint below zero.
    ("below_zero_saturation", -100.0, 1.0, 8),
)


@pytest.mark.parametrize("packing,cls", PACKERS)
@pytest.mark.parametrize("name,weight_value,scale_value,zero_value", CASES)
def test_gsq_awq_codes_match_fp64_oracle_and_packed_bytes(
    packing, cls, name, weight_value, scale_value, zero_value
):
    weight, scales, zeros, groups = _tensors(weight_value, scale_value, zero_value, torch.float16)
    predicted = affine_codes(
        weight, scales, zeros, groups, BITS, packing=packing, scale_dtype=torch.float16
    )
    oracle = _oracle(weight, scales, zeros, groups, torch.float16)
    assert torch.equal(predicted, oracle), name

    # The exported weight must reproduce the source payload byte for byte.
    exported = _decode(
        predicted, scales, zeros, groups, torch.float16, packing
    ).to(weight.dtype)
    assert torch.equal(_pack(cls, exported, scales, zeros), _pack(cls, weight, scales, zeros)), name


def test_awq_gemm_resolves_wider_source_scale_to_fp16_storage():
    """A wider source scale still packs into the FP16 scale buffer."""
    source = torch.full((OUT_FEATURES, 1), 0.1, dtype=torch.float32)
    stored = source.to(torch.float16)
    zeros = torch.full((OUT_FEATURES, 1), 8, dtype=torch.int32)
    groups = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE

    # The correct quotient (FP16 divisor) sits just above a half step while
    # dividing by the FP32 source would land below it.
    sample = (8.501 * stored.double() - 8 * source.double()).to(torch.float16)
    weights = sample.expand(OUT_FEATURES, IN_FEATURES)
    predicted = affine_codes(
        weights, source, zeros, groups, BITS, packing="awq_gemm", scale_dtype=torch.float32
    )
    oracle = _oracle(weights, source, zeros, groups, torch.float16)
    assert torch.equal(predicted, oracle)
    assert torch.unique(predicted).tolist() == [9.0]

    legacy = (
        (weights.double() + zeros.double() * source.double()[:, groups])
        / source.double()[:, groups]
    ).round().clamp(0, (1 << BITS) - 1)
    assert not torch.equal(predicted, legacy)

    exported = _decode(
        predicted, source, zeros, groups, torch.float16, "awq_gemm"
    ).to(weights.dtype)
    assert torch.equal(
        _pack(AwqGEMMLinear, exported, source, zeros),
        _pack(AwqGEMMLinear, weights, source, zeros),
    )


@pytest.mark.parametrize("packing,cls", PACKERS)
def test_refine_affine_scalar_accepts_finite_fp16_overflow(packing, cls):
    """A finite FP16 sum may exceed FP16 range without being an illegal code."""
    weight, scales, zeros, groups = _tensors(30000.0, 5000.0, 8, torch.float16)
    result = refine_affine_scalar(
        weight,
        scales,
        zeros.float(),
        groups,
        target=weight.float(),
        bits=BITS,
        config=GSQConfig(enabled=True, steps=1, candidates=2, seed=0),
        packing=packing,
        scale_dtype=torch.float16,
    )
    assert torch.isfinite(result.weight).all()
    codes = affine_codes(
        result.weight, result.scales, result.zeros, groups, BITS, packing=packing
    )
    assert bool(((codes >= 0) & (codes < 1 << BITS)).all())
    assert codes.max().item() == 14
    assert torch.equal(
        _pack(cls, result.weight, result.scales, result.zeros),
        _pack(cls, weight, scales, zeros),
    )
