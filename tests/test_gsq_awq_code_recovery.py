# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GSQ AWQ code recovery must name the code the real packer stores.

The AWQ packers build the affine offset from the checkpoint-stored scale in
FP32, divide by that same scale in FP32, round, and saturate. GSQ scores hard
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


def _packed_module(cls, weight, scales, zeros):
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False,
                       device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
    module = _bare(cls)
    module.pack(linear, scales.clone(), zeros.clone())
    return module


def _pack(cls, weight, scales, zeros):
    return _packed_module(cls, weight, scales, zeros).qweight


def _unpack(module, packing):
    """Read every code from the actual native packed layout independently."""
    words = module.qweight.long()
    if packing == "awq_gemv_fast":
        out = torch.arange(OUT_FEATURES, device=words.device)[:, None]
        col = torch.arange(IN_FEATURES, device=words.device)[None, :]
        within = col % 32
        transposed = ((within // 2) % 4) * 8 + (within // 8) * 2 + within % 2
        permuted = (transposed // 8) * 8 + (transposed % 8) // 2 + (transposed % 2) * 4
        index = (col % 64) // 32 * 32 + permuted
        word_col = col // 64 * 64 + out % 4 * 16 + index // 4
        return (words[out // 4, word_col] >> (4 * (index % 4))) & 15
    shifts = torch.arange(8, device=words.device) * 4
    codes = (words.unsqueeze(-1) >> shifts) & 15
    if packing == "awq_gemm":
        codes = codes[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        return codes.reshape(IN_FEATURES, OUT_FEATURES).T
    return codes.reshape(OUT_FEATURES, IN_FEATURES)


def _oracle(weight, scales, zeros, groups, scale_dtype):
    """FP64 oracle with explicit rounding at each specified FP32 operation."""
    stored_source = scales.to(scale_dtype)
    stored = stored_source.double()[:, groups]
    offset = (zeros.float().double() * stored_source.float().double()).float().double()
    numerator = (weight.float().double() + offset[:, groups]).float().double()
    quotient = (numerator / stored).float().double()
    return quotient.round().clamp(0, (1 << BITS) - 1)


def _decode(codes, scales, zeros, groups, scale_dtype, packing):
    """Decode with the checkpoint storage the packer writes."""
    stored = scales.to(scale_dtype).float()
    if packing == "awq_gemv_fast":
        offset = -(stored * zeros.float()).half().float()
        return stored[:, groups] * codes + offset[:, groups]
    return stored[:, groups] * (codes - zeros.float()[:, groups])


def _tensors(weight_value, scale_value, zero_value, scale_dtype, *, device="cpu", weight_dtype=torch.float16):
    weight = torch.full((OUT_FEATURES, IN_FEATURES), weight_value, dtype=weight_dtype, device=device)
    scales = torch.full((OUT_FEATURES, 1), scale_value, dtype=scale_dtype, device=device)
    zeros = torch.full((OUT_FEATURES, 1), zero_value, dtype=torch.int32, device=device)
    groups = torch.arange(IN_FEATURES, dtype=torch.int32, device=device) // GROUP_SIZE
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
    ("above_grid_saturation", 100.0, 1.0, 8),
    ("positive_zero", 0.0, 1.0, 0),
    ("negative_zero", -0.0, 1.0, 0),
)


@pytest.mark.parametrize("packing,cls", PACKERS)
@pytest.mark.parametrize("name,weight_value,scale_value,zero_value", CASES)
def test_gsq_awq_codes_match_fp64_oracle_and_packed_bytes(
    device, packing, cls, name, weight_value, scale_value, zero_value
):
    weight, scales, zeros, groups = _tensors(
        weight_value, scale_value, zero_value, torch.float16, device=device,
    )
    predicted = affine_codes(
        weight, scales, zeros, groups, BITS, packing=packing, scale_dtype=torch.float16
    )
    oracle = _oracle(weight, scales, zeros, groups, torch.float16)
    assert torch.equal(predicted, oracle), name
    assert torch.equal(predicted, _unpack(_packed_module(cls, weight, scales, zeros), packing)), name

    # The exported weight must reproduce the source payload byte for byte.
    exported = _decode(
        predicted, scales, zeros, groups, torch.float16, packing
    ).to(weight.dtype)
    assert torch.equal(_pack(cls, exported, scales, zeros), _pack(cls, weight, scales, zeros)), name


def test_awq_gemm_resolves_wider_source_scale_to_fp16_storage(device):
    """A wider source scale still packs into the FP16 scale buffer."""
    source = torch.full((OUT_FEATURES, 1), 1.4 * 2.0**-24, dtype=torch.float32, device=device)
    stored = source.to(torch.float16)
    assert stored.flatten()[0].item() == 2.0**-24
    zeros = torch.full((OUT_FEATURES, 1), 8, dtype=torch.int32, device=device)
    groups = torch.arange(IN_FEATURES, dtype=torch.int32, device=device) // GROUP_SIZE

    # The FP32 source scale rounds to the FP16 subnormal the checkpoint stores.
    # The affine offset must use that stored grid, so an exact zero weight
    # keeps the zero point instead of being pushed off it by the wider scale.
    weights = torch.zeros((OUT_FEATURES, IN_FEATURES), dtype=torch.float16, device=device)
    predicted = affine_codes(
        weights, source, zeros, groups, BITS, packing="awq_gemm", scale_dtype=torch.float32
    )
    oracle = _oracle(weights, source, zeros, groups, torch.float16)
    assert torch.equal(predicted, oracle)
    assert torch.unique(predicted).tolist() == [8.0]

    exported = _decode(
        predicted, source, zeros, groups, torch.float16, "awq_gemm"
    ).to(weights.dtype)
    assert torch.equal(
        _pack(AwqGEMMLinear, exported, source, zeros),
        _pack(AwqGEMMLinear, weights, source, zeros),
    )


@pytest.mark.parametrize("packing,cls", PACKERS)
def test_refine_affine_scalar_accepts_finite_fp16_overflow(device, packing, cls):
    """A finite FP16 sum may exceed FP16 range without being an illegal code."""
    weight, scales, zeros, groups = _tensors(30000.0, 5000.0, 8, torch.float16, device=device)
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


@pytest.fixture(params=["cpu", "cuda:0", "cuda:1"])
def device(request):
    target = torch.device(request.param)
    if target.type == "cuda" and torch.cuda.device_count() <= target.index:
        pytest.skip("A second accelerator is required" if target.index else "An accelerator is required")
    yield target
    if target.type == "cuda":
        torch.cuda.synchronize(target)


@pytest.mark.parametrize("packing,cls", PACKERS)
def test_wide_source_codes_follow_each_fp32_rounding_step(device, packing, cls):
    # A raw FP64 quotient on the stored grid falls below 1.5, while the
    # packer's FP32 addition lands exactly on 1.5. The format specifies FP32
    # round-to-even: code 2.
    stored = torch.tensor(0.1, dtype=torch.float32).to(torch.float16)
    weight_below_half_step = torch.nextafter(
        0.5 * stored.float(), torch.full((), float("-inf"))
    ).item()
    weight, scales, zeros, groups = _tensors(
        weight_below_half_step, 0.1, 1, torch.float32, device=device, weight_dtype=torch.float32,
    )
    predicted = affine_codes(weight, scales, zeros, groups, BITS, packing=packing)
    expected = _oracle(weight, scales, zeros, groups, torch.float16)
    ideal = ((weight.double() + zeros.double()[:, groups] * scales.half().double()[:, groups])
             / scales.half().double()[:, groups]).round()
    assert torch.equal(predicted, expected)
    assert torch.unique(predicted).tolist() == [2.0]
    assert torch.unique(ideal).tolist() == [1.0]
    assert torch.equal(predicted, _unpack(_packed_module(cls, weight, scales, zeros), packing))


@pytest.mark.parametrize("packing,cls", PACKERS)
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("chunked", [False, True])
def test_learned_scale_score_matches_actual_packed_export(device, packing, cls, source_dtype, chunked):
    gen = torch.Generator().manual_seed(1)
    # Repeated small rows keep the CPU and accelerator optimization tractable
    # while exercising every word in a valid native packing shape.
    sample = torch.randn((4, 16), generator=gen).to(source_dtype)
    weight = sample.repeat(16, 8).to(device)
    scales = torch.full((OUT_FEATURES, 1), 0.1, dtype=source_dtype, device=device)
    zeros = torch.full_like(scales, 8)
    groups = torch.zeros(IN_FEATURES, dtype=torch.int32, device=device)
    target = weight.float() * 1.15
    scale_dtype = source_dtype if packing == "awq_gemm" and source_dtype != torch.float32 else torch.float16
    result = refine_affine_scalar(
        weight, scales, zeros, groups, target=target, bits=BITS, packing=packing, scale_dtype=scale_dtype,
        config=GSQConfig(enabled=True, steps=4, candidates=4, seed=1, learn_scales=True, learning_rate=0.02,
                         max_candidate_bytes=(16 if chunked else OUT_FEATURES) * IN_FEATURES * 4 * 4),
    )
    assert result.after < result.before
    assert not torch.equal(result.scales, scales)
    assert result.scales.dtype == source_dtype
    packed = _packed_module(cls, result.weight, result.scales, result.zeros)
    codes = _unpack(packed, packing).float()
    assert torch.equal(codes, _oracle(result.weight, result.scales, result.zeros, groups, scale_dtype))
    if packing == "awq_gemm":
        stored = packed.scales.T.float()
    elif packing == "awq_gemv":
        stored = packed.scales[:, :1].float()
    else:
        stored = packed.scales[:1].T.float()
    if packing == "awq_gemv_fast":
        decoded = stored[:, groups] * codes + packed.qzeros[:1].T.float()[:, groups]
    else:
        decoded = stored[:, groups] * (codes - result.zeros.float()[:, groups])
    measured = ((decoded.double() - target.double()).square().sum() / target.double().square().sum()).item()
    assert result.after == pytest.approx(measured, rel=1e-6, abs=1e-6)
    assert result.after <= result.before


@pytest.mark.parametrize("packing,cls", PACKERS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
def test_complete_packed_codes_at_dtype_thresholds(device, packing, cls, dtype):
    half = torch.tensor([-8.5, -7.5, -0.5, 0.5, 6.5, 7.5], dtype=dtype, device=device)
    samples = torch.cat((
        torch.nextafter(half, torch.full_like(half, -torch.inf)), half,
        torch.nextafter(half, torch.full_like(half, torch.inf)),
    ))
    weight = samples.repeat((OUT_FEATURES * IN_FEATURES + samples.numel() - 1) // samples.numel())
    weight = weight[:OUT_FEATURES * IN_FEATURES].reshape(OUT_FEATURES, IN_FEATURES)
    scales = torch.ones((OUT_FEATURES, 1), dtype=dtype, device=device)
    zeros = torch.full_like(scales, 8)
    groups = torch.zeros(IN_FEATURES, dtype=torch.int32, device=device)
    storage_dtype = dtype if packing == "awq_gemm" and dtype != torch.float32 else torch.float16
    predicted = affine_codes(weight, scales, zeros, groups, BITS, packing=packing, scale_dtype=storage_dtype)
    expected = _oracle(weight, scales, zeros, groups, storage_dtype)
    assert torch.equal(predicted, expected)
    assert torch.equal(predicted, _unpack(_packed_module(cls, weight, scales, zeros), packing))
