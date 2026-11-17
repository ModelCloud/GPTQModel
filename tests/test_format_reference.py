# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from gptqmodel.quantization.format_reference import (
    FORMATS,
    FormatGrid,
    decode_elements,
    encode_elements,
    fit_grid,
    format_spec,
    quantize_rtn,
)


def scalar_decode(code, element):
    """Independent Python specification arithmetic, not the implementation LUT."""
    if element == "int4":
        return float(code)
    exponent_bits, mantissa_bits, bias = {"e2m1": (2, 1, 1), "e4m3": (4, 3, 7),
                                           "e5m2": (5, 2, 15)}[element]
    sign = -1 if code >> (exponent_bits + mantissa_bits) else 1
    exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1)
    fraction = code & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        return sign * math.ldexp(fraction / (1 << mantissa_bits), 1 - bias)
    return sign * math.ldexp(1 + fraction / (1 << mantissa_bits), exponent - bias)


def scalar_encode(value, element):
    last = {"e2m1": 7, "e4m3": 126, "e5m2": 123}[element]
    magnitude = abs(value)
    codes = range(last + 1)
    code = min(codes, key=lambda c: (abs(magnitude - scalar_decode(c, element)), c % 2))
    return code | ((8 if element == "e2m1" else 128) if math.copysign(1., value) < 0 else 0)


@pytest.mark.parametrize("element,last", [("e2m1", 7), ("e4m3", 126), ("e5m2", 123)])
def test_complete_finite_codebook_and_round_trip(element, last):
    sign = 8 if element == "e2m1" else 128
    codes = torch.tensor(list(range(last + 1)) + list(range(sign, sign + last + 1)), dtype=torch.uint8)
    expected = torch.tensor([scalar_decode(int(c), element) for c in codes], dtype=torch.float64)
    actual = decode_elements(codes, element)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    assert torch.equal(encode_elements(expected, element), codes)


@pytest.mark.parametrize("element,last", [("e2m1", 7), ("e4m3", 126), ("e5m2", 123)])
def test_every_midpoint_and_nextafter_neighbours(element, last):
    for code in range(last):
        midpoint = (scalar_decode(code, element) + scalar_decode(code + 1, element)) / 2
        tie = code if code % 2 == 0 else code + 1
        positive = torch.tensor([math.nextafter(midpoint, -math.inf), midpoint,
                                 math.nextafter(midpoint, math.inf)], dtype=torch.float64)
        for sign in (1, -1):
            sign_bit = 0 if sign == 1 else (8 if element == "e2m1" else 128)
            expected = torch.tensor([code | sign_bit, tie | sign_bit, (code + 1) | sign_bit], dtype=torch.uint8)
            assert torch.equal(encode_elements(sign * positive, element), expected)


@pytest.mark.parametrize("element,last", [("e2m1", 7), ("e4m3", 126), ("e5m2", 123)])
def test_saturating_finite_overflow_and_nonfinite_input_rejection(element, last):
    sign = 8 if element == "e2m1" else 128
    actual = encode_elements(torch.tensor([1e100, -1e100], dtype=torch.float64), element)
    assert actual.tolist() == [last, last | sign]
    with pytest.raises(ValueError, match="finite CPU"):
        encode_elements(torch.tensor([float("inf")]), element)


@pytest.mark.parametrize("element,dtype,last", [("e4m3", torch.float8_e4m3fn, 126),
                                               ("e5m2", torch.float8_e5m2, 123)])
def test_emulation_matches_native_torch_fp8_when_available(element, dtype, last):
    # Stay inside the finite range: native casts need not use saturation.
    maximum = scalar_decode(last, element)
    x = torch.linspace(-maximum, maximum, 10001, dtype=torch.float32)
    assert torch.equal(encode_elements(x, element), x.to(dtype).view(torch.uint8))


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_packed_payload_independent_decode_and_second_consumer(spec, dtype):
    rng = torch.Generator().manual_seed(27)
    source = torch.randn(64, 8, generator=rng).to(dtype).T
    assert not source.is_contiguous()
    packed = quantize_rtn(source, spec.name)
    grid = packed.grid
    # Decode the actual stored words, not grid.decode or FormatTensor.codes.
    words = packed.payload.tolist()
    expected = torch.empty(source.shape, dtype=torch.float64)
    for row in range(source.shape[0]):
        for col in range(source.shape[1]):
            if spec.element == "int4":
                code = (words[row][col // 8] >> (4 * (col % 8))) & 15
            elif spec.bits == 4:
                code = (words[row][col // 2] >> (4 * (col % 2))) & 15
            else:
                code = words[row][col]
            group = col // spec.block_size if spec.block_size else 0
            raw_scale = grid.scales[row if spec.block_size else 0, group].item()
            if spec.scale == "e8m0":
                scale = math.ldexp(1., raw_scale - 127)
            elif spec.scale == "e4m3":
                scale = scalar_decode(raw_scale, "e4m3") * grid.tensor_scale.item()
            else:
                scale = raw_scale
            zero = grid.zeros[row, group].item() if grid.zeros is not None else 0
            expected[row, col] = (scalar_decode(code, spec.element) - zero) * scale
    torch.testing.assert_close(packed.dequantize(), expected, rtol=0, atol=0)
    repacked = grid.pack(packed.codes())
    assert torch.equal(repacked.payload, packed.payload)
    inputs = torch.randn(11, 64, generator=rng, dtype=torch.float64)
    torch.testing.assert_close(inputs @ repacked.dequantize().T, inputs @ expected.T, rtol=0, atol=0)
    assert set(vars(packed)) == {"grid", "payload"}
    if spec.name == "int4":
        assert packed.payload.dtype == torch.int32
    else:
        assert packed.payload.dtype == torch.uint8


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
def test_zero_tiny_outliers_and_rtn_grid_idempotence(spec):
    weight = torch.zeros(3, 64, dtype=torch.float64)
    weight[1, :8] = torch.tensor([1e-40, -1e-40, .25, -.25, 8, -8, 1e6, -1e6], dtype=torch.float64)
    weight[2] = 1e-40
    packed = quantize_rtn(weight, spec.name)
    assert torch.isfinite(packed.dequantize()).all()
    assert not packed.dequantize()[0].any()
    assert torch.equal(packed.grid.pack(packed.grid.encode(packed.dequantize())).payload, packed.payload)


def test_e8m0_all_finite_scales_and_invalid_reserved_code():
    spec = format_spec("mxfp4")
    scales = torch.arange(255, dtype=torch.uint8).reshape(255, 1)
    grid = FormatGrid(spec, (255, 32), scales)
    expected = torch.tensor([math.ldexp(1., c - 127) for c in range(255)], dtype=torch.float64)
    torch.testing.assert_close(grid.expanded()[:, 0], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="255 is NaN"):
        FormatGrid(spec, (1, 32), torch.tensor([[255]], dtype=torch.uint8))


def test_scale_storage_rounding_happens_before_payload_rounding():
    x = torch.linspace(-3.12345, 2.07891, 64, dtype=torch.float64).repeat(2, 1)
    grid = fit_grid(x, "nvfp4")
    assert grid.tensor_scale.dtype == torch.float32
    assert grid.scales.dtype == torch.uint8
    for row in range(2):
        for col in range(64):
            scale = scalar_decode(grid.scales[row, col // 16].item(), "e4m3") * grid.tensor_scale.item()
            assert grid.encode(x)[row, col].item() == scalar_encode(x[row, col].item() / scale, "e2m1")


@pytest.mark.parametrize("name,maximum", [("mxfp4", 6), ("mxfp8_e4m3", 448), ("mxfp8_e5m2", 57344)])
@pytest.mark.parametrize("exponent", [-127, -10, 0, 10, 127])
def test_e8m0_scale_selection_at_power_boundary_neighbours(name, maximum, exponent):
    boundary = math.ldexp(float(maximum), exponent)
    for value, expected_exp in ((math.nextafter(boundary, 0.), exponent), (boundary, exponent),
                                (math.nextafter(boundary, math.inf), min(exponent + 1, 127))):
        weight = torch.full((1, 32), value, dtype=torch.float64)
        grid = fit_grid(weight, name)
        assert grid.scales.item() == expected_exp + 127


@pytest.mark.parametrize("name", ["nvfp8", "unknown"])
def test_unknown_format_is_not_silently_aliased(name):
    with pytest.raises(ValueError, match="Unknown reference format"):
        quantize_rtn(torch.ones(2, 64), name)


@pytest.mark.parametrize("weight", [torch.ones(2, 33), torch.empty(0, 32),
                                    torch.full((2, 32), float("nan")), torch.ones(2, 32, dtype=torch.int32)])
def test_invalid_weights_fail_before_quantization(weight):
    with pytest.raises(ValueError):
        quantize_rtn(weight, "mxfp4")


@pytest.mark.parametrize("element,code", [("e2m1", 16), ("e4m3", 127), ("e5m2", 124), ("int4", 16)])
def test_nonfinite_and_invalid_codes_rejected(element, code):
    with pytest.raises(ValueError):
        decode_elements(torch.tensor([code], dtype=torch.uint8), element)
