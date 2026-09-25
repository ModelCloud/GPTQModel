# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for Metal GGUF block packing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

_source = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_gguf.py"
_spec = importlib.util.spec_from_file_location("gptqmodel_mlx_gguf_test", _source)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def _torch_gguf_q4_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 32)
    selected = blocks.gather(1, blocks.abs().argmax(dim=1, keepdim=True))
    scale = selected / -8.0
    reciprocal = torch.where(scale == 0, 0, 1.0 / scale)
    # GGUF's Q4_0 reference multiplies in float64 after float32 scale
    # selection and reciprocal calculation.
    codes = torch.trunc(blocks.double() * reciprocal.double() + 8.5)
    codes = codes.clamp(0, 15).to(torch.uint8)
    payload = codes[:, :16] | (codes[:, 16:] << 4)
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((scale_bytes, payload.numpy()), axis=1)
    return packed.reshape(rows, columns // 32 * packed.shape[1])


def _torch_gguf_q8_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 32)
    scale = blocks.abs().amax(dim=1, keepdim=True) / 127.0
    reciprocal = torch.where(scale == 0, 0, 1.0 / scale)
    codes = torch.round(blocks * reciprocal).clamp(-128, 127).to(torch.int8)
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((scale_bytes, codes.view(torch.uint8).numpy()), axis=1)
    return packed.reshape(rows, columns // 32 * 34)


def _torch_gguf_q1_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 128)
    magnitudes = blocks.abs()
    # The existing checkpoint packer accumulates eight float32 lanes before
    # combining them in this order. A generic mean changes fp16 scale bytes
    # for blocks whose true mean is near a rounding midpoint.
    lanes = magnitudes[:, :8]
    for offset in range(8, 128, 8):
        lanes = lanes + magnitudes[:, offset : offset + 8]
    total = ((lanes[:, 0] + lanes[:, 1]) + (lanes[:, 2] + lanes[:, 3])) + (
        (lanes[:, 4] + lanes[:, 5]) + (lanes[:, 6] + lanes[:, 7])
    )
    scale = (total / 128.0).to(torch.float16)
    scale_bytes = scale.numpy().view(np.uint8).reshape(-1, 2)
    sign = (blocks >= 0).to(torch.uint8).reshape(-1, 16, 8)
    powers = (1 << torch.arange(8, dtype=torch.int32)).reshape(1, 1, 8)
    payload = (sign.to(torch.int32) * powers).sum(dim=2).to(torch.uint8)
    packed = np.concatenate((scale_bytes, payload.numpy()), axis=1)
    return packed.reshape(rows, columns // 128 * 18)


def _torch_gguf_q2_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 64)
    scale = blocks.abs().amax(dim=1, keepdim=True)
    inverse = torch.where(scale == 0, 0, 1.0 / scale)
    normalized = blocks * inverse
    magnitude = normalized.abs()
    whole = torch.floor(magnitude)
    rounded = torch.sign(normalized) * (
        whole + torch.floor(2 * (magnitude - whole))
    )
    codes = (rounded.to(torch.int8) + 1).clamp(0, 3).to(torch.uint8)
    powers = (1 << (2 * torch.arange(4, dtype=torch.int32))).reshape(1, 1, 4)
    payload = (
        codes.reshape(-1, 16, 4).to(torch.int32) * powers
    ).sum(dim=2).to(torch.uint8)
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((scale_bytes, payload.numpy()), axis=1)
    return packed.reshape(rows, columns // 64 * 18)


def _torch_gguf_q4_k_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 8, 32)
    minima = (-blocks.amin(dim=-1)).clamp(min=0)
    scales = (blocks.amax(dim=-1) + minima) / 15.0
    base = scales.amax(dim=-1, keepdim=True) / 63.0
    min_base = minima.amax(dim=-1, keepdim=True) / 63.0
    scale_codes = (
        torch.where(base > 0, torch.round(scales / base), 0)
        .clamp(0, 63)
        .to(torch.uint8)
    )
    min_codes = (
        torch.where(min_base > 0, torch.round(minima / min_base), 0)
        .clamp(0, 63)
        .to(torch.uint8)
    )
    steps = base * scale_codes.float()
    biases = min_base * min_codes.float()
    codes = torch.where(
        steps[..., None] > 0,
        torch.round((blocks + biases[..., None]) / steps[..., None]),
        0,
    ).clamp(0, 15).to(torch.uint8)
    scale_bytes = base.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    min_bytes = min_base.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed_scales = torch.cat(
        (
            (scale_codes[:, :4] & 63) | ((scale_codes[:, 4:] & 48) << 2),
            (min_codes[:, :4] & 63) | ((min_codes[:, 4:] & 48) << 2),
            (scale_codes[:, 4:] & 15) | ((min_codes[:, 4:] & 15) << 4),
        ),
        dim=-1,
    ).numpy()
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(-1, 128).numpy()
    packed = np.concatenate((scale_bytes, min_bytes, packed_scales, payload), axis=-1)
    return packed.reshape(rows, columns // 256 * 144)


def _torch_gguf_q5_k_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 8, 32)
    minima = (-blocks.amin(dim=-1)).clamp(min=0)
    scales = (blocks.amax(dim=-1) + minima) / 31.0
    base = scales.amax(dim=-1, keepdim=True) / 63.0
    min_base = minima.amax(dim=-1, keepdim=True) / 63.0
    scale_codes = torch.where(
        base > 0, torch.round(scales / base), 0
    ).clamp(0, 63).to(torch.uint8)
    min_codes = torch.where(
        min_base > 0, torch.round(minima / min_base), 0
    ).clamp(0, 63).to(torch.uint8)
    steps = base * scale_codes.float()
    biases = min_base * min_codes.float()
    codes = torch.where(
        steps[..., None] > 0,
        torch.round((blocks + biases[..., None]) / steps[..., None]),
        0,
    ).clamp(0, 31).to(torch.uint8)
    scale_bytes = base.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    min_bytes = min_base.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed_scales = torch.cat(
        (
            (scale_codes[:, :4] & 63) | ((scale_codes[:, 4:] & 48) << 2),
            (min_codes[:, :4] & 63) | ((min_codes[:, 4:] & 48) << 2),
            (scale_codes[:, 4:] & 15) | ((min_codes[:, 4:] & 15) << 4),
        ), dim=-1,
    ).numpy()
    bit_positions = torch.arange(8, dtype=torch.uint8).reshape(1, 8, 1)
    high = (((codes >> 4) & 1) << bit_positions).sum(dim=1).to(torch.uint8).numpy()
    low = ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4))
    payload = low.reshape(-1, 128).numpy()
    packed = np.concatenate(
        (scale_bytes, min_bytes, packed_scales, high, payload), axis=-1
    )
    return packed.reshape(rows, columns // 256 * 176)


def _torch_gguf_q6_k_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 16, 16)
    scales = blocks.abs().amax(dim=-1) / 31.0
    base = scales.amax(dim=-1, keepdim=True) / 127.0
    scale_codes = torch.where(
        base > 0, torch.round(scales / base), 0
    ).clamp(0, 127).to(torch.uint8)
    steps = base * scale_codes.float()
    codes = torch.where(
        steps[..., None] > 0,
        torch.round(blocks / steps[..., None]),
        0,
    ).clamp(-32, 31).to(torch.int16)
    raw = (codes + 32).to(torch.uint8).reshape(-1, 2, 4, 32)
    low = torch.cat(
        (
            (raw[:, :, 0] & 15) | ((raw[:, :, 2] & 15) << 4),
            (raw[:, :, 1] & 15) | ((raw[:, :, 3] & 15) << 4),
        ), dim=-1,
    ).reshape(-1, 128).numpy()
    high = torch.zeros((raw.shape[0], 2, 32), dtype=torch.uint8)
    for lane in range(4):
        high |= ((raw[:, :, lane] >> 4) & 3) << (2 * lane)
    scale_bytes = scale_codes.reshape(-1, 16).numpy()
    base_bytes = base.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate(
        (low, high.reshape(-1, 64).numpy(), scale_bytes, base_bytes), axis=-1
    )
    return packed.reshape(rows, columns // 256 * 210)


def _torch_gguf_tq1_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 256)
    scale = blocks.abs().amax(dim=-1, keepdim=True)
    inverse = torch.where(scale == 0, 0, 1.0 / scale)
    normalized = blocks * inverse
    magnitude = normalized.abs()
    whole = torch.floor(magnitude)
    rounded = torch.sign(normalized) * (
        whole + torch.floor(2 * (magnitude - whole))
    )
    codes = (rounded.to(torch.int8) + 1).to(torch.int32)
    powers = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)
    first = (
        codes[:, :160].reshape(-1, 5, 32) * powers.reshape(1, 5, 1)
    ).sum(dim=1)
    second = (
        codes[:, 160:240].reshape(-1, 5, 16) * powers.reshape(1, 5, 1)
    ).sum(dim=1)
    last = (
        codes[:, 240:].reshape(-1, 4, 4) * powers[:4].reshape(1, 4, 1)
    ).sum(dim=1)
    trits = torch.cat((first, second, last), dim=1)
    payload = ((trits * 256 + 242) // 243).to(torch.uint8).numpy()
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((payload, scale_bytes), axis=-1)
    return packed.reshape(rows, columns // 256 * 54)


def _torch_gguf_tq2_0_oracle(weight):
    rows, columns = weight.shape
    blocks = torch.from_numpy(weight).reshape(-1, 256)
    scale = blocks.abs().amax(dim=-1, keepdim=True)
    inverse = torch.where(scale == 0, 0, 1.0 / scale)
    normalized = blocks * inverse
    magnitude = normalized.abs()
    whole = torch.floor(magnitude)
    rounded = torch.sign(normalized) * (
        whole + torch.floor(2 * (magnitude - whole))
    )
    codes = (rounded.to(torch.int8) + 1).to(torch.int32).reshape(-1, 2, 4, 32)
    powers = (1 << (2 * torch.arange(4, dtype=torch.int32))).reshape(1, 1, 4, 1)
    payload = (codes * powers).sum(dim=2).to(torch.uint8).reshape(-1, 64).numpy()
    scale_bytes = scale.to(torch.float16).numpy().view(np.uint8).reshape(-1, 2)
    packed = np.concatenate((payload, scale_bytes), axis=-1)
    return packed.reshape(rows, columns // 256 * 66)


@pytest.mark.parametrize("rows,width", [(9, 256), (13, 512), (129, 1024)])
def test_gguf_tq2_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(115).standard_normal((rows, width)).astype(
        np.float32
    )
    weight[0] = 0
    weight[1, :256] = 2
    weight[2, :256] = np.linspace(-3, 3, 256, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "TQ2_0")
    expected = _torch_gguf_tq2_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_tq2_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(116).standard_normal((4, 512))).astype(
        dtype
    )
    expected = _torch_gguf_tq2_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "TQ2_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_tq2_0_packing_at_float32_rounding_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    maxima = np.array([0.03, 1.0, 20.0], dtype=np.float32)
    weight = np.zeros((18, 256), dtype=np.float32)
    row = 0
    for maximum in maxima:
        reciprocal = np.float32(1.0) / maximum
        for sign in (-1, 1):
            threshold = np.float32(sign * 0.5 / np.float64(reciprocal))
            for direction in (-np.inf, None, np.inf):
                weight[row, 0] = maximum
                value = (
                    threshold if direction is None
                    else np.nextafter(threshold, np.float32(direction))
                )
                weight[row, 1:] = value
                row += 1
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ2_0"))
    np.testing.assert_array_equal(actual, _torch_gguf_tq2_0_oracle(weight))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(weight, "TQ2_0"))


def test_gguf_tq2_0_packing_at_fp16_scale_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    midpoint = np.float32(1.0 + 2.0**-11)
    overflow = np.float32(65520.0)
    values = [
        np.float32(0.0),
        np.float32(-0.0),
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
        np.float32(2.0**-25),
        np.nextafter(overflow, np.float32(-np.inf)),
        overflow,
        np.nextafter(overflow, np.float32(np.inf)),
    ]
    weight = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 256, axis=1)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ2_0"))
    np.testing.assert_array_equal(actual, _torch_gguf_tq2_0_oracle(weight))
    with np.errstate(over="ignore"):
        expected = _fallback_gguf_quantize(weight, "TQ2_0")
    np.testing.assert_array_equal(actual, expected)


def test_gguf_tq2_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _fallback_gguf_quantize,
    )

    weight = np.random.default_rng(117).standard_normal((4, 512)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ2_0"))
    np.testing.assert_array_equal(packed, _fallback_gguf_quantize(weight, "TQ2_0"))
    decoded = _dequantize_gguf_tensor_numpy(packed, "TQ2_0")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_tq2_0_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 256"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128)), "TQ2_0")


@pytest.mark.parametrize("rows,width", [(9, 256), (13, 512), (129, 1024)])
def test_gguf_tq1_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(112).standard_normal((rows, width)).astype(
        np.float32
    )
    weight[0] = 0
    weight[1, :256] = 2
    weight[2, :256] = np.linspace(-3, 3, 256, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "TQ1_0")
    expected = _torch_gguf_tq1_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_tq1_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(113).standard_normal((4, 512))).astype(
        dtype
    )
    expected = _torch_gguf_tq1_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "TQ1_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_tq1_0_packing_at_float32_rounding_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    maxima = np.array([0.03, 1.0, 20.0], dtype=np.float32)
    weight = np.zeros((18, 256), dtype=np.float32)
    row = 0
    for maximum in maxima:
        reciprocal = np.float32(1.0) / maximum
        for sign in (-1, 1):
            threshold = np.float32(sign * 0.5 / np.float64(reciprocal))
            for direction in (-np.inf, None, np.inf):
                weight[row, 0] = maximum
                value = (
                    threshold if direction is None
                    else np.nextafter(threshold, np.float32(direction))
                )
                weight[row, 1:] = value
                row += 1
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ1_0"))
    np.testing.assert_array_equal(actual, _torch_gguf_tq1_0_oracle(weight))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(weight, "TQ1_0"))


def test_gguf_tq1_0_packing_at_fp16_scale_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    midpoint = np.float32(1.0 + 2.0**-11)
    overflow = np.float32(65520.0)
    values = [
        np.float32(0.0),
        np.float32(-0.0),
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
        np.float32(2.0**-25),
        np.nextafter(overflow, np.float32(-np.inf)),
        overflow,
        np.nextafter(overflow, np.float32(np.inf)),
    ]
    weight = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 256, axis=1)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ1_0"))
    np.testing.assert_array_equal(actual, _torch_gguf_tq1_0_oracle(weight))
    with np.errstate(over="ignore"):
        expected = _fallback_gguf_quantize(weight, "TQ1_0")
    np.testing.assert_array_equal(actual, expected)


def test_gguf_tq1_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _fallback_gguf_quantize,
    )

    weight = np.random.default_rng(114).standard_normal((4, 512)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "TQ1_0"))
    np.testing.assert_array_equal(packed, _fallback_gguf_quantize(weight, "TQ1_0"))
    decoded = _dequantize_gguf_tensor_numpy(packed, "TQ1_0")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_tq1_0_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 256"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128)), "TQ1_0")


@pytest.mark.parametrize("rows,width", [(9, 256), (13, 512), (129, 1024)])
def test_gguf_q6_k_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(108).standard_normal((rows, width)).astype(
        np.float32
    )
    weight[0] = 0
    weight[1, :256] = 2
    weight[2, :256] = np.linspace(-3, 3, 256, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q6_K")
    expected = _torch_gguf_q6_k_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q6_k_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(109).standard_normal((4, 512))).astype(
        dtype
    )
    expected = _torch_gguf_q6_k_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q6_K")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q6_k_packing_one_ulp_across_signed_code_thresholds():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    weight = np.zeros((183, 16, 16), dtype=np.float32)
    weight[:, :, 0] = -1.0
    weight[:, :, 1] = 1.0
    step = np.float32(np.float32(1.0 / 31.0) / np.float32(127.0)) * np.float32(127.0)
    for row, code in enumerate(range(-30, 31)):
        threshold = np.float32((code + 0.5) * float(step))
        weight[3 * row, :, 2] = np.nextafter(threshold, np.float32(-np.inf))
        weight[3 * row + 1, :, 2] = threshold
        weight[3 * row + 2, :, 2] = np.nextafter(threshold, np.float32(np.inf))
    flat = weight.reshape(183, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q6_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q6_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q6_K"))


def test_gguf_q6_k_packing_at_scale_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    rng = np.random.default_rng(111)
    rows = 3 * 126
    weight = np.zeros((rows, 16, 16), dtype=np.float32)
    weight[:, 0, 0] = 2.0
    weight[:, 0, 1] = -2.0
    for group in range(1, 16):
        code = np.repeat(rng.permutation(np.arange(1, 127)), 3)
        threshold = (2.0 * (code + 0.5) / 127).astype(np.float32)
        maximum = threshold.copy()
        maximum[::3] = np.nextafter(
            threshold[::3], np.float32(-np.inf)
        )
        maximum[2::3] = np.nextafter(
            threshold[2::3], np.float32(np.inf)
        )
        weight[:, group, 0] = maximum
        weight[:, group, 1] = -maximum
        weight[:, group, 2:] = rng.uniform(-1, 1, (rows, 14)) * maximum[:, None]
    flat = weight.reshape(rows, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q6_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q6_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q6_K"))


def test_gguf_q6_k_packing_saturates_signed_codes():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    weight = np.zeros((1, 16, 16), dtype=np.float32)
    weight[0, 0, :2] = [-2.0, 2.0]
    small_maximum = np.float32(2.0 * 1.4 / 127.0)
    weight[0, 1, :2] = [-small_maximum, small_maximum]
    flat = weight.reshape(1, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q6_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q6_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q6_K"))


def test_gguf_q6_k_packing_at_fp16_scale_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    midpoint = np.float32(1.0 + 2.0**-11)
    overflow = np.float32(65520.0)
    values = [
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
        np.float32(2.0**-25),
        np.nextafter(overflow, np.float32(-np.inf)),
        overflow,
        np.nextafter(overflow, np.float32(np.inf)),
    ]
    # Q6_K stores max(abs(weight)) / (31 * 127) as its fp16 base scale.
    magnitudes = np.asarray(values, dtype=np.float32) * np.float32(31 * 127)
    weight = np.repeat(magnitudes[:, None], 256, axis=1)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q6_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q6_k_oracle(weight))
    with np.errstate(over="ignore"):
        expected = _fallback_gguf_quantize(weight, "Q6_K")
    np.testing.assert_array_equal(actual, expected)


def test_gguf_q6_k_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _fallback_gguf_quantize,
    )

    weight = np.random.default_rng(110).standard_normal((4, 512)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q6_K"))
    np.testing.assert_array_equal(packed, _fallback_gguf_quantize(weight, "Q6_K"))
    decoded = _dequantize_gguf_tensor_numpy(packed, "Q6_K")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_q6_k_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 256"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128)), "Q6_K")


@pytest.mark.parametrize("qtype", ["Q5_K", "q5_k_s", "q5_k_m"])
@pytest.mark.parametrize("rows,width", [(9, 256), (13, 512), (129, 1024)])
def test_gguf_q5_k_packing_matches_torch_oracle(qtype, rows, width):
    weight = np.random.default_rng(104).standard_normal((rows, width)).astype(
        np.float32
    )
    weight[0] = 0
    weight[1, :256] = 2
    weight[2, :256] = np.linspace(-3, 3, 256, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), qtype)
    expected = _torch_gguf_q5_k_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q5_k_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(105).standard_normal((4, 512))).astype(
        dtype
    )
    expected = _torch_gguf_q5_k_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q5_K")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q5_k_packing_one_ulp_across_each_code_threshold():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    weight = np.full((93, 8, 32), -1.0, dtype=np.float32)
    weight[:, :, 1] = 1.0
    weight[:, :, 3:] = 0.0
    for code in range(31):
        threshold = np.float32((code + 0.5) * (2.0 / 31.0) - 1.0)
        weight[3 * code, :, 2] = np.nextafter(threshold, np.float32(-np.inf))
        weight[3 * code + 1, :, 2] = threshold
        weight[3 * code + 2, :, 2] = np.nextafter(threshold, np.float32(np.inf))
    flat = weight.reshape(93, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q5_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q5_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q5_K"))


def test_gguf_q5_k_packing_at_scale_and_minimum_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    rng = np.random.default_rng(107)
    rows = 256
    weight = np.empty((rows, 8, 32), dtype=np.float32)
    weight[:, 0, 0] = -1.0
    weight[:, 0, 1] = 2.0
    weight[:, 0, 2:] = rng.uniform(-1, 2, (rows, 30))
    for group in range(1, 8):
        scale_code = rng.integers(1, 63, rows)
        min_code = rng.integers(1, 63, rows)
        step = ((3.0 / 31.0) * (scale_code + 0.5) / 63).astype(np.float32)
        minimum = ((min_code + 0.5) / 63).astype(np.float32)
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        step = np.nextafter(step, direction)
        minimum = np.nextafter(minimum, direction)
        high = np.float32(31) * step - minimum
        weight[:, group, 0] = -minimum
        weight[:, group, 1] = high
        weight[:, group, 2:] = (
            -minimum[:, None]
            + rng.uniform(0, 1, (rows, 30)) * (high + minimum)[:, None]
        ).astype(np.float32)
    flat = weight.reshape(rows, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q5_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q5_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q5_K"))


def test_gguf_q5_k_packing_at_fp16_scale_boundaries():
    midpoint = np.float32(1.0 + 2.0**-11)
    overflow = np.float32(65520.0)
    values = [
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
        np.float32(2.0**-25),
        np.nextafter(overflow, np.float32(-np.inf)),
        overflow,
        np.nextafter(overflow, np.float32(np.inf)),
    ]
    weight = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 256, axis=1)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q5_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q5_k_oracle(weight))


def test_gguf_q5_k_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _fallback_gguf_quantize,
    )

    weight = np.random.default_rng(106).standard_normal((4, 512)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q5_K"))
    np.testing.assert_array_equal(packed, _fallback_gguf_quantize(weight, "Q5_K"))
    decoded = _dequantize_gguf_tensor_numpy(packed, "Q5_K")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_q5_k_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 256"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128)), "Q5_K")


@pytest.mark.parametrize("qtype", ["Q4_K", "q4_k_s", "q4_k_m"])
@pytest.mark.parametrize("rows,width", [(9, 256), (13, 512), (129, 1024)])
def test_gguf_q4_k_packing_matches_torch_oracle(qtype, rows, width):
    weight = np.random.default_rng(99).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :256] = 2
    weight[2, :256] = np.linspace(-3, 3, 256, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), qtype)
    expected = _torch_gguf_q4_k_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q4_k_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(100).standard_normal((4, 512))).astype(
        dtype
    )
    expected = _torch_gguf_q4_k_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q4_K")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_k_packing_at_rounding_boundaries():
    rng = np.random.default_rng(101)
    rows = 128
    weight = rng.uniform(-1, 1, (rows, 8, 32)).astype(np.float32)
    weight[:, :, 0] = -1.0
    weight[:, :, 1] = 1.0
    for group in range(8):
        for column in range(2, 32):
            code = rng.integers(1, 15, rows)
            threshold = ((code + 0.5) * (2.0 / 15.0) - 1.0).astype(np.float32)
            direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
                np.float32
            )
            weight[:, group, column] = np.where(
                rng.integers(0, 3, rows) == 0,
                np.nextafter(threshold, direction),
                threshold,
            )
    flat = weight.reshape(rows, 256)
    actual = native.gguf_quantize_weight_mlx(mx.array(flat), "Q4_K")
    expected = _torch_gguf_q4_k_oracle(flat)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_k_packing_one_ulp_across_each_code_threshold():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    base = np.full((45, 8, 32), -1.0, dtype=np.float32)
    base[:, :, 1] = 1.0
    base[:, :, 3:] = 0.0
    step = np.float32(np.float32(2.0 / 15.0) / np.float32(63.0)) * np.float32(63.0)
    bias = np.float32(np.float32(1.0 / 63.0) * np.float32(63.0))
    for code in range(15):
        threshold = np.float32((code + 0.5) * float(step) - float(bias))
        base[3 * code, :, 2] = np.nextafter(threshold, np.float32(-np.inf))
        base[3 * code + 1, :, 2] = threshold
        base[3 * code + 2, :, 2] = np.nextafter(threshold, np.float32(np.inf))

    weight = base.reshape(45, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q4_k_oracle(weight))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(weight, "Q4_K"))


def test_gguf_q4_k_packing_at_scale_and_minimum_boundaries():
    from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize

    rng = np.random.default_rng(103)
    rows = 256
    weight = np.empty((rows, 8, 32), dtype=np.float32)
    weight[:, 0, 0] = -1
    weight[:, 0, 1] = 2
    weight[:, 0, 2:] = rng.uniform(-1, 2, (rows, 30))
    for group in range(1, 8):
        scale_code = rng.integers(1, 63, rows)
        min_code = rng.integers(1, 63, rows)
        step = (0.2 * (scale_code + 0.5) / 63).astype(np.float32)
        minimum = ((min_code + 0.5) / 63).astype(np.float32)
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        step = np.nextafter(step, direction)
        minimum = np.nextafter(minimum, direction)
        high = np.float32(15) * step - minimum
        weight[:, group, 0] = -minimum
        weight[:, group, 1] = high
        weight[:, group, 2:] = (
            -minimum[:, None]
            + rng.uniform(0, 1, (rows, 30)) * (high + minimum)[:, None]
        ).astype(np.float32)

    flat = weight.reshape(rows, 256)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(flat), "Q4_K"))
    np.testing.assert_array_equal(actual, _torch_gguf_q4_k_oracle(flat))
    np.testing.assert_array_equal(actual, _fallback_gguf_quantize(flat, "Q4_K"))


def test_gguf_q4_k_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _fallback_gguf_quantize,
    )

    weight = np.random.default_rng(102).standard_normal((4, 512)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_K"))
    np.testing.assert_array_equal(packed, _fallback_gguf_quantize(weight, "Q4_K"))
    decoded = _dequantize_gguf_tensor_numpy(packed, "Q4_K")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_q4_k_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 256"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128)), "Q4_K")


@pytest.mark.parametrize("rows,width", [(13, 64), (13, 256), (257, 512)])
def test_gguf_q2_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(95).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :64] = np.tile([-1.0, -0.5, 0.5, 1.0], 16)
    weight[2, :64] = np.tile([-0.0, 0.0, 0.49, -0.49], 16)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q2_0")
    expected = _torch_gguf_q2_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q2_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(96).standard_normal((8, 128))).astype(
        dtype
    )
    expected = _torch_gguf_q2_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q2_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q2_0_packing_at_float32_rounding_boundaries():
    rng = np.random.default_rng(97)
    rows = 128
    maxima = rng.uniform(0.03, 20.0, rows).astype(np.float32)
    weight = np.zeros((rows, 64), dtype=np.float32)
    weight[:, 0] = np.where(rng.integers(0, 2, rows), maxima, -maxima)
    inverse = np.float32(1.0) / maxima
    for column in range(1, 64):
        sign = np.where(rng.integers(0, 2, rows), 1, -1)
        threshold = (sign * 0.5 / inverse.astype(np.float64)).astype(np.float32)
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        weight[:, column] = np.where(
            rng.integers(0, 3, rows) == 0,
            np.nextafter(threshold, direction),
            threshold,
        )
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q2_0")
    expected = _torch_gguf_q2_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q2_0_packing_at_fp16_scale_boundaries():
    midpoint = np.float32(1.0 + 2.0**-11)
    overflow = np.float32(65520.0)
    values = [
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
        np.float32(2.0**-25),
        np.nextafter(overflow, np.float32(-np.inf)),
        overflow,
        np.nextafter(overflow, np.float32(np.inf)),
    ]
    weight = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 64, axis=1)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q2_0")
    expected = _torch_gguf_q2_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q2_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _quantize_gguf_tensor_numpy,
    )

    weight = np.random.default_rng(98).standard_normal((4, 128)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q2_0"))
    expected = _quantize_gguf_tensor_numpy(weight, "Q2_0")
    np.testing.assert_array_equal(packed, expected)
    decoded = _dequantize_gguf_tensor_numpy(packed, "Q2_0")
    assert decoded.shape == weight.shape
    assert np.isfinite(decoded).all()


def test_gguf_q2_0_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 64"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 32)), "Q2_0")


@pytest.mark.parametrize("qtype", ["Q1_0", "Q1_0_g128"])
@pytest.mark.parametrize("rows,width", [(13, 128), (13, 256), (257, 512)])
def test_gguf_q1_0_packing_matches_torch_oracle(qtype, rows, width):
    weight = np.random.default_rng(91).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :128] = np.where(np.arange(128) % 3 == 0, 1.0, -1.0)
    weight[2, :128] = np.linspace(-2.0, 2.0, 128, dtype=np.float32)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), qtype)
    expected = _torch_gguf_q1_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q1_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(92).standard_normal((8, 256))).astype(
        dtype
    )
    expected = _torch_gguf_q1_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q1_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q1_0_packing_at_sign_and_scale_boundaries():
    midpoint = np.float32(1.0 + 2.0**-11)
    below = np.nextafter(midpoint, np.float32(-np.inf))
    above = np.nextafter(midpoint, np.float32(np.inf))
    weight = np.zeros((8, 128), dtype=np.float32)
    for row, value in enumerate((below, midpoint, above)):
        weight[row] = value
        weight[row + 3] = -value
    weight[6, :8] = [0.0, -0.0, below, -below, midpoint, -midpoint, above, -above]
    weight[7, :] = np.float32(2.0**-25)
    for qtype in ("Q1_0", "Q1_0_g128"):
        actual = native.gguf_quantize_weight_mlx(mx.array(weight), qtype)
        expected = _torch_gguf_q1_0_oracle(weight)
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q1_0_packing_at_fp16_range_boundaries():
    thresholds = (np.float32(2.0**-25), np.float32(65520.0))
    values = [np.float32(0.0)]
    for threshold in thresholds:
        values.extend(
            [
                np.nextafter(threshold, np.float32(-np.inf)),
                threshold,
                np.nextafter(threshold, np.float32(np.inf)),
            ]
        )
    weight = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 128, axis=1)
    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q1_0"))
    expected = _torch_gguf_q1_0_oracle(weight)
    np.testing.assert_array_equal(actual, expected)


def test_gguf_q1_0_packing_at_irregular_scale_midpoints():
    from gptqmodel.nn_modules.qlinear.gguf import _quantize_gguf_tensor_numpy

    rng = np.random.default_rng(34)
    rows = 256
    half_values = np.linspace(0.25, 8.0, 2000, dtype=np.float16)
    indices = rng.integers(0, 1999, rows)
    midpoint = (
        half_values[indices].astype(np.float32)
        + half_values[indices + 1].astype(np.float32)
    ) / 2
    weight = (midpoint[:, None] * rng.uniform(0.94, 1.06, (rows, 128))).astype(
        np.float32
    )
    weight[:, -1] = (
        midpoint.astype(np.float64) * 128
        - weight[:, :-1].astype(np.float64).sum(axis=1)
    ).astype(np.float32)
    weight *= rng.choice([-1, 1], size=(rows, 128)).astype(np.float32)

    actual = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q1_0"))
    expected = _torch_gguf_q1_0_oracle(weight)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, _quantize_gguf_tensor_numpy(weight, "Q1_0"))


def test_gguf_q1_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _quantize_gguf_tensor_numpy,
    )

    weight = np.random.default_rng(93).standard_normal((4, 256)).astype(np.float32)
    for qtype in ("Q1_0", "Q1_0_g128"):
        packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), qtype))
        expected = _quantize_gguf_tensor_numpy(weight, qtype)
        np.testing.assert_array_equal(packed, expected)
        decoded = _dequantize_gguf_tensor_numpy(packed, qtype)
        assert decoded.shape == weight.shape
        assert np.isfinite(decoded).all()


def test_gguf_q1_0_packing_accepts_transposed_weights():
    source = np.random.default_rng(94).standard_normal((128, 5)).astype(np.float32)
    weight = mx.array(source).T
    actual = native.gguf_quantize_weight_mlx(weight, "Q1_0")
    expected = _torch_gguf_q1_0_oracle(source.T.copy())
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q1_0_packing_validates_shape():
    with pytest.raises(ValueError, match="divisible by 128"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 32)), "Q1_0")
    with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 128), dtype=mx.int32), "Q1_0")


@pytest.mark.parametrize("rows,width", [(13, 32), (13, 256), (257, 512)])
def test_gguf_q4_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(86).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :32] = 1
    weight[2, :32] = np.arange(-16, 16, dtype=np.float32) / 8
    weight[3, :32] = np.tile(
        (np.arange(-8, 8, dtype=np.float32) + 0.5) * 0.25, 2
    )
    weight[3, 0] = -2.0
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0")
    expected = _torch_gguf_q4_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q4_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(19).standard_normal((8, 64))).astype(
        dtype
    )
    expected = _torch_gguf_q4_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q4_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_0_packing_at_float32_rounding_boundaries():
    rng = np.random.default_rng(819)
    rows = 64
    maxima = rng.uniform(0.03, 20.0, rows).astype(np.float32)
    signed_maxima = np.where(rng.integers(0, 2, rows), maxima, -maxima)
    weight = np.zeros((rows, 32), dtype=np.float32)
    weight[:, 0] = signed_maxima
    reciprocal = np.float32(1.0) / (signed_maxima / np.float32(-8.0))
    for column in range(1, 32):
        code = rng.integers(2, 14, rows)
        boundary = ((code - 8.5) / reciprocal.astype(np.float64)).astype(
            np.float32
        )
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        weight[:, column] = np.where(
            rng.integers(0, 3, rows) == 0,
            np.nextafter(boundary, direction),
            boundary,
        )

    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0")
    expected = _torch_gguf_q4_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_0_packing_preserves_zero_scale_sign():
    weight = np.zeros((2, 32), dtype=np.float32)
    weight[1, 0] = np.float32(-0.0)
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0")
    expected = _torch_gguf_q4_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q4_0_packing_validates_format_and_shape():
    with pytest.raises(ValueError, match="supports Q1_0.*Q4_0.*Q8_0"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 32)), "MXFP4")
    with pytest.raises(ValueError, match="divisible by 32"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 33)), "Q4_0")
    with pytest.raises(ValueError, match="nonzero input width"):
        native.gguf_quantize_weight_mlx(mx.zeros((2, 0)), "Q4_0")


def test_gguf_q4_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _quantize_gguf_tensor_numpy,
    )

    weight = np.random.default_rng(17).standard_normal((4, 128)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q4_0"))
    expected = _quantize_gguf_tensor_numpy(weight, "Q4_0")
    np.testing.assert_array_equal(packed, expected)
    dequantized = _dequantize_gguf_tensor_numpy(packed, "Q4_0")
    assert dequantized.shape == weight.shape
    assert np.isfinite(dequantized).all()


@pytest.mark.parametrize("rows,width", [(13, 32), (13, 256), (257, 512)])
def test_gguf_q8_0_packing_matches_torch_oracle(rows, width):
    weight = np.random.default_rng(87).standard_normal((rows, width)).astype(np.float32)
    weight[0] = 0
    weight[1, :32] = 1
    weight[2, :32] = np.arange(-16, 16, dtype=np.float32) + 0.5
    weight[2, 0] = -127.0
    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q8_0")
    expected = _torch_gguf_q8_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gguf_q8_0_packing_promotes_low_precision_inputs(dtype):
    weight = mx.array(np.random.default_rng(20).standard_normal((8, 64))).astype(
        dtype
    )
    expected = _torch_gguf_q8_0_oracle(np.asarray(weight.astype(mx.float32)))
    actual = native.gguf_quantize_weight_mlx(weight, "Q8_0")
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q8_0_packing_at_float32_rounding_boundaries():
    rng = np.random.default_rng(820)
    rows = 64
    maxima = rng.uniform(0.03, 20.0, rows).astype(np.float32)
    signed_maxima = np.where(rng.integers(0, 2, rows), maxima, -maxima)
    weight = np.zeros((rows, 32), dtype=np.float32)
    weight[:, 0] = signed_maxima
    weight[:, 1] = -signed_maxima
    reciprocal = np.float32(1.0) / (maxima / np.float32(127.0))
    for column in range(2, 32):
        code = rng.integers(-120, 120, rows)
        boundary = ((code + 0.5) / reciprocal.astype(np.float64)).astype(
            np.float32
        )
        direction = np.where(rng.integers(0, 2, rows), np.inf, -np.inf).astype(
            np.float32
        )
        weight[:, column] = np.where(
            rng.integers(0, 3, rows) == 0,
            np.nextafter(boundary, direction),
            boundary,
        )

    actual = native.gguf_quantize_weight_mlx(mx.array(weight), "Q8_0")
    expected = _torch_gguf_q8_0_oracle(weight)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_gguf_q8_0_bytes_are_accepted_by_existing_runtime():
    from gptqmodel.nn_modules.qlinear.gguf import (
        _dequantize_gguf_tensor_numpy,
        _quantize_gguf_tensor_numpy,
    )

    weight = np.random.default_rng(18).standard_normal((4, 128)).astype(np.float32)
    packed = np.asarray(native.gguf_quantize_weight_mlx(mx.array(weight), "Q8_0"))
    expected = _quantize_gguf_tensor_numpy(weight, "Q8_0")
    np.testing.assert_array_equal(packed, expected)
    dequantized = _dequantize_gguf_tensor_numpy(packed, "Q8_0")
    assert dequantized.shape == weight.shape
    assert np.isfinite(dequantized).all()
