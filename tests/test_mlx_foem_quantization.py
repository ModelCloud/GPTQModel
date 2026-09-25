# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch-oracle checks for native MLX FOEM quantization."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.quantization.mlx_foem import foem_quantize_weight_mlx  # noqa: E402


def _torch_params(group, bits, sym):
    """Reproduce Quantizer.find_params without calling the implementation."""
    minimum = torch.minimum(group.amin(dim=1), torch.zeros(group.shape[0]))
    maximum = torch.maximum(group.amax(dim=1), torch.zeros(group.shape[0]))
    if sym:
        maximum = torch.maximum(minimum.abs(), maximum)
        minimum = torch.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = torch.where(empty, -1, minimum)
    maximum = torch.where(empty, 1, maximum)
    scale = (maximum - minimum) / (2**bits - 1)
    zero = (
        torch.full_like(scale, 2 ** (bits - 1))
        if sym else torch.round(-minimum / scale)
    )
    return scale, zero


def _torch_foem_oracle(weight, inverse_hessian, bits, group_size, beta, sym):
    """Follow FOEM's sequential column corrections using independent Torch ops."""
    raw = torch.from_numpy(weight.copy())
    factor = torch.from_numpy(inverse_hessian.copy())
    rows, columns = raw.shape
    remaining = raw.clone()
    quantized, scales, zeros = [], [], []
    for start in range(0, columns, group_size):
        end = start + group_size
        group = remaining[:, :group_size].clone()
        scale, zero = _torch_params(group, bits, sym)
        errors = torch.empty_like(group)
        output = torch.empty_like(group)
        for offset in range(group_size):
            value = group[:, offset].clone()
            code = (torch.round(value / scale) + zero).clamp(0, 2**bits - 1)
            q = scale * (code - zero)
            error = ((value - q) - (value - raw[:, start + offset]) * beta) / factor[
                start + offset, start + offset
            ]
            output[:, offset] = q
            errors[:, offset] = error
            group[:, offset:] -= error[:, None] * factor[
                start + offset, start + offset : end
            ]
            if offset + 1 < group_size:
                group[:, offset + 1] -= beta * (
                    group[:, offset + 1] - raw[:, start + offset + 1]
                )
        quantized.append(output)
        scales.append(scale[:, None])
        zeros.append(zero[:, None])
        if end < columns:
            remaining = remaining[:, group_size:] - errors @ factor[start:end, end:]
    return tuple(torch.cat(values, dim=1).numpy() for values in (quantized, scales, zeros))


def _codes_from_dequantized(result, group_size):
    quantized, scales, zeros = result
    rows, columns = quantized.shape
    codes = np.rint(
        quantized.reshape(rows, columns // group_size, group_size)
        / scales[:, :, None] + zeros[:, :, None]
    )
    return codes.astype(np.uint8)


@pytest.mark.parametrize(
    "bits,sym", [(bits, sym) for bits in range(2, 9) for sym in (True, False)],
)
@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("beta", [0.0, 0.2])
def test_foem_group_updates_match_torch(bits, sym, group_size, beta):
    rng = np.random.default_rng(600 + bits + group_size)
    rows, columns = 7, group_size * 2
    weight = rng.normal(0, 0.3, (rows, columns)).astype(np.float32)
    factor = np.eye(columns, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.002, (columns, columns)).astype(np.float32), 1)
    expected = _torch_foem_oracle(weight, factor, bits, group_size, beta, sym)
    actual = foem_quantize_weight_mlx(
        mx.array(weight), mx.array(factor), bits=bits, group_size=group_size,
        beta=beta, sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), group_size),
        _codes_from_dequantized(expected, group_size),
    )


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("sym", [True, False])
def test_foem_code_boundaries_one_float32_step(bits, sym):
    weight = np.zeros((4, 32), dtype=np.float32)
    weight[:, 0], weight[:, 1] = -1, 1
    scale = np.float32(2 / (2**bits - 1))
    midpoint = np.float32(scale / 2)
    weight[0, 2:5] = [
        np.nextafter(midpoint, np.float32(-np.inf)), midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
    ]
    weight[1, 2:5] = -weight[0, 2:5]
    weight[2, 2:5] = [0, -0.0, 1]
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(weight, factor, bits, 32, 0.2, sym)
    actual = foem_quantize_weight_mlx(
        mx.array(weight), mx.array(factor), bits=bits, group_size=32, sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 32),
        _codes_from_dequantized(expected, 32),
    )


def test_foem_asymmetric_zero_point_tie_and_neighbors():
    weight = np.zeros((3, 32), dtype=np.float32)
    minimum = np.float32(-17)
    weight[0, 0], weight[0, 1] = minimum, 13
    weight[1, 0], weight[1, 1] = np.nextafter(minimum, np.float32(-np.inf)), 13
    weight[2, 0], weight[2, 1] = np.nextafter(minimum, np.float32(np.inf)), 13
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(weight, factor, 4, 32, 0.2, False)
    actual = foem_quantize_weight_mlx(
        mx.array(weight), mx.array(factor), group_size=32, sym=False,
    )
    assert expected[2][0, 0] == 8  # 8.5 rounds to even.
    for index in range(3):
        np.testing.assert_allclose(np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype,torch_dtype", [
    (mx.float16, torch.float16), (mx.bfloat16, torch.bfloat16),
])
@pytest.mark.parametrize("sym", [True, False])
def test_foem_low_precision_code_midpoint(dtype, torch_dtype, sym):
    midpoint = torch.tensor(1 / 15, dtype=torch_dtype)
    lower = torch.nextafter(midpoint, torch.tensor(-float("inf"), dtype=torch_dtype))
    upper = torch.nextafter(midpoint, torch.tensor(float("inf"), dtype=torch_dtype))
    source = np.zeros((2, 32), dtype=np.float32)
    source[:, 0], source[:, 1] = -1, 1
    source[0, 2:5] = torch.stack((lower, midpoint, upper)).float().numpy()
    source[1, 2:5] = -source[0, 2:5]
    weight = mx.array(source).astype(dtype)
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(
        np.asarray(weight.astype(mx.float32)), factor, 4, 32, 0.2, sym,
    )
    actual = foem_quantize_weight_mlx(weight, mx.array(factor), group_size=32, sym=sym)
    for index in range(3):
        np.testing.assert_allclose(np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 32),
        _codes_from_dequantized(expected, 32),
    )


@pytest.mark.parametrize("sym", [True, False])
def test_foem_bfloat16_with_cross_group_updates(sym):
    rng = np.random.default_rng(7900)
    source = rng.normal(0, 0.3, (9, 128)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    factor = np.eye(128, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.002, (128, 128)).astype(np.float32), 1)
    expected = _torch_foem_oracle(
        np.asarray(weight.astype(mx.float32)), factor, 4, 64, 0.2, sym,
    )
    actual = foem_quantize_weight_mlx(
        weight, mx.array(factor), bits=4, group_size=64, beta=0.2, sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 64),
        _codes_from_dequantized(expected, 64),
    )


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_foem_qwen38_27b_full_projection(name, out_features, in_features):
    """Compare every BF16 weight, scale, and zero on model-scale projections."""
    del name
    group_size, bits = 128, 4
    rng = np.random.default_rng(7700 + out_features + in_features)
    source = rng.normal(0, 0.2, (out_features, in_features)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    source = np.asarray(weight.astype(mx.float32))
    oracle = torch.from_numpy(source).reshape(out_features, -1, group_size)
    minimum = torch.minimum(oracle.amin(dim=-1), torch.zeros((out_features, in_features // group_size)))
    maximum = torch.maximum(oracle.amax(dim=-1), torch.zeros_like(minimum))
    maximum = torch.maximum(minimum.abs(), maximum)
    minimum = torch.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = torch.where(empty, -1, minimum)
    maximum = torch.where(empty, 1, maximum)
    scales = (maximum - minimum) / 15
    zeros = torch.full_like(scales, 8)
    codes = (torch.round(oracle / scales[..., None]) + zeros[..., None]).clamp(0, 15)
    expected = scales[..., None] * (codes - zeros[..., None])
    actual = foem_quantize_weight_mlx(
        weight, mx.eye(in_features), bits=bits, group_size=group_size,
        beta=0.2, sym=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual[0]), expected.reshape(out_features, in_features).numpy(),
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(actual[1]), scales.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[2]), zeros.numpy())
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), group_size),
        codes.numpy().astype(np.uint8),
    )


def test_foem_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        foem_quantize_weight_mlx(mx.zeros((0, 32)), mx.eye(32), group_size=32)
    with pytest.raises(ValueError, match="group_size"):
        foem_quantize_weight_mlx(mx.zeros((2, 33)), mx.eye(33), group_size=32)
    with pytest.raises(ValueError, match="diagonal"):
        foem_quantize_weight_mlx(mx.zeros((2, 32)), mx.zeros((32, 32)), group_size=32)
