# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch-oracle checks for ParoQuant pseudo-quantization on MLX."""

import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_paroquant_quant import (
    paroquant_quantize_weight_mlx,
)


def _torch_oracle(weight, scales, *, bits, group_size, sym, zero_point_float=None):
    """Independent reproduction of pseudo_quantize_dequant(use_ste=False)."""
    rows, columns = weight.shape
    groups = columns // group_size
    weight_view = torch.from_numpy(weight).reshape(rows * groups, group_size)
    scale = torch.from_numpy(scales).reshape(rows * groups, 1).clamp(1e-5, 1e5)
    if sym:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        code = torch.round(weight_view / scale).clamp(qmin, qmax)
        output = code * scale
    else:
        qmin, qmax = 0, 2**bits - 1
        zero_float = torch.from_numpy(zero_point_float).reshape(rows * groups, 1)
        zero = torch.round(-zero_float).clamp(qmin, qmax)
        code = (torch.round(weight_view / scale) + zero).clamp(qmin, qmax)
        output = (code - zero) * scale
    return output.reshape(rows, columns).numpy()


def _run(weight, scales, *, bits, group_size, sym, zero_point_float=None):
    zeros = None if zero_point_float is None else mx.array(zero_point_float)
    result = paroquant_quantize_weight_mlx(
        mx.array(weight),
        mx.array(scales),
        bits=bits,
        group_size=group_size,
        sym=sym,
        zero_point_float=zeros,
    )
    return np.asarray(result.astype(mx.float32))


def _patterned_rounding_inputs(group_size, bits, sym):
    qmin = -(2 ** (bits - 1)) if sym else 0
    qmax = 2 ** (bits - 1) - 1 if sym else 2**bits - 1
    step = np.float32(0.125)
    values = np.zeros((3, group_size), dtype=np.float32)
    midpoints = (np.arange(qmin, qmax, dtype=np.float32) + np.float32(0.5)) * step
    count = min(midpoints.size, group_size - 3)
    values[0, :count] = np.nextafter(midpoints[:count], np.float32(-np.inf))
    values[1, :count] = midpoints[:count]
    values[2, :count] = np.nextafter(midpoints[:count], np.float32(np.inf))
    values[:, -3:] = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    scales = np.full((3, 1), step, dtype=np.float32)
    zeros = None if sym else np.full((3, 1), -7.5, dtype=np.float32)
    return values, scales, zeros


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("sym", [True, False])
def test_paroquant_rounding_thresholds_and_saturation(bits, sym):
    weight, scales, zeros = _patterned_rounding_inputs(32, bits, sym)
    actual = _run(
        weight,
        scales,
        bits=bits,
        group_size=32,
        sym=sym,
        zero_point_float=zeros,
    )
    expected = _torch_oracle(
        weight,
        scales,
        bits=bits,
        group_size=32,
        sym=sym,
        zero_point_float=zeros,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("sym", [True, False])
def test_paroquant_low_precision_inputs(dtype, sym):
    source = np.zeros((2, 64), dtype=np.float32)
    source[:, :3] = [-1, 1, 0.0625]
    weight = mx.array(source).astype(dtype)
    resident = np.asarray(weight.astype(mx.float32))
    scales = np.full((2, 1), 1 / 7, dtype=np.float32)
    zeros = None if sym else np.full((2, 1), -7.0, dtype=np.float32)
    expected = _torch_oracle(
        resident,
        scales,
        bits=4,
        group_size=64,
        sym=sym,
        zero_point_float=zeros,
    )
    torch_dtype = {
        mx.float32: torch.float32,
        mx.float16: torch.float16,
        mx.bfloat16: torch.bfloat16,
    }[dtype]
    expected = torch.from_numpy(expected).to(torch_dtype).float().numpy()
    actual = _run(
        weight,
        scales,
        bits=4,
        group_size=64,
        sym=sym,
        zero_point_float=zeros,
    )
    np.testing.assert_array_equal(actual, expected)


def test_paroquant_accepts_learned_parameter_layout():
    rng = np.random.default_rng(413)
    weight = rng.normal(0, 0.2, (5, 128)).astype(np.float32)
    scales = rng.uniform(0.01, 0.08, (5, 4)).astype(np.float32)
    zeros = rng.uniform(-9, 0, (5, 4)).astype(np.float32)
    grouped = paroquant_quantize_weight_mlx(
        mx.array(weight),
        mx.array(scales),
        group_size=32,
        sym=False,
        zero_point_float=mx.array(zeros),
    )
    flattened = paroquant_quantize_weight_mlx(
        mx.array(weight),
        mx.array(scales.reshape(-1, 1)),
        group_size=32,
        sym=False,
        zero_point_float=mx.array(zeros.reshape(-1, 1)),
    )
    np.testing.assert_array_equal(
        np.asarray(grouped.astype(mx.float32)),
        np.asarray(flattened.astype(mx.float32)),
    )


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("name,rows,columns", QWEN38_27B_PROJECTIONS)
def test_paroquant_qwen38_projection_oracle(name, rows, columns, sym):
    group_size = 128
    rng = np.random.default_rng(8821 + rows + columns)
    source = rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    resident = np.asarray(weight.astype(mx.float32))
    groups = columns // group_size
    scales = rng.uniform(0.002, 0.06, (rows, groups)).astype(np.float32)
    zero_float = None if sym else rng.uniform(-12, 0, (rows, groups)).astype(np.float32)
    expected = _torch_oracle(
        resident,
        scales,
        bits=4,
        group_size=group_size,
        sym=sym,
        zero_point_float=zero_float,
    )
    expected = torch.from_numpy(expected).to(torch.bfloat16).float().numpy()
    actual = _run(
        weight,
        scales,
        bits=4,
        group_size=group_size,
        sym=sym,
        zero_point_float=zero_float,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} symmetric={sym}",
    )


def test_paroquant_invalid_inputs():
    weight = mx.zeros((2, 64), dtype=mx.float32)
    scales = mx.ones((2, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="bits"):
        paroquant_quantize_weight_mlx(weight, scales, bits=1)
    with pytest.raises(ValueError, match="group_size"):
        paroquant_quantize_weight_mlx(weight, scales, group_size=63)
    with pytest.raises(ValueError, match="scales"):
        paroquant_quantize_weight_mlx(weight, mx.ones((2, 1)), group_size=32)
    with pytest.raises(ValueError, match="zero_point_float"):
        paroquant_quantize_weight_mlx(weight, scales, group_size=64, sym=False)
