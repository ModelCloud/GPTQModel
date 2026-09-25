# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle and byte-boundary checks for MLX FP8 weight quantization."""

import gc
import sys

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import quantize_fp8_weight
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_fp8 import quantize_fp8_weight_mlx  # noqa: E402


FORMATS = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")


def _compare(weight, fmt, method, block_size=None):
    source = torch.from_numpy(weight.copy())
    expected, expected_scale = quantize_fp8_weight(
        source, format=fmt, weight_scale_method=method, weight_block_size=block_size,
    )
    actual, actual_scale = quantize_fp8_weight_mlx(
        mx.array(weight), format=fmt, weight_scale_method=method,
        weight_block_size=block_size,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected.view(torch.uint8).numpy())
    np.testing.assert_allclose(np.asarray(actual_scale), expected_scale.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, "bfloat16"])
@pytest.mark.parametrize("method,block_size", [("tensor", None), ("row", None), ("block", (2, 64))])
def test_fp8_matches_torch_small(fmt, dtype, method, block_size):
    rng = np.random.default_rng(500)
    weight = rng.normal(0, 0.3, (4, 256)).astype(np.float32)
    weight[0] = 0
    weight[0, 0] = -0.0
    if dtype == "bfloat16":
        source = torch.from_numpy(weight).to(torch.bfloat16)
        expected, expected_scale = quantize_fp8_weight(
            source, format=fmt, weight_scale_method=method, weight_block_size=block_size,
        )
        actual, actual_scale = quantize_fp8_weight_mlx(
            mx.array(weight).astype(mx.bfloat16), format=fmt, weight_scale_method=method,
            weight_block_size=block_size,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected.view(torch.uint8).numpy())
        np.testing.assert_allclose(np.asarray(actual_scale), expected_scale.numpy(), rtol=1e-6, atol=1e-6)
    else:
        _compare(weight.astype(dtype), fmt, method, block_size)


@pytest.mark.parametrize("fmt", FORMATS)
def test_fp8_boundaries_and_one_float32_step(fmt):
    dtype = getattr(torch, fmt)
    count = 124 if fmt == "float8_e5m2" else 127 if fmt == "float8_e4m3fn" else 128
    values = torch.arange(count, dtype=torch.uint8).view(dtype).float()
    midpoint = (values[:-1] + values[1:]) / 2
    lower = torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf")))
    upper = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
    samples = torch.cat((lower, midpoint, upper, -lower, -midpoint, -upper))
    weight = torch.cat((torch.tensor([torch.finfo(dtype).max, 0.0, -0.0]), samples))[None, :].numpy()
    _compare(weight, fmt, "row")


@pytest.mark.parametrize("fmt", FORMATS)
def test_fp8_all_zero_groups(fmt):
    weight = np.zeros((4, 128), dtype=np.float32)
    weight[0, 0] = -0.0
    _compare(weight, fmt, "tensor")
    _compare(weight, fmt, "row")
    _compare(weight, fmt, "block", (2, 64))


@pytest.mark.parametrize("fmt", FORMATS)
def test_fp8_subnormal_groups_and_infinite_inverse_scale(fmt):
    tiny = np.float32(1e-40)
    weight = np.zeros((4, 128), dtype=np.float32)
    weight[0, :4] = [tiny, -tiny, 0, -0.0]
    weight[1, :4] = [2 * tiny, -2 * tiny, 0, -0.0]
    _compare(weight, fmt, "tensor")
    weight[2, 64] = 1
    _compare(weight, fmt, "row")
    _compare(weight, fmt, "block", (2, 64))
    weight[0, 4] = np.float32(1e-35)
    _compare(weight, fmt, "row")
    _compare(weight, fmt, "block", (2, 64))
    samples = np.linspace(1, 0x7fffff, 128, dtype=np.uint32).view(np.float32)
    weight[0] = samples
    weight[1] = -samples
    weight[0, 0] = np.float32(1e-35)
    _compare(weight, fmt, "row")
    _compare(weight, fmt, "block", (2, 64))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("method", ["tensor", "row", "block"])
@pytest.mark.parametrize("name,rows,cols", QWEN38_27B_PROJECTIONS)
def test_fp8_qwen38_27b_full_projection(fmt, method, name, rows, cols):
    """Compare every FP8 byte and scale for BF16 checkpoint-size weights."""
    del name
    block_size = (128, 128) if method == "block" else None
    rng = np.random.default_rng(380027 + rows + cols)
    weight = torch.from_numpy(rng.normal(0, 0.2, (rows, cols)).astype(np.float32)).to(torch.bfloat16)
    expected, expected_scale = quantize_fp8_weight(
        weight, format=fmt, weight_scale_method=method, weight_block_size=block_size,
    )
    actual, actual_scale = quantize_fp8_weight_mlx(
        mx.array(weight.float().numpy()).astype(mx.bfloat16),
        format=fmt, weight_scale_method=method, weight_block_size=block_size,
    )
    np.testing.assert_array_equal(np.asarray(actual), expected.view(torch.uint8).numpy())
    np.testing.assert_allclose(np.asarray(actual_scale), expected_scale.numpy(), rtol=1e-6, atol=1e-6)
    del weight, expected, expected_scale, actual, actual_scale
    mx.clear_cache()
    gc.collect()


def test_fp8_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        quantize_fp8_weight_mlx(mx.zeros((0, 128)))
    with pytest.raises(ValueError, match="divisible"):
        quantize_fp8_weight_mlx(mx.zeros((3, 128)), weight_scale_method="block", weight_block_size=(2, 64))
    with pytest.raises(ValueError, match="finite"):
        quantize_fp8_weight_mlx(mx.array([[float("nan"), 1]]))
