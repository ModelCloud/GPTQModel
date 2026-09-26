# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle checks for EXL3's native MLX RMS reduction."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_rms import exl3_block_rms_mlx


def _torch_block_rms_oracle(matrix, axis, block_size=32):
    source = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    squared_sum = None
    for block in torch.split(source, block_size, dim=axis):
        block_sum = block.square().sum(dim=axis, keepdim=True)
        squared_sum = block_sum if squared_sum is None else squared_sum + block_sum
    return (squared_sum / source.shape[axis]).sqrt().numpy()


def _normalized_rms_drift(actual, expected):
    error = actual.astype(np.float64) - expected.astype(np.float64)
    numerator = float(np.dot(error.reshape(-1), error.reshape(-1)))
    reference = expected.astype(np.float64).reshape(-1)
    denominator = float(np.dot(reference, reference))
    return float(np.sqrt(numerator / max(denominator, 1e-30)))


def _assert_rms_matches(actual, expected, *, err_msg=None):
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
        err_msg=err_msg,
    )
    assert np.isfinite(actual).all(), err_msg
    assert _normalized_rms_drift(actual, expected) <= 1e-6, err_msg


@pytest.mark.parametrize("axis", [0, 1])
def test_exl3_block_rms_small_torch_oracle(axis):
    rng = np.random.default_rng(3371 + axis)
    source = rng.normal(0.0, 0.4, (79, 113)).astype(np.float32)
    expected = _torch_block_rms_oracle(source, axis)
    actual = np.asarray(exl3_block_rms_mlx(mx.array(source), axis=axis))
    _assert_rms_matches(actual, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_exl3_block_rms_float32_boundaries(axis):
    tiny_positive = np.nextafter(np.float32(0), np.float32(1))
    tiny_negative = np.nextafter(np.float32(0), np.float32(-1))
    near_one_low = np.nextafter(np.float32(1), np.float32(0))
    near_one_high = np.nextafter(np.float32(1), np.float32(np.inf))
    boundary = np.array(
        [
            -near_one_high,
            -1.0,
            -near_one_low,
            tiny_negative,
            -0.0,
            0.0,
            tiny_positive,
            near_one_low,
            1.0,
            near_one_high,
        ],
        dtype=np.float32,
    )
    source = np.resize(boundary, (67, 101)).astype(np.float32, copy=False)
    expected = _torch_block_rms_oracle(source, axis)
    actual = np.asarray(exl3_block_rms_mlx(mx.array(source), axis=axis))
    _assert_rms_matches(actual, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_block_rms_qwen38_projection_oracle(name, out_features, in_features, axis):
    rng = np.random.default_rng(14111 + out_features + in_features + axis)
    source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
    expected = _torch_block_rms_oracle(source, axis)
    actual = np.asarray(exl3_block_rms_mlx(mx.array(source), axis=axis))
    _assert_rms_matches(actual, expected, err_msg=f"{name} axis={axis}")

    error = np.abs(actual - expected)
    allowed = 1e-6 + 1e-6 * np.abs(expected)
    outside_tolerance = int(np.count_nonzero(error > allowed))
    assert outside_tolerance == 0, (
        f"{name} axis={axis}: {outside_tolerance} scales exceed tolerance"
    )

    del source, expected, actual, error
    gc.collect()
    mx.clear_cache()


def test_exl3_block_rms_zero_matrix():
    source = np.full((64, 96), -0.0, dtype=np.float32)
    for axis in (0, 1):
        expected = _torch_block_rms_oracle(source, axis)
        actual = np.asarray(exl3_block_rms_mlx(mx.array(source), axis=axis))
        np.testing.assert_array_equal(actual, expected)


def test_exl3_block_rms_rejects_invalid_inputs():
    valid = mx.zeros((32, 32), dtype=mx.float32)
    for axis in (-1, 2, 0.0, True, None):
        with pytest.raises(ValueError, match="axis"):
            exl3_block_rms_mlx(valid, axis=axis)
    with pytest.raises(ValueError, match="rank-two"):
        exl3_block_rms_mlx(mx.zeros((32,), dtype=mx.float32), axis=0)
    with pytest.raises(ValueError, match="nonempty"):
        exl3_block_rms_mlx(mx.zeros((0, 32), dtype=mx.float32), axis=0)
    with pytest.raises(ValueError, match="float32"):
        exl3_block_rms_mlx(mx.zeros((32, 32), dtype=mx.float16), axis=0)
    nonfinite = valid.at[0, 0].add(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        exl3_block_rms_mlx(nonfinite, axis=0)
