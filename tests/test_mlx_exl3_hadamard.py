# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for EXL3's native MLX Hadamard kernel."""

import gc
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_hadamard import (  # noqa: E402
    exl3_hadamard_128_mlx,
)


@lru_cache(maxsize=1)
def _torch_hadamard_128():
    """Construct the normalized Sylvester matrix independently from EXL3."""
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (1.0 / np.sqrt(128.0))


def _torch_hadamard_oracle(matrix, axis):
    """Apply independent dense Torch matmuls with EXL3's block layout."""
    source = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    output = torch.empty_like(source)
    hadamard = _torch_hadamard_128()
    if axis == 1:
        for start in range(0, source.shape[1], 128):
            output[:, start : start + 128] = source[:, start : start + 128] @ hadamard
    else:
        for start in range(0, source.shape[0], 128):
            output[start : start + 128] = hadamard @ source[start : start + 128]
    return output.numpy()


def _normalized_rms_drift(actual, expected):
    squared_error = 0.0
    squared_expected = 0.0
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    for start in range(0, actual_flat.size, 1 << 20):
        stop = min(start + (1 << 20), actual_flat.size)
        error = actual_flat[start:stop].astype(np.float64) - expected_flat[
            start:stop
        ].astype(np.float64)
        reference = expected_flat[start:stop].astype(np.float64)
        squared_error += float(np.dot(error, error))
        squared_expected += float(np.dot(reference, reference))
    return float(np.sqrt(squared_error / squared_expected))


@pytest.mark.parametrize("axis", [0, 1])
def test_exl3_hadamard_small_torch_oracle(axis):
    rng = np.random.default_rng(9341 + axis)
    source = rng.normal(0.0, 1.0, (256, 384)).astype(np.float32)
    expected = _torch_hadamard_oracle(source, axis)
    actual = np.asarray(exl3_hadamard_128_mlx(mx.array(source), axis=axis))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert _normalized_rms_drift(actual, expected) <= 1e-6


@pytest.mark.parametrize("axis", [0, 1])
def test_exl3_hadamard_cancellation_and_float32_boundaries(axis):
    tiny_positive = np.nextafter(np.float32(0), np.float32(1))
    tiny_negative = np.nextafter(np.float32(0), np.float32(-1))
    near_one_low = np.nextafter(np.float32(1), np.float32(-np.inf))
    near_one_high = np.nextafter(np.float32(1), np.float32(np.inf))
    boundary = np.array(
        [
            -1.0,
            -0.0,
            0.0,
            1.0,
            tiny_negative,
            tiny_positive,
            near_one_low,
            near_one_high,
        ],
        dtype=np.float32,
    )
    source = np.resize(boundary, (256, 256)).astype(np.float32, copy=False)
    source[1::2] *= -1
    expected = _torch_hadamard_oracle(source, axis)
    actual = np.asarray(exl3_hadamard_128_mlx(mx.array(source), axis=axis))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert _normalized_rms_drift(actual, expected) <= 1e-6


@pytest.mark.parametrize("axis", [0, 1])
def test_exl3_hadamard_block_lane_boundaries(axis):
    source = np.zeros((128, 128), dtype=np.float32)
    boundaries = (0, 31, 32, 63, 64, 95, 96, 127)
    values = np.array(
        [1.0, -1.0, np.nextafter(1.0, 0.0), -0.0, 0.0, 0.5, -0.5, 2.0],
        dtype=np.float32,
    )
    if axis == 1:
        source[17, boundaries] = values
    else:
        source[boundaries, 17] = values

    expected = _torch_hadamard_oracle(source, axis)
    actual = np.asarray(exl3_hadamard_128_mlx(mx.array(source), axis=axis))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert _normalized_rms_drift(actual, expected) <= 1e-6


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("name,rows,columns", QWEN38_27B_PROJECTIONS)
def test_exl3_hadamard_qwen38_projection_oracle(name, rows, columns, axis):
    rng = np.random.default_rng(10421 + rows + columns + axis)
    source = rng.normal(0.0, 0.2, (rows, columns)).astype(np.float32)
    expected = _torch_hadamard_oracle(source, axis)
    actual = np.asarray(exl3_hadamard_128_mlx(mx.array(source), axis=axis))
    error = np.abs(actual - expected)
    allowed = 1e-6 + 1e-6 * np.abs(expected)
    outside_tolerance = int(np.count_nonzero(error > allowed))
    assert outside_tolerance == 0, (
        f"{name} axis={axis}: {outside_tolerance} outputs exceed tolerance"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"{name} axis={axis}",
    )
    assert _normalized_rms_drift(actual, expected) <= 1e-6
    del source, expected, actual, error
    gc.collect()
    mx.clear_cache()


def test_exl3_hadamard_is_self_inverse():
    rng = np.random.default_rng(641)
    source = rng.normal(0.0, 0.3, (256, 256)).astype(np.float32)
    transformed = exl3_hadamard_128_mlx(mx.array(source), axis=1)
    restored = np.asarray(exl3_hadamard_128_mlx(transformed, axis=1))
    np.testing.assert_allclose(restored, source, atol=1e-6, rtol=1e-6)


def test_exl3_hadamard_axis_zero_accepts_narrow_matrix():
    source = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(128, 1)
    expected = _torch_hadamard_oracle(source, axis=0)
    actual = np.asarray(exl3_hadamard_128_mlx(mx.array(source), axis=0))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_exl3_hadamard_rejects_invalid_inputs():
    valid = mx.zeros((128, 128), dtype=mx.float32)
    for axis in (-1, 2, 0.0, True, None):
        with pytest.raises(ValueError, match="axis"):
            exl3_hadamard_128_mlx(valid, axis=axis)
    with pytest.raises(ValueError, match="rank-two"):
        exl3_hadamard_128_mlx(mx.zeros((128,), dtype=mx.float32), axis=0)
    with pytest.raises(ValueError, match="nonempty"):
        exl3_hadamard_128_mlx(mx.zeros((0, 128), dtype=mx.float32), axis=1)
    with pytest.raises(ValueError, match="float32"):
        exl3_hadamard_128_mlx(mx.zeros((128, 128), dtype=mx.float16), axis=1)
    with pytest.raises(ValueError, match="divisible by 128"):
        exl3_hadamard_128_mlx(mx.zeros((129, 128), dtype=mx.float32), axis=0)
    nonfinite = mx.zeros((128, 128), dtype=mx.float32)
    nonfinite = nonfinite.at[0, 0].add(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        exl3_hadamard_128_mlx(nonfinite, axis=1)
