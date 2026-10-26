# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for native MLX EXL3 block-LDL."""

import gc
import math
import sys
from functools import cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_block_ldl import exl3_block_ldl_mlx


def _torch_block_ldl_oracle(hessian, *, block_size=16, dtype=torch.float32):
    """Compute the EXL3 factor in Torch without calling the MLX implementation."""
    source = torch.as_tensor(hessian, dtype=dtype)
    factor = torch.linalg.cholesky(source)
    columns = factor.shape[0]
    block_count = columns // block_size
    diagonal = torch.diagonal(
        factor.reshape(block_count, block_size, block_count, block_size),
        dim1=0,
        dim2=2,
    ).permute(2, 0, 1)
    diagonal_inverse = torch.linalg.inv(diagonal)
    factor = factor.reshape(columns, block_count, block_size)
    for block in range(block_count):
        factor[:, block, :] = factor[:, block, :] @ diagonal_inverse[block]
    factor = factor.reshape(columns, columns).contiguous()
    factor_blocks = factor.reshape(
        block_count, block_size, block_count, block_size
    ).permute(0, 2, 1, 3)
    indices = torch.arange(block_count)
    factor_blocks[indices, indices] = torch.stack(
        [torch.eye(block_size, dtype=dtype)] * block_count
    )
    return factor


def _drift_metrics(actual, expected, *, row_chunk=128):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    outside = 0
    nonfinite = 0
    max_abs = 0.0
    difference_square_sum = 0.0
    expected_square_sum = 0.0
    for start in range(0, actual.shape[0], row_chunk):
        actual_chunk = actual[start : start + row_chunk]
        expected_chunk = expected[start : start + row_chunk]
        nonfinite += int(np.count_nonzero(~np.isfinite(actual_chunk)))
        difference = actual_chunk.astype(np.float64) - expected_chunk.astype(np.float64)
        absolute = np.abs(difference)
        allowed = 1e-6 + 1e-6 * np.abs(expected_chunk)
        outside += int(np.count_nonzero(absolute > allowed))
        max_abs = max(max_abs, float(absolute.max(initial=0.0)))
        difference_square_sum += float(np.sum(difference * difference))
        expected64 = expected_chunk.astype(np.float64)
        expected_square_sum += float(np.sum(expected64 * expected64))
    normalized = math.sqrt(
        difference_square_sum / max(expected_square_sum, np.finfo(np.float64).tiny)
    )
    return outside, max_abs, normalized, nonfinite


def _dense_spd(size, *, seed):
    generator = np.random.default_rng(seed)
    source = generator.normal(0.0, 0.25, (size, size)).astype(np.float32)
    source_t = torch.from_numpy(source)
    return (source_t @ source_t.T + torch.eye(size, dtype=torch.float32) * 2.0).numpy()


@pytest.mark.parametrize("block_size", (8, 16, 32))
def test_exl3_block_ldl_matches_torch_float32_oracle(block_size):
    hessian = _dense_spd(96, seed=3172 + block_size)
    expected = _torch_block_ldl_oracle(hessian, block_size=block_size).numpy()
    actual = np.asarray(exl3_block_ldl_mlx(mx.array(hessian), block_size=block_size))
    outside, max_abs, normalized, nonfinite = _drift_metrics(actual, expected)
    assert nonfinite == 0
    assert outside == 0
    assert max_abs <= 1e-6
    assert normalized <= 1e-6


def test_exl3_block_ldl_meets_float64_torch_oracle_limit():
    hessian = _dense_spd(128, seed=73172)
    expected = _torch_block_ldl_oracle(
        hessian.astype(np.float64), dtype=torch.float64
    ).numpy()
    actual = np.asarray(exl3_block_ldl_mlx(mx.array(hessian)))
    outside, max_abs, normalized, nonfinite = _drift_metrics(actual, expected)
    assert nonfinite == 0
    assert outside == 0
    assert max_abs <= 1e-6
    assert normalized <= 1e-6


def test_exl3_block_ldl_float32_normalization_boundaries():
    factor = np.eye(32, dtype=np.float32)
    factor[1:16, :15] += np.tril(
        np.full((15, 15), np.float32(2**-7), dtype=np.float32), k=-1
    )
    boundaries = np.array(
        [
            np.nextafter(np.float32(0.5), np.float32(0.0)),
            np.float32(0.5),
            np.nextafter(np.float32(0.5), np.float32(np.inf)),
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            -0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    factor[16:, :16] = np.resize(boundaries, (16, 16))
    factor_t = torch.from_numpy(factor)
    hessian = (factor_t @ factor_t.T).numpy()
    expected = _torch_block_ldl_oracle(hessian).numpy()
    actual = np.asarray(exl3_block_ldl_mlx(mx.array(hessian)))
    outside, max_abs, normalized, nonfinite = _drift_metrics(actual, expected)
    assert nonfinite == 0
    assert outside == 0
    assert max_abs <= 1e-6
    assert normalized <= 1e-6


def test_exl3_block_ldl_checks_positive_definite_boundary():
    below_one = np.nextafter(np.float32(1.0), np.float32(0.0))
    positive_definite = np.eye(16, dtype=np.float32)
    positive_definite[0, 1] = below_one
    positive_definite[1, 0] = below_one
    actual = np.asarray(exl3_block_ldl_mlx(mx.array(positive_definite)))
    np.testing.assert_array_equal(actual, np.eye(16, dtype=np.float32))

    for boundary in (
        np.float32(1.0),
        np.nextafter(np.float32(1.0), np.float32(np.inf)),
    ):
        invalid = np.eye(16, dtype=np.float32)
        invalid[0, 1] = boundary
        invalid[1, 0] = boundary
        with pytest.raises(ValueError, match="positive definite"):
            exl3_block_ldl_mlx(mx.array(invalid))


def _qwen_hessian(width):
    indices = np.arange(width, dtype=np.int32)
    distance = np.abs(indices[:, None] - indices[None, :])
    return np.power(np.float32(0.2), distance, dtype=np.float32)


@cache
def _qwen_width_metrics(in_features):
    hessian = _qwen_hessian(in_features)
    expected = _torch_block_ldl_oracle(hessian).numpy()
    actual_array = exl3_block_ldl_mlx(mx.array(hessian))
    metrics = _drift_metrics(np.asarray(actual_array), expected)
    del hessian, expected, actual_array
    gc.collect()
    mx.clear_cache()
    return metrics


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_block_ldl_qwen38_projection_oracle(name, out_features, in_features):
    del out_features
    outside, max_abs, normalized, nonfinite = _qwen_width_metrics(in_features)
    assert nonfinite == 0, name
    assert outside == 0, name
    assert max_abs <= 1e-6, name
    assert normalized <= 1e-6, name


def test_exl3_block_ldl_rejects_invalid_inputs():
    valid = mx.eye(16, dtype=mx.float32)
    with pytest.raises(ValueError, match="square rank-two"):
        exl3_block_ldl_mlx(mx.ones((16,), dtype=mx.float32))
    with pytest.raises(ValueError, match="square rank-two"):
        exl3_block_ldl_mlx(mx.ones((16, 32), dtype=mx.float32))
    with pytest.raises(ValueError, match="nonempty"):
        exl3_block_ldl_mlx(mx.zeros((0, 0), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32"):
        exl3_block_ldl_mlx(valid.astype(mx.float16))
    for block_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            exl3_block_ldl_mlx(valid, block_size=block_size)
    with pytest.raises(ValueError, match="divisible"):
        exl3_block_ldl_mlx(valid, block_size=3)
    for value in (np.nan, np.inf, -np.inf):
        invalid = np.eye(16, dtype=np.float32)
        invalid[3, 3] = value
        with pytest.raises(ValueError, match="finite"):
            exl3_block_ldl_mlx(mx.array(invalid))
