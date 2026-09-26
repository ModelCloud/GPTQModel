# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Torch-oracle checks for native MLX EXL3 Hessian finalization."""

import gc
import math
import sys
from functools import cache, lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_hessian import exl3_finalize_hessian_mlx


@lru_cache(maxsize=2)
def _torch_hadamard_128(dtype):
    matrix = torch.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (1.0 / math.sqrt(128.0))


def _torch_blockwise_hadamard_(matrix, *, axis):
    hadamard = _torch_hadamard_128(matrix.dtype)
    if axis == 1:
        for start in range(0, matrix.shape[1], 128):
            block = matrix[:, start : start + 128]
            matrix[:, start : start + 128] = block @ hadamard
    else:
        for start in range(0, matrix.shape[0], 128):
            block = matrix[start : start + 128]
            matrix[start : start + 128] = hadamard @ block
    return matrix


def _torch_block_ldl(hessian, *, block_size=16):
    factor = torch.linalg.cholesky(hessian)
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
    block_indices = torch.arange(block_count)
    factor_blocks[block_indices, block_indices] = torch.stack(
        [torch.eye(block_size, dtype=factor.dtype)] * block_count
    )
    diagonal_indices = torch.arange(columns)
    factor[diagonal_indices, diagonal_indices] = 0
    return factor


def _torch_finalize_hessian_oracle(
    hessian,
    signs,
    *,
    sample_count,
    sigma_reg=0.025,
    block_size=16,
    dtype=torch.float32,
):
    source = torch.as_tensor(hessian, dtype=dtype).clone()
    signs = torch.as_tensor(signs, dtype=dtype)
    if sample_count:
        source /= sample_count
        diagonal_mean = torch.diagonal(source).mean()
        fallback = diagonal_mean.item() < 1e-20
        diagonal_indices = torch.arange(source.shape[0])
        source[diagonal_indices, diagonal_indices] += sigma_reg * diagonal_mean
    else:
        source.zero_()
        fallback = True
    diagonal = torch.diagonal(source).clone()
    source *= signs.unsqueeze(0)
    _torch_blockwise_hadamard_(source, axis=1)
    source *= signs.unsqueeze(1)
    _torch_blockwise_hadamard_(source, axis=0)
    factor = None if fallback else _torch_block_ldl(source, block_size=block_size)
    return fallback, source, factor, diagonal


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


def _qwen_hessian(width, *, sample_count=8):
    indices = np.arange(width, dtype=np.int32)
    distance = np.abs(indices[:, None] - indices[None, :])
    normalized = np.power(np.float32(0.2), distance, dtype=np.float32)
    return normalized * np.float32(sample_count)


def _qwen_signs(width):
    generator = np.random.default_rng(73172 + width)
    return np.where(generator.integers(0, 2, size=width), -1.0, 1.0).astype(np.float32)


@cache
def _qwen_width_metrics(width):
    sample_count = 8
    hessian = _qwen_hessian(width, sample_count=sample_count)
    signs = _qwen_signs(width)
    reference = _torch_finalize_hessian_oracle(
        hessian,
        signs,
        sample_count=sample_count,
        dtype=torch.float64,
    )
    torch_float32 = _torch_finalize_hessian_oracle(
        hessian,
        signs,
        sample_count=sample_count,
    )
    actual = exl3_finalize_hessian_mlx(
        mx.array(hessian), mx.array(signs), sample_count=sample_count
    )
    metrics = (
        _drift_metrics(actual[1], reference[1].numpy()),
        _drift_metrics(actual[2], reference[2].numpy()),
        _drift_metrics(np.asarray(actual[3])[None, :], reference[3].numpy()[None, :]),
        _drift_metrics(torch_float32[1].numpy(), reference[1].numpy()),
        _drift_metrics(torch_float32[2].numpy(), reference[2].numpy()),
    )
    del hessian, signs, reference, torch_float32, actual
    gc.collect()
    mx.clear_cache()
    return metrics


def test_exl3_finalize_hessian_matches_torch_float32_oracle():
    rng = np.random.default_rng(93175)
    source = rng.normal(0.0, 0.1, (256, 256)).astype(np.float32)
    source_torch = torch.from_numpy(source)
    hessian = (
        source_torch @ source_torch.T + torch.eye(256, dtype=torch.float32) * 4.0
    ).numpy()
    signs = _qwen_signs(256)
    expected = _torch_finalize_hessian_oracle(hessian, signs, sample_count=7)
    actual = exl3_finalize_hessian_mlx(
        mx.array(hessian), mx.array(signs), sample_count=7
    )
    assert actual[0] is expected[0]
    for actual_array, expected_tensor in zip(actual[1:], expected[1:]):
        metrics = _drift_metrics(actual_array, expected_tensor.numpy())
        assert metrics[0] == 0
        assert metrics[3] == 0
        assert metrics[2] <= 1e-6


@pytest.mark.parametrize(
    "diagonal_value,expected_fallback",
    (
        (np.nextafter(np.float32(1e-20), np.float32(0.0)), True),
        (np.nextafter(np.float32(1e-20), np.float32(np.inf)), False),
    ),
)
def test_exl3_finalize_hessian_fallback_threshold_boundary(
    diagonal_value, expected_fallback
):
    hessian = np.eye(128, dtype=np.float32) * diagonal_value
    signs = np.ones((128,), dtype=np.float32)
    fallback, transformed, factor, diagonal = exl3_finalize_hessian_mlx(
        mx.array(hessian), mx.array(signs), sample_count=1
    )
    assert fallback is expected_fallback
    assert (factor is None) is expected_fallback
    assert np.isfinite(np.asarray(transformed)).all()
    assert np.isfinite(np.asarray(diagonal)).all()


def test_exl3_finalize_hessian_empty_capture_falls_back():
    hessian = mx.zeros((128, 128), dtype=mx.float32)
    signs = mx.ones((128,), dtype=mx.float32)
    fallback, transformed, factor, diagonal = exl3_finalize_hessian_mlx(
        hessian, signs, sample_count=0
    )
    assert fallback is True
    assert factor is None
    np.testing.assert_array_equal(np.asarray(transformed), np.zeros((128, 128)))
    np.testing.assert_array_equal(np.asarray(diagonal), np.zeros((128,)))


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_finalize_hessian_qwen38_float64_oracle(name, out_features, in_features):
    del out_features
    hessian, factor, diagonal, torch_hessian, torch_factor = _qwen_width_metrics(
        in_features
    )
    for label, metrics in (
        ("hessian", hessian),
        ("factor", factor),
        ("diagonal", diagonal),
    ):
        assert metrics[0] == 0, f"{name} {label}: {metrics}"
        assert metrics[2] <= 1e-6, f"{name} {label}: {metrics}"
        assert metrics[3] == 0, f"{name} {label}: {metrics}"
    assert hessian[2] <= torch_hessian[2], name
    assert factor[2] <= torch_factor[2], name


def test_exl3_finalize_hessian_rejects_invalid_inputs():
    valid = mx.eye(128, dtype=mx.float32)
    signs = mx.ones((128,), dtype=mx.float32)
    invalid_calls = (
        (mx.ones((128,), dtype=mx.float32), signs, {}, "square rank-two"),
        (mx.ones((128, 256), dtype=mx.float32), signs, {}, "square rank-two"),
        (
            mx.zeros((0, 0), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.float32),
            {},
            "nonempty",
        ),
        (valid.astype(mx.float16), signs, {}, "float32"),
        (valid.astype(mx.bfloat16), signs, {}, "float32"),
        (
            mx.eye(144, dtype=mx.float32),
            mx.ones((144,), dtype=mx.float32),
            {},
            "divisible by 128",
        ),
        (valid, mx.ones((127,), dtype=mx.float32), {}, "one value"),
        (valid, signs.astype(mx.float16), {}, "float32"),
        (valid, signs, {"sample_count": -1}, "nonnegative integer"),
        (valid, signs, {"sample_count": True}, "nonnegative integer"),
        (valid, signs, {"sigma_reg": -0.1}, "finite nonnegative"),
        (valid, signs, {"sigma_reg": np.inf}, "finite nonnegative"),
        (valid, signs, {"block_size": 0}, "positive integer"),
        (valid, signs, {"block_size": True}, "positive integer"),
        (valid, signs, {"block_size": 48}, "divisible by block_size"),
    )
    for hessian, input_signs, kwargs, message in invalid_calls:
        parameters = {"sample_count": 1, **kwargs}
        with pytest.raises(ValueError, match=message):
            exl3_finalize_hessian_mlx(hessian, input_signs, **parameters)

    invalid_signs = np.ones((128,), dtype=np.float32)
    invalid_signs[3] = 0
    with pytest.raises(ValueError, match=r"only -1 or \+1"):
        exl3_finalize_hessian_mlx(valid, mx.array(invalid_signs), sample_count=1)
    for value in (np.nan, np.inf, -np.inf):
        invalid = np.eye(128, dtype=np.float32)
        invalid[3, 3] = value
        with pytest.raises(ValueError, match="finite"):
            exl3_finalize_hessian_mlx(mx.array(invalid), signs, sample_count=1)
