# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for native EXL3 global-scale search."""

import gc
import math
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3_gss import (  # noqa: E402
    exl3_global_scale_search_mlx,
    exl3_sample_global_scale_tiles_mlx,
)
from tests.test_mlx_exl3_viterbi import (  # noqa: E402
    _torch_viterbi_oracle_tensors,
)


@lru_cache(maxsize=1)
def _torch_tensor_core_permutation():
    permutation = torch.empty(256, dtype=torch.int64)
    for thread in range(32):
        rows = (
            (thread % 4) * 2,
            (thread % 4) * 2 + 1,
            (thread % 4) * 2 + 8,
            (thread % 4) * 2 + 9,
        )
        columns = (thread // 4, thread // 4 + 8)
        for item in range(8):
            permutation[thread * 8 + item] = rows[item % 4] * 16 + columns[item // 4]
    return permutation


def _torch_gss_sample_oracle(weight, *, width=3):
    source = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32))
    tile_rows = source.shape[0] // 16
    tile_columns = source.shape[1] // 16
    permutation = _torch_tensor_core_permutation()
    tiles = []
    for diagonal in range(max(tile_rows, tile_columns)):
        for offset in range(width):
            row = (diagonal % tile_rows) * 16
            column = ((diagonal + offset) % tile_columns) * 16
            tile = source[row : row + 16, column : column + 16].clone().view(256)
            tiles.append(tile[permutation])
    return torch.stack(tiles).contiguous().numpy()


def _torch_global_scale_search_oracle(tiles, *, bits, codebook):
    source = torch.from_numpy(np.ascontiguousarray(tiles, dtype=np.float32))
    unique, inverse = torch.unique(source, dim=0, return_inverse=True)

    def test_scale(scale):
        quantized, _ = _torch_viterbi_oracle_tensors(unique * scale, bits, codebook)
        quantized = quantized[inverse]
        return float(((quantized / scale - source) ** 2).mean().item())

    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    lower = 0.1
    upper = 1.9
    tolerance = 0.01
    x1 = lower + resphi * (upper - lower)
    x2 = upper - resphi * (upper - lower)
    f1 = test_scale(x1)
    f2 = test_scale(x2)
    while abs(upper - lower) > tolerance:
        if f1 < f2:
            upper = x2
            x2 = x1
            f2 = f1
            x1 = lower + resphi * (upper - lower)
            f1 = test_scale(x1)
        else:
            lower = x1
            x1 = x2
            f1 = f2
            x2 = upper - resphi * (upper - lower)
            f2 = test_scale(x2)
    final_mse = (
        torch.tensor(f1, dtype=torch.float32) + torch.tensor(f2, dtype=torch.float32)
    ) / 2
    return (lower + upper) / 2, float(final_mse.item())


def _patterned_weight(rows, columns, *, seed):
    rng = np.random.default_rng(seed)
    lane_tiles = rng.normal(0.0, 1.75, (4, 256)).astype(np.float32)
    raw_tiles = np.empty_like(lane_tiles)
    permutation = _torch_tensor_core_permutation().numpy()
    raw_tiles[:, permutation] = lane_tiles
    strip = np.concatenate([tile.reshape(16, 16) for tile in raw_tiles], axis=1)
    repeats = (columns + strip.shape[1] - 1) // strip.shape[1]
    return np.tile(strip, (rows // 16, repeats))[:, :columns].copy()


def _assert_float32_bits_equal(actual, expected, *, err_msg=None):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
        err_msg=err_msg,
    )


def _assert_scale_search_matches(
    actual_scale, actual_mse, expected_scale, expected_mse, *, err_msg=None
):
    assert actual_scale == expected_scale, err_msg
    absolute_drift = abs(actual_mse - expected_mse)
    relative_drift = absolute_drift / max(abs(expected_mse), 1e-20)
    assert absolute_drift <= 1e-6, err_msg
    assert relative_drift <= 1e-6, err_msg


@pytest.mark.parametrize("shape", ((48, 80), (80, 48)))
@pytest.mark.parametrize("width", (1, 3, 7))
def test_exl3_gss_sample_small_torch_oracle(shape, width):
    rng = np.random.default_rng(3164 + shape[0] + width)
    source = rng.normal(0.0, 0.4, shape).astype(np.float32)
    expected = _torch_gss_sample_oracle(source, width=width)
    actual = np.asarray(
        exl3_sample_global_scale_tiles_mlx(mx.array(source), width=width)
    )
    _assert_float32_bits_equal(actual, expected)


def test_exl3_gss_sample_boundaries_preserve_exact_bits_and_wrapping():
    boundary = np.array(
        [
            -np.finfo(np.float32).max,
            -1.0,
            np.nextafter(np.float32(-1), np.float32(-np.inf)),
            np.nextafter(np.float32(-1), np.float32(0)),
            np.nextafter(np.float32(0), np.float32(-1)),
            -0.0,
            0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            np.nextafter(np.float32(1), np.float32(0)),
            1.0,
            np.nextafter(np.float32(1), np.float32(np.inf)),
            np.finfo(np.float32).max,
        ],
        dtype=np.float32,
    )
    source = np.resize(boundary, (32, 48)).astype(np.float32, copy=False)
    expected = _torch_gss_sample_oracle(source, width=5)
    actual = np.asarray(exl3_sample_global_scale_tiles_mlx(mx.array(source), width=5))
    _assert_float32_bits_equal(actual, expected)


@pytest.mark.parametrize("bits", (2, 4, 8))
def test_exl3_global_scale_search_small_torch_oracle(bits):
    source = _patterned_weight(32, 48, seed=8164 + bits)
    sampled = _torch_gss_sample_oracle(source)
    expected_scale, expected_mse = _torch_global_scale_search_oracle(
        sampled, bits=bits, codebook="mcg"
    )
    actual_scale, actual_mse = exl3_global_scale_search_mlx(mx.array(source), bits=bits)
    _assert_scale_search_matches(actual_scale, actual_mse, expected_scale, expected_mse)


@pytest.mark.parametrize("codebook", ("3inst", "mcg", "mul1"))
def test_exl3_global_scale_search_rounding_boundaries(codebook):
    values = np.array(
        [
            np.nextafter(np.float16(-1), np.float16(-np.inf)),
            np.float16(-1),
            np.nextafter(np.float16(-1), np.float16(0)),
            np.nextafter(np.float16(0), np.float16(-1)),
            np.float16(-0.0),
            np.float16(0.0),
            np.nextafter(np.float16(0), np.float16(1)),
            np.nextafter(np.float16(1), np.float16(0)),
            np.float16(1),
            np.nextafter(np.float16(1), np.float16(np.inf)),
        ],
        dtype=np.float16,
    ).astype(np.float32)
    source = np.resize(values, (16, 16)).astype(np.float32, copy=False)
    sampled = _torch_gss_sample_oracle(source)
    expected_scale, expected_mse = _torch_global_scale_search_oracle(
        sampled, bits=4, codebook=codebook
    )
    actual_scale, actual_mse = exl3_global_scale_search_mlx(
        mx.array(source), bits=4, codebook=codebook
    )
    _assert_scale_search_matches(actual_scale, actual_mse, expected_scale, expected_mse)


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_gss_sample_qwen38_projection_oracle(name, out_features, in_features):
    rng = np.random.default_rng(23164 + out_features + in_features)
    source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
    expected = _torch_gss_sample_oracle(source)
    actual = np.asarray(exl3_sample_global_scale_tiles_mlx(mx.array(source)))
    _assert_float32_bits_equal(actual, expected, err_msg=name)

    del source, expected, actual
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_global_scale_search_qwen38_projection_oracle(
    name, out_features, in_features
):
    source = _patterned_weight(
        in_features,
        out_features,
        seed=33164 + out_features + in_features,
    )
    sampled = _torch_gss_sample_oracle(source)
    expected_scale, expected_mse = _torch_global_scale_search_oracle(
        sampled, bits=4, codebook="mcg"
    )
    actual_scale, actual_mse = exl3_global_scale_search_mlx(mx.array(source), bits=4)
    _assert_scale_search_matches(
        actual_scale,
        actual_mse,
        expected_scale,
        expected_mse,
        err_msg=name,
    )

    del source, sampled
    gc.collect()
    mx.clear_cache()


def test_exl3_gss_sample_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="rank-two"):
        exl3_sample_global_scale_tiles_mlx(mx.zeros((256,), dtype=mx.float32))
    with pytest.raises(ValueError, match="nonempty"):
        exl3_sample_global_scale_tiles_mlx(mx.zeros((0, 16), dtype=mx.float32))
    with pytest.raises(ValueError, match="divisible by 16"):
        exl3_sample_global_scale_tiles_mlx(mx.zeros((17, 16), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32"):
        exl3_sample_global_scale_tiles_mlx(mx.zeros((16, 16), dtype=mx.float16))
    for width in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            exl3_sample_global_scale_tiles_mlx(
                mx.zeros((16, 16), dtype=mx.float32), width=width
            )


def test_exl3_global_scale_search_rejects_invalid_quantization_arguments():
    valid = mx.zeros((16, 16), dtype=mx.float32)
    for bits in (0, 9, 4.0, True):
        with pytest.raises(ValueError, match="bits"):
            exl3_global_scale_search_mlx(valid, bits=bits)
    for codebook in ("unknown", 1, None):
        with pytest.raises(ValueError, match="codebook"):
            exl3_global_scale_search_mlx(valid, bits=4, codebook=codebook)
    with pytest.raises(ValueError, match="positive integer"):
        exl3_global_scale_search_mlx(valid, bits=4, workspace_bytes=0)
