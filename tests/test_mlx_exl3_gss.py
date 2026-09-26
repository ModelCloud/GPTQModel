# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle checks for native EXL3 scale-search sampling."""

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

from gptqmodel.quantization.mlx_exl3_gss import (  # noqa: E402
    exl3_sample_global_scale_tiles_mlx,
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


def _assert_float32_bits_equal(actual, expected, *, err_msg=None):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
        err_msg=err_msg,
    )


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
