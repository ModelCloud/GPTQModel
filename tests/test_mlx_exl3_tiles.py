# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle checks for EXL3's native MLX tile layout."""

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

from gptqmodel.quantization.mlx_exl3_tiles import (
    exl3_from_tensor_core_tiles_mlx,
    exl3_to_tensor_core_tiles_mlx,
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


def _torch_to_tiles_oracle(weight):
    source = torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32))
    rows, columns = source.shape
    tiles = (
        source.reshape(rows // 16, 16, columns // 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(rows // 16, columns // 16, 256)
    )
    return tiles[:, :, _torch_tensor_core_permutation()].contiguous().numpy()


def _torch_from_tiles_oracle(tiles):
    source = torch.from_numpy(np.ascontiguousarray(tiles, dtype=np.float32))
    tile_rows, tile_columns, _ = source.shape
    inverse = torch.argsort(_torch_tensor_core_permutation())
    return (
        source[:, :, inverse]
        .reshape(tile_rows, tile_columns, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(tile_rows * 16, tile_columns * 16)
        .contiguous()
        .numpy()
    )


def _assert_float32_bits_equal(actual, expected, *, err_msg=None):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
        err_msg=err_msg,
    )


def test_exl3_tile_layout_small_torch_oracle_and_roundtrip():
    rng = np.random.default_rng(1551)
    source = rng.normal(0.0, 0.4, (48, 80)).astype(np.float32)
    expected_tiles = _torch_to_tiles_oracle(source)
    actual_tiles = np.asarray(exl3_to_tensor_core_tiles_mlx(mx.array(source)))
    _assert_float32_bits_equal(actual_tiles, expected_tiles)

    expected_weight = _torch_from_tiles_oracle(expected_tiles)
    actual_weight = np.asarray(
        exl3_from_tensor_core_tiles_mlx(mx.array(expected_tiles))
    )
    _assert_float32_bits_equal(actual_weight, expected_weight)
    _assert_float32_bits_equal(actual_weight, source)


def test_exl3_tile_layout_boundaries_preserve_exact_bits():
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
    source = np.resize(boundary, (32, 32)).astype(np.float32, copy=False)
    expected_tiles = _torch_to_tiles_oracle(source)
    actual_tiles = np.asarray(exl3_to_tensor_core_tiles_mlx(mx.array(source)))
    _assert_float32_bits_equal(actual_tiles, expected_tiles)

    restored = np.asarray(exl3_from_tensor_core_tiles_mlx(mx.array(actual_tiles)))
    _assert_float32_bits_equal(restored, source)


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_tile_layout_qwen38_projection_oracle(name, out_features, in_features):
    rng = np.random.default_rng(12611 + out_features + in_features)
    source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
    expected_tiles = _torch_to_tiles_oracle(source)
    actual_tiles = np.asarray(exl3_to_tensor_core_tiles_mlx(mx.array(source)))
    _assert_float32_bits_equal(actual_tiles, expected_tiles, err_msg=name)

    expected_weight = _torch_from_tiles_oracle(expected_tiles)
    actual_weight = np.asarray(
        exl3_from_tensor_core_tiles_mlx(mx.array(expected_tiles))
    )
    _assert_float32_bits_equal(actual_weight, expected_weight, err_msg=name)
    _assert_float32_bits_equal(actual_weight, source, err_msg=f"{name} roundtrip")

    del source, expected_tiles, actual_tiles, expected_weight, actual_weight
    gc.collect()
    mx.clear_cache()


def test_exl3_tile_layout_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="rank-two"):
        exl3_to_tensor_core_tiles_mlx(mx.zeros((256,), dtype=mx.float32))
    with pytest.raises(ValueError, match="nonempty"):
        exl3_to_tensor_core_tiles_mlx(mx.zeros((0, 16), dtype=mx.float32))
    with pytest.raises(ValueError, match="divisible by 16"):
        exl3_to_tensor_core_tiles_mlx(mx.zeros((17, 16), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32"):
        exl3_to_tensor_core_tiles_mlx(mx.zeros((16, 16), dtype=mx.float16))

    with pytest.raises(ValueError, match="shape"):
        exl3_from_tensor_core_tiles_mlx(mx.zeros((1, 256), dtype=mx.float32))
    with pytest.raises(ValueError, match="shape"):
        exl3_from_tensor_core_tiles_mlx(mx.zeros((1, 1, 255), dtype=mx.float32))
    with pytest.raises(ValueError, match="at least one tile"):
        exl3_from_tensor_core_tiles_mlx(mx.zeros((0, 1, 256), dtype=mx.float32))
    with pytest.raises(ValueError, match="float32"):
        exl3_from_tensor_core_tiles_mlx(mx.zeros((1, 1, 256), dtype=mx.float16))
