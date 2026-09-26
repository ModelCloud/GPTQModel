# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark native EXL3 scale-search sampling against main's Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_gss import (
    exl3_sample_global_scale_tiles_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_gss import (
    _torch_gss_sample_oracle,
    _torch_tensor_core_permutation,
)


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_gss_sample(weight, *, width=3):
    tile_rows = weight.shape[0] // 16
    tile_columns = weight.shape[1] // 16
    permutation = _torch_tensor_core_permutation()
    tiles = []
    for diagonal in range(max(tile_rows, tile_columns)):
        for offset in range(width):
            row = (diagonal % tile_rows) * 16
            column = ((diagonal + offset) % tile_columns) * 16
            tile = weight[row : row + 16, column : column + 16].clone().view(256)
            tiles.append(tile[permutation])
    return torch.stack(tiles)


def _bit_mismatches(actual, expected):
    return int(
        np.count_nonzero(
            np.asarray(actual).view(np.uint32) != np.asarray(expected).view(np.uint32)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | checkpoint shape | EXL3 matrix shape | sampled values | "
        "bit mismatches | max abs error | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(23164 + out_features + in_features)
        source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
        expected = _torch_gss_sample_oracle(source)
        mlx_weight = mx.array(source)
        mx.eval(mlx_weight)
        actual = np.asarray(exl3_sample_global_scale_tiles_mlx(mlx_weight))
        mismatches = _bit_mismatches(actual, expected)
        max_abs_error = float(np.max(np.abs(actual - expected)))
        if mismatches:
            raise AssertionError(f"{name}: {mismatches} bit mismatches")

        torch_weight = torch.from_numpy(source)
        torch_ms = _median_ms(
            partial(_main_torch_gss_sample, torch_weight), args.repeats
        )
        mlx_ms = _median_ms(
            partial(exl3_sample_global_scale_tiles_mlx, mlx_weight), args.repeats
        )
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{out_features} | {expected.size} | {mismatches} | "
            f"{max_abs_error:.3e} | {torch_ms:.3f} | {mlx_ms:.3f} | "
            f"{torch_ms / mlx_ms:.2f}x",
            flush=True,
        )

        del source, expected, actual, mlx_weight, torch_weight
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
