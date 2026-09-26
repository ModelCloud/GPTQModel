# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native MLX EXL3 tile layout against main's Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_tiles import (
    exl3_from_tensor_core_tiles_mlx,
    exl3_to_tensor_core_tiles_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_tiles import (
    _torch_from_tiles_oracle,
    _torch_tensor_core_permutation,
    _torch_to_tiles_oracle,
)


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_to_tiles(weight):
    rows, columns = weight.shape
    tiles = (
        weight.reshape(rows // 16, 16, columns // 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(rows // 16, columns // 16, 256)
    )
    return tiles[:, :, _torch_tensor_core_permutation()].contiguous()


def _main_torch_from_tiles(tiles):
    tile_rows, tile_columns, _ = tiles.shape
    inverse = torch.argsort(_torch_tensor_core_permutation())
    return (
        tiles[:, :, inverse]
        .reshape(tile_rows, tile_columns, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(tile_rows * 16, tile_columns * 16)
        .contiguous()
    )


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
        "projection | direction | checkpoint shape | EXL3 matrix shape | "
        "bit mismatches | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(12611 + out_features + in_features)
        source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
        expected_tiles = _torch_to_tiles_oracle(source)
        expected_weight = _torch_from_tiles_oracle(expected_tiles)

        mlx_weight = mx.array(source)
        mlx_tiles = mx.array(expected_tiles)
        mx.eval(mlx_weight, mlx_tiles)
        actual_tiles = exl3_to_tensor_core_tiles_mlx(mlx_weight)
        actual_weight = exl3_from_tensor_core_tiles_mlx(mlx_tiles)
        forward_mismatches = _bit_mismatches(actual_tiles, expected_tiles)
        reverse_mismatches = _bit_mismatches(actual_weight, expected_weight)
        if forward_mismatches or reverse_mismatches:
            raise AssertionError(
                f"{name}: {forward_mismatches} forward and "
                f"{reverse_mismatches} reverse bit mismatches"
            )

        torch_weight = torch.from_numpy(source)
        torch_tiles = torch.from_numpy(expected_tiles)
        directions = (
            (
                "to_tiles",
                forward_mismatches,
                partial(_main_torch_to_tiles, torch_weight),
                partial(exl3_to_tensor_core_tiles_mlx, mlx_weight),
            ),
            (
                "from_tiles",
                reverse_mismatches,
                partial(_main_torch_from_tiles, torch_tiles),
                partial(exl3_from_tensor_core_tiles_mlx, mlx_tiles),
            ),
        )
        for direction, mismatches, torch_call, mlx_call in directions:
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            print(
                f"{name} | {direction} | {out_features}x{in_features} | "
                f"{in_features}x{out_features} | {mismatches} | {torch_ms:.3f} | "
                f"{mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
                flush=True,
            )

        del source, expected_tiles, expected_weight, actual_tiles, actual_weight
        del mlx_weight, mlx_tiles, torch_weight, torch_tiles, directions
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
