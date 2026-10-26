# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native EXL3 scale-search sampling against main's Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import torch

from gptqmodel.quantization.mlx_exl3_gss import exl3_global_scale_search_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_gss import (
    _patterned_weight,
    _torch_global_scale_search_oracle,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | checkpoint shape | EXL3 matrix shape | sampled tiles | "
        "scale drift | MSE abs drift | MSE relative drift | "
        "Torch oracle extrapolated ms | MLX measured ms | estimated speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        source = _patterned_weight(
            in_features,
            out_features,
            seed=33164 + out_features + in_features,
        )
        sampled = _torch_gss_sample_oracle(source)
        expected_scale, expected_mse = _torch_global_scale_search_oracle(
            sampled, bits=4, codebook="mcg"
        )
        mlx_weight = mx.array(source)
        mx.eval(mlx_weight)
        actual_scale, actual_mse = exl3_global_scale_search_mlx(mlx_weight, bits=4)
        scale_drift = abs(actual_scale - expected_scale)
        mse_drift = abs(actual_mse - expected_mse)
        mse_relative_drift = mse_drift / max(abs(expected_mse), 1e-20)
        if scale_drift > 1e-6 or mse_drift > 1e-6 or mse_relative_drift > 1e-6:
            raise AssertionError(
                f"{name}: scale drift {scale_drift}, MSE drift {mse_drift}, "
                f"relative MSE drift {mse_relative_drift}"
            )

        torch_weight = torch.from_numpy(source)
        torch_sample_ms = _median_ms(
            partial(_main_torch_gss_sample, torch_weight), args.repeats
        )
        one_tile = sampled[:1]
        torch_tile_ms = _median_ms(
            partial(
                _torch_global_scale_search_oracle,
                one_tile,
                bits=4,
                codebook="mcg",
            ),
            args.repeats,
        )
        sample_count = sampled.shape[0]
        torch_ms = torch_sample_ms + torch_tile_ms * sample_count
        mlx_ms = _median_ms(
            partial(exl3_global_scale_search_mlx, mlx_weight, bits=4),
            args.repeats,
        )
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{out_features} | {sample_count} | "
            f"{scale_drift:.9g} | {mse_drift:.9g} | "
            f"{mse_relative_drift:.9g} | "
            f"{torch_ms:.3f} | {mlx_ms:.3f} | "
            f"{torch_ms / mlx_ms:.2f}x",
            flush=True,
        )

        del source, sampled, mlx_weight, torch_weight, one_tile
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
