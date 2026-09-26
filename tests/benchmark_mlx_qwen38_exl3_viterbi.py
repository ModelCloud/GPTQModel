# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native MLX EXL3 path search on Qwen3.8-27B projections."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3 import exl3_quantize_tiles_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_viterbi import _torch_viterbi_oracle_tensors


def _median_ms(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _accuracy(actual_values, actual_indices, expected_values, expected_indices):
    actual_values = np.asarray(actual_values).reshape(-1, 256)
    actual_indices = np.asarray(actual_indices).reshape(-1, 256)
    tile_count = actual_values.shape[0]
    bank_size = expected_values.shape[0]
    index_mismatches = 0
    value_mismatches = 0
    max_abs_drift = 0.0
    for start in range(0, tile_count, 8192):
        stop = min(start + 8192, tile_count)
        rows = np.arange(start, stop) % bank_size
        expected_i = expected_indices[rows]
        expected_v = expected_values[rows]
        index_mismatches += int(
            np.count_nonzero(actual_indices[start:stop] != expected_i)
        )
        delta = np.abs(actual_values[start:stop] - expected_v)
        value_mismatches += int(np.count_nonzero(delta))
        max_abs_drift = max(max_abs_drift, float(delta.max()))
    return index_mismatches, value_mismatches, max_abs_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workspace-mib", type=int, default=256)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.workspace_mib < 1:
        parser.error("--workspace-mib must be positive")

    bits = 4
    codebook = "mcg"
    workspace_bytes = args.workspace_mib << 20
    print(
        "projection | shape | tiles | values | index mismatches | value mismatches | "
        "max abs drift | Torch oracle extrapolated ms | MLX measured ms | estimated speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(7100 + out_features + in_features)
        source_bank = rng.normal(0.0, 1.75, (4, 256)).astype(np.float32)
        torch_bank = torch.from_numpy(source_bank)
        torch_call = partial(_torch_viterbi_oracle_tensors, torch_bank, bits, codebook)
        expected_values_t, expected_indices_t = torch_call()
        expected_values = expected_values_t.numpy()
        expected_indices = expected_indices_t.numpy()

        tile_shape = (in_features // 16, out_features // 16, 256)
        tile_count = tile_shape[0] * tile_shape[1]
        values = np.resize(source_bank, (tile_count, 256)).reshape(tile_shape)
        mlx_values = mx.array(values)
        mx.eval(mlx_values)
        mlx_call = partial(
            exl3_quantize_tiles_mlx,
            mlx_values,
            bits=bits,
            codebook=codebook,
            workspace_bytes=workspace_bytes,
        )
        actual_values, actual_indices = mlx_call()
        index_mismatches, value_mismatches, max_abs_drift = _accuracy(
            actual_values,
            actual_indices,
            expected_values,
            expected_indices,
        )
        if index_mismatches or value_mismatches:
            raise AssertionError(
                f"{name}: {index_mismatches} indices and {value_mismatches} values differ"
            )

        torch_bank_ms = _median_ms(torch_call, args.repeats)
        torch_extrapolated_ms = torch_bank_ms / source_bank.shape[0] * tile_count
        mlx_ms = _median_ms(mlx_call, args.repeats)
        print(
            f"{name} | {out_features}x{in_features} | {tile_count} | {tile_count * 256} | "
            f"{index_mismatches} | {value_mismatches} | {max_abs_drift:.9g} | "
            f"{torch_extrapolated_ms:.3f} | {mlx_ms:.3f} | "
            f"{torch_extrapolated_ms / mlx_ms:.2f}x",
            flush=True,
        )

        del values, mlx_values, actual_values, actual_indices
        del expected_values_t, expected_indices_t, expected_values, expected_indices
        del torch_bank, torch_call, mlx_call
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
