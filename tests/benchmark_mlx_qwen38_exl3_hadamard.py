# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark EXL3's native MLX Hadamard transform against its Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_hadamard import exl3_hadamard_128_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_hadamard import (
    _normalized_rms_drift,
    _torch_hadamard_128,
    _torch_hadamard_oracle,
)


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_blockwise_(matrix, axis):
    """Mirror EXL3 main's in-place blockwise Torch implementation."""
    hadamard = _torch_hadamard_128()
    if axis == 1:
        for start in range(0, matrix.shape[1], 128):
            matrix[:, start : start + 128] = matrix[:, start : start + 128] @ hadamard
    else:
        for start in range(0, matrix.shape[0], 128):
            matrix[start : start + 128] = hadamard @ matrix[start : start + 128]
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | axis | shape | outside 1e-6 | max abs drift | "
        "normalized RMS drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(10421 + rows + columns)
        source = rng.normal(0.0, 0.2, (rows, columns)).astype(np.float32)
        mlx_source = mx.array(source)
        mx.eval(mlx_source)

        for axis in (0, 1):
            expected = _torch_hadamard_oracle(source, axis)
            actual = np.asarray(exl3_hadamard_128_mlx(mlx_source, axis=axis))
            error = np.abs(actual - expected)
            allowed = 1e-6 + 1e-6 * np.abs(expected)
            outside = int(np.count_nonzero(error > allowed))
            if outside:
                raise AssertionError(
                    f"{name} axis={axis}: {outside} outputs exceed tolerance"
                )
            max_abs = float(error.max())
            normalized_rms = _normalized_rms_drift(actual, expected)

            torch_source = torch.from_numpy(source.copy())
            torch_call = partial(_main_torch_blockwise_, torch_source, axis)
            mlx_call = partial(exl3_hadamard_128_mlx, mlx_source, axis=axis)
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            print(
                f"{name} | {axis} | {rows}x{columns} | {outside} | "
                f"{max_abs:.9g} | {normalized_rms:.9g} | {torch_ms:.3f} | "
                f"{mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
                flush=True,
            )
            del expected, actual, error, torch_source, torch_call, mlx_call
            gc.collect()

        del source, mlx_source
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
