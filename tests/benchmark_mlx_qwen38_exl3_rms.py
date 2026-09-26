# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark native MLX EXL3 block RMS against main's Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_rms import exl3_block_rms_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_rms import (
    _normalized_rms_drift,
    _torch_block_rms_oracle,
)


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_block_rms(matrix, axis, block_size=32):
    squared_sum = None
    for block in torch.split(matrix, block_size, dim=axis):
        block_sum = block.square().sum(dim=axis, keepdim=True)
        squared_sum = block_sum if squared_sum is None else squared_sum + block_sum
    return (squared_sum / matrix.shape[axis]).sqrt()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | axis | checkpoint shape | EXL3 matrix shape | outside 1e-6 | "
        "max abs drift | normalized RMS drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(14111 + out_features + in_features)
        source = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
        torch_source = torch.from_numpy(source)
        mlx_source = mx.array(source)
        mx.eval(mlx_source)

        for axis in (0, 1):
            expected = _torch_block_rms_oracle(source, axis)
            actual = np.asarray(exl3_block_rms_mlx(mlx_source, axis=axis))
            error = np.abs(actual - expected)
            allowed = 1e-6 + 1e-6 * np.abs(expected)
            outside = int(np.count_nonzero(error > allowed))
            max_abs = float(error.max())
            normalized_rms = _normalized_rms_drift(actual, expected)
            if outside or normalized_rms > 1e-6:
                raise AssertionError(
                    f"{name} axis={axis}: outside={outside}, "
                    f"normalized RMS={normalized_rms}"
                )

            torch_call = partial(_main_torch_block_rms, torch_source, axis)
            mlx_call = partial(exl3_block_rms_mlx, mlx_source, axis=axis)
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            print(
                f"{name} | {axis} | {out_features}x{in_features} | "
                f"{in_features}x{out_features} | {outside} | {max_abs:.9g} | "
                f"{normalized_rms:.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | "
                f"{torch_ms / mlx_ms:.2f}x",
                flush=True,
            )

        del source, torch_source, mlx_source, expected, actual, error
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
