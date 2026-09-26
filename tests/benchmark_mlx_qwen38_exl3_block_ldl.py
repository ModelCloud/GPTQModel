# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native MLX EXL3 block-LDL on Qwen3.8-27B Hessian shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import torch

from gptqmodel.quantization.mlx_exl3_block_ldl import exl3_block_ldl_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_block_ldl import (
    _drift_metrics,
    _qwen_hessian,
    _torch_block_ldl_oracle,
)


def _median_ms(function, repeats):
    result = function()
    del result
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1000)
        del result
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | checkpoint shape | Hessian shape | outside tolerance | "
        "max abs drift | normalized drift | nonfinite | Torch median ms | "
        "MLX median ms | speedup",
        flush=True,
    )
    width_results = {}
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if in_features not in width_results:
            hessian = _qwen_hessian(in_features)
            torch_hessian = torch.from_numpy(hessian)
            mlx_hessian = mx.array(hessian)
            mx.eval(mlx_hessian)

            expected = _torch_block_ldl_oracle(torch_hessian).numpy()
            actual = exl3_block_ldl_mlx(mlx_hessian)
            accuracy = _drift_metrics(actual, expected)
            if accuracy[0] or accuracy[2] > 1e-6 or accuracy[3]:
                raise AssertionError(f"input width {in_features}: accuracy={accuracy}")
            del expected, actual
            gc.collect()

            torch_ms = _median_ms(
                partial(_torch_block_ldl_oracle, torch_hessian), args.repeats
            )
            mlx_ms = _median_ms(partial(exl3_block_ldl_mlx, mlx_hessian), args.repeats)
            width_results[in_features] = (accuracy, torch_ms, mlx_ms)
            del hessian, torch_hessian, mlx_hessian
            gc.collect()
            mx.clear_cache()

        accuracy, torch_ms, mlx_ms = width_results[in_features]
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{in_features} | {accuracy[0]} | "
            f"{accuracy[1]:.9g} | {accuracy[2]:.9g} | {accuracy[3]} | "
            f"{torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
