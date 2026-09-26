# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native MLX EXL3 Hessian finalization on Qwen3.8-27B."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import torch

from gptqmodel.quantization.mlx_exl3_hessian import exl3_finalize_hessian_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_hessian import (
    _qwen_hessian,
    _qwen_signs,
    _qwen_width_metrics,
    _torch_finalize_hessian_oracle,
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
        "projection | checkpoint shape | Hessian shape | MLX H max abs | "
        "MLX H normalized | MLX L max abs | MLX L normalized | "
        "Torch L normalized | outside | nonfinite | Torch median ms | "
        "MLX median ms | speedup",
        flush=True,
    )
    width_timings = {}
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if in_features not in width_timings:
            sample_count = 8
            hessian = _qwen_hessian(in_features, sample_count=sample_count)
            signs = _qwen_signs(in_features)
            torch_hessian = torch.from_numpy(hessian)
            torch_signs = torch.from_numpy(signs)
            mlx_hessian = mx.array(hessian)
            mlx_signs = mx.array(signs)
            mx.eval(mlx_hessian, mlx_signs)
            torch_call = partial(
                _torch_finalize_hessian_oracle,
                torch_hessian,
                torch_signs,
                sample_count=sample_count,
            )
            mlx_call = partial(
                exl3_finalize_hessian_mlx,
                mlx_hessian,
                mlx_signs,
                sample_count=sample_count,
            )
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            width_timings[in_features] = (torch_ms, mlx_ms)
            del hessian, signs, torch_hessian, torch_signs, mlx_hessian, mlx_signs
            gc.collect()
            mx.clear_cache()

        metrics = _qwen_width_metrics(in_features)
        hessian_metrics, factor_metrics, _, _, torch_factor_metrics = metrics
        torch_ms, mlx_ms = width_timings[in_features]
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{in_features} | {hessian_metrics[1]:.9g} | "
            f"{hessian_metrics[2]:.9g} | {factor_metrics[1]:.9g} | "
            f"{factor_metrics[2]:.9g} | {torch_factor_metrics[2]:.9g} | "
            f"{hessian_metrics[0] + factor_metrics[0]} | "
            f"{hessian_metrics[3] + factor_metrics[3]} | {torch_ms:.3f} | "
            f"{mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
