# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark fused native MLX EXL3 regularization against main's Torch path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_regularize import (
    exl3_input_regularize_mlx,
    exl3_output_regularize_mlx,
)
from tests.benchmark_mlx_qwen38_exl3_hadamard import _main_torch_blockwise_
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_regularize import (
    _torch_input_regularize_oracle,
    _torch_output_regularize_oracle,
)
from tests.test_mlx_exl3_rms import _normalized_rms_drift, _torch_block_rms_oracle

_CODEBOOK_SCALE = 1.24371088


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_output(weight, signs, rms, mean):
    scales = rms.clone()
    if mean > 1e-30:
        scales /= mean
    zero_channels = scales.abs() < 1e-30
    scales[zero_channels] = 0.1
    stored = (signs * scales + 1e-10).float()
    scaled = weight / stored
    stored[zero_channels] = 0.0
    return _main_torch_blockwise_(scaled, axis=1), stored


def _main_torch_input(weight, signs, rms):
    scales = rms.clone()
    scales[scales.abs() < 1e-30] = 0.1
    stored = (signs * scales / -_CODEBOOK_SCALE + 1e-10).float()
    scaled = weight / stored
    return _main_torch_blockwise_(scaled, axis=0), stored


def _accuracy(actual, expected):
    error = np.abs(actual - expected)
    allowed = 1e-6 + 1e-6 * np.abs(expected)
    return (
        int(np.count_nonzero(error > allowed)),
        float(error.max()),
        _normalized_rms_drift(actual, expected),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | mode | shape | weight outside | weight max abs | "
        "weight normalized RMS | scale outside | scale max abs | scale normalized RMS | "
        "Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(15331 + out_features + in_features)
        weight = rng.normal(0.0, 0.2, (in_features, out_features)).astype(np.float32)
        torch_weight = torch.from_numpy(weight)
        mlx_weight = mx.array(weight)
        mx.eval(mlx_weight)

        for mode in ("output", "input"):
            axis = 0 if mode == "output" else 1
            shape = (1, out_features) if mode == "output" else (in_features, 1)
            signs = rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
            rms = _torch_block_rms_oracle(weight, axis)
            torch_signs = torch.from_numpy(signs)
            torch_rms = torch.from_numpy(rms)
            mlx_signs = mx.array(signs)
            mlx_rms = mx.array(rms)
            mx.eval(mlx_signs, mlx_rms)

            if mode == "output":
                mean = float(torch_rms.mean().item())
                expected_weight, expected_scales, _ = _torch_output_regularize_oracle(
                    weight, signs, rms, mean, True
                )
                actual_weight, actual_scales, _ = exl3_output_regularize_mlx(
                    mlx_weight,
                    mlx_signs,
                    mlx_rms,
                    mean=mean,
                    apply_scales=True,
                )
                torch_call = partial(
                    _main_torch_output, torch_weight, torch_signs, torch_rms, mean
                )
                mlx_call = partial(
                    exl3_output_regularize_mlx,
                    mlx_weight,
                    mlx_signs,
                    mlx_rms,
                    mean=mean,
                    apply_scales=True,
                )
            else:
                expected_weight, expected_scales = _torch_input_regularize_oracle(
                    weight, signs, rms
                )
                actual_weight, actual_scales = exl3_input_regularize_mlx(
                    mlx_weight, mlx_signs, mlx_rms
                )
                torch_call = partial(
                    _main_torch_input, torch_weight, torch_signs, torch_rms
                )
                mlx_call = partial(
                    exl3_input_regularize_mlx, mlx_weight, mlx_signs, mlx_rms
                )

            weight_accuracy = _accuracy(np.asarray(actual_weight), expected_weight)
            scale_accuracy = _accuracy(np.asarray(actual_scales), expected_scales)
            if weight_accuracy[0] or scale_accuracy[0]:
                raise AssertionError(
                    f"{name} {mode}: weight={weight_accuracy}, scale={scale_accuracy}"
                )
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            print(
                f"{name} | {mode} | {out_features}x{in_features} | "
                f"{weight_accuracy[0]} | {weight_accuracy[1]:.9g} | "
                f"{weight_accuracy[2]:.9g} | {scale_accuracy[0]} | "
                f"{scale_accuracy[1]:.9g} | {scale_accuracy[2]:.9g} | "
                f"{torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
                flush=True,
            )
            del signs, rms, torch_signs, torch_rms, mlx_signs, mlx_rms
            del expected_weight, expected_scales, actual_weight, actual_scales
            del torch_call, mlx_call
            gc.collect()

        del weight, torch_weight, mlx_weight
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
