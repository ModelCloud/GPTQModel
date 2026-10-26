# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.

"""Benchmark the complete native EXL3 transform stage on Qwen3.8-27B."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_regularize import (
    exl3_regularize_transforms_mlx,
)
from tests.benchmark_mlx_qwen38_exl3_regularize import (
    _main_torch_input,
    _main_torch_output,
)
from tests.benchmark_mlx_qwen38_exl3_rms import _main_torch_block_rms
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_regularize import (
    _torch_regularize_transforms_oracle,
)
from tests.test_mlx_exl3_rms import _normalized_rms_drift


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _main_torch_regularize_transforms(
    weight, input_signs, output_signs, hessian_diagonal
):
    diagonal, _ = torch.sort(hessian_diagonal.sqrt(), descending=True)
    cutoff = diagonal.shape[0] // 50
    skew = diagonal[:cutoff].sum() / diagonal.sum()
    apply_output_scales = skew.item() < 0.15

    output_rms = _main_torch_block_rms(weight, 0)
    output_mean = float(output_rms.mean().item())
    transformed, output_scales = _main_torch_output(
        weight, output_signs, output_rms, output_mean
    )
    input_rms = _main_torch_block_rms(transformed, 1)
    transformed, input_scales = _main_torch_input(
        transformed, input_signs, input_rms
    )
    return apply_output_scales, transformed, input_scales, output_scales


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
        "projection | checkpoint shape | EXL3 shape | policy match | "
        "weight outside 1e-6 | weight max abs | weight normalized RMS | "
        "input-scale max abs | input-scale normalized RMS | "
        "output-scale max abs | output-scale normalized RMS | "
        "Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(25401 + out_features + in_features)
        weight = rng.normal(
            0.0, 0.2, (in_features, out_features)
        ).astype(np.float32)
        input_signs = rng.choice(
            [-1.0, 1.0], size=(in_features, 1)
        ).astype(np.float32)
        output_signs = rng.choice(
            [-1.0, 1.0], size=(1, out_features)
        ).astype(np.float32)
        hessian_diagonal = rng.uniform(
            0.01, 2.0, size=in_features
        ).astype(np.float32)

        expected = _torch_regularize_transforms_oracle(
            weight,
            input_signs,
            output_signs,
            hessian_diagonal=hessian_diagonal,
        )
        torch_inputs = tuple(
            torch.from_numpy(value)
            for value in (weight, input_signs, output_signs, hessian_diagonal)
        )
        mlx_inputs = tuple(
            mx.array(value)
            for value in (weight, input_signs, output_signs, hessian_diagonal)
        )
        mx.eval(*mlx_inputs)
        actual = exl3_regularize_transforms_mlx(
            mlx_inputs[0],
            mlx_inputs[1],
            mlx_inputs[2],
            hessian_diagonal=mlx_inputs[3],
        )
        actual_arrays = tuple(np.asarray(value) for value in actual[1:])
        weight_accuracy = _accuracy(actual_arrays[0], expected[1])
        input_accuracy = _accuracy(actual_arrays[1], expected[2])
        output_accuracy = _accuracy(actual_arrays[2], expected[3])
        policy_match = actual[0] is expected[0]
        if (
            not policy_match
            or weight_accuracy[1] > 4e-6
            or weight_accuracy[2] > 1e-6
            or input_accuracy[0]
            or input_accuracy[2] > 1e-6
            or output_accuracy[0]
            or output_accuracy[2] > 1e-6
        ):
            raise AssertionError(
                f"{name}: policy={policy_match}, weight={weight_accuracy}, "
                f"input={input_accuracy}, output={output_accuracy}"
            )

        torch_ms = _median_ms(
            partial(_main_torch_regularize_transforms, *torch_inputs),
            args.repeats,
        )
        mlx_ms = _median_ms(
            partial(
                exl3_regularize_transforms_mlx,
                mlx_inputs[0],
                mlx_inputs[1],
                mlx_inputs[2],
                hessian_diagonal=mlx_inputs[3],
            ),
            args.repeats,
        )
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{out_features} | {policy_match} | "
            f"{weight_accuracy[0]} | {weight_accuracy[1]:.9g} | "
            f"{weight_accuracy[2]:.9g} | {input_accuracy[1]:.9g} | "
            f"{input_accuracy[2]:.9g} | {output_accuracy[1]:.9g} | "
            f"{output_accuracy[2]:.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | "
            f"{torch_ms / mlx_ms:.2f}x",
            flush=True,
        )

        del weight, input_signs, output_signs, hessian_diagonal
        del expected, torch_inputs, mlx_inputs, actual, actual_arrays
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
