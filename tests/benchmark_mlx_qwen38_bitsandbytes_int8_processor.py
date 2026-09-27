# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark BitsAndBytes INT8 processor packing through the MLX bridge."""

import argparse
import gc
from statistics import median
from time import perf_counter

import bitsandbytes as bnb
import mlx.core as mx
import torch

from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    _MLX_INT8_MIN_ELEMENTS,
    _quantize_int8_weight_mlx_to_torch,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _torch_quantize(weight):
    return bnb.functional.int8_vectorwise_quant(
        weight.to(torch.float16),
        threshold=0.0,
    )[:2]


def _mlx_quantize(weight):
    return _quantize_int8_weight_mlx_to_torch(weight.to(torch.float16).contiguous())


def _measure_pair(torch_fn, mlx_fn, samples):
    for function in (torch_fn, mlx_fn):
        result = function()
        del result
    timings = [[], []]
    functions = (torch_fn, mlx_fn)
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            started = perf_counter()
            result = functions[index]()
            timings[index].append((perf_counter() - started) * 1000)
            del result
    return median(timings[0]), median(timings[1])


def _accuracy(source, torch_result, mlx_result):
    expected_codes, expected_scales = torch_result
    actual_codes, actual_scales = mlx_result
    differing = torch.nonzero(actual_codes != expected_codes, as_tuple=False)
    mismatch_count = differing.shape[0]
    tie_count = 0
    maximum_code_delta = 0
    if mismatch_count:
        rows, columns = differing.T
        values = source[rows, columns].to(torch.float16).float().double()
        maxima = expected_scales[rows].double()
        exact_codes = values * 127.0 / maxima
        tie_count = int(torch.count_nonzero(exact_codes - torch.floor(exact_codes) == 0.5))
        maximum_code_delta = int(
            (actual_codes[rows, columns].short() - expected_codes[rows, columns].short())
            .abs()
            .max()
        )
    scale_drift = float((actual_scales - expected_scales).abs().max())
    dequant_drift = 0.0
    for start in range(0, source.shape[0], 1024):
        stop = min(start + 1024, source.shape[0])
        expected_weight = (
            expected_codes[start:stop].float() * expected_scales[start:stop, None] / 127.0
        )
        actual_weight = actual_codes[start:stop].float() * actual_scales[start:stop, None] / 127.0
        dequant_drift = max(
            dequant_drift,
            float((actual_weight - expected_weight).abs().max()),
        )
    return (
        mismatch_count,
        tie_count,
        mismatch_count - tie_count,
        maximum_code_delta,
        scale_drift,
        dequant_drift,
    )


def run(samples, projection):
    if samples <= 0:
        raise ValueError("samples must be positive")
    results = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        torch.manual_seed(823500 + index)
        source = torch.randn((rows, columns), dtype=torch.float32)
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            weight = source.to(dtype).contiguous()
            torch_result = _torch_quantize(weight)
            mlx_result = _mlx_quantize(weight)
            accuracy = _accuracy(weight, torch_result, mlx_result)
            if accuracy[2] or accuracy[3] > 1 or accuracy[4] > 1e-6:
                raise AssertionError(f"{name} {dtype_name}: {accuracy}")
            torch_ms, mlx_ms = _measure_pair(
                lambda weight=weight: _torch_quantize(weight),
                lambda weight=weight: _mlx_quantize(weight),
                samples,
            )
            selected = "MLX" if weight.numel() >= _MLX_INT8_MIN_ELEMENTS else "Torch"
            selected_ms = mlx_ms if selected == "MLX" else torch_ms
            row = (
                name,
                dtype_name,
                rows,
                columns,
                selected,
                torch_ms,
                mlx_ms,
                selected_ms,
                torch_ms / selected_ms,
                *accuracy,
            )
            results.append(row)
            print(" | ".join(str(value) for value in row), flush=True)
            del weight, torch_result, mlx_result
            gc.collect()
            mx.clear_cache()
        del source
        gc.collect()

    print(
        "\n| Projection | Dtype | Shape | Selected | Main ms | Raw MLX ms | "
        "Selected ms | Speedup | Code mismatches | Tie mismatches | Non-tie "
        "mismatches | Max code delta | Max scale drift | Max dequant drift |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]}x{row[3]} | {row[4]} | "
            f"{row[5]:.4f} | {row[6]:.4f} | {row[7]:.4f} | {row[8]:.3f}x | "
            f"{row[9]} | {row[10]} | {row[11]} | {row[12]} | "
            f"{row[13]:.9g} | {row[14]:.9g} |"
        )
    print(f"\nMedian selected speedup: {median(row[8] for row in results):.3f}x")
    print(f"Minimum selected speedup: {min(row[8] for row in results):.3f}x")
    print(f"Maximum selected speedup: {max(row[8] for row in results):.3f}x")
    print(f"Total code mismatches: {sum(row[9] for row in results)}")
    print(f"Total adjudicated ties: {sum(row[10] for row in results)}")
    print(f"Total non-tie mismatches: {sum(row[11] for row in results)}")
    print(f"Maximum scale drift: {max(row[13] for row in results):.9g}")
    print(f"Maximum dequant drift: {max(row[14] for row in results):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
