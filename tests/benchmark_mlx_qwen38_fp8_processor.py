# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark production FP8 packing through the native MLX bridge."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import torch

from gptqmodel.nn_modules.qlinear.fp8 import (
    _MLX_FP8_MIN_ELEMENTS,
    _quantize_fp8_weight_mlx_to_torch,
    quantize_fp8_weight,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


FORMATS = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
)
METHODS = ("tensor", "row", "block")


def _torch_quantize(weight, format_name, method):
    return quantize_fp8_weight(
        weight,
        format=format_name,
        weight_scale_method=method,
        weight_block_size=(128, 128) if method == "block" else None,
    )


def _mlx_quantize(weight, format_name, method):
    if weight.numel() < _MLX_FP8_MIN_ELEMENTS:
        return _torch_quantize(weight, format_name, method)
    return _quantize_fp8_weight_mlx_to_torch(
        weight,
        format=format_name,
        weight_scale_method=method,
        weight_block_size=(128, 128) if method == "block" else None,
    )


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


def run(samples, projection, format_filter, method_filter):
    if samples <= 0:
        raise ValueError("samples must be positive")
    results = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        torch.manual_seed(824000 + index)
        source = torch.randn((rows, columns), dtype=torch.float32)
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            weight = source.to(dtype).float().contiguous()
            for format_name in FORMATS:
                if format_filter and format_name != format_filter:
                    continue
                for method in METHODS:
                    if method_filter and method != method_filter:
                        continue

                    def torch_fn(weight=weight, format_name=format_name, method=method):
                        return _torch_quantize(weight, format_name, method)

                    def mlx_fn(weight=weight, format_name=format_name, method=method):
                        return _mlx_quantize(weight, format_name, method)

                    expected_weight, expected_scale = torch_fn()
                    actual_weight, actual_scale = mlx_fn()
                    code_mismatches = int(
                        torch.count_nonzero(
                            actual_weight.view(torch.uint8)
                            != expected_weight.view(torch.uint8)
                        )
                    )
                    scale_delta = (actual_scale - expected_scale).abs()
                    max_scale_abs = float(scale_delta.max())
                    nonzero = expected_scale != 0
                    max_scale_rel = float(
                        (scale_delta[nonzero] / expected_scale[nonzero].abs()).max()
                    ) if bool(nonzero.any()) else 0.0
                    if code_mismatches or max_scale_abs > 1e-6 or max_scale_rel > 1e-6:
                        raise AssertionError(
                            f"{name} {dtype_name} {format_name} {method}: "
                            f"{code_mismatches} code mismatches, scale abs {max_scale_abs}, "
                            f"scale rel {max_scale_rel}"
                        )
                    torch_ms, mlx_ms = _measure_pair(torch_fn, mlx_fn, samples)
                    row = (
                        name,
                        dtype_name,
                        rows,
                        columns,
                        format_name,
                        method,
                        code_mismatches,
                        max_scale_abs,
                        max_scale_rel,
                        torch_ms,
                        mlx_ms,
                        torch_ms / mlx_ms,
                    )
                    results.append(row)
                    print(" | ".join(str(value) for value in row), flush=True)
                    del expected_weight, expected_scale, actual_weight, actual_scale
            del weight
            gc.collect()
            mx.clear_cache()
        del source
        gc.collect()

    print(
        "\n| Projection | Dtype | Shape | Format | Method | Code mismatches | "
        "Max scale abs | Max scale rel | Main ms | MLX ms | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]}x{row[3]} | {row[4]} | {row[5]} | "
            f"{row[6]} | {row[7]:.8g} | {row[8]:.8g} | {row[9]:.4f} | "
            f"{row[10]:.4f} | {row[11]:.3f}x |"
        )

    print(
        "\n| Format | Method | MLX cases | Median speedup | Minimum speedup | "
        "Maximum speedup | Code mismatches | Max scale abs | Max scale rel |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for format_name in FORMATS:
        for method in METHODS:
            matching = [row for row in results if row[4:6] == (format_name, method)]
            matching = [
                row
                for row in matching
                if row[2] * row[3] >= _MLX_FP8_MIN_ELEMENTS
            ]
            if not matching:
                continue
            print(
                f"| {format_name} | {method} | {len(matching)} | "
                f"{median(row[11] for row in matching):.3f}x | "
                f"{min(row[11] for row in matching):.3f}x | "
                f"{max(row[11] for row in matching):.3f}x | "
                f"{sum(row[6] for row in matching)} | "
                f"{max(row[7] for row in matching):.8g} | "
                f"{max(row[8] for row in matching):.8g} |"
            )
    mlx_results = [
        row for row in results if row[2] * row[3] >= _MLX_FP8_MIN_ELEMENTS
    ]
    print(f"\nMLX-selected cases: {len(mlx_results)}")
    print(f"Torch-fallback cases: {len(results) - len(mlx_results)}")
    print(f"MLX-selected median speedup: {median(row[11] for row in mlx_results):.3f}x")
    print(f"MLX-selected minimum speedup: {min(row[11] for row in mlx_results):.3f}x")
    print(f"MLX-selected maximum speedup: {max(row[11] for row in mlx_results):.3f}x")
    print(f"Total code mismatches: {sum(row[6] for row in results)}")
    print(f"Maximum scale absolute error: {max(row[7] for row in results):.8g}")
    print(f"Maximum scale relative error: {max(row[8] for row in results):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--projection")
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    run(args.samples, args.projection, args.format, args.method)
