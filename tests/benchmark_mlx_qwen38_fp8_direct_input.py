# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 encoding oracle: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark FP8's FP32-temporary path against direct MLX input."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_fp8 import quantize_fp8_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


FORMATS = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
)
METHODS = ("tensor", "row", "block")


def _quantize(weight, format_name, method):
    return quantize_fp8_weight_mlx(
        weight,
        format=format_name,
        weight_scale_method=method,
        weight_block_size=(128, 128) if method == "block" else None,
    )


def _measure_pair(main_fn, direct_fn, samples):
    for _ in range(10):
        main_fn()
        direct_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = perf_counter()
            (main_fn, direct_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, formats, methods, projection):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(319900 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)
            for format_name in formats:
                for method in methods:
                    def main_fn(
                        value=weight, fmt=format_name, scale_method=method
                    ):
                        result = _quantize(
                            value.astype(mx.float32), fmt, scale_method
                        )
                        mx.eval(*result)
                        return result

                    def direct_fn(
                        value=weight, fmt=format_name, scale_method=method
                    ):
                        result = _quantize(value, fmt, scale_method)
                        mx.eval(*result)
                        return result

                    baseline = main_fn()
                    direct = direct_fn()
                    code_mismatches = int(
                        mx.sum(baseline[0] != direct[0]).item()
                    )
                    scale_error = float(
                        mx.max(mx.abs(baseline[1] - direct[1])).item()
                    )
                    main_ms, direct_ms = _measure_pair(
                        main_fn, direct_fn, samples
                    )
                    rows_out.append(
                        (
                            name,
                            dtype_name,
                            format_name,
                            method,
                            main_ms,
                            direct_ms,
                            main_ms / direct_ms,
                            code_mismatches,
                            scale_error,
                        )
                    )
                    print(
                        f"{name} {dtype_name} {format_name} {method}: "
                        f"{main_ms:.4f} -> {direct_ms:.4f} ms "
                        f"({main_ms / direct_ms:.3f}x), "
                        f"code mismatches {code_mismatches}, "
                        f"max scale error {scale_error:.8g}",
                        flush=True,
                    )
        del source, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Format | Scale method | Rows | Median speedup | Minimum speedup | "
        "Maximum speedup | Code mismatches | Maximum scale error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for format_name in formats:
        for method in methods:
            selected = [
                row
                for row in rows_out
                if row[2] == format_name and row[3] == method
            ]
            print(
                f"| {format_name} | {method} | {len(selected)} | "
                f"{median(row[6] for row in selected):.3f}x | "
                f"{min(row[6] for row in selected):.3f}x | "
                f"{max(row[6] for row in selected):.3f}x | "
                f"{sum(row[7] for row in selected)} | "
                f"{max(row[8] for row in selected):.8g} |"
            )
    print(f"\nOverall median speedup: {median(row[6] for row in rows_out):.3f}x")
    print(f"Overall minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"Overall maximum speedup: {max(row[6] for row in rows_out):.3f}x")
    print(f"Total code mismatches: {sum(row[7] for row in rows_out)}")
    print(f"Maximum scale error: {max(row[8] for row in rows_out):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--format", choices=("all", *FORMATS), default="all")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(
        args.samples,
        FORMATS if args.format == "all" else (args.format,),
        METHODS if args.method == "all" else (args.method,),
        args.projection,
    )
