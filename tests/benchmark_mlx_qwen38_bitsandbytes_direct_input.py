# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark BitsAndBytes FP32-temporary paths against direct MLX input."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_bitsandbytes import (
    quantize_4bit_weight_mlx,
    quantize_int8_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


FORMATS = ("nf4", "fp4", "int8")


def _quantize(weight, format_name):
    if format_name == "int8":
        return quantize_int8_weight_mlx(weight)
    return quantize_4bit_weight_mlx(
        weight, quant_type=format_name, block_size=64
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


def run(samples, formats, projection):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(319600 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)
            for format_name in formats:
                def main_fn(value=weight, fmt=format_name):
                    result = _quantize(value.astype(mx.float32), fmt)
                    mx.eval(*result)
                    return result

                def direct_fn(value=weight, fmt=format_name):
                    result = _quantize(value, fmt)
                    mx.eval(*result)
                    return result

                baseline = main_fn()
                direct = direct_fn()
                code_mismatches = int(mx.sum(baseline[0] != direct[0]).item())
                scale_error = float(
                    mx.max(
                        mx.abs(
                            baseline[1].astype(mx.float32)
                            - direct[1].astype(mx.float32)
                        )
                    ).item()
                )
                main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
                rows_out.append(
                    (
                        name,
                        dtype_name,
                        format_name,
                        main_ms,
                        direct_ms,
                        main_ms / direct_ms,
                        code_mismatches,
                        scale_error,
                    )
                )
                print(
                    f"{name} {dtype_name} {format_name}: "
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
        "\n| Projection | Dtype | Format | Main ms | PR ms | Speedup | "
        "Code mismatches | Maximum scale error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | "
            f"{row[4]:.4f} | {row[5]:.3f}x | {row[6]} | {row[7]:.8g} |"
        )
    print(f"\nMedian speedup: {median(row[5] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in rows_out):.3f}x")
    print(f"Total code mismatches: {sum(row[6] for row in rows_out)}")
    print(f"Maximum scale error: {max(row[7] for row in rows_out):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--format", choices=("all", *FORMATS), default="all")
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(
        args.samples,
        FORMATS if args.format == "all" else (args.format,),
        args.projection,
    )
