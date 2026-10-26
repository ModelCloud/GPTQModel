# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark main's FP32-temporary RTN path against direct MLX input."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_rtn import _rtn_kernel, quantize_rtn_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _main_quantize(weight):
    rows, columns = weight.shape
    group_size = 128
    groups = (columns + group_size - 1) // group_size
    quantized, scales, zeros = _rtn_kernel()(
        inputs=[weight.astype(mx.float32)],
        template=[
            ("ROWS", rows),
            ("COLS", columns),
            ("GROUPS", groups),
            ("GROUP_SIZE", group_size),
            ("MAXQ", 15),
            ("SYMMETRIC", 1),
        ],
        grid=(32, rows * groups, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, columns), (rows, groups), (rows, groups)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    result = (
        quantized.astype(weight.dtype),
        scales,
        zeros,
        mx.arange(columns, dtype=mx.int32) // group_size,
    )
    mx.eval(*result)
    return result


def _measure_pair(main_fn, direct_fn, samples):
    for _ in range(10):
        main_fn()
        direct_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            (main_fn, direct_fn)[path]()
            timings[path].append((perf_counter() - start) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples):
    rows_out = []
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        mx.random.seed(380028 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.025
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_quantize(value)

            def direct_fn(value=weight):
                return quantize_rtn_weight_mlx(
                    value, bits=4, group_size=128, sym=True
                )

            main = main_fn()
            direct = direct_fn()
            errors = [
                float(mx.max(mx.abs(
                    observed.astype(mx.float32) - expected.astype(mx.float32)
                )).item())
                for observed, expected in zip(direct, main)
            ]
            main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
            rows_out.append(
                (name, label, main_ms, direct_ms, main_ms / direct_ms, max(errors))
            )
            print(
                f"{name} {label}: {main_ms:.4f} -> {direct_ms:.4f} ms "
                f"({main_ms / direct_ms:.3f}x), max error {max(errors):.8g}",
                flush=True,
            )
        del source, weight
        mx.clear_cache()
        gc.collect()

    print("\n| Projection | Dtype | Main ms | PR ms | Speedup | Maximum error |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]:.8g} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Maximum error: {max(row[5] for row in rows_out):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    run(parser.parse_args().samples)
