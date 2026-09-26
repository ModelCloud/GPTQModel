# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ParoQuant's main FP32 input copy against direct MLX input."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_paroquant_quant import (
    _paroquant_quantize_kernel,
    paroquant_quantize_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _main_quantize(weight, scales, *, sym, zero_points):
    rows, columns = weight.shape
    group_size = 128
    groups = columns // group_size
    kernel_zeros = mx.zeros((rows, groups), dtype=mx.float32) if sym else zero_points
    valid = mx.all(mx.isfinite(weight)) & mx.all(mx.isfinite(scales))
    if not sym:
        valid = valid & mx.all(mx.isfinite(kernel_zeros))
    mx.eval(valid)
    if not bool(valid.item()):
        raise ValueError("weight and quantization parameters must be finite")
    result = _paroquant_quantize_kernel()(
        inputs=[
            mx.contiguous(weight.astype(mx.float32).reshape(-1)),
            mx.contiguous(scales.reshape(-1)),
            mx.contiguous(kernel_zeros.reshape(-1)),
        ],
        template=[
            ("BITS", 4),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("GROUP_SIZE", group_size),
            ("SYMMETRIC", sym),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(min(weight.size, 256), 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.float32],
    )[0].astype(weight.dtype)
    mx.eval(result)
    return result


def _measure_pair(main_fn, direct_fn, samples):
    for _ in range(10):
        main_fn()
        direct_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter()
            (main_fn, direct_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, projection):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        rng = np.random.default_rng(319500 + index)
        source = mx.array(rng.normal(0, 0.2, (rows, columns)).astype(np.float32))
        groups = columns // 128
        scales = mx.array(rng.uniform(0.002, 0.06, (rows, groups)).astype(np.float32))
        asymmetric_zeros = mx.array(
            rng.uniform(-12, 0, (rows, groups)).astype(np.float32)
        )
        mx.eval(source, scales, asymmetric_zeros)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)
            for sym in (True, False):
                zero_points = None if sym else asymmetric_zeros

                def main_fn(
                    value=weight,
                    scale_values=scales,
                    symmetric=sym,
                    zeros=zero_points,
                ):
                    return _main_quantize(
                        value,
                        scale_values,
                        sym=symmetric,
                        zero_points=zeros,
                    )

                def direct_fn(
                    value=weight,
                    scale_values=scales,
                    symmetric=sym,
                    zeros=zero_points,
                ):
                    return paroquant_quantize_weight_mlx(
                        value,
                        scale_values,
                        bits=4,
                        group_size=128,
                        sym=symmetric,
                        zero_point_float=zeros,
                    )

                baseline = main_fn()
                direct = direct_fn()
                delta = mx.abs(baseline.astype(mx.float32) - direct.astype(mx.float32))
                max_error = float(mx.max(delta).item())
                mismatches = int(mx.sum(delta != 0).item())
                main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
                rows_out.append(
                    (
                        name,
                        dtype_name,
                        sym,
                        main_ms,
                        direct_ms,
                        main_ms / direct_ms,
                        max_error,
                        mismatches,
                    )
                )
                print(
                    f"{name} {dtype_name} sym={sym}: "
                    f"{main_ms:.4f} -> {direct_ms:.4f} ms "
                    f"({main_ms / direct_ms:.3f}x), max error {max_error:.8g}, "
                    f"mismatches {mismatches}",
                    flush=True,
                )
        del source, scales, asymmetric_zeros, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Symmetric | Main ms | PR ms | Speedup | "
        "Maximum error | Mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | "
            f"{row[4]:.4f} | {row[5]:.3f}x | {row[6]:.8g} | {row[7]} |"
        )
    print(f"\nMedian speedup: {median(row[5] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in rows_out):.3f}x")
    print(f"Maximum error: {max(row[6] for row in rows_out):.8g}")
    print(f"Total mismatches: {sum(row[7] for row in rows_out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
