# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark direct ParoQuant output stores against current main."""

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
    groups = columns // 128
    zeros = mx.zeros((rows, groups), dtype=mx.float32) if sym else zero_points
    valid = mx.all(mx.isfinite(weight)) & mx.all(mx.isfinite(scales))
    if not sym:
        valid = valid & mx.all(mx.isfinite(zeros))
    mx.eval(valid)
    if not bool(valid.item()):
        raise ValueError("weight and quantization parameters must be finite")
    result = _paroquant_quantize_kernel()(
        inputs=[
            mx.contiguous(weight.reshape(-1)),
            mx.contiguous(scales.reshape(-1)),
            mx.contiguous(zeros.reshape(-1)),
        ],
        template=[
            ("BITS", 4),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("GROUP_SIZE", 128),
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
    for _ in range(5):
        main_fn()
        direct_fn()
    timings = [[], []]
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for path in order:
            started = perf_counter()
            (main_fn, direct_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def _codes(result, scales, zero_points, sym):
    rows, columns = result.shape
    grouped = result.astype(mx.float32).reshape(rows, columns // 128, 128)
    codes = mx.round(grouped / scales[:, :, None])
    if not sym:
        codes += mx.clip(mx.round(-zero_points), 0, 15)[:, :, None]
    return codes


def run(samples):
    rows_out = []
    print(
        "projection | dtype | symmetric | shape | main ms | PR ms | speedup | "
        "code mismatches | output mismatches | max output drift",
        flush=True,
    )
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        rng = np.random.default_rng(320400 + index)
        source = mx.array(rng.normal(0, 0.2, (rows, columns)).astype(np.float32))
        scales = mx.array(
            rng.uniform(0.002, 0.06, (rows, columns // 128)).astype(np.float32)
        )
        asymmetric_zeros = mx.array(
            rng.uniform(-12, 0, (rows, columns // 128)).astype(np.float32)
        )
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight, scales, asymmetric_zeros)
            for sym in (True, False):
                zero_points = None if sym else asymmetric_zeros

                def main_fn(
                    value=weight,
                    scale_values=scales,
                    symmetric=sym,
                    zeros=zero_points,
                ):
                    return _main_quantize(
                        value, scale_values, sym=symmetric, zero_points=zeros
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

                expected = main_fn()
                actual = direct_fn()
                delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
                code_mismatches = int(
                    mx.sum(
                        _codes(actual, scales, asymmetric_zeros, sym)
                        != _codes(expected, scales, asymmetric_zeros, sym)
                    ).item()
                )
                output_mismatches = int(mx.sum(delta != 0).item())
                max_drift = float(mx.max(delta).item())
                main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
                row = (
                    name,
                    label,
                    sym,
                    rows,
                    columns,
                    main_ms,
                    direct_ms,
                    main_ms / direct_ms,
                    code_mismatches,
                    output_mismatches,
                    max_drift,
                )
                rows_out.append(row)
                print(
                    f"{name} | {label} | {sym} | {rows}x{columns} | "
                    f"{main_ms:.4f} | {direct_ms:.4f} | "
                    f"{main_ms / direct_ms:.3f}x | {code_mismatches} | "
                    f"{output_mismatches} | {max_drift:.9g}",
                    flush=True,
                )
        del source, scales, asymmetric_zeros, weight, expected, actual
        mx.clear_cache()
        gc.collect()

    print(f"median speedup: {median(row[7] for row in rows_out):.3f}x")
    print(f"minimum speedup: {min(row[7] for row in rows_out):.3f}x")
    print(f"maximum speedup: {max(row[7] for row in rows_out):.3f}x")
    print(f"total code mismatches: {sum(row[8] for row in rows_out)}")
    print(f"total output mismatches: {sum(row[9] for row in rows_out)}")
    print(f"maximum output drift: {max(row[10] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    run(parser.parse_args().samples)
