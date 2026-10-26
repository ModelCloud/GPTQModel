# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark direct RTN output stores against current main on Qwen shapes."""

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
        inputs=[mx.contiguous(weight)],
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
    for _ in range(5):
        main_fn()
        direct_fn()
    timings = [[], []]
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for path in order:
            start = perf_counter()
            (main_fn, direct_fn)[path]()
            timings[path].append((perf_counter() - start) * 1000)
    return median(timings[0]), median(timings[1])


def _codes(result, columns):
    quantized, scales, zeros, _ = result
    expanded_scales = mx.repeat(scales, 128, axis=1)[:, :columns]
    expanded_zeros = mx.repeat(zeros, 128, axis=1)[:, :columns]
    return mx.round(quantized.astype(mx.float32) / expanded_scales + expanded_zeros)


def run(samples):
    rows_out = []
    print(
        "projection | dtype | shape | main ms | PR ms | speedup | "
        "code mismatches | max weight drift | max scale drift | max zero drift",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        mx.random.seed(380029 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.025
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_quantize(value)

            def direct_fn(value=weight):
                return quantize_rtn_weight_mlx(value, bits=4, group_size=128, sym=True)

            expected = main_fn()
            actual = direct_fn()
            mismatches = int(
                mx.sum(_codes(actual, columns) != _codes(expected, columns)).item()
            )
            drifts = [
                float(
                    mx.max(
                        mx.abs(
                            observed.astype(mx.float32) - reference.astype(mx.float32)
                        )
                    ).item()
                )
                for observed, reference in zip(actual[:3], expected[:3])
            ]
            main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
            row = (
                name,
                label,
                rows,
                columns,
                main_ms,
                direct_ms,
                main_ms / direct_ms,
                mismatches,
                *drifts,
            )
            rows_out.append(row)
            print(
                f"{name} | {label} | {rows}x{columns} | {main_ms:.4f} | "
                f"{direct_ms:.4f} | {main_ms / direct_ms:.3f}x | "
                f"{mismatches} | {drifts[0]:.9g} | {drifts[1]:.9g} | "
                f"{drifts[2]:.9g}",
                flush=True,
            )
        del source, weight, expected, actual
        mx.clear_cache()
        gc.collect()

    print(f"median speedup: {median(row[6] for row in rows_out):.3f}x")
    print(f"minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"maximum speedup: {max(row[6] for row in rows_out):.3f}x")
    print(f"total code mismatches: {sum(row[7] for row in rows_out)}")
    print(f"maximum weight drift: {max(row[8] for row in rows_out):.9g}")
    print(f"maximum scale drift: {max(row[9] for row in rows_out):.9g}")
    print(f"maximum zero drift: {max(row[10] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    run(parser.parse_args().samples)
