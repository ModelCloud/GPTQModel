# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark main's FP8 threshold search against direct bit encoding."""

import argparse
import gc
from functools import lru_cache, partial
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_fp8 import (
    _FORMATS,
    _positive_values,
    quantize_fp8_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


FORMATS = tuple(_FORMATS)
METHODS = ("tensor", "row", "block")


@lru_cache(maxsize=4)
def _thresholds(format_name):
    values = _positive_values(format_name)
    return mx.array(
        [(left + right) / 2 for left, right in zip(values, values[1:])],
        dtype=mx.float32,
    )


@lru_cache(maxsize=1)
def _main_encode_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_fp8_threshold_encode_benchmark",
        input_names=["raw", "thresholds", "scales"],
        output_names=["codes"],
        source="""
            uint index = thread_position_in_grid.x;
            if (index >= SIZE) return;
            uint row = index / COLS;
            uint scale_index = MODE == 0 ? 0 :
                (MODE == 1 ? row :
                (row / BLOCK_ROWS) * GROUP_COLS
                    + (index % COLS) / BLOCK_COLS);
            float source = float(raw[index]);
            uint bits = as_type<uint>(source);
            uint magnitude_bits = bits & 0x7fffffffu;
            if (metal::isinf(scales[scale_index])) {
                codes[index] = magnitude_bits == 0 ? uchar(NAN_CODE) :
                    uchar((COUNT - 1) | ((bits >> 31) << 7));
                return;
            }
            float value = source * scales[scale_index];
            value = metal::clamp(value, -float(MAXQ), float(MAXQ));
            if (magnitude_bits > 0 && magnitude_bits < 0x00800000u) {
                float unit = scales[scale_index] * 0x1.0p-126f;
                float product = float(magnitude_bits) * unit * 0x1.0p-23f;
                value = metal::clamp((bits >> 31) ? -product : product,
                                     -float(MAXQ), float(MAXQ));
            }
            float magnitude = metal::abs(value);
            uint low = 0;
            uint high = COUNT - 1;
            while (low < high) {
                uint midpoint = (low + high) / 2;
                float boundary = thresholds[midpoint];
                if (magnitude > boundary ||
                    (magnitude == boundary && (midpoint & 1))) {
                    low = midpoint + 1;
                } else {
                    high = midpoint;
                }
            }
            uint sign = as_type<uint>(value) >> 31;
            codes[index] = uchar(
                low | ((sign && (SIGNED_ZERO || low)) ? 128 : 0));
        """,
    )


def _main_quantize(weight, format_name, method):
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("weight must be finite")
    fp8_max = _positive_values(format_name)[-1]

    def maximum_bits(values, axis=None):
        maximum = mx.max(mx.abs(values), axis=axis).astype(mx.float32)
        return maximum.view(mx.uint32)

    def inverse_scale(bits):
        maximum = bits.view(mx.float32)
        return mx.where(
            bits > 0,
            fp8_max / mx.maximum(maximum, 2.0 ** -126),
            1.0,
        )

    if method == "tensor":
        scales = inverse_scale(maximum_bits(weight))
    elif method == "row":
        scales = inverse_scale(maximum_bits(weight, axis=1))
    else:
        rows, columns = weight.shape
        blocks = weight.reshape(
            rows // 128, 128, columns // 128, 128
        )
        scales = inverse_scale(maximum_bits(blocks, axis=(1, 3)))

    _, _, _, count, signed_zero = _FORMATS[format_name]
    block_rows, block_cols = (128, 128) if method == "block" else (1, 1)
    codes = _main_encode_kernel()(
        inputs=[
            mx.contiguous(weight),
            _thresholds(format_name),
            mx.contiguous(scales.reshape(-1)),
        ],
        template=[
            ("SIZE", weight.size),
            ("COUNT", count),
            ("SIGNED_ZERO", int(signed_zero)),
            ("MAXQ", int(fp8_max)),
            ("NAN_CODE", 127 if signed_zero else 128),
            ("COLS", weight.shape[1]),
            ("MODE", {"tensor": 0, "row": 1, "block": 2}[method]),
            ("BLOCK_ROWS", block_rows),
            ("BLOCK_COLS", block_cols),
            ("GROUP_COLS", weight.shape[1] // block_cols),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(codes, scales)
    return codes, scales


def _pr_quantize(weight, format_name, method):
    return quantize_fp8_weight_mlx(
        weight,
        format=format_name,
        weight_scale_method=method,
        weight_block_size=(128, 128) if method == "block" else None,
    )


def _measure_pair(main_fn, pr_fn, samples):
    for _ in range(10):
        main_fn()
        pr_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = perf_counter()
            (main_fn, pr_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, formats, methods, projection):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(320600 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)
            for format_name in formats:
                for method in methods:
                    main_fn = partial(
                        _main_quantize, weight, format_name, method
                    )
                    pr_fn = partial(
                        _pr_quantize, weight, format_name, method
                    )
                    baseline = main_fn()
                    actual = pr_fn()
                    code_mismatches = int(
                        mx.sum(baseline[0] != actual[0]).item()
                    )
                    scale_error = float(
                        mx.max(mx.abs(baseline[1] - actual[1])).item()
                    )
                    main_ms, pr_ms = _measure_pair(main_fn, pr_fn, samples)
                    speedup = main_ms / pr_ms
                    rows_out.append(
                        (
                            name,
                            dtype_name,
                            format_name,
                            method,
                            main_ms,
                            pr_ms,
                            speedup,
                            code_mismatches,
                            scale_error,
                        )
                    )
                    print(
                        f"{name} {dtype_name} {format_name} {method}: "
                        f"{main_ms:.4f} -> {pr_ms:.4f} ms ({speedup:.3f}x), "
                        f"code mismatches {code_mismatches}, "
                        f"max scale error {scale_error:.8g}",
                        flush=True,
                    )
        del source, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Format | Scale method | Main ms | PR ms | "
        "Speedup | Code mismatches | Maximum scale error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"{row[4]:.4f} | {row[5]:.4f} | {row[6]:.3f}x | "
            f"{row[7]} | {row[8]:.8g} |"
        )
    print(f"\nMedian speedup: {median(row[6] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[6] for row in rows_out):.3f}x")
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
