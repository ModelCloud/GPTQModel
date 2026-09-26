# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native and CPU subnormal BitsAndBytes statistics on Qwen shapes."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization import mlx_bitsandbytes as bnb_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_pack_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_4bit_pack_main_baseline",
        input_names=["weight", "absmax", "bounds", "order"],
        output_names=["packed"],
        header="""
            inline float bnb_scaled(uint bits, uint maximum_bits) {
                uint mantissa = bits & 0x7fffffffu;
                if (mantissa == 0 || maximum_bits == 0) return 0.0f;
                float inverse = maximum_bits < 0x00800000u ?
                    as_type<float>(0x7e96769au) : 1.0f / as_type<float>(maximum_bits);
                if (mantissa > 0 && mantissa < 0x00800000u) {
                    float magnitude = float(mantissa) * (0x1.0p-126f * inverse) * 0x1.0p-23f;
                    if (magnitude == 0.0f && float(mantissa) * inverse > 0.5f)
                        magnitude = 0x1.0p-126f;
                    return (bits >> 31) ? -magnitude : magnitude;
                }
                float value = as_type<float>(bits);
                if (as_type<float>(maximum_bits) > 1.0e37f)
                    return value / as_type<float>(maximum_bits);
                return value * inverse;
            }
        """,
        source="""
            uint pair = thread_position_in_grid.x;
            if (pair >= PAIRS) return;
            uint first = pair * 2;
            uint block0 = first / BLOCK;
            float a = metal::clamp(
                bnb_scaled(as_type<uint>(float(weight[first])), absmax[block0]),
                -1.0f, 1.0f);
            uint code0 = 0;
            for (uint i = 0; i < 15; ++i) code0 += a > bounds[i];
            code0 = uint(order[code0]);
            uint code1 = 0;
            if (first + 1 < SIZE) {
                uint block1 = (first + 1) / BLOCK;
                float b = metal::clamp(
                    bnb_scaled(as_type<uint>(float(weight[first + 1])), absmax[block1]),
                    -1.0f, 1.0f);
                for (uint i = 0; i < 15; ++i) code1 += b > bounds[i];
                code1 = uint(order[code1]);
            } else {
                for (uint i = 0; i < 15; ++i) code1 += 0.0f > bounds[i];
                code1 = uint(order[code1]);
            }
            packed[pair] = uchar((code0 << 4) | code1);
        """,
    )


def _main_subnormal_nested_scales(absmax):
    raw = np.asarray(absmax).astype(np.float32)
    offset = np.float32(raw.astype(np.float64).mean())
    centered = (raw - offset).astype(np.float32)
    padded = (-centered.size) % 256
    blocks = np.pad(centered, (0, padded)).reshape(-1, 256)
    nested_absmax = np.max(np.abs(blocks), axis=1)
    lookup, code = bnb_mlx._dynamic_lookup_np()
    nested_codes = np.zeros(centered.size, dtype=np.uint8)
    for block, maximum in enumerate(nested_absmax):
        start = block * 256
        end = min(start + 256, centered.size)
        if maximum:
            scaled = np.clip(
                centered[start:end].astype(np.float64) / float(maximum), -1, 1
            )
            indices = np.floor((scaled + 1) * 32767.5 + 0.5).astype(np.int32)
            nested_codes[start:end] = lookup[indices]
    return {
        "absmax": mx.array(nested_codes),
        "nested_absmax": mx.array(nested_absmax),
        "offset": mx.array(offset),
        "nested_code": mx.array(code),
    }


def _quantize(weight, quant_type, nested_fn, pack_kernel):
    block_size = 64
    flat = mx.contiguous(weight.reshape(-1))
    absmax = bnb_mlx._block_absmax(weight, block_size, clamp_remainder=True)
    bounds, order = bnb_mlx._codebook(quant_type)
    pairs = (flat.size + 1) // 2
    packed = pack_kernel(
        inputs=[flat, absmax.view(mx.uint32), bounds, order],
        template=[("SIZE", flat.size), ("PAIRS", pairs), ("BLOCK", block_size)],
        grid=(pairs, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(pairs, 1)],
        output_dtypes=[mx.uint8],
    )[0]
    state = nested_fn(absmax)
    mx.eval(packed, state["absmax"], state["nested_absmax"], state["offset"])
    return packed, state


def _measure_pair(main_fn, native_fn, samples):
    for _ in range(5):
        main_fn()
        native_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter()
            (main_fn, native_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def _subnormal_weight(rows, columns, seed):
    mx.random.seed(seed)
    magnitude = mx.random.randint(
        0, 0x00800000, (rows, columns), dtype=mx.uint32
    )
    sign = mx.random.randint(0, 2, (rows, columns), dtype=mx.uint32) << 31
    weight = (magnitude | sign).view(mx.float32)
    mx.eval(weight)
    return weight


def _fp64_oracle_errors(weight, quant_type, native):
    raw_bits = np.asarray(weight).reshape(-1).view(np.uint32)
    magnitude = raw_bits & np.uint32(0x7FFFFFFF)
    block_maximum = magnitude.reshape(-1, 64).max(axis=1)
    denominator = np.maximum(
        block_maximum, np.float32(1e-38).view(np.uint32)
    )
    code = np.array(
        bnb_mlx._NF4 if quant_type == "nf4" else bnb_mlx._FP4,
        dtype=np.float32,
    )
    order = np.argsort(code, kind="stable")
    bounds = (code[order[:-1]] + code[order[1:]]) / np.float32(2)
    observed_packed = np.asarray(native[0]).reshape(-1)
    packed_mismatches = 0
    blocks_per_chunk = 16384
    for block_start in range(0, block_maximum.size, blocks_per_chunk):
        block_stop = min(block_start + blocks_per_chunk, block_maximum.size)
        value_start = block_start * 64
        value_stop = block_stop * 64
        scaled = magnitude[value_start:value_stop].astype(np.float64)
        scaled /= np.repeat(denominator[block_start:block_stop], 64)
        scaled[(raw_bits[value_start:value_stop] >> 31) != 0] *= -1
        codes = order[np.searchsorted(bounds, scaled, side="left")].astype(
            np.uint8
        )
        expected = (codes[0::2] << 4) | codes[1::2]
        packed_mismatches += int(
            np.count_nonzero(
                observed_packed[value_start // 2 : value_stop // 2] != expected
            )
        )

    total = int(block_maximum.astype(np.uint64).sum())
    quotient, remainder = divmod(total, block_maximum.size)
    offset_bits = quotient + int(
        2 * remainder > block_maximum.size
        or (2 * remainder == block_maximum.size and quotient % 2)
    )
    centered = block_maximum.astype(np.int64) - offset_bits
    padded = (-centered.size) % 256
    nested_maximum = np.max(
        np.abs(np.pad(centered, (0, padded))).reshape(-1, 256), axis=1
    )
    scaled = centered.astype(np.float64)
    scaled /= nested_maximum[np.arange(centered.size) // 256]
    _, dynamic_code = bnb_mlx._dynamic_lookup_np()
    dynamic_bounds = (
        dynamic_code[:-1] + dynamic_code[1:]
    ) / np.float32(2)
    expected_nested = np.searchsorted(
        dynamic_bounds, scaled, side="left"
    ).astype(np.uint8)
    state = native[1]
    nested_mismatches = int(
        np.count_nonzero(np.asarray(state["absmax"]) != expected_nested)
    )
    nested_scale_mismatches = int(
        np.count_nonzero(
            np.asarray(state["nested_absmax"]).view(np.uint32)
            != nested_maximum.astype(np.uint32)
        )
    )
    offset_error = abs(
        float(np.asarray(state["offset"]))
        - float(np.array(offset_bits, dtype=np.uint32).view(np.float32))
    )
    return (
        packed_mismatches,
        nested_mismatches,
        nested_scale_mismatches,
        offset_error,
    )


def run(samples, projections, formats):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projections and name not in projections:
            continue
        weight = _subnormal_weight(rows, columns, 322000 + index)
        for quant_type in formats:
            def main_fn(value=weight, fmt=quant_type):
                return _quantize(
                    value, fmt, _main_subnormal_nested_scales, _main_pack_kernel()
                )

            def native_fn(value=weight, fmt=quant_type):
                return _quantize(
                    value,
                    fmt,
                    bnb_mlx._subnormal_nested_scales,
                    bnb_mlx._subnormal_pack_kernel(),
                )

            baseline = main_fn()
            native = native_fn()
            packed_mismatches = int(mx.sum(baseline[0] != native[0]).item())
            nested_mismatches = int(
                mx.sum(baseline[1]["absmax"] != native[1]["absmax"]).item()
            )
            nested_scale_mismatches = int(
                mx.sum(
                    baseline[1]["nested_absmax"]
                    != native[1]["nested_absmax"]
                ).item()
            )
            offset_error = float(
                mx.abs(baseline[1]["offset"] - native[1]["offset"]).item()
            )
            oracle_errors = _fp64_oracle_errors(weight, quant_type, native)
            main_ms, native_ms = _measure_pair(main_fn, native_fn, samples)
            rows_out.append(
                (
                    name,
                    quant_type,
                    main_ms,
                    native_ms,
                    main_ms / native_ms,
                    packed_mismatches,
                    nested_mismatches,
                    nested_scale_mismatches,
                    offset_error,
                    *oracle_errors,
                )
            )
            print(
                f"{name} {quant_type}: {main_ms:.4f} -> {native_ms:.4f} ms "
                f"({main_ms / native_ms:.3f}x), mismatches "
                f"{packed_mismatches + nested_mismatches + nested_scale_mismatches}, "
                f"offset error {offset_error:.9g}",
                flush=True,
            )
        del weight, baseline, native
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Format | Main ms | PR ms | Speedup | Packed | "
        "Nested codes | Nested scales | Offset max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]} | {row[6]} | {row[7]} | "
            f"{row[8]:.9g} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Total packed mismatches: {sum(row[5] for row in rows_out)}")
    print(f"Total nested-code mismatches: {sum(row[6] for row in rows_out)}")
    print(f"Total nested-scale mismatches: {sum(row[7] for row in rows_out)}")
    print(f"Maximum offset error: {max(row[8] for row in rows_out):.9g}")

    print(
        "\n| Projection | Format | FP64 packed | FP64 nested codes | "
        "FP64 nested scales | FP64 offset max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[9]} | {row[10]} | {row[11]} | "
            f"{row[12]:.9g} |"
        )
    print(f"\nTotal FP64 packed mismatches: {sum(row[9] for row in rows_out)}")
    print(f"Total FP64 nested-code mismatches: {sum(row[10] for row in rows_out)}")
    print(f"Total FP64 nested-scale mismatches: {sum(row[11] for row in rows_out)}")
    print(f"Maximum FP64 offset error: {max(row[12] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--projection", action="append")
    parser.add_argument("--format", action="append", choices=("nf4", "fp4"))
    args = parser.parse_args()
    run(args.samples, args.projection, args.format or ("nf4", "fp4"))
