# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark direct ParoQuant pack codes against current main."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_paroquant import (
    _paroquant_pack_kernel,
    paroquant_pack_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _main_pack(weight, scales, *, group_size):
    out_features, in_features = weight.shape
    groups = in_features // group_size
    scale_zeros = scales * 8
    runtime_scales = scales.astype(mx.float16) if scales.dtype == mx.float32 else scales
    view = weight.reshape(out_features, groups, group_size)
    codes = mx.round((view + scale_zeros[:, :, None]) / runtime_scales[:, :, None])
    valid_scales = mx.all(mx.isfinite(scales) & (scales > 0))
    valid_stored_scales = mx.all(mx.isfinite(runtime_scales) & (runtime_scales > 0))
    valid_weight = mx.all(mx.isfinite(weight))
    valid_codes = mx.all(mx.isfinite(codes) & (codes >= 0) & (codes <= 15))
    mx.eval(valid_scales, valid_stored_scales, valid_weight, valid_codes)
    if not bool(valid_scales.item()):
        raise ValueError("scales must be finite and positive")
    if not bool(valid_stored_scales.item()):
        raise ValueError("stored scales must be finite and positive")
    if not bool(valid_weight.item()):
        raise ValueError("weight must be finite")
    if not bool(valid_codes.item()):
        raise ValueError("ParoQuant four-bit codes must be in [0, 15]")

    out_packs = out_features // 8
    packed = _paroquant_pack_kernel()(
        inputs=[codes.astype(mx.uint32)],
        grid=(in_features * out_packs, 1, 1),
        threadgroup=(min(in_features * out_packs, 256), 1, 1),
        output_shapes=[(in_features, out_packs)],
        output_dtypes=[mx.int32],
        template=[("OUT_PACKS", out_packs), ("IN_FEATURES", in_features)],
    )[0]
    qzeros = mx.full((groups, out_packs), -0x77777778, dtype=mx.int32)
    result_scales = mx.contiguous(runtime_scales.T)
    mx.eval(packed, qzeros, result_scales)
    return packed, qzeros, result_scales


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


def run(samples):
    rows_out = []
    group_size = 64
    print(
        "projection | dtype | shape | main ms | PR ms | speedup | "
        "packed byte mismatches | qzero mismatches | scale mismatches | "
        "max scale drift",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(320600 + rows + columns)
        raw_scales = rng.uniform(0.01, 0.03, (rows, columns // group_size)).astype(
            np.float32
        )
        raw_codes = rng.integers(-7, 8, (rows, columns), dtype=np.int8)
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            scales = mx.array(raw_scales).astype(dtype)
            weight = (
                mx.array(raw_codes).astype(dtype).reshape(rows, -1, group_size)
                * scales[:, :, None]
            ).reshape(rows, columns)
            mx.eval(weight, scales)

            def main_fn(value=weight, scale_values=scales):
                return _main_pack(value, scale_values, group_size=group_size)

            def direct_fn(value=weight, scale_values=scales):
                return paroquant_pack_weight_mlx(
                    value, scale_values, group_size=group_size
                )

            expected = main_fn()
            actual = direct_fn()
            packed_byte_mismatches = int(
                np.count_nonzero(
                    np.asarray(actual[0]).view(np.uint8)
                    != np.asarray(expected[0]).view(np.uint8)
                )
            )
            qzero_mismatches = int(mx.sum(actual[1] != expected[1]).item())
            scale_delta = mx.abs(
                actual[2].astype(mx.float32) - expected[2].astype(mx.float32)
            )
            scale_mismatches = int(mx.sum(scale_delta != 0).item())
            max_scale_drift = float(mx.max(scale_delta).item())
            main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
            row = (
                name,
                label,
                rows,
                columns,
                main_ms,
                direct_ms,
                main_ms / direct_ms,
                packed_byte_mismatches,
                qzero_mismatches,
                scale_mismatches,
                max_scale_drift,
            )
            rows_out.append(row)
            print(
                f"{name} | {label} | {rows}x{columns} | {main_ms:.4f} | "
                f"{direct_ms:.4f} | {main_ms / direct_ms:.3f}x | "
                f"{packed_byte_mismatches} | {qzero_mismatches} | "
                f"{scale_mismatches} | {max_scale_drift:.9g}",
                flush=True,
            )
        del raw_scales, raw_codes, scales, weight, expected, actual
        mx.clear_cache()
        gc.collect()

    print(f"median speedup: {median(row[6] for row in rows_out):.3f}x")
    print(f"minimum speedup: {min(row[6] for row in rows_out):.3f}x")
    print(f"maximum speedup: {max(row[6] for row in rows_out):.3f}x")
    print(f"total packed byte mismatches: {sum(row[7] for row in rows_out)}")
    print(f"total qzero mismatches: {sum(row[8] for row in rows_out)}")
    print(f"total scale mismatches: {sum(row[9] for row in rows_out)}")
    print(f"maximum scale drift: {max(row[10] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    run(parser.parse_args().samples)
