# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark main's ParoQuant metadata loop against vectorized construction."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_paroquant_rotation import (
    _rotation_kernel,
    paroquant_rotate_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_paroquant_rotation import _pair_schedule


def _main_rotation_metadata(pairs, *, columns, group_size):
    pair_values = np.asarray(pairs)
    krot = pair_values.shape[0]
    groups = columns // group_size
    half_group = group_size // 2
    pair_groups = pair_values.astype(np.int64, copy=False).reshape(
        krot, groups, group_size
    )
    expected = np.arange(group_size)
    partners = np.empty((krot, columns), dtype=np.int32)
    pair_indices = np.empty((krot, columns), dtype=np.int32)
    sine_signs = np.empty((krot, columns), dtype=np.float32)
    for stage in range(krot):
        for group in range(groups):
            members = pair_groups[stage, group]
            if not np.array_equal(np.sort(members), expected):
                raise ValueError(
                    "each ParoQuant group must contain every local channel index once"
                )
            for pair_index, (left, right) in enumerate(members.reshape(half_group, 2)):
                left_column = group * group_size + int(left)
                right_column = group * group_size + int(right)
                partners[stage, left_column] = int(right)
                partners[stage, right_column] = int(left)
                pair_indices[stage, left_column] = pair_index
                pair_indices[stage, right_column] = pair_index
                sine_signs[stage, left_column] = 1.0
                sine_signs[stage, right_column] = -1.0
    return partners, pair_indices, sine_signs


def _main_rotate(weight, pairs, theta, scales, *, inverse):
    columns = weight.shape[-1]
    group_size = 128
    pair_values = np.asarray(pairs)
    krot = pair_values.shape[0]
    scale_values = mx.array(scales).astype(mx.float32).reshape((columns,))
    theta_values = mx.array(theta).astype(mx.float32)
    valid_scales = mx.all(mx.isfinite(scale_values) & (scale_values > 0))
    valid_weight = mx.all(mx.isfinite(weight))
    valid_theta = mx.all(mx.isfinite(theta_values))
    mx.eval(valid_scales, valid_weight, valid_theta)
    if not bool(valid_scales.item()):
        raise ValueError("channel_scales must be finite and positive")
    if not bool(valid_weight.item()):
        raise ValueError("weight must be finite")
    if not bool(valid_theta.item()):
        raise ValueError("theta must be finite")
    partners, pair_indices, sine_signs = _main_rotation_metadata(
        pair_values, columns=columns, group_size=group_size
    )
    groups = columns // group_size
    rows = weight.size // columns
    result = _rotation_kernel(group_size, columns, groups, krot)(
        inputs=[
            mx.contiguous(weight.astype(mx.float32).reshape(rows, columns)),
            mx.array(partners.reshape(-1)),
            mx.array(pair_indices.reshape(-1)),
            mx.array(sine_signs.reshape(-1)),
            mx.contiguous(theta_values.reshape(-1)),
            scale_values,
        ],
        template=[
            ("GROUP_SIZE", group_size),
            ("COLUMNS", columns),
            ("GROUPS", groups),
            ("KROT", krot),
            ("INVERSE", inverse),
            ("HAS_SCALES", True),
        ],
        grid=(weight.size, 1, 1),
        threadgroup=(group_size, 1, 1),
        output_shapes=[weight.shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(result)
    return result


def _measure_pair(main_fn, vector_fn, samples):
    for _ in range(3):
        main_fn()
        vector_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter()
            (main_fn, vector_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, projection):
    rows_out = []
    group_size, krot = 128, 8
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        if projection and name != projection:
            continue
        seed = 319600 + rows + columns
        rng = np.random.default_rng(seed)
        source = mx.array(rng.normal(0, 0.2, (rows, columns)).astype(np.float32))
        theta = mx.array(
            rng.uniform(-0.35, 0.35, (krot, columns // 2)).astype(np.float32)
        )
        scales = mx.array(rng.uniform(0.7, 1.3, columns).astype(np.float32))
        pairs = _pair_schedule(columns, group_size, krot, seed=seed + 17)
        mx.eval(source, theta, scales)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)
            for inverse in (False, True):

                def main_fn(
                    value=weight,
                    reverse=inverse,
                    pair_values=pairs,
                    theta_values=theta,
                    scale_values=scales,
                ):
                    return _main_rotate(
                        value,
                        pair_values,
                        theta_values,
                        scale_values,
                        inverse=reverse,
                    )

                def vector_fn(
                    value=weight,
                    reverse=inverse,
                    pair_values=pairs,
                    theta_values=theta,
                    scale_values=scales,
                ):
                    return paroquant_rotate_mlx(
                        value,
                        pair_values,
                        theta_values,
                        group_size=group_size,
                        channel_scales=scale_values,
                        inverse=reverse,
                    )

                baseline = main_fn()
                vectorized = vector_fn()
                delta = mx.abs(baseline - vectorized)
                max_error = float(mx.max(delta).item())
                mismatches = int(mx.sum(delta != 0).item())
                main_ms, vector_ms = _measure_pair(main_fn, vector_fn, samples)
                rows_out.append(
                    (
                        name,
                        dtype_name,
                        inverse,
                        main_ms,
                        vector_ms,
                        main_ms / vector_ms,
                        max_error,
                        mismatches,
                    )
                )
                print(
                    f"{name} {dtype_name} inverse={inverse}: "
                    f"{main_ms:.3f} -> {vector_ms:.3f} ms "
                    f"({main_ms / vector_ms:.3f}x), max error {max_error:.9g}, "
                    f"mismatches {mismatches}",
                    flush=True,
                )
        del source, theta, scales, pairs, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Inverse | Main ms | PR ms | Speedup | "
        "Maximum error | Mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.3f} | "
            f"{row[4]:.3f} | {row[5]:.3f}x | {row[6]:.9g} | {row[7]} |"
        )
    print(f"\nMedian speedup: {median(row[5] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in rows_out):.3f}x")
    print(f"Maximum error: {max(row[6] for row in rows_out):.9g}")
    print(f"Total mismatches: {sum(row[7] for row in rows_out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
