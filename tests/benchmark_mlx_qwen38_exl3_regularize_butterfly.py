# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark butterfly EXL3 regularization against main."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_exl3_regularize import (
    exl3_input_regularize_mlx,
    exl3_output_regularize_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_regularize import (
    _torch_input_regularize_oracle,
    _torch_output_regularize_oracle,
)
from tests.test_mlx_exl3_rms import _torch_block_rms_oracle


@lru_cache(maxsize=1)
def _main_regularize_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_exl3_regularize_main_butterfly_benchmark",
        input_names=["weight", "signs", "rms", "mean"],
        output_names=["transformed", "channel_scales"],
        source="""
            threadgroup float current[128];

            uint lane = thread_position_in_threadgroup.x;
            uint vector = threadgroup_position_in_grid.x;
            uint row;
            uint column;
            if (INPUT_MODE) {
                row = (vector / COLUMNS) * 128u + lane;
                column = vector % COLUMNS;
            } else {
                row = vector / COLUMN_BLOCKS;
                column = (vector % COLUMN_BLOCKS) * 128u + lane;
            }

            uint channel = INPUT_MODE ? row : column;
            float root_mean_square = rms[channel];
            float scale;
            bool zero_channel;
            if (INPUT_MODE) {
                zero_channel = metal::abs(root_mean_square) < 1.0e-30f;
                float effective_rms = zero_channel ? 0.1f : root_mean_square;
                scale = signs[channel] * effective_rms / -1.24371088f + 1.0e-10f;
            } else {
                float normalized_rms = HAS_MEAN
                    ? root_mean_square / mean[0]
                    : root_mean_square;
                zero_channel = metal::abs(normalized_rms) < 1.0e-30f;
                if (APPLY_OUTPUT_SCALES) {
                    float effective_rms = zero_channel ? 0.1f : normalized_rms;
                    scale = signs[channel] * effective_rms + 1.0e-10f;
                } else {
                    scale = signs[channel];
                }
            }

            uint element = row * COLUMNS + column;
            current[lane] = weight[element] / scale;
            if ((!INPUT_MODE && row == 0u) || (INPUT_MODE && column == 0u)) {
                channel_scales[channel] = (!INPUT_MODE && zero_channel)
                    ? 0.0f
                    : scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total = 0.0f;
            for (uint source = 0u; source < 128u; ++source) {
                uint bits = lane & source;
                bits ^= bits >> 4u;
                bits ^= bits >> 2u;
                bits ^= bits >> 1u;
                float coefficient = (bits & 1u)
                    ? -0.08838834764831845f
                    : 0.08838834764831845f;
                total = metal::fma(current[source], coefficient, total);
            }
            transformed[element] = total;
        """,
    )


def _main_regularize(weight, signs, rms, *, input_mode, mean):
    rows, columns = weight.shape
    column_blocks = max(1, columns // 128)
    vector_count = (rows // 128) * columns if input_mode else rows * column_blocks
    outputs = _main_regularize_kernel()(
        inputs=[weight, signs, rms, mean],
        template=[
            ("INPUT_MODE", input_mode),
            ("APPLY_OUTPUT_SCALES", True),
            ("HAS_MEAN", not input_mode),
            ("COLUMNS", columns),
            ("COLUMN_BLOCKS", column_blocks),
        ],
        grid=(vector_count * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[weight.shape, signs.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    mx.eval(*outputs)
    return tuple(outputs)


def _measure_pair(main_fn, candidate_fn, samples):
    for _ in range(5):
        main_fn()
        candidate_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = perf_counter()
            (main_fn, candidate_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, projection):
    rows_out = []
    for index, (name, out_features, in_features) in enumerate(
        QWEN38_27B_PROJECTIONS
    ):
        if projection and name != projection:
            continue
        rng = np.random.default_rng(321300 + index)
        source = rng.normal(
            0.0, 0.2, (in_features, out_features)
        ).astype(np.float32)
        weight = mx.array(source)
        mx.eval(weight)
        for mode in ("output", "input"):
            axis = 0 if mode == "output" else 1
            shape = (1, out_features) if mode == "output" else (in_features, 1)
            signs = mx.array(
                rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
            )
            rms_values = _torch_block_rms_oracle(source, axis)
            rms = mx.array(rms_values)
            mean_value = float(np.mean(rms_values)) if mode == "output" else 1.0
            mean = mx.array([mean_value], dtype=mx.float32)
            mx.eval(signs, rms, mean)

            def main_fn(input_mode=mode == "input"):
                return _main_regularize(
                    weight, signs, rms, input_mode=input_mode, mean=mean
                )

            if mode == "output":
                expected_weight, expected_scales, _ = (
                    _torch_output_regularize_oracle(
                        source, np.asarray(signs), rms_values, mean_value, True
                    )
                )

                def candidate_fn():
                    return exl3_output_regularize_mlx(
                        weight,
                        signs,
                        rms,
                        mean=mean_value,
                        apply_scales=True,
                    )[:2]

            else:
                expected_weight, expected_scales = _torch_input_regularize_oracle(
                    source, np.asarray(signs), rms_values
                )

                def candidate_fn():
                    return exl3_input_regularize_mlx(weight, signs, rms)

            baseline = main_fn()
            candidate = candidate_fn()
            actual_weight = np.asarray(candidate[0])
            oracle_difference = actual_weight - expected_weight
            main_difference = actual_weight - np.asarray(baseline[0])
            max_abs_drift = float(np.max(np.abs(oracle_difference)))
            normalized_drift = float(
                np.linalg.norm(oracle_difference.astype(np.float64))
                / np.linalg.norm(expected_weight.astype(np.float64))
            )
            outside_tolerance = int(
                np.count_nonzero(
                    np.abs(oracle_difference)
                    > 1e-6 + 1e-6 * np.abs(expected_weight)
                )
            )
            scale_drift = float(
                np.max(np.abs(np.asarray(candidate[1]) - expected_scales))
            )
            max_main_drift = float(np.max(np.abs(main_difference)))
            main_ms, candidate_ms = _measure_pair(
                main_fn, candidate_fn, samples
            )
            rows_out.append(
                (
                    name,
                    mode,
                    main_ms,
                    candidate_ms,
                    main_ms / candidate_ms,
                    outside_tolerance,
                    max_abs_drift,
                    normalized_drift,
                    scale_drift,
                    max_main_drift,
                )
            )
            print(
                f"{name} {mode}: {main_ms:.4f} -> {candidate_ms:.4f} ms "
                f"({main_ms / candidate_ms:.3f}x), max abs drift "
                f"{max_abs_drift:.9g}, normalized drift {normalized_drift:.9g}, "
                f"outside tolerance {outside_tolerance}, scale drift "
                f"{scale_drift:.9g}, max drift from main {max_main_drift:.9g}",
                flush=True,
            )
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Mode | Main ms | PR ms | Speedup | "
        "Torch outside tolerance | Maximum absolute Torch drift | "
        "Normalized Torch drift | Maximum scale drift | Maximum drift from main |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]} | {row[6]:.9g} | {row[7]:.9g} | "
            f"{row[8]:.9g} | {row[9]:.9g} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Torch values outside tolerance: {sum(row[5] for row in rows_out)}")
    print(f"Maximum absolute Torch drift: {max(row[6] for row in rows_out):.9g}")
    print(f"Maximum normalized Torch drift: {max(row[7] for row in rows_out):.9g}")
    print(f"Maximum scale drift: {max(row[8] for row in rows_out):.9g}")
    print(f"Maximum drift from main: {max(row[9] for row in rows_out):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
