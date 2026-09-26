# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark main's GPTAQ parameter reductions against fused Metal selection."""

import argparse
import gc
from decimal import Decimal
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_gptaq import gptaq_quantize_weight_mlx
from gptqmodel.quantization.mlx_native import _accurate_matmul_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_gptaq_quantization import (
    _codes,
    _exact_banded_tie_margin,
    _torch_banded_oracle,
)


@lru_cache(maxsize=1)
def _main_group_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_gptaq_group_update_main_baseline",
        input_names=["weights", "hinv", "correction", "scales", "zeros"],
        output_names=["quantized", "errors", "corrected"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
            }
            float scale = scales[row];
            float zero = zeros[row];
            for (int k = 0; k < G; ++k) {
                float value = values[k];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                float q = scale * (code - zero);
                float error = (value - q) / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j]
                               - value * correction[k * G + j];
                }
            }
            for (int k = 0; k < G; ++k) {
                corrected[row * G + k] = values[k];
            }
        """,
    )


def _main_group_params(group, bits, sym):
    minimum = mx.minimum(mx.min(group, axis=1), 0)
    maximum = mx.maximum(mx.max(group, axis=1), 0)
    if sym:
        maximum = mx.maximum(mx.abs(minimum), maximum)
        minimum = mx.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = mx.where(empty, -1, minimum)
    maximum = mx.where(empty, 1, maximum)
    scale = (maximum - minimum) / (2**bits - 1)
    zero = (
        mx.full(scale.shape, 2 ** (bits - 1), dtype=mx.float32)
        if sym
        else mx.round(-minimum / scale)
    )
    return scale[:, None], zero[:, None]


def _main_quantize(weight, inverse_hessian, correction):
    rows, columns = weight.shape
    bits, group_size, sym = 4, 128, True
    remaining = weight.astype(mx.float32)
    quantized, scales, zeros = [], [], []
    kernel = _main_group_kernel()
    for start in range(0, columns, group_size):
        end = start + group_size
        group = mx.contiguous(remaining[:, :group_size])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        correction_group = mx.contiguous(correction[start:end, start:end])
        scale, zero = _main_group_params(group, bits, sym)
        q, error, corrected = kernel(
            inputs=[group, factor, correction_group, scale, zero],
            template=[("G", group_size), ("MAXQ", 2**bits - 1)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, group_size)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        quantized.append(q)
        scales.append(scale)
        zeros.append(zero)
        if end < columns:
            remaining = (
                remaining[:, group_size:]
                - _accurate_matmul_mlx(error, inverse_hessian[start:end, end:])
                + _accurate_matmul_mlx(corrected, correction[start:end, end:])
            )
            mx.eval(remaining)
    result = (
        mx.concatenate(quantized, axis=1),
        mx.concatenate(scales, axis=1),
        mx.concatenate(zeros, axis=1),
    )
    mx.eval(*result)
    return result


def _measure_pair(main_fn, fused_fn, samples):
    for _ in range(3):
        main_fn()
        fused_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter()
            (main_fn, fused_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, projection):
    rows_out = []
    oracle_rows = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(320000 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        factor = mx.eye(columns, dtype=mx.float32)
        factor = factor + 0.05 * mx.eye(columns, k=1, dtype=mx.float32)
        correction = 0.002 * mx.eye(columns, k=1, dtype=mx.float32)
        mx.eval(source, factor, correction)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(
                value=weight,
                factor_input=factor,
                correction_input=correction,
            ):
                return _main_quantize(value, factor_input, correction_input)

            def fused_fn(
                value=weight,
                factor_input=factor,
                correction_input=correction,
            ):
                return gptaq_quantize_weight_mlx(
                    value,
                    factor_input,
                    correction_input,
                )

            baseline = main_fn()
            fused = fused_fn()
            errors, mismatches = [], []
            for expected, actual in zip(baseline, fused):
                delta = mx.abs(expected - actual)
                errors.append(float(mx.max(delta).item()))
                mismatches.append(int(mx.sum(delta != 0).item()))
            source_array = np.asarray(weight.astype(mx.float32))
            oracle = _torch_banded_oracle(source_array)
            observed = tuple(np.asarray(value) for value in fused)
            observed_codes = _codes(observed, 128)
            oracle_codes = _codes(oracle, 128)
            changed = observed_codes != oracle_codes
            coordinates = np.argwhere(changed)
            direct_ties = sum(
                _exact_banded_tie_margin(source_array[row], int(column))
                < Decimal("1e-30")
                for row, column in coordinates
            )
            drift = np.abs(observed[0] - oracle[0])
            away = ~changed
            allowed = 1e-6 + 1e-6 * np.abs(oracle[0])
            oracle_rows.append(
                (
                    name,
                    dtype_name,
                    len(coordinates),
                    direct_ties,
                    len(coordinates) - direct_ties,
                    float(np.max(drift)),
                    float(np.max(drift[away])),
                    int(np.count_nonzero((drift > allowed) & away)),
                    float(np.max(np.abs(observed[1] - oracle[1]))),
                    int(np.count_nonzero(observed[2] != oracle[2])),
                )
            )
            main_ms, fused_ms = _measure_pair(main_fn, fused_fn, samples)
            rows_out.append(
                (
                    name,
                    dtype_name,
                    main_ms,
                    fused_ms,
                    main_ms / fused_ms,
                    max(errors),
                    sum(mismatches),
                )
            )
            print(
                f"{name} {dtype_name}: {main_ms:.3f} -> {fused_ms:.3f} ms "
                f"({main_ms / fused_ms:.3f}x), max error "
                f"{max(errors):.9g}, mismatches {sum(mismatches)}",
                flush=True,
            )
            del (
                baseline,
                fused,
                source_array,
                oracle,
                observed,
                observed_codes,
                oracle_codes,
                changed,
                coordinates,
                drift,
                away,
                allowed,
            )
        del source, factor, correction, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Main ms | PR ms | Speedup | Maximum error | "
        "Mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | "
            f"{row[4]:.3f}x | {row[5]:.9g} | {row[6]} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Maximum error: {max(row[5] for row in rows_out):.9g}")
    print(f"Total mismatches: {sum(row[6] for row in rows_out)}")

    print(
        "\n| Projection | Dtype | Changed codes | Direct ties | Tie descendants | "
        "Weight max abs | Max abs away | Outside tolerance away | "
        "Scale max abs | Zero mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in oracle_rows:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]:.9g} | {row[6]:.9g} | {row[7]} | {row[8]:.9g} | "
            f"{row[9]} |"
        )
    print(f"\nTotal changed codes: {sum(row[2] for row in oracle_rows)}")
    print(f"Total direct ties: {sum(row[3] for row in oracle_rows)}")
    print(f"Total tie descendants: {sum(row[4] for row in oracle_rows)}")
    print(f"Maximum weight error: {max(row[5] for row in oracle_rows):.9g}")
    print(
        f"Maximum error away from changed codes: {max(row[6] for row in oracle_rows):.9g}"
    )
    print(f"Total values outside tolerance away: {sum(row[7] for row in oracle_rows)}")
    print(f"Maximum scale error: {max(row[8] for row in oracle_rows):.9g}")
    print(f"Total zero mismatches: {sum(row[9] for row in oracle_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
