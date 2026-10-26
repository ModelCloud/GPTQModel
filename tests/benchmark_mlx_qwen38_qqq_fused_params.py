# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ method: Meituan, Ying Zhang et al., https://arxiv.org/abs/2406.09904
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark QQQ's separate parameter reductions against fused selection."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_native import _accurate_matmul_mlx
from gptqmodel.quantization.mlx_qqq import (
    _qqq_group_kernel,
    qqq_quantize_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_qqq_quantization import (
    _boundary_margin,
    _codes,
    _torch_banded_oracle,
)


def _main_params(group):
    minimum = mx.minimum(mx.min(group, axis=1), 0)
    maximum = mx.maximum(mx.max(group, axis=1), 0)
    maximum = mx.maximum(mx.abs(minimum), maximum)
    minimum = mx.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = mx.where(empty, -1, minimum)
    maximum = mx.where(empty, 1, maximum)
    return (maximum - minimum)[:, None] / 15, mx.full(
        (group.shape[0], 1), 8, dtype=mx.float32
    )


def _main_quantize(weight, inverse_hessian):
    rows, columns = weight.shape
    original = weight.astype(mx.float32)
    maximum = mx.max(mx.abs(original), axis=1)
    scale_extra = mx.where(maximum == 0, 1, maximum)[:, None] / 127
    remaining = original
    quantized, scales, zeros = [], [], []
    kernel = _qqq_group_kernel()
    for start in range(0, columns, 128):
        end = start + 128
        group = mx.contiguous(remaining[:, :128])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        scale, zero = _main_params(group)
        q, error = kernel(
            inputs=[group, factor, scale, zero],
            template=[("G", 128), ("SIGNED", False)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, 128), (rows, 128)],
            output_dtypes=[mx.float32, mx.float32],
        )
        quantized.append(q)
        scales.append(scale)
        zeros.append(zero)
        if end < columns:
            remaining = remaining[:, 128:] - _accurate_matmul_mlx(
                error, inverse_hessian[start:end, end:]
            )
            mx.eval(remaining)
    result = (
        mx.concatenate(quantized, axis=1),
        mx.concatenate(scales, axis=1),
        mx.concatenate(zeros, axis=1),
        scale_extra,
    )
    mx.eval(*result)
    return result


def _measure_pair(main_fn, fused_fn, samples):
    for _ in range(3):
        main_fn()
        fused_fn()
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = perf_counter()
            (main_fn, fused_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples, projection):
    performance_rows = []
    oracle_rows = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(320100 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        factor = mx.eye(columns, dtype=mx.float32)
        factor = factor + 0.05 * mx.eye(columns, k=1, dtype=mx.float32)
        mx.eval(source, factor)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight, factor_input=factor):
                return _main_quantize(value, factor_input)

            def fused_fn(value=weight, factor_input=factor):
                return qqq_quantize_weight_mlx(
                    value, factor_input, group_size=128
                )

            baseline = main_fn()
            fused = fused_fn()
            errors = [
                float(mx.max(mx.abs(expected - actual)).item())
                for expected, actual in zip(baseline, fused)
            ]
            mismatches = [
                int(mx.sum(expected != actual).item())
                for expected, actual in zip(baseline, fused)
            ]

            resident = np.asarray(weight.astype(mx.float32))
            expected = _torch_banded_oracle(resident, 128)
            observed = tuple(np.asarray(value) for value in fused)
            observed_codes = _codes(observed, 128)
            expected_codes = _codes(expected, 128)
            changed = observed_codes != expected_codes
            coordinates = np.argwhere(changed)
            margins = [
                float(
                    _boundary_margin(
                        resident[row],
                        expected[0][row],
                        expected[1][row, column // 128],
                        int(column),
                    )
                )
                for row, column in coordinates
            ]
            direct_ties = sum(margin < 0.0002 for margin in margins)
            drift = np.abs(observed[0] - expected[0])
            away = ~changed
            oracle_rows.append(
                (
                    name,
                    dtype_name,
                    len(coordinates),
                    direct_ties,
                    len(coordinates) - direct_ties,
                    float(np.max(drift[away])),
                    float(np.max(np.abs(observed[1] - expected[1]))),
                    int(np.count_nonzero(observed[2] != expected[2])),
                    float(np.max(np.abs(observed[3] - expected[3]))),
                )
            )

            main_ms, fused_ms = _measure_pair(main_fn, fused_fn, samples)
            performance_rows.append(
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
        del source, factor, weight
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Main ms | PR ms | Speedup | Maximum error | "
        "Mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in performance_rows:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | "
            f"{row[4]:.3f}x | {row[5]:.9g} | {row[6]} |"
        )
    print(
        f"\nMedian speedup: {median(row[4] for row in performance_rows):.3f}x"
    )
    print(f"Minimum speedup: {min(row[4] for row in performance_rows):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in performance_rows):.3f}x")
    print(f"Maximum error: {max(row[5] for row in performance_rows):.9g}")
    print(f"Total mismatches: {sum(row[6] for row in performance_rows)}")

    print(
        "\n| Projection | Dtype | Changed codes | Direct ties | Tie descendants | "
        "Max weight drift away | Maximum scale drift | Zero mismatches | "
        "Maximum extra-scale drift |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in oracle_rows:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]:.9g} | {row[6]:.9g} | {row[7]} | {row[8]:.9g} |"
        )
    print(f"\nTorch changed codes: {sum(row[2] for row in oracle_rows)}")
    print(f"Torch direct ties: {sum(row[3] for row in oracle_rows)}")
    print(f"Torch tie descendants: {sum(row[4] for row in oracle_rows)}")
    print(f"Max weight drift away: {max(row[5] for row in oracle_rows):.9g}")
    print(f"Maximum scale drift: {max(row[6] for row in oracle_rows):.9g}")
    print(f"Zero mismatches: {sum(row[7] for row in oracle_rows)}")
    print(f"Maximum extra-scale drift: {max(row[8] for row in oracle_rows):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
