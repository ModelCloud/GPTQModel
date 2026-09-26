# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark main's FOEM parameter reductions against fused Metal selection."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_foem import foem_quantize_weight_mlx
from gptqmodel.quantization.mlx_native import _accurate_matmul_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_group_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_foem_group_update_main_baseline",
        input_names=["weights", "raw", "hinv", "scales", "zeros", "beta"],
        output_names=["quantized", "errors"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            float original[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
                original[k] = raw[row * G + k];
            }
            float scale = scales[row];
            float zero = zeros[row];
            float correction = beta;
            for (int k = 0; k < G; ++k) {
                float value = values[k];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                float q = scale * (code - zero);
                float error = ((value - q) - (value - original[k]) * correction)
                              / hinv[k * G + k];
                quantized[row * G + k] = q;
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
                if (k + 1 < G) {
                    values[k + 1] -= correction * (values[k + 1] - original[k + 1]);
                }
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


def _main_quantize(weight, inverse_hessian, bits=4, group_size=128, beta=0.2, sym=True):
    if not bool(mx.all(mx.isfinite(weight)).item()):
        raise ValueError("weight must be finite")
    if not bool(mx.all(mx.isfinite(inverse_hessian)).item()):
        raise ValueError("inverse_hessian must be finite")
    if not bool(mx.all(mx.diag(inverse_hessian) > 0).item()):
        raise ValueError("inverse_hessian diagonal must be positive")
    rows, columns = weight.shape
    raw = weight.astype(mx.float32)
    beta_input = mx.array(beta, dtype=mx.float32)
    remaining = raw
    quantized_groups, scales, zeros = [], [], []
    kernel = _main_group_kernel()
    for start in range(0, columns, group_size):
        end = start + group_size
        group = mx.contiguous(remaining[:, :group_size])
        original = mx.contiguous(raw[:, start:end])
        factor = mx.contiguous(inverse_hessian[start:end, start:end])
        scale, zero = _main_group_params(group, bits, sym)
        quantized, errors = kernel(
            inputs=[group, original, factor, scale, zero, beta_input],
            template=[("G", group_size), ("MAXQ", 2**bits - 1)],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[(rows, group_size), (rows, group_size)],
            output_dtypes=[mx.float32, mx.float32],
        )
        quantized_groups.append(quantized)
        scales.append(scale)
        zeros.append(zero)
        if end < columns:
            remaining = remaining[:, group_size:] - _accurate_matmul_mlx(
                errors, inverse_hessian[start:end, end:]
            )
            mx.eval(remaining)
    result = tuple(
        mx.concatenate(parts, axis=1) for parts in (quantized_groups, scales, zeros)
    )
    mx.eval(*result)
    return result


def _measure_pair(main_fn, fused_fn, samples):
    main_fn()
    fused_fn()
    timings = [[], []]
    results = [None, None]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter()
            results[path] = (main_fn, fused_fn)[path]()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1]), results


def run(samples, projection):
    rows_out = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(319300 + index)
        source = mx.random.normal((rows, columns))
        factor = mx.eye(columns, dtype=mx.float32)
        factor = factor + 0.05 * mx.eye(columns, k=1, dtype=mx.float32)
        mx.eval(source, factor)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight, factor_input=factor):
                return _main_quantize(value, factor_input)

            def fused_fn(value=weight, factor_input=factor):
                return foem_quantize_weight_mlx(value, factor_input)

            main_ms, fused_ms, results = _measure_pair(main_fn, fused_fn, samples)
            errors = []
            mismatches = []
            for main_value, fused_value in zip(*results):
                delta = mx.abs(main_value - fused_value)
                errors.append(float(mx.max(delta).item()))
                mismatches.append(int(mx.sum(delta != 0).item()))
            rows_out.append(
                (
                    name,
                    dtype_name,
                    main_ms,
                    fused_ms,
                    main_ms / fused_ms,
                    *errors,
                    *mismatches,
                )
            )
            print(
                f"{name} {dtype_name}: {main_ms:.3f} -> {fused_ms:.3f} ms "
                f"({main_ms / fused_ms:.3f}x), max errors {errors}, "
                f"mismatches {mismatches}",
                flush=True,
            )
            del weight, results
        del source, factor
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Main ms | PR ms | Speedup | Q max abs | Scale max abs | Zero max abs | Q mismatches | Scale mismatches | Zero mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | "
            f"{row[4]:.3f}x | {row[5]:.3e} | {row[6]:.3e} | "
            f"{row[7]:.3e} | {row[8]} | {row[9]} | {row[10]} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
