# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ method: Elias Frantar et al., https://arxiv.org/abs/2210.17323
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark separate GPTQ affine reductions against fused MLX selection."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_native import (
    _accurate_matmul_mlx,
    gptq_quantize_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


BITS = (2, 4, 8)
GROUP_SIZES = (32, 64, 128)
DTYPES = (("FP16", mx.float16), ("BF16", mx.bfloat16))
DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64
CONFIG_PROJECTION = "full_attn.k_proj"


@lru_cache(maxsize=1)
def _main_group_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_main_group_update",
        input_names=["weights", "hinv", "scales", "biases"],
        output_names=["packed", "errors"],
        source="""
            uint row = thread_position_in_grid.x;
            float values[G];
            for (int k = 0; k < G; ++k) {
                values[k] = weights[row * G + k];
            }
            float scale = scales[row];
            float bias = biases[row];
            uint word = 0;
            for (int k = 0; k < G; ++k) {
                float code = scale == 0.0f ? 0.0f :
                    metal::clamp(metal::rint((values[k] - bias) / scale),
                                 0.0f, float((1 << BITS) - 1));
                float quantized = code * scale + bias;
                float error = (values[k] - quantized) / hinv[k * G + k];
                word |= uint(code) << ((k % PACK) * BITS);
                if (k % PACK == PACK - 1) {
                    packed[row * (G / PACK) + k / PACK] = word;
                    word = 0;
                }
                errors[row * G + k] = error;
                for (int j = k + 1; j < G; ++j) {
                    values[j] -= error * hinv[k * G + j];
                }
            }
        """,
    )


def _main_affine_params(group, bits):
    minimum = mx.min(group, axis=1)
    maximum = mx.maximum(mx.max(group, axis=1), 0)
    scale = mx.maximum((maximum - minimum) / (2**bits - 1), 1e-7)
    use_minimum = mx.abs(minimum) > mx.abs(maximum)
    scale = mx.where(use_minimum, scale, -scale)
    edge = mx.where(use_minimum, minimum, maximum)
    zero_code = mx.round(edge / scale)
    at_zero = zero_code == 0
    scale = mx.where(at_zero, scale, edge / mx.where(at_zero, 1, zero_code))
    bias = mx.where(at_zero, 0, edge)
    return scale[:, None], bias[:, None]


def _main_quantize(weight, inverse_hessian, bits, group_size):
    """Reproduce main's separate affine reductions and retained group kernel."""
    rows, columns = weight.shape
    remaining = weight.astype(mx.float32)
    packed_groups, all_scales, all_biases = [], [], []
    values_per_word = 32 // bits
    kernel = _main_group_kernel()
    for start in range(0, columns, group_size):
        end = start + group_size
        group = remaining[:, :group_size]
        scales, biases = _main_affine_params(group, bits)
        packed, errors = kernel(
            inputs=[group, inverse_hessian[start:end, start:end], scales, biases],
            template=[
                ("G", group_size),
                ("BITS", bits),
                ("PACK", values_per_word),
            ],
            grid=(rows, 1, 1),
            threadgroup=(min(rows, 64), 1, 1),
            output_shapes=[
                (rows, group_size // values_per_word),
                (rows, group_size),
            ],
            output_dtypes=[mx.uint32, mx.float32],
        )
        packed_groups.append(packed)
        all_scales.append(scales)
        all_biases.append(biases)
        if end < columns:
            remaining = remaining[:, group_size:] - _accurate_matmul_mlx(
                errors, inverse_hessian[start:end, end:]
            )
            mx.eval(remaining)
    result = (
        mx.concatenate(packed_groups, axis=1),
        mx.concatenate(all_scales, axis=1),
        mx.concatenate(all_biases, axis=1),
    )
    mx.eval(*result)
    return result


def _torch_diagonal_oracle(source, bits, group_size):
    rows, columns = source.shape
    grouped = torch.from_numpy(source).reshape(rows, columns // group_size, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1).clamp_min(0)
    scales = ((maximum - minimum) / (2**bits - 1)).clamp_min(1e-7)
    use_minimum = minimum.abs() > maximum.abs()
    scales = torch.where(use_minimum, scales, -scales)
    edge = torch.where(use_minimum, minimum, maximum)
    zero_code = torch.round(edge / scales)
    at_zero = zero_code == 0
    scales = torch.where(
        at_zero,
        scales,
        edge / torch.where(at_zero, torch.ones_like(zero_code), zero_code),
    )
    biases = torch.where(at_zero, torch.zeros_like(edge), edge)
    codes = torch.round(
        (grouped - biases[..., None]) / scales[..., None]
    ).clamp(0, 2**bits - 1).to(torch.int64)
    values_per_word = 32 // bits
    shifts = torch.arange(values_per_word, dtype=torch.int64) * bits
    packed = (
        (codes.reshape(rows, -1, values_per_word) << shifts).sum(dim=-1)
        .numpy().astype(np.uint32)
    )
    return packed, scales.numpy(), biases.numpy()


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


def _comparison(main, fused):
    errors = [
        float(mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))).item())
        for left, right in zip(main, fused)
    ]
    mismatches = [
        int(mx.sum(left != right).item()) for left, right in zip(main, fused)
    ]
    return max(errors), sum(mismatches)


def _performance_row(name, dtype_name, weight, factor, bits, group_size, samples):
    def main_fn():
        return _main_quantize(weight, factor, bits, group_size)

    def fused_fn():
        return gptq_quantize_weight_mlx(weight, factor, bits, group_size)

    error, mismatches = _comparison(main_fn(), fused_fn())
    main_ms, fused_ms = _measure_pair(main_fn, fused_fn, samples)
    return (
        name,
        dtype_name,
        bits,
        group_size,
        main_ms,
        fused_ms,
        main_ms / fused_ms,
        error,
        mismatches,
    )


def _print_performance(title, rows):
    print(f"\n### {title}")
    print(
        "| Projection | Dtype | Bits | Group | Main ms | PR ms | Speedup | "
        "Maximum error | Mismatches |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.3f} | "
            f"{row[5]:.3f} | {row[6]:.3f}x | {row[7]:.9g} | {row[8]} |"
        )
    print(f"\nMedian speedup: {median(row[6] for row in rows):.3f}x")
    print(f"Minimum speedup: {min(row[6] for row in rows):.3f}x")
    print(f"Maximum speedup: {max(row[6] for row in rows):.3f}x")
    print(f"Maximum error: {max(row[7] for row in rows):.9g}")
    print(f"Total mismatches: {sum(row[8] for row in rows)}")


def run(samples, projection):
    shape_rows = []
    config_rows = []
    config_projection = projection or CONFIG_PROJECTION
    accuracy = {
        (dtype_name, bits, group_size): [0, 0.0, 0.0]
        for dtype_name, _ in DTYPES
        for bits in BITS
        for group_size in GROUP_SIZES
    }
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        mx.random.seed(410200 + index)
        source = mx.random.normal((rows, columns)) * 0.2
        factor = mx.eye(columns, dtype=mx.float32)
        factor = factor + 0.05 * mx.eye(columns, k=1, dtype=mx.float32)
        mx.eval(source, factor)
        for dtype_name, dtype in DTYPES:
            weight = source.astype(dtype)
            mx.eval(weight)
            shape_rows.append(
                _performance_row(
                    name,
                    dtype_name,
                    weight,
                    factor,
                    DEFAULT_BITS,
                    DEFAULT_GROUP_SIZE,
                    samples,
                )
            )
            if name == config_projection:
                for bits in BITS:
                    for group_size in GROUP_SIZES:
                        config_rows.append(
                            _performance_row(
                                name,
                                dtype_name,
                                weight,
                                factor,
                                bits,
                                group_size,
                                samples,
                            )
                        )

            resident = np.asarray(weight.astype(mx.float32))
            identity = mx.eye(columns, dtype=mx.float32)
            for bits in BITS:
                for group_size in GROUP_SIZES:
                    actual = gptq_quantize_weight_mlx(
                        weight, identity, bits=bits, group_size=group_size
                    )
                    observed = tuple(np.asarray(value) for value in actual)
                    expected = _torch_diagonal_oracle(resident, bits, group_size)
                    summary = accuracy[(dtype_name, bits, group_size)]
                    summary[0] += int(np.count_nonzero(observed[0] != expected[0]))
                    summary[1] = max(
                        summary[1], float(np.max(np.abs(observed[1] - expected[1])))
                    )
                    summary[2] = max(
                        summary[2], float(np.max(np.abs(observed[2] - expected[2])))
                    )
                    del actual, observed, expected
            del resident, identity, weight
        del source, factor
        mx.clear_cache()
        gc.collect()

    _print_performance("All Qwen3.8-27B projections, default GPTQ", shape_rows)
    _print_performance(
        f"{config_projection} supported configuration matrix", config_rows
    )
    print(
        "\n### Independent Torch diagonal-Hessian oracle across every Qwen shape"
    )
    print(
        "| Dtype | Bits | Group | Projection cases | Packed mismatches | "
        "Maximum scale drift | Maximum bias drift |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    case_count = sum(
        1 for name, _, _ in QWEN38_27B_PROJECTIONS if not projection or name == projection
    )
    for dtype_name, _ in DTYPES:
        for bits in BITS:
            for group_size in GROUP_SIZES:
                mismatches, scale_drift, bias_drift = accuracy[
                    (dtype_name, bits, group_size)
                ]
                print(
                    f"| {dtype_name} | {bits} | {group_size} | {case_count} | "
                    f"{mismatches} | {scale_drift:.9g} | {bias_drift:.9g} |"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
