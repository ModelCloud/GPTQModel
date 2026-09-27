# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark BitsAndBytes processor packing through the MLX 4-bit bridge."""

import argparse
import gc
from functools import partial
from statistics import median
from time import perf_counter

import bitsandbytes as bnb
import torch

from gptqmodel.nn_modules.qlinear.bitsandbytes import _quantize_4bit_weight_mlx_to_torch
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _torch_quantize(weight, quant_type, compressed):
    return bnb.functional.quantize_4bit(
        weight,
        quant_type=quant_type,
        blocksize=64,
        compress_statistics=compressed,
        quant_storage=torch.uint8,
    )


def _mlx_quantize(weight, quant_type, compressed):
    return _quantize_4bit_weight_mlx_to_torch(
        weight,
        bnb=bnb,
        quant_type=quant_type,
        block_size=64,
        compress_statistics=compressed,
    )


def _measure_pair(torch_fn, mlx_fn, samples):
    for _ in range(10):
        torch_fn()
        mlx_fn()
    timings = [[], []]
    for sample in range(samples):
        order = (
            ((0, torch_fn), (1, mlx_fn))
            if sample % 2 == 0
            else ((1, mlx_fn), (0, torch_fn))
        )
        for path, function in order:
            started = perf_counter()
            function()
            timings[path].append((perf_counter() - started) * 1000)
    return median(timings[0]), median(timings[1])


def _accuracy(torch_result, mlx_result, compressed):
    torch_weight, torch_state = torch_result
    mlx_weight, mlx_state = mlx_result
    packed_mismatches = int(torch.count_nonzero(torch_weight != mlx_weight))
    nested_code_mismatches = 0
    if compressed:
        nested_code_mismatches = int(
            torch.count_nonzero(torch_state.absmax != mlx_state.absmax)
        )
        maximum_scale_error = max(
            float(
                torch.max(
                    torch.abs(torch_state.state2.absmax - mlx_state.state2.absmax)
                )
            ),
            abs(float(torch_state.offset) - float(mlx_state.offset)),
        )
    else:
        maximum_scale_error = float(
            torch.max(torch.abs(torch_state.absmax - mlx_state.absmax))
        )
    return packed_mismatches, nested_code_mismatches, maximum_scale_error


def run(samples, projection):
    results = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        torch.manual_seed(380027 + index)
        source = torch.randn((rows, columns), dtype=torch.float32)
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            weight = source.to(dtype).contiguous()
            for quant_type in ("nf4", "fp4"):
                for compressed in (False, True):
                    torch_fn = partial(_torch_quantize, weight, quant_type, compressed)
                    mlx_fn = partial(_mlx_quantize, weight, quant_type, compressed)
                    accuracy = _accuracy(torch_fn(), mlx_fn(), compressed)
                    torch_ms, mlx_ms = _measure_pair(torch_fn, mlx_fn, samples)
                    row = (
                        name,
                        dtype_name,
                        quant_type,
                        compressed,
                        torch_ms,
                        mlx_ms,
                        torch_ms / mlx_ms,
                        *accuracy,
                    )
                    results.append(row)
                    print(
                        f"{name} {dtype_name} {quant_type} compressed={compressed}: "
                        f"{torch_ms:.4f} -> {mlx_ms:.4f} ms ({torch_ms / mlx_ms:.3f}x), "
                        f"packed mismatches {accuracy[0]}, nested code mismatches {accuracy[1]}, "
                        f"max scale error {accuracy[2]:.8g}",
                        flush=True,
                    )
        del source, weight
        gc.collect()

    print(
        "\n| Projection | Dtype | Format | Compressed | Torch ms | MLX ms | Speedup | "
        "Packed mismatches | Nested code mismatches | Maximum scale error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.4f} | "
            f"{row[5]:.4f} | {row[6]:.3f}x | {row[7]} | {row[8]} | {row[9]:.8g} |"
        )
    print(f"\nMedian speedup: {median(row[6] for row in results):.3f}x")
    print(f"Minimum speedup: {min(row[6] for row in results):.3f}x")
    print(f"Maximum speedup: {max(row[6] for row in results):.3f}x")
    print(f"Total packed mismatches: {sum(row[7] for row in results)}")
    print(f"Total nested code mismatches: {sum(row[8] for row in results)}")
    print(f"Maximum scale error: {max(row[9] for row in results):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    parser.add_argument("--projection")
    args = parser.parse_args()
    run(args.samples, args.projection)
