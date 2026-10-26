# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark production GGUF packing through the native MLX bridge."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    _gguf_quantize,
    _gguf_quantize_weight_mlx_to_torch,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


QTYPES = (
    "Q1_0",
    "Q1_0_g128",
    "Q2_0",
    "Q4_0",
    "Q8_0",
    "Q4_K",
    "Q5_K",
    "Q6_K",
)


def _cpu_quantize(weight, qtype):
    packed = _gguf_quantize(weight.numpy(), qtype)
    return torch.from_numpy(np.ascontiguousarray(packed)).to(torch.uint8)


def _mlx_quantize(weight, qtype):
    return _gguf_quantize_weight_mlx_to_torch(weight, qtype)


def _measure_pair(cpu_fn, mlx_fn, samples):
    for function in (cpu_fn, mlx_fn):
        result = function()
        del result
    timings = [[], []]
    functions = (cpu_fn, mlx_fn)
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            started = perf_counter()
            result = functions[index]()
            timings[index].append((perf_counter() - started) * 1000)
            del result
    return median(timings[0]), median(timings[1])


def run(samples, projection, qtype_filter):
    if samples <= 0:
        raise ValueError("samples must be positive")
    results = []
    for index, (name, rows, columns) in enumerate(QWEN38_27B_PROJECTIONS):
        if projection and name != projection:
            continue
        torch.manual_seed(823800 + index)
        source = torch.randn((rows, columns), dtype=torch.float32)
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            weight = source.to(dtype).float().contiguous()
            for qtype in QTYPES:
                if qtype_filter and qtype.lower() != qtype_filter.lower():
                    continue
                def cpu_fn(weight=weight, qtype=qtype):
                    return _cpu_quantize(weight, qtype)

                def mlx_fn(weight=weight, qtype=qtype):
                    return _mlx_quantize(weight, qtype)
                expected = cpu_fn()
                actual = mlx_fn()
                differing = actual != expected
                mismatches = int(torch.count_nonzero(differing))
                max_byte_drift = (
                    int((actual.short() - expected.short()).abs().max()) if mismatches else 0
                )
                if mismatches:
                    raise AssertionError(
                        f"{name} {dtype_name} {qtype}: {mismatches} byte mismatches, "
                        f"maximum byte drift {max_byte_drift}"
                    )
                cpu_ms, mlx_ms = _measure_pair(cpu_fn, mlx_fn, samples)
                row = (
                    name,
                    dtype_name,
                    rows,
                    columns,
                    qtype,
                    mismatches,
                    max_byte_drift,
                    cpu_ms,
                    mlx_ms,
                    cpu_ms / mlx_ms,
                )
                results.append(row)
                print(" | ".join(str(value) for value in row), flush=True)
                del expected, actual, differing
                gc.collect()
                mx.clear_cache()
            del weight
            gc.collect()
        del source
        gc.collect()

    print(
        "\n| Projection | Dtype | Shape | Qtype | Byte mismatches | Max byte drift | "
        "Main ms | MLX ms | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]}x{row[3]} | {row[4]} | "
            f"{row[5]} | {row[6]} | {row[7]:.4f} | {row[8]:.4f} | {row[9]:.3f}x |"
        )

    print("\n| Qtype | Median speedup | Minimum speedup | Maximum speedup | Byte mismatches |")
    print("|---|---:|---:|---:|---:|")
    for qtype in QTYPES:
        matching = [row for row in results if row[4] == qtype]
        if not matching:
            continue
        print(
            f"| {qtype} | {median(row[9] for row in matching):.3f}x | "
            f"{min(row[9] for row in matching):.3f}x | "
            f"{max(row[9] for row in matching):.3f}x | "
            f"{sum(row[5] for row in matching)} |"
        )
    print(f"\nOverall median speedup: {median(row[9] for row in results):.3f}x")
    print(f"Overall minimum speedup: {min(row[9] for row in results):.3f}x")
    print(f"Overall maximum speedup: {max(row[9] for row in results):.3f}x")
    print(f"Total byte mismatches: {sum(row[5] for row in results)}")
    print(f"Maximum byte drift: {max(row[6] for row in results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--projection")
    parser.add_argument("--qtype", choices=QTYPES)
    args = parser.parse_args()
    run(args.samples, args.projection, args.qtype)
