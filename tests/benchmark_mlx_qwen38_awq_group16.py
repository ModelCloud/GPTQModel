# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark main and direct-dtype AWQ group-16 decode on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_awq import _awq_group16_kernel
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_awq_group16 import _awq_group16_fixture, _rounded_oracle


def measure_pair(baseline, candidate, x, warmups, samples):
    for _ in range(warmups):
        mx.eval(baseline(x), candidate(x))
    timings = {"main": [], "pr": []}
    functions = (("main", baseline), ("pr", candidate))
    for sample in range(samples):
        ordered = functions if sample % 2 == 0 else functions[::-1]
        for label, function in ordered:
            start = time.perf_counter_ns()
            mx.eval(function(x))
            timings[label].append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(timings["main"]), statistics.median(timings["pr"])


def main_decode(layer, x):
    """Reconstruct main's FP32 kernel output followed by an activation-dtype cast."""
    output_dims = layer.weight.shape[0]
    threads = 128 if layer.input_dims >= 8192 or output_dims >= 16384 else 64
    bias = layer.bias if "bias" in layer else layer.zero_bias[0]
    output = _awq_group16_kernel(mx.float32)(
        inputs=[
            x, layer.weight, layer.scales_even, layer.scales_odd,
            layer.biases_even, layer.biases_odd, bias,
        ],
        template=[
            ("K", layer.input_dims), ("THREADS", threads),
            ("GROUPS", threads // 32),
        ],
        grid=(threads, output_dims, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(1, output_dims)], output_dtypes=[mx.float32],
    )[0]
    return output.astype(x.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        source = (np.sin(positions * 0.013) * 0.08 + np.cos(positions * 0.007) * 0.04)[None]
        candidate, reference, output_pattern, layer_bias = _awq_group16_fixture(
            out_features, in_features,
        )
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            x = mx.array(source).astype(dtype)
            mx.eval(x)
            baseline = partial(main_decode, candidate)
            main_ms, pr_ms = measure_pair(baseline, candidate, x, args.warmups, args.samples)
            main_output, pr_output = baseline(x), candidate(x)
            mx.eval(main_output, pr_output)
            expected, _ = _rounded_oracle(x, reference, output_pattern, layer_bias, dtype)
            main_visible = np.asarray(main_output.astype(mx.float32))
            pr_visible = np.asarray(pr_output.astype(mx.float32))
            record = (
                name, dtype_name, main_ms, pr_ms, main_ms / pr_ms,
                float(np.max(np.abs(main_visible - expected))),
                float(np.max(np.abs(pr_visible - expected))),
                float(np.max(np.abs(pr_visible - main_visible))),
            )
            records.append(record)
            print(
                f"{name} {dtype_name}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                f"({main_ms / pr_ms:.3f}x), errors "
                f"{record[5]:.8f}/{record[6]:.8f}/{record[7]:.8f}",
                flush=True,
            )
        del candidate
        gc.collect()
        mx.clear_cache()

    print("\n| Projection | Dtype | Main ms | PR ms | Speedup | Main error | PR error | PR-main |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]:.8f} | {row[6]:.8f} | {row[7]:.8f} |",
        )
    print(f"\nMedian speedup: {statistics.median(row[4] for row in records):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in records):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in records):.3f}x")
    print(f"Maximum PR error: {max(row[6] for row in records):.8f}")
    print(f"Maximum PR-main difference: {max(row[7] for row in records):.8f}")


if __name__ == "__main__":
    main()
