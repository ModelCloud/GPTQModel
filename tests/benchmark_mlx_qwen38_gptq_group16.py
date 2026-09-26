# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
# MLX Metal runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark main and single-pass GPTQ group-16 decode on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_gptq_group16 import SOURCE_BITS, _group16_fixture, _rounded_oracle


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=101)
    args = parser.parse_args()
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        source = (np.sin(positions * 0.013) * 0.08 + np.cos(positions * 0.007) * 0.04)[None]
        for source_bits in SOURCE_BITS:
            candidate, reference, output_pattern, layer_bias = _group16_fixture(
                out_features, in_features, source_bits,
            )
            baseline = MlxGroup16Linear(
                in_features, out_features, bits=candidate.bits, bias=True,
            )
            for parameter in (
                "weight", "scales_even", "scales_odd", "biases_even", "biases_odd", "bias",
            ):
                setattr(baseline, parameter, getattr(candidate, parameter))
            for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                x = mx.array(source).astype(dtype)
                mx.eval(x)
                main_ms, pr_ms = measure_pair(
                    baseline, candidate, x, args.warmups, args.samples,
                )
                main_output, pr_output = baseline(x), candidate(x)
                mx.eval(main_output, pr_output)
                expected, _ = _rounded_oracle(
                    x, reference, output_pattern, layer_bias, dtype,
                )
                main_visible = np.asarray(main_output.astype(mx.float32))
                pr_visible = np.asarray(pr_output.astype(mx.float32))
                record = (
                    name, source_bits, dtype_name, main_ms, pr_ms, main_ms / pr_ms,
                    float(np.max(np.abs(main_visible - expected))),
                    float(np.max(np.abs(pr_visible - expected))),
                    float(np.max(np.abs(pr_visible - main_visible))),
                )
                records.append(record)
                print(
                    f"{name} {source_bits}-bit {dtype_name}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                    f"({main_ms / pr_ms:.3f}x), errors "
                    f"{record[6]:.8f}/{record[7]:.8f}/{record[8]:.8f}",
                    flush=True,
                )
            del baseline, candidate
            gc.collect()
            mx.clear_cache()

    print("\n| Projection | Bits | Dtype | Main ms | PR ms | Speedup | Main error | PR error | PR-main |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {row[4]:.4f} | "
            f"{row[5]:.3f}x | {row[6]:.8f} | {row[7]:.8f} | {row[8]:.8f} |",
        )
    print(f"\nMedian speedup: {statistics.median(row[5] for row in records):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in records):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in records):.3f}x")
    print(f"Maximum PR error: {max(row[7] for row in records):.8f}")
    print(f"Maximum PR-main difference: {max(row[8] for row in records):.8f}")


if __name__ == "__main__":
    main()
