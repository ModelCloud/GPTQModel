# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark merged and packed-kernel GGUF Q6_K paths on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFQ6KLinear
from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
from gptqmodel.utils.mlx_packing import _pack_rows
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def make_layers(out_features, in_features, patterns=4):
    positions = np.arange(in_features, dtype=np.uint32)
    phases = np.arange(patterns, dtype=np.uint32)[:, None]
    codes = ((positions[None, :] * (phases * 2 + 5) + phases * 11 + 3) & 63).astype(np.uint8)
    groups = np.arange(in_features // 16, dtype=np.float32)[None, :]
    scales = (0.0007 + ((groups + phases) % 13) * 0.000025).astype(np.float32)
    biases = (-31.5 * scales + ((groups + phases) % 3 - 1) * 0.0001).astype(np.float32)
    output_pattern = np.arange(out_features) % patterns

    baseline = MlxGroup16Linear(in_features, out_features, bits=6)
    candidate = MlxGGUFQ6KLinear(in_features, out_features)
    values = {
        "weight": mx.array(_pack_rows(codes, 6)[output_pattern]),
        "scales_even": mx.array(scales[:, ::2][output_pattern]),
        "scales_odd": mx.array(scales[:, 1::2][output_pattern]),
        "biases_even": mx.array(biases[:, ::2][output_pattern]),
        "biases_odd": mx.array(biases[:, 1::2][output_pattern]),
    }
    for layer in (baseline, candidate):
        for name, value in values.items():
            setattr(layer, name, value)
    reference = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 16, axis=1)
    reference += np.repeat(biases.astype(np.float64), 16, axis=1)
    return baseline, candidate, reference, output_pattern


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


def oracle(x, reference, output_pattern, dtype):
    source = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    raw = (source @ torch.from_numpy(reference).double().T)[:, torch.from_numpy(output_pattern)]
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return raw.to(target).float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    args = parser.parse_args()
    rows_out = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        baseline, candidate, reference, output_pattern = make_layers(out_features, in_features)
        positions = np.arange(in_features, dtype=np.float32)
        for rows in (1, 16):
            source = np.stack([
                np.sin(positions * (0.011 + row * 0.0001)) * 0.08
                + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
                for row in range(rows)
            ])
            for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                x = mx.array(source).astype(dtype)
                mx.eval(x)
                main_ms, pr_ms = measure_pair(
                    baseline, candidate, x, args.warmups, args.samples,
                )
                main_output, pr_output = baseline(x), candidate(x)
                mx.eval(main_output, pr_output)
                expected = oracle(x, reference, output_pattern, dtype)
                main_visible = np.asarray(main_output.astype(mx.float32))
                pr_visible = np.asarray(pr_output.astype(mx.float32))
                rows_out.append((
                    name, rows, label, main_ms, pr_ms, main_ms / pr_ms,
                    float(np.max(np.abs(main_visible - expected))),
                    float(np.max(np.abs(pr_visible - expected))),
                    float(np.max(np.abs(pr_visible - main_visible))),
                ))
                print(
                    f"{name} rows={rows} {label}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                    f"({main_ms / pr_ms:.3f}x), errors "
                    f"{rows_out[-1][6]:.8f}/{rows_out[-1][7]:.8f}/{rows_out[-1][8]:.8f}",
                    flush=True,
                )
        del baseline, candidate
        gc.collect()
        mx.clear_cache()

    print("\n| Projection | Rows | Dtype | Main ms | PR ms | Speedup | Main error | PR error | PR-main |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {row[4]:.4f} | "
            f"{row[5]:.3f}x | {row[6]:.8f} | {row[7]:.8f} | {row[8]:.8f} |",
        )
    decode = [row for row in rows_out if row[1] == 1]
    prefill = [row for row in rows_out if row[1] == 16]
    print(f"\nDecode median speedup: {statistics.median(row[5] for row in decode):.3f}x")
    print(f"Prefill median speedup: {statistics.median(row[5] for row in prefill):.3f}x")
    print(f"Maximum PR error: {max(row[7] for row in rows_out):.8f}")
    print(f"Maximum PR-main difference: {max(row[8] for row in rows_out):.8f}")


if __name__ == "__main__":
    main()
