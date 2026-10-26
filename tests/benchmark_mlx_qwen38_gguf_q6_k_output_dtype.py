# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Compare main and target-dtype GGUF Q6_K decode on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFQ6KLinear, _q6_k_kernel
from gptqmodel.utils.mlx_packing import _pack_rows
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _fixture(out_features, in_features, patterns=2):
    positions = np.arange(in_features, dtype=np.uint32)
    phases = np.arange(patterns, dtype=np.uint32)[:, None]
    codes = ((positions[None, :] * (phases * 2 + 5) + phases * 11 + 3) & 63).astype(np.uint8)
    groups = np.arange(in_features // 16, dtype=np.float32)[None, :]
    scales = (0.0007 + ((groups + phases) % 13) * 0.000025).astype(np.float32)
    biases = (-31.5 * scales + ((groups + phases) % 3 - 1) * 0.0001).astype(np.float32)
    output_pattern = np.arange(out_features) % patterns

    layer = MlxGGUFQ6KLinear(in_features, out_features)
    layer.weight = mx.array(_pack_rows(codes, 6)[output_pattern])
    layer.scales_even = mx.array(scales[:, ::2][output_pattern])
    layer.scales_odd = mx.array(scales[:, 1::2][output_pattern])
    layer.biases_even = mx.array(biases[:, ::2][output_pattern])
    layer.biases_odd = mx.array(biases[:, 1::2][output_pattern])
    return layer


def _main_forward(layer, x):
    """Reproduce main's FP32 output allocation followed by an activation cast."""
    output_dims = layer.weight.shape[0]
    if layer.input_dims >= 8192 or output_dims <= 2048:
        threads = 128
    elif output_dims >= 16384:
        threads = 32
    else:
        threads = 64
    output = _q6_k_kernel(mx.float32)(
        inputs=[
            x, layer.weight, layer.scales_even, layer.scales_odd,
            layer.biases_even, layer.biases_odd, layer.zero_bias[0],
        ],
        template=[
            ("K", layer.input_dims), ("N", output_dims), ("ROWS", 1),
            ("RTILE", 1), ("THREADS", threads), ("GROUPS", threads // 32),
        ],
        grid=(threads, output_dims, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(1, output_dims)], output_dtypes=[mx.float32],
    )[0]
    return output.astype(x.dtype)


def _measure_pair(main_fn, pr_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), pr_fn())
    timings = ([], [])
    functions = (main_fn, pr_fn)
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter_ns()
            mx.eval(functions[index]())
            timings[index].append((time.perf_counter_ns() - start) / 1e6)
    return tuple(statistics.median(values) for values in timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=501)
    args = parser.parse_args()
    results = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        layer = _fixture(out_features, in_features)
        positions = np.arange(in_features, dtype=np.float32)
        source = (np.sin(positions * 0.013) * 0.08)[None]
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            x = mx.array(source).astype(dtype)
            main_fn = partial(_main_forward, layer, x)
            pr_fn = partial(layer, x)
            main_ms, pr_ms = _measure_pair(main_fn, pr_fn, args.warmups, args.samples)
            main_output, pr_output = main_fn(), pr_fn()
            mx.eval(main_output, pr_output)
            difference = float(np.max(np.abs(
                np.asarray(main_output.astype(mx.float32))
                - np.asarray(pr_output.astype(mx.float32))
            )))
            speedup = main_ms / pr_ms
            results.append((name, dtype_name, main_ms, pr_ms, speedup, difference))
            print(
                f"{name} {dtype_name}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                f"({speedup:.3f}x), PR-main {difference:.8f}", flush=True,
            )
        gc.collect()
        mx.clear_cache()

    print("\n| Projection | Dtype | Main ms | PR ms | Speedup | PR-main |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]:.8f} |",
        )
    for dtype_name in ("FP16", "BF16"):
        values = [row[4] for row in results if row[1] == dtype_name]
        print(f"{dtype_name} median speedup: {statistics.median(values):.3f}x")
    print(f"Overall median speedup: {statistics.median(row[4] for row in results):.3f}x")
    print(f"Maximum PR-main difference: {max(row[5] for row in results):.8f}")


if __name__ == "__main__":
    main()
