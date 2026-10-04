# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ arithmetic reference: vLLM contributors, Apache-2.0, https://github.com/vllm-project/vllm
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark fused QQQ decode finalization against the current MLX path."""

import argparse
import gc
import json
from statistics import geometric_mean, median
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx import nn

from gptqmodel.nn_modules.qlinear.mlx_qqq import MlxQQQLinear, _dynamic_quant
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _main_decode(layer, x):
    """Reproduce main's separate QQQ scaling, casts, and bias addition."""
    half = x.astype(mx.float16)
    quantized, input_scale = _dynamic_quant(half)
    output = layer.linear(quantized) * input_scale * layer.channel_scale
    output = output.astype(mx.float16)
    if "bias" in layer:
        output = output + layer.bias
    return output.astype(x.dtype)


def _measure_pair(main_fn, pr_fn, samples):
    for _ in range(5):
        mx.eval(main_fn(), pr_fn())
    timings = [[], []]
    for sample in range(samples):
        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, pr_fn)[index]())
            timings[index].append((perf_counter() - start) * 1_000)
    return median(timings[0]), median(timings[1])


def _layer(out_features, in_features, bits):
    if bits == 4:
        packed = np.full(
            (out_features, in_features // 8), np.uint32(0x76543210),
        )
        weight_scale = 16.0
    else:
        packed = np.full(
            (out_features, in_features // 4), np.uint32(0x03020100),
        )
        weight_scale = 1.0
    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=128, bits=bits,
    )
    linear.load_weights([
        ("weight", mx.array(packed)),
        ("scales", mx.full(
            (out_features, in_features // 128), weight_scale,
            dtype=mx.float32,
        )),
        ("biases", mx.full(
            (out_features, in_features // 128), -128.0,
            dtype=mx.float32,
        )),
    ])
    channel_scale = np.linspace(
        0.0008, 0.0012, out_features, dtype=np.float32,
    )[None, :]
    bias = (np.sin(np.arange(out_features) * 0.1) * 0.002).astype(np.float16)
    return MlxQQQLinear(linear, channel_scale, bias)


def run(samples):
    mx.random.seed(3233)
    results = []
    for bits in (4, 8):
        for dtype in (mx.float16, mx.bfloat16):
            for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
                layer = _layer(out_features, in_features, bits)
                x = (mx.random.normal((1, in_features)) * 0.05).astype(dtype)
                main_output = _main_decode(layer, x)
                pr_output = layer(x)
                mx.eval(main_output, pr_output)
                difference = mx.abs(
                    main_output.astype(mx.float32) - pr_output.astype(mx.float32),
                )
                main_ms, pr_ms = _measure_pair(
                    lambda layer=layer, x=x: _main_decode(layer, x),
                    lambda layer=layer, x=x: layer(x),
                    samples,
                )
                result = {
                    "bits": bits,
                    "dtype": str(dtype),
                    "projection": name,
                    "main_ms": main_ms,
                    "pr_ms": pr_ms,
                    "speedup": main_ms / pr_ms,
                    "max_abs_vs_main": float(mx.max(difference).item()),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                del layer, x, main_output, pr_output
                mx.clear_cache()
                gc.collect()

    for bits in (4, 8):
        for dtype in ("mlx.core.float16", "mlx.core.bfloat16"):
            group = [
                row for row in results
                if row["bits"] == bits and row["dtype"] == dtype
            ]
            print(json.dumps({
                "summary": True,
                "bits": bits,
                "dtype": dtype,
                "main_median_ms": median(row["main_ms"] for row in group),
                "pr_median_ms": median(row["pr_ms"] for row in group),
                "geomean_speedup": geometric_mean(
                    row["speedup"] for row in group
                ),
                "min_speedup": min(row["speedup"] for row in group),
                "max_speedup": max(row["speedup"] for row in group),
                "max_abs_vs_main": max(
                    row["max_abs_vs_main"] for row in group
                ),
            }), flush=True)
    print(json.dumps({
        "summary": "overall",
        "cases": len(results),
        "geomean_speedup": geometric_mean(row["speedup"] for row in results),
        "min_speedup": min(row["speedup"] for row in results),
        "max_speedup": max(row["speedup"] for row in results),
        "max_abs_vs_main": max(row["max_abs_vs_main"] for row in results),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    run(parser.parse_args().samples)
