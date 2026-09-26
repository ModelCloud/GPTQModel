# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Compare group-16 activation dtype preservation with the merged main path."""

import argparse
import json
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def main_forward(layer, x):
    """Exact group-16 forward from main before the output cast."""
    first_half = mx.arange(layer.input_dims) % 32 < 16
    even = mx.quantized_matmul(
        mx.where(first_half, x, 0), layer.weight, layer.scales_even,
        layer.biases_even, group_size=32, bits=layer.bits,
    )
    odd = mx.quantized_matmul(
        mx.where(first_half, 0, x), layer.weight, layer.scales_odd,
        layer.biases_odd, group_size=32, bits=layer.bits,
    )
    result = even + odd
    if "bias" in layer:
        result += layer.bias
    return result


def measure_pair(main_fn, new_fn, repeats):
    for _ in range(5):
        mx.eval(main_fn(), new_fn())
    samples = [[], []]
    for index in range(repeats):
        for path in ((0, 1) if index % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, new_fn)[path]())
            samples[path].append((perf_counter() - start) * 1000)
    return median(samples[0]), median(samples[1])


def run(repeats):
    mx.random.seed(42)
    for dtype in (mx.float16, mx.bfloat16):
        for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
            layer = MlxGroup16Linear(in_features, out_features, 4)
            layer.weight = mx.random.randint(0, 2**31, layer.weight.shape).astype(mx.uint32)
            layer.scales_even = mx.full(layer.scales_even.shape, 0.001, mx.float32)
            layer.scales_odd = mx.full(layer.scales_odd.shape, 0.001, mx.float32)
            layer.biases_even = mx.full(layer.biases_even.shape, -0.006, mx.float32)
            layer.biases_odd = mx.full(layer.biases_odd.shape, -0.006, mx.float32)
            for rows in (1, 16):
                x = (mx.random.normal((rows, in_features)) * 0.1).astype(dtype)
                mx.eval(x, layer.weight, layer.scales_even, layer.scales_odd,
                        layer.biases_even, layer.biases_odd)
                old, new = measure_pair(
                    lambda: main_forward(layer, x), lambda: layer(x), repeats,
                )
                print(json.dumps(dict(dtype=str(dtype), projection=name, rows=rows,
                                      main_ms=old, new_ms=new, speedup=old / new)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=21)
    run(parser.parse_args().repeats)
