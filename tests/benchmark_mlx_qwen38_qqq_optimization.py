# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ arithmetic reference: vLLM, Apache-2.0, https://github.com/vllm-project/vllm
# MLX packed matmul: Apple Inc., MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
"""Compare the merged QQQ MLX path with the optimized path on Qwen shapes."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_qqq import MlxQQQLinear
from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _paired_milliseconds(main_fn, optimized_fn, repeats):
    for _ in range(3):
        mx.eval(main_fn(), optimized_fn())
    samples = ([], [])
    for iteration in range(repeats):
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            started = time.perf_counter()
            mx.eval((main_fn if index == 0 else optimized_fn)())
            samples[index].append((time.perf_counter() - started) * 1000)
    return statistics.median(samples[0]), statistics.median(samples[1])


def benchmark(name, output_dims, input_dims, rows, group_size, repeats):
    phase = np.arange(output_dims, dtype=np.uint8)[:, None] % 16
    index = np.arange(input_dims, dtype=np.uint8)[None, :] % 16
    codes = ((index + phase) % 16).astype(np.uint8)
    signed = codes.astype(np.int16) - 8
    grouped = group_size == 128
    old_codes = (signed * (1 if grouped else 16) + 128).astype(np.uint8)
    old_packed = np.ascontiguousarray(old_codes).view(np.uint32).reshape(output_dims, input_dims // 4)
    old = nn.QuantizedLinear(input_dims, output_dims, bias=False, group_size=128, bits=8)
    shape = (output_dims, input_dims // 128)
    old.load_weights([
        ("weight", mx.array(old_packed)),
        ("scales", mx.ones(shape, dtype=mx.float32)),
        ("biases", mx.full(shape, -128, dtype=mx.float32)),
    ])

    if grouped:
        optimized = old
    else:
        lanes = codes.reshape(output_dims, input_dims // 8, 8)
        packed = np.zeros((output_dims, input_dims // 8), dtype=np.uint32)
        for lane in range(8):
            packed |= lanes[:, :, lane].astype(np.uint32) << (4 * lane)
        optimized = nn.QuantizedLinear(input_dims, output_dims, bias=False, group_size=128, bits=4)
        optimized.load_weights([
            ("weight", mx.array(packed)),
            ("scales", mx.full(shape, 16, dtype=mx.float32)),
            ("biases", mx.full(shape, -128, dtype=mx.float32)),
        ])

    rng = np.random.default_rng(380027 + rows)
    x = mx.array(rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16))
    channel_scale = mx.full((1, output_dims), 0.001, dtype=mx.float32)
    new_layer = MlxQQQLinear(optimized, channel_scale)

    def main_path():
        half = x.astype(mx.float16)
        scale = (mx.max(mx.abs(half), axis=-1, keepdims=True) / 127).astype(mx.float32)
        quantized = mx.clip(mx.round(half.astype(mx.float32) / scale), -128, 127)
        return (old(quantized) * scale * channel_scale).astype(mx.float16)

    main_result, new_result = main_path(), new_layer(x)
    mx.eval(main_result, new_result)
    absolute_error = np.abs(np.asarray(main_result).astype(np.float32)
                            - np.asarray(new_result).astype(np.float32))
    np.testing.assert_allclose(np.asarray(new_result), np.asarray(main_result),
                               rtol=0.002, atol=0.002, err_msg=name)
    main_ms, new_ms = _paired_milliseconds(main_path, lambda: new_layer(x), repeats)
    result = {
        "projection": name, "out": output_dims, "in": input_dims,
        "rows": rows, "group_size": group_size,
        "main_mlx_ms": round(main_ms, 3), "optimized_mlx_ms": round(new_ms, 3),
        "speedup": round(main_ms / new_ms, 2),
        "max_abs_error": float(absolute_error.max()),
        "packed_weight_bytes_main": int(old_packed.nbytes),
        "packed_weight_bytes_optimized": int(old_packed.nbytes if grouped else packed.nbytes),
    }
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--group-size", type=int, nargs="+", default=[-1, 128])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for group_size in args.group_size:
            for rows in args.rows:
                print(json.dumps(benchmark(name, output_dims, input_dims, rows,
                                           group_size, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
