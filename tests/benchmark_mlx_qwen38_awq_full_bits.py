# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark all native MLX AWQ widths on Qwen3.8-27B projection shapes."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _pack_constant(code, bits, count):
    words = np.zeros(bits, dtype=np.uint32)
    for index in range(32):
        bit = index * bits
        words[bit // 32] |= np.uint32((code << (bit % 32)) & 0xFFFFFFFF)
        if bit % 32 + bits > 32:
            words[bit // 32 + 1] |= np.uint32(code >> (32 - bit % 32))
    return np.tile(words, count // 32)


def _measure(function, warmups, repeats):
    for _ in range(warmups):
        mx.eval(function())
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        mx.eval(function())
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(timings)


def run(dtype_name, rows, warmups, repeats):
    dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_name]
    rng = np.random.default_rng(380027 + rows)
    scale_value = float(np.float16(0.001))
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        source = rng.normal(0, 0.01, (rows, in_features)).astype(np.float16)
        x = mx.array(source).astype(dtype)
        dense_weight = mx.full((out_features, in_features), scale_value, dtype=dtype)
        mx.eval(x, dense_weight)

        def dense_call(value=x, weight=dense_weight):
            return value @ weight.T

        dense_ms = _measure(dense_call, warmups, repeats)
        expected = dense_call()
        mx.eval(expected)
        expected_visible = np.asarray(expected.astype(mx.float32))
        for bits in (2, 3, 4, 5, 6, 7, 8):
            storage_bits = 8 if bits == 7 else bits
            code, zero = (1 << bits) - 1, (1 << bits) - 2
            packed_row = _pack_constant(code, storage_bits, in_features)
            packed = np.broadcast_to(packed_row, (out_features, packed_row.size)).copy()
            layer = nn.QuantizedLinear(
                in_features, out_features, bias=False,
                group_size=128, bits=storage_bits,
            )
            layer.load_weights([
                ("weight", mx.array(packed)),
                ("scales", mx.full((out_features, in_features // 128), scale_value, dtype=mx.float16)),
                ("biases", mx.full((out_features, in_features // 128), -zero * scale_value, dtype=mx.float32)),
            ])

            def packed_call(value=x, quantized=layer):
                return quantized(value).astype(dtype)

            packed_ms = _measure(packed_call, warmups, repeats)
            actual = packed_call()
            mx.eval(actual)
            visible = np.asarray(actual.astype(mx.float32))
            delta = visible - expected_visible
            record = {
                "projection": name,
                "out_features": out_features,
                "in_features": in_features,
                "rows": rows,
                "dtype": dtype_name,
                "bits": bits,
                "runtime_bits": storage_bits,
                "main": "supported" if bits == 4 else "unsupported",
                "packed_ms": packed_ms,
                "dense_ms": dense_ms,
                "speed_vs_dense": dense_ms / packed_ms,
                "max_abs": float(np.max(np.abs(delta))),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected_visible), 1e-12)),
                "packed_bytes": int(packed.nbytes),
                "dense_bytes": int(out_features * in_features * 2),
            }
            print(json.dumps(record), flush=True)
            del layer, packed
            mx.clear_cache()
        del dense_weight
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--rows", type=int, choices=(1, 16), required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    run(args.dtype, args.rows, args.warmups, args.repeats)
