# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark packed GPTQ inference on Qwen3.8-27B projection shapes."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQLinear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _time(module, inputs):
    started = time.perf_counter()
    mx.eval(module(inputs))
    return (time.perf_counter() - started) * 1000


def benchmark(name, out_features, in_features, bits, repeats, dtype_name):
    group_size = 128
    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=group_size, bits=bits,
    )
    codes_per_word = 32 // bits
    shifts = np.arange(codes_per_word, dtype=np.uint32) * bits
    codes = np.resize(np.array([0, 2], dtype=np.uint32), in_features)
    packed_row = np.bitwise_or.reduce(
        codes.reshape(-1, codes_per_word) << shifts,
        axis=-1,
    )
    linear.weight = mx.array(
        np.broadcast_to(packed_row, (out_features, packed_row.size)).copy(),
    )
    linear.scales = mx.full(
        (out_features, in_features // group_size), 0.002, dtype=mx.float32,
    )
    linear.biases = mx.full(
        (out_features, in_features // group_size), -0.002, dtype=mx.float32,
    )
    packed = MlxGPTQLinear(linear)

    dense = nn.Linear(in_features, out_features, bias=False)
    pattern = mx.array(np.tile(np.array([-0.002, 0.002], dtype=np.float16), in_features // 2))
    dense.weight = mx.broadcast_to(pattern, (out_features, in_features))
    mlx_dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
    rng = np.random.default_rng(380027 + out_features + in_features)
    records = []
    for rows in (1, 16):
        inputs = mx.array(
            rng.normal(0, 0.01, (rows, in_features)).astype(np.float32),
        ).astype(mlx_dtype)
        mx.eval(inputs, packed.linear.weight, dense.weight)
        for _ in range(7):
            mx.eval(packed(inputs), dense(inputs))
        packed_times, dense_times = [], []
        for index in range(repeats):
            if index % 2:
                packed_times.append(_time(packed, inputs))
                dense_times.append(_time(dense, inputs))
            else:
                dense_times.append(_time(dense, inputs))
                packed_times.append(_time(packed, inputs))
        packed_ms = statistics.median(packed_times)
        dense_ms = statistics.median(dense_times)
        actual = np.asarray(packed(inputs).astype(mx.float32)).astype(np.float64)
        reference = np.asarray(dense(inputs).astype(mx.float32)).astype(np.float64)
        residual = actual - reference
        records.append({
            "projection": name,
            "out": out_features,
            "in": in_features,
            "bits": bits,
            "rows": rows,
            "dtype": dtype_name,
            "main_format_runtime": "unsupported",
            "dense_reference_ms": round(dense_ms, 4),
            "packed_ms": round(packed_ms, 4),
            "speedup_vs_dense": round(dense_ms / packed_ms, 3),
            "dense_weight_mib": round(dense.weight.nbytes / (1024 ** 2), 3),
            "packed_weight_mib": round(packed.linear.weight.nbytes / (1024 ** 2), 3),
            "max_abs": float(np.max(np.abs(residual))),
            "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(reference)),
        })
    del packed, dense
    mx.clear_cache()
    gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), action="append")
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for bits in args.bits or (2, 4, 8):
            for record in benchmark(name, out_features, in_features, bits, args.repeats, args.dtype):
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
