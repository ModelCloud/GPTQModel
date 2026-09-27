# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 encoding oracle: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark main's dense E5M2 fallback against packed MLX inference."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _time(module, inputs):
    started = time.perf_counter()
    mx.eval(module(inputs))
    return (time.perf_counter() - started) * 1000


def _payload(out_features, in_features):
    codebook = torch.arange(256, dtype=torch.uint8).view(torch.float8_e5m2).float().numpy()
    columns = np.arange(in_features, dtype=np.uint32)[None, :]
    rows = np.arange(out_features, dtype=np.uint32)[:, None]
    magnitude = (columns * 13 + rows * 17) % 120
    signs = ((columns + rows) & 1) << 7
    weight = (magnitude | signs).astype(np.uint8)
    scales = (600000 + np.arange(out_features) % 10000).astype(np.float32)
    return weight, scales, codebook


def _oracle(inputs, scales, codebook, out_features):
    in_features = inputs.shape[1]
    columns = np.arange(in_features, dtype=np.uint32)[None, :]
    phases = np.arange(240, dtype=np.uint32)[:, None]
    magnitude = (columns * 13 + phases * 17) % 120
    signs = ((columns + phases) & 1) << 7
    templates = codebook[(magnitude | signs).astype(np.uint8)].astype(np.float64)
    phase = np.arange(out_features) % 240
    with np.errstate(all="ignore"):
        return (inputs.astype(np.float64) @ templates.T)[:, phase] / scales


def benchmark(name, out_features, in_features, repeats, dtype_name):
    weight, scales, codebook = _payload(out_features, in_features)
    packed = MlxFP8PackedLinear(
        weight, scales, codebook,
        in_features=in_features, out_features=out_features, scale_method="row",
    )

    # This reproduces origin/main: dequantize_weight(..., dtype=torch.float16)
    # casts E5M2 inverse scales above FP16_MAX to infinity before division.
    with np.errstate(over="ignore"):
        main_weight = (
            codebook[weight].astype(np.float16)
            / scales[:, None].astype(np.float16)
        )
    dense = nn.Linear(in_features, out_features, bias=False)
    dense.weight = mx.array(main_weight)
    mlx_dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
    rng = np.random.default_rng(380027 + out_features + in_features)
    records = []
    for rows in (1, 16):
        inputs = mx.array(
            rng.normal(0, 0.01, (rows, in_features)).astype(np.float32),
        ).astype(mlx_dtype)
        mx.eval(inputs, packed.weight, dense.weight)
        for _ in range(7):
            mx.eval(packed(inputs), dense(inputs))
        packed_times, main_times = [], []
        for index in range(repeats):
            if index % 2:
                packed_times.append(_time(packed, inputs))
                main_times.append(_time(dense, inputs))
            else:
                main_times.append(_time(dense, inputs))
                packed_times.append(_time(packed, inputs))
        packed_ms = statistics.median(packed_times)
        main_ms = statistics.median(main_times)
        actual = np.asarray(packed(inputs).astype(mx.float32)).astype(np.float64)
        baseline = np.asarray(dense(inputs).astype(mx.float32)).astype(np.float64)
        expected = _oracle(
            np.asarray(inputs.astype(mx.float32)), scales, codebook, out_features,
        )
        packed_residual = actual - expected
        main_residual = baseline - expected
        records.append({
            "projection": name,
            "out": out_features,
            "in": in_features,
            "rows": rows,
            "dtype": dtype_name,
            "main_dense_ms": round(main_ms, 4),
            "packed_ms": round(packed_ms, 4),
            "speedup": round(main_ms / packed_ms, 3),
            "main_weight_mib": round(dense.weight.nbytes / (1024 ** 2), 3),
            "packed_weight_mib": round(packed.weight.nbytes / (1024 ** 2), 3),
            "packed_max_abs": float(np.max(np.abs(packed_residual))),
            "packed_relative_l2": float(
                np.linalg.norm(packed_residual) / np.linalg.norm(expected)
            ),
            "main_max_abs": float(np.max(np.abs(main_residual))),
            "main_relative_l2": float(
                np.linalg.norm(main_residual) / np.linalg.norm(expected)
            ),
        })
    del packed, dense, weight, main_weight
    mx.clear_cache()
    gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for record in benchmark(name, out_features, in_features, args.repeats, args.dtype):
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
