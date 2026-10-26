# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 E4M3 reference: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
"""Paired Metal FP8 benchmark: origin/main dense transfer vs native MXFP8."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx import FP8MlxQuantLinear
from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8Linear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_fp8_native import make_source, torch_oracle


def time_once(module, inputs):
    started = time.perf_counter()
    mx.eval(module(inputs))
    return (time.perf_counter() - started) * 1000


def benchmark(name, output_dims, input_dims, repeats, dtype):
    source = make_source(output_dims, input_dims)
    packed, scales, output_scale, _ = FP8MlxQuantLinear.pack_source(source)
    native = MlxFP8Linear(input_dims, output_dims, output_scale, source.bias.numpy())
    native.linear.load_weights([("weight", mx.array(packed)), ("scales", mx.array(scales))])

    # This is the origin/main load path: decode each FP8 weight to FP16 once.
    dense = nn.Linear(input_dims, output_dims, bias=True)
    dense.load_weights([
        ("weight", mx.array(source.dequantize_weight(device="cpu", dtype=torch.float16).T.numpy())),
        ("bias", mx.array(source.bias.numpy())),
    ])
    rng = np.random.default_rng(380027 + output_dims + input_dims)
    mlx_dtype = mx.float16 if dtype == "float16" else mx.bfloat16
    inputs = {rows: mx.array(rng.normal(0, 0.05, (rows, input_dims)).astype(np.float16)).astype(mlx_dtype)
              for rows in (1, 16)}
    mx.eval(*inputs.values(), dense.weight, native.linear.weight)
    result = []
    for rows, x in inputs.items():
        for _ in range(5):
            mx.eval(dense(x), native(x))
        baseline_times, native_times = [], []
        for index in range(repeats):
            if index % 2:
                native_times.append(time_once(native, x))
                baseline_times.append(time_once(dense, x))
            else:
                baseline_times.append(time_once(dense, x))
                native_times.append(time_once(native, x))
        dense_ms, native_ms = statistics.median(baseline_times), statistics.median(native_times)
        metric = {"projection": name, "out": output_dims, "in": input_dims,
                  "rows": rows, "input_dtype": dtype,
                  "main_output_dtype": str(dense(x).dtype).split(".")[-1],
                  "native_output_dtype": str(native(x).dtype).split(".")[-1],
                  "main_dense_ms": round(dense_ms, 4),
                  "native_mxfp8_ms": round(native_ms, 4),
                  "speedup": round(dense_ms / native_ms, 3)}
        oracle_tensor = torch_oracle(source, torch.from_numpy(np.asarray(x.astype(mx.float32))))
        oracle = oracle_tensor.numpy()
        actual = np.asarray(native(x).astype(mx.float32)).astype(np.float64)
        baseline = np.asarray(dense(x).astype(mx.float32)).astype(np.float64)
        internal = native._unscaled_dot(x) * native.output_scale + native.bias
        mx.eval(internal)
        residual = actual - oracle
        metric.update({
            "max_abs": float(np.max(np.abs(residual))),
            "relative_l2": float(np.linalg.norm(residual) / np.linalg.norm(oracle)),
            "max_tolerance_ratio": float(np.max(np.abs(residual) / (0.002 + 0.002 * np.abs(oracle)))),
            "max_internal_abs": float(np.max(np.abs(np.asarray(internal).astype(np.float64) - oracle))),
            "main_max_abs": float(np.max(np.abs(baseline - oracle))),
            "max_abs_vs_main": float(np.max(np.abs(actual - baseline))),
        })
        if dtype == "float16" and metric["max_tolerance_ratio"] > 1:
            raise AssertionError(f"{name} rows={rows}: FP8 inference exceeds the 2e-3 tolerance")
        if dtype == "bfloat16":
            rounded = oracle_tensor.to(torch.bfloat16)
            rounded_np = rounded.float().numpy()
            upper = torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).float().numpy()
            lower = torch.nextafter(rounded, torch.full_like(rounded, float("-inf"))).float().numpy()
            ulp = np.maximum(upper - rounded_np, rounded_np - lower)
            allowed = np.maximum(ulp, 0.002 + 0.002 * np.abs(rounded_np))
            metric["bf16_rounding_violations"] = int(np.count_nonzero(np.abs(actual - rounded_np) > allowed))
            if metric["bf16_rounding_violations"]:
                raise AssertionError(f"{name} rows={rows}: BF16 result exceeds one ULP or 2e-3")
        result.append(metric)
    del source, dense, native, inputs
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--projection", action="append")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    for name, output_dims, input_dims in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        for record in benchmark(name, output_dims, input_dims, args.repeats, args.dtype):
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
