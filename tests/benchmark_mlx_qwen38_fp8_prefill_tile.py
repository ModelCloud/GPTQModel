# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# FP8 encoding oracle: PyTorch contributors, BSD-3-Clause, https://github.com/pytorch/pytorch
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark the 32-column packed FP8 prefill tile against main's 16-column tile."""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_fp8 import MlxFP8PackedLinear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _time(call):
    started = time.perf_counter()
    mx.eval(call())
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


def _main_call(layer, inputs):
    rows = inputs.size // layer.in_features
    selected = (
        rows != 1
        and layer.in_features <= 8192
        and layer.out_features >= 8192
        and layer.in_features % 64 == 0
        and layer.out_features % 32 == 0
    )
    if not selected:
        return layer(inputs)
    output = layer._prefill(inputs, rows, tile_columns=16)
    return (output + layer.bias).reshape(*inputs.shape[:-1], layer.out_features).astype(inputs.dtype)


def benchmark(name, out_features, in_features, dtype_name, warmups, samples):
    weight, scales, codebook = _payload(out_features, in_features)
    layer = MlxFP8PackedLinear(
        weight, scales, codebook,
        in_features=in_features, out_features=out_features, scale_method="row",
    )
    mlx_dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
    inputs = mx.array(
        np.random.default_rng(380027 + out_features + in_features).normal(
            0, 0.01, (16, in_features),
        ).astype(np.float32),
    ).astype(mlx_dtype)
    selected = in_features <= 8192 and out_features >= 8192
    mx.eval(inputs, layer.weight, layer.scales, layer.codebook)
    for _ in range(warmups):
        mx.eval(_main_call(layer, inputs), layer(inputs))
    main_times, candidate_times = [], []
    for index in range(samples):
        if index % 2:
            candidate_times.append(_time(lambda: layer(inputs)))
            main_times.append(_time(lambda: _main_call(layer, inputs)))
        else:
            main_times.append(_time(lambda: _main_call(layer, inputs)))
            candidate_times.append(_time(lambda: layer(inputs)))
    main_output = _main_call(layer, inputs)
    candidate = layer(inputs)
    mx.eval(main_output, candidate)
    main_array = np.asarray(main_output.astype(mx.float32)).astype(np.float64)
    candidate_array = np.asarray(candidate.astype(mx.float32)).astype(np.float64)
    expected = _oracle(
        np.asarray(inputs.astype(mx.float32)), scales, codebook, out_features,
    )
    main_ms = statistics.median(main_times)
    candidate_ms = statistics.median(candidate_times)
    record = {
        "projection": name,
        "shape": f"{out_features}x{in_features}",
        "dtype": dtype_name,
        "selected": selected,
        "main_ms": main_ms,
        "candidate_ms": candidate_ms,
        "speedup": main_ms / candidate_ms,
        "max_abs_vs_main": float(np.max(np.abs(candidate_array - main_array))),
        "max_abs_vs_fp64": float(np.max(np.abs(candidate_array - expected))),
    }
    mx.clear_cache()
    gc.collect()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    records = []
    for dtype_name in ("float16", "bfloat16"):
        for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
            record = benchmark(
                name, out_features, in_features, dtype_name, args.warmups, args.samples,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    selected = [record for record in records if record["selected"]]
    print(json.dumps({
        "selected_cases": len(selected),
        "geomean_speedup": float(np.exp(np.mean(np.log([
            record["speedup"] for record in selected
        ])))),
        "minimum_speedup": min(record["speedup"] for record in selected),
        "maximum_speedup": max(record["speedup"] for record in selected),
        "max_abs_vs_main": max(record["max_abs_vs_main"] for record in records),
        "selected_fp16_max_abs_vs_fp64": max(
            record["max_abs_vs_fp64"] for record in selected
            if record["dtype"] == "float16"
        ),
        "selected_bf16_max_abs_vs_fp64": max(
            record["max_abs_vs_fp64"] for record in selected
            if record["dtype"] == "bfloat16"
        ),
        "fp16_max_abs_vs_fp64": max(
            record["max_abs_vs_fp64"] for record in records
            if record["dtype"] == "float16"
        ),
        "bf16_max_abs_vs_fp64": max(
            record["max_abs_vs_fp64"] for record in records
            if record["dtype"] == "bfloat16"
        ),
    }), flush=True)


if __name__ == "__main__":
    main()
