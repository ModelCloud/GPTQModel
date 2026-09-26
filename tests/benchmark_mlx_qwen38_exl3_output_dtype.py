# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Measure EXL3 decoded dense inference with dtype preservation."""

import argparse
import json
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn

from gptqmodel.nn_modules.qlinear.mlx_exl3 import MlxEXL3Linear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def measure_pair(main_fn, new_fn, repeats):
    for _ in range(5):
        mx.eval(main_fn(), new_fn())
    samples = [[], []]
    for index in range(repeats):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for path in order:
            start = perf_counter()
            mx.eval((main_fn, new_fn)[path]())
            samples[path].append((perf_counter() - start) * 1000)
    return median(samples[0]), median(samples[1])


def run(repeats):
    mx.random.seed(173)
    for dtype in (mx.float16, mx.bfloat16):
        for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
            linear = nn.Linear(in_features, out_features, bias=True)
            linear.weight = (mx.random.normal((out_features, in_features)) * 0.015).astype(mx.float16)
            linear.bias = (mx.random.normal((out_features,)) * 0.002).astype(mx.float16)
            wrapped = MlxEXL3Linear(linear)
            for rows in (1, 16):
                x = (mx.random.normal((rows, in_features)) * 0.15).astype(dtype)
                mx.eval(x, linear.weight, linear.bias)
                main_ms, wrapped_ms = measure_pair(
                    lambda: linear(x), lambda: wrapped(x), repeats,
                )
                print(json.dumps({
                    "dtype": str(dtype), "projection": name, "rows": rows,
                    "main_ms": main_ms, "wrapped_ms": wrapped_ms,
                    "speedup": main_ms / wrapped_ms,
                }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=21)
    run(parser.parse_args().repeats)
