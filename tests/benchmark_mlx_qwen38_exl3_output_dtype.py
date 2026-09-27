# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Compare packed EXL3 inference with main's decoded FP16 MLX fallback."""

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
            main = nn.Linear(in_features, out_features, bias=True)
            main.weight = (mx.random.normal((out_features, in_features)) * 0.015).astype(mx.float16)
            main.bias = (mx.random.normal((out_features,)) * 0.002).astype(mx.float16)
            packed = MlxEXL3Linear(in_features, out_features, 4, "mcg", bias=True)
            packed.trellis = mx.random.randint(
                -32768, 32767,
                (in_features // 16, out_features // 16, 64),
            ).astype(mx.int16)
            packed.suh = mx.where(mx.arange(in_features) % 3, 1, -1).astype(mx.float16)
            packed.svh = mx.where(mx.arange(out_features) % 5, 1, -1).astype(mx.float16)
            packed.bias = main.bias
            for rows in (1, 16):
                x = (mx.random.normal((rows, in_features)) * 0.15).astype(dtype)
                mx.eval(x, main.weight, main.bias, packed.trellis, packed.suh, packed.svh)
                main_ms, packed_ms = measure_pair(
                    lambda: main(x).astype(dtype), lambda: packed(x), repeats,
                )
                dense_mib = out_features * in_features * 2 / (1 << 20)
                packed_mib = out_features * in_features * 4 / 8 / (1 << 20)
                print(json.dumps({
                    "dtype": str(dtype), "projection": name, "rows": rows,
                    "main_ms": main_ms, "packed_ms": packed_ms,
                    "speedup": main_ms / packed_ms,
                    "dense_weight_mib": dense_mib, "packed_weight_mib": packed_mib,
                    "weight_reduction": dense_mib / packed_mib,
                }), flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=21)
    run(parser.parse_args().repeats)
