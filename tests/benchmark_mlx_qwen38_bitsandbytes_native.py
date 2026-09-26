# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# MLX Metal kernel API: Apple Inc., MIT, https://github.com/ml-explore/mlx
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark packed BitsAndBytes kernels against main's decoded FP16 path."""

import argparse
import gc
import json
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import MlxBitsAndBytesLinear, repack_int8_affine
from gptqmodel.quantization.mlx_bitsandbytes import _FP4, _NF4
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def measure_pair(main_fn, native_fn, repeats):
    for _ in range(5):
        mx.eval(main_fn(), native_fn())
    samples = [[], []]
    for index in range(repeats):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for path in order:
            start = perf_counter()
            mx.eval((main_fn, native_fn)[path]())
            samples[path].append((perf_counter() - start) * 1000)
    return median(samples[0]), median(samples[1])


def weights(format_name, out_features, in_features):
    scale = np.float32(0.01875)
    if format_name in ("nf4", "fp4"):
        codebook = np.array(_NF4 if format_name == "nf4" else _FP4, dtype=np.float32)
        codes = np.arange(in_features, dtype=np.uint8) & np.uint8(15)
        packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
        packed = np.tile(packed_row, out_features)
        scales = np.full(out_features * in_features // 64, scale, dtype=np.float32)
        dense = np.tile((codebook[codes] * scale).astype(np.float16), (out_features, 1))
        return packed, scales, codebook, dense, 4
    codes = ((np.arange(in_features, dtype=np.int32) % 255) - 127).astype(np.int8)
    packed = np.tile(codes, out_features)
    scales = np.full(out_features, scale, dtype=np.float32)
    dense = np.tile((codes.astype(np.float32) * (scale / 127)).astype(np.float16), (out_features, 1))
    packed, scales, affine_biases, group_size = repack_int8_affine(
        packed, scales, in_features, out_features,
    )
    return packed, scales, affine_biases, dense, 8, group_size


def run(repeats, formats=("nf4", "fp4", "int8")):
    mx.random.seed(117)
    for format_name in formats:
        for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
            result = weights(format_name, out_features, in_features)
            if format_name == "int8":
                packed, scales, affine_biases, dense, bits, group_size = result
                codebook = None
            else:
                packed, scales, codebook, dense, bits = result
                affine_biases, group_size = None, 64
            bias = np.linspace(-0.002, 0.002, out_features, dtype=np.float32).astype(np.float16)
            main = nn.Linear(in_features, out_features, bias=True)
            main.weight = mx.array(dense)
            main.bias = mx.array(bias)
            native = MlxBitsAndBytesLinear(
                packed, scales, in_features=in_features, out_features=out_features,
                bits=bits, block_size=group_size, codebook=codebook,
                affine_biases=affine_biases, bias=bias,
            )
            del dense
            for dtype in (mx.float16, mx.bfloat16):
                for rows in (1, 16):
                    x = (mx.random.normal((rows, in_features)) * 0.15).astype(dtype)
                    mx.eval(x, main.weight, native.weight)
                    main_ms, native_ms = measure_pair(
                        lambda layer=main, value=x, output_dtype=dtype: layer(value).astype(output_dtype),
                        lambda layer=native, value=x: layer(value),
                        repeats,
                    )
                    print(json.dumps({
                        "format": format_name,
                        "dtype": str(dtype),
                        "projection": name,
                        "rows": rows,
                        "main_ms": main_ms,
                        "native_ms": native_ms,
                        "speedup": main_ms / native_ms,
                        "main_weight_bytes": out_features * in_features * 2,
                        "native_weight_bytes": native.weight.nbytes + native.scales.nbytes
                        + (native.affine_biases.nbytes if "affine_biases" in native else 0),
                    }), flush=True)
            del main, native, packed
            mx.clear_cache()
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--format", choices=("all", "nf4", "fp4", "int8"), default="all")
    args = parser.parse_args()
    run(args.repeats, ("nf4", "fp4", "int8") if args.format == "all" else (args.format,))
