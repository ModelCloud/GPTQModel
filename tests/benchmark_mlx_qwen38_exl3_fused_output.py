# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark fused EXL3 decode output against the current packed MLX path."""

import argparse
import json
from statistics import geometric_mean, median
from time import perf_counter

import mlx.core as mx

from gptqmodel.nn_modules.qlinear.mlx_exl3 import (
    _CODEBOOK_IDS,
    _ROW_THREADS,
    MlxEXL3Linear,
    _exl3_row_matmul_kernel,
    _hadamard_128,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _main_row(layer, x):
    """Reproduce main's separate packed matmul and output transform."""
    transformed = _hadamard_128(x.astype(mx.float32) * layer.suh)
    inner = _exl3_row_matmul_kernel(
        layer.bits, _CODEBOOK_IDS[layer.codebook], False,
    )(
        inputs=[transformed, layer.trellis, layer.svh, layer.bias],
        template=[
            ("BITS", layer.bits),
            ("CODEBOOK", _CODEBOOK_IDS[layer.codebook]),
            ("FUSE_OUTPUT", False),
            ("PACKED_WORDS", 16 * layer.bits),
            ("THREADS", _ROW_THREADS),
            ("K", layer.in_features),
            ("N", layer.out_features),
            ("K_TILES", layer.in_features // 16),
            ("N_TILES", layer.out_features // 16),
            ("ROWS", 1),
        ],
        grid=(layer.out_features, 1, 1),
        threadgroup=(_ROW_THREADS, 1, 1),
        output_shapes=[(1, layer.out_features)],
        output_dtypes=[mx.float32],
    )[0]
    output = _hadamard_128(inner) * layer.svh
    if "bias" in layer:
        output = output + layer.bias
    return output.astype(x.dtype)


def _measure_pair(main_fn, pr_fn, samples):
    for _ in range(5):
        mx.eval(main_fn(), pr_fn())
    timings = [[], []]
    for sample in range(samples):
        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, pr_fn)[index]())
            timings[index].append((perf_counter() - start) * 1_000)
    return median(timings[0]), median(timings[1])


def run(samples):
    mx.random.seed(313)
    wide = [shape for shape in QWEN38_27B_PROJECTIONS if shape[1] > shape[2]]
    results = []
    for bits in range(1, 9):
        for dtype in (mx.float16, mx.bfloat16):
            for name, out_features, in_features in wide:
                layer = MlxEXL3Linear(in_features, out_features, bits, "mcg", bias=True)
                layer.trellis = mx.random.randint(
                    -32768, 32767,
                    (in_features // 16, out_features // 16, bits * 16),
                ).astype(mx.int16)
                layer.suh = mx.where(mx.arange(in_features) % 3, 1, -1).astype(mx.float16)
                layer.svh = mx.where(mx.arange(out_features) % 5, 1, -1).astype(mx.float16)
                layer.bias = (mx.random.normal((out_features,)) * 0.002).astype(mx.float16)
                x = (mx.random.normal((1, in_features)) * 0.15).astype(dtype)
                main_output = _main_row(layer, x)
                pr_output = layer(x)
                mx.eval(main_output, pr_output)
                difference = mx.abs(main_output.astype(mx.float32) - pr_output.astype(mx.float32))
                main_ms, pr_ms = _measure_pair(
                    lambda layer=layer, x=x: _main_row(layer, x),
                    lambda layer=layer, x=x: layer(x),
                    samples,
                )
                result = {
                    "bits": bits,
                    "dtype": str(dtype),
                    "projection": name,
                    "main_ms": main_ms,
                    "pr_ms": pr_ms,
                    "speedup": main_ms / pr_ms,
                    "max_abs_vs_main": float(mx.max(difference).item()),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                mx.clear_cache()

    for bits in range(1, 9):
        for dtype in ("mlx.core.float16", "mlx.core.bfloat16"):
            group = [row for row in results if row["bits"] == bits and row["dtype"] == dtype]
            print(json.dumps({
                "summary": True,
                "bits": bits,
                "dtype": dtype,
                "main_median_ms": median(row["main_ms"] for row in group),
                "pr_median_ms": median(row["pr_ms"] for row in group),
                "geomean_speedup": geometric_mean(row["speedup"] for row in group),
                "min_speedup": min(row["speedup"] for row in group),
                "max_speedup": max(row["speedup"] for row in group),
                "max_abs_vs_main": max(row["max_abs_vs_main"] for row in group),
            }), flush=True)
    print(json.dumps({
        "summary": "overall",
        "cases": len(results),
        "geomean_speedup": geometric_mean(row["speedup"] for row in results),
        "min_speedup": min(row["speedup"] for row in results),
        "max_speedup": max(row["speedup"] for row in results),
        "max_abs_vs_main": max(row["max_abs_vs_main"] for row in results),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    run(parser.parse_args().samples)
