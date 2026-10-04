# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq
# Marlin format: IST-DASLab contributors, MIT, https://github.com/IST-DASLab/marlin
# BitBLAS format: Microsoft Research contributors, Apache-2.0, https://github.com/microsoft/BitBLAS
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark AWQ format transfer against main's packed AWQ GEMM runtime."""

import argparse
import gc
import json
from statistics import geometric_mean, median
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch
from mlx import nn

from gptqmodel.nn_modules.qlinear.mlx import (
    AWQBitBLASMlxQuantLinear,
    AWQMarlinMlxQuantLinear,
)
from gptqmodel.nn_modules.qlinear.mlx_awq import MlxAWQLinear
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


FORMAT_CASES = (
    ("marlin", 4, AWQMarlinMlxQuantLinear),
    ("marlin", 8, AWQMarlinMlxQuantLinear),
    *(("bitblas", bits, AWQBitBLASMlxQuantLinear) for bits in range(2, 9)),
)


def _repeated_word(code, bits):
    value = sum(int(code) << shift for shift in range(0, 32, bits))
    return np.asarray(value, dtype=np.uint32).view(np.int32).item()


def _pack_bytes(codes, bits):
    code_bits = (
        codes.astype(np.uint16)[..., None]
        >> np.arange(bits, dtype=np.uint16)
    ) & 1
    return np.packbits(
        code_bits.reshape(*codes.shape[:-1], -1), axis=-1, bitorder="little",
    ).view(np.int8)


def _source(fmt, bits, source_cls, in_features, out_features):
    source = source_cls(
        bits=bits,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        register_buffers=True,
    )
    low = (1 << (bits - 1)) - 1
    high = low + 2
    zero = 1 << (bits - 1)
    if fmt == "marlin":
        source.qweight[0::2].fill_(_repeated_word(low, bits))
        source.qweight[1::2].fill_(_repeated_word(high, bits))
        source.qzeros.fill_(_repeated_word(zero, bits))
        source.scales.fill_(0.002)
    else:
        weight_row = np.resize(
            np.asarray((low, high), dtype=np.uint8), in_features,
        )[None, :]
        zero_row = np.full((1, out_features), zero, dtype=np.uint8)
        source.qweight.copy_(
            torch.from_numpy(np.broadcast_to(
                _pack_bytes(weight_row, bits), source.qweight.shape,
            ).copy()),
        )
        source.qzeros.copy_(
            torch.from_numpy(np.broadcast_to(
                _pack_bytes(zero_row, bits), source.qzeros.shape,
            ).copy()),
        )
        source.scales.fill_(0.002)
    source.bias.copy_(
        torch.linspace(-0.01, 0.01, out_features, dtype=torch.float16),
    )
    return source


def _runtime(source_cls, source):
    weight, scales, biases, params = source_cls.pack_source(source)
    linear = nn.QuantizedLinear(
        source.in_features,
        source.out_features,
        bias=True,
        group_size=params["group_size"],
        bits=params["bits"],
    )
    linear.load_weights([
        ("weight", mx.array(weight)),
        ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
        ("bias", mx.array(source.bias.numpy())),
    ])
    return MlxAWQLinear(linear)


def _measure_pair(main_fn, format_fn, samples):
    for _ in range(5):
        mx.eval(main_fn(), format_fn())
    timings = [[], []]
    for sample in range(samples):
        for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, format_fn)[index]())
            timings[index].append((perf_counter() - start) * 1_000)
    return median(timings[0]), median(timings[1])


def _rounded_oracle(x, bias, in_features, dtype):
    values = torch.tensor((-0.002, 0.002), dtype=torch.float64).repeat(
        in_features // 2,
    )
    projection = torch.from_numpy(
        np.asarray(x.astype(mx.float32)),
    ).double() @ values
    output = projection[:, None] + bias.double()[None, :]
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return output.to(target).float().numpy()


def run(samples, summary_only=False):
    mx.random.seed(3234)
    results = []
    for fmt, bits, source_cls in FORMAT_CASES:
        for dtype in (mx.float16, mx.bfloat16):
            for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
                source = _source(fmt, bits, source_cls, in_features, out_features)
                format_layer = _runtime(source_cls, source)
                # Main supports this exact packed MLX runtime through AWQ GEMM,
                # but cannot transfer MARLIN/BITBLAS AWQ checkpoint layouts.
                main_layer = _runtime(source_cls, source)
                x = (mx.random.normal((1, in_features)) * 0.01).astype(dtype)
                main_output = main_layer(x)
                format_output = format_layer(x)
                mx.eval(main_output, format_output)
                if main_output.dtype != dtype or format_output.dtype != dtype:
                    raise AssertionError("AWQ MLX benchmark output dtype changed")
                difference = mx.abs(
                    main_output.astype(mx.float32)
                    - format_output.astype(mx.float32),
                )
                expected = _rounded_oracle(x, source.bias, in_features, dtype)
                oracle_difference = np.abs(
                    np.asarray(format_output.astype(mx.float32)) - expected,
                )
                main_ms, format_ms = _measure_pair(
                    lambda layer=main_layer, x=x: layer(x),
                    lambda layer=format_layer, x=x: layer(x),
                    samples,
                )
                result = {
                    "format": fmt,
                    "bits": bits,
                    "dtype": str(dtype),
                    "projection": name,
                    "main_awq_gemm_ms": main_ms,
                    "format_ms": format_ms,
                    "runtime_ratio": main_ms / format_ms,
                    "max_abs_vs_main_runtime": float(mx.max(difference).item()),
                    "max_abs_vs_rounded_torch": float(oracle_difference.max()),
                }
                results.append(result)
                if not summary_only:
                    print(json.dumps(result), flush=True)
                del source, main_layer, format_layer, x, main_output, format_output
                mx.clear_cache()
                gc.collect()

    for fmt, bits, _source_cls in FORMAT_CASES:
        for dtype in ("mlx.core.float16", "mlx.core.bfloat16"):
            group = [
                row for row in results
                if row["format"] == fmt
                and row["bits"] == bits
                and row["dtype"] == dtype
            ]
            print(json.dumps({
                "summary": True,
                "format": fmt,
                "bits": bits,
                "dtype": dtype,
                "main_awq_gemm_median_ms": median(
                    row["main_awq_gemm_ms"] for row in group
                ),
                "format_median_ms": median(row["format_ms"] for row in group),
                "geomean_runtime_ratio": geometric_mean(
                    row["runtime_ratio"] for row in group
                ),
                "min_runtime_ratio": min(row["runtime_ratio"] for row in group),
                "max_runtime_ratio": max(row["runtime_ratio"] for row in group),
                "max_abs_vs_main_runtime": max(
                    row["max_abs_vs_main_runtime"] for row in group
                ),
                "max_abs_vs_rounded_torch": max(
                    row["max_abs_vs_rounded_torch"] for row in group
                ),
            }), flush=True)
    print(json.dumps({
        "summary": "overall",
        "cases": len(results),
        "geomean_runtime_ratio": geometric_mean(
            row["runtime_ratio"] for row in results
        ),
        "min_runtime_ratio": min(row["runtime_ratio"] for row in results),
        "max_runtime_ratio": max(row["runtime_ratio"] for row in results),
        "max_abs_vs_main_runtime": max(
            row["max_abs_vs_main_runtime"] for row in results
        ),
        "max_abs_vs_rounded_torch": max(
            row["max_abs_vs_rounded_torch"] for row in results
        ),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    run(args.samples, args.summary_only)
