# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark four-code BNB decode against current main."""

import argparse
import gc
import statistics
import time
from functools import lru_cache, partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import (
    MlxBitsAndBytesLinear,
    _four_bit_kernel,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_bnb_output_dtype import _CODEBOOKS


@lru_cache(maxsize=1)
def _main_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_bnb_4bit_matmul_main_benchmark",
        input_names=["x", "weight", "scales", "codebook", "bias"],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            uint row_base = threadgroup_position_in_grid.z * RTILE;
            threadgroup float partials[RTILE * GROUPS];
            float sums[RTILE];
            for (uint r = 0; r < RTILE; ++r) sums[r] = 0.0f;
            uint weight_offset = column * K;
            if (EVEN) {
                for (uint k = lane * 2; k < K; k += THREADS * 2) {
                    uint index = weight_offset + k;
                    uchar packed = weight[index >> 1];
                    float scale = scales[index / BLOCK];
                    float first = float(half(codebook[uint(packed >> 4)] * scale));
                    float second = float(half(codebook[uint(packed & 15)] * scale));
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                        uint input_offset = (row_base + r) * K + k;
                        sums[r] = metal::fma(float(x[input_offset]), first, sums[r]);
                        sums[r] = metal::fma(float(x[input_offset + 1]), second, sums[r]);
                    }
                }
            } else {
                for (uint k = lane; k < K; k += THREADS) {
                    uint index = weight_offset + k;
                    uchar packed = weight[index >> 1];
                    uint code = (index & 1) ? uint(packed & 15) : uint(packed >> 4);
                    float value = float(half(codebook[code] * scales[index / BLOCK]));
                    for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r)
                        sums[r] = metal::fma(float(x[(row_base + r) * K + k]), value, sums[r]);
                }
            }
            for (uint r = 0; r < RTILE; ++r) {
                float reduced = simd_sum(sums[r]);
                if ((lane & 31) == 0) partials[r * GROUPS + (lane >> 5)] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RTILE && row_base + r < ROWS; ++r) {
                float reduced = lane < GROUPS ? partials[r * GROUPS + lane] : 0.0f;
                reduced = simd_sum(reduced);
                if (lane == 0) output[(row_base + r) * N + column] = reduced + bias[column];
            }
        """,
    )


def _fixture(format_name, out_features, in_features):
    codebook = _CODEBOOKS[format_name]
    codes = np.arange(in_features, dtype=np.uint8) & np.uint8(15)
    packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
    packed = np.tile(packed_row, out_features)
    scale = np.float32(0.01875)
    scales = np.full(out_features * in_features // 64, scale, dtype=np.float32)
    bias = np.linspace(-0.002, 0.002, out_features, dtype=np.float32).astype(np.float16)
    layer = MlxBitsAndBytesLinear(
        packed, scales, in_features=in_features, out_features=out_features,
        bits=4, block_size=64, codebook=codebook, bias=bias,
    )
    return layer, (codebook[codes] * scale).astype(np.float64), bias


def _threads(layer, x, rows):
    small = rows == 1 and layer.out_features <= 2048 and layer.in_features >= 4096
    if small and x.dtype == mx.float16:
        return 128
    if small and x.dtype == mx.bfloat16:
        return 256
    return 32


def _decode(layer, x, kernel):
    rows = x.size // layer.in_features
    row_tile = 1 if rows == 1 else min(8, rows)
    threads = _threads(layer, x, rows)
    return kernel()(
        inputs=[x, layer.weight, layer.scales, layer.codebook, layer.bias],
        template=[
            ("K", layer.in_features), ("N", layer.out_features),
            ("BLOCK", layer.block_size), ("ROWS", rows),
            ("RTILE", row_tile), ("THREADS", threads),
            ("EVEN", True), ("GROUPS", threads // 32),
        ],
        grid=(threads, layer.out_features, (rows + row_tile - 1) // row_tile),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, layer.out_features)],
        output_dtypes=[mx.float32],
    )[0]


def _visible(layer, x, kernel):
    return _decode(layer, x, kernel).astype(x.dtype)


def _measure_pair(main_call, candidate_call, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_call(), candidate_call())
    timings = [[], []]
    calls = (main_call, candidate_call)
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = time.perf_counter_ns()
            mx.eval(calls[path]())
            timings[path].append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(timings[0]), statistics.median(timings[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 16))
    args = parser.parse_args()
    print(
        "projection | rows | format | dtype | main ms | candidate ms | speedup | "
        "changed | FP32 arithmetic max abs | visible/rounded max abs | "
        "visible/raw max abs",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        for format_name in ("nf4", "fp4"):
            layer, row_weight, bias = _fixture(format_name, out_features, in_features)
            positions = np.arange(in_features, dtype=np.float32)
            for rows in args.rows:
                source = np.stack([
                    np.sin(positions * (0.013 + row * 0.0001)) * 0.08
                    + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
                    for row in range(rows)
                ])
                for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                    x = mx.array(source).astype(dtype)
                    mx.eval(x)
                    main_output = _visible(layer, x, _main_kernel)
                    candidate_internal = _decode(layer, x, _four_bit_kernel)
                    candidate_output = candidate_internal.astype(dtype)
                    mx.eval(main_output, candidate_internal, candidate_output)
                    inputs = np.asarray(x.astype(mx.float32)).astype(np.float64)
                    raw_row = np.sum(inputs * row_weight[None, :], axis=1)
                    raw = raw_row[:, None] + bias.astype(np.float64)[None, :]
                    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
                    rounded = torch.from_numpy(raw).to(target).float().numpy()
                    main_values = np.asarray(main_output.astype(mx.float32))
                    internal_values = np.asarray(candidate_internal)
                    candidate_values = np.asarray(candidate_output.astype(mx.float32))
                    changed = int(np.count_nonzero(candidate_values != main_values))
                    arithmetic_error = float(np.max(np.abs(internal_values - raw)))
                    rounded_error = float(np.max(np.abs(candidate_values - rounded)))
                    raw_error = float(np.max(np.abs(candidate_values - raw)))
                    allowed = 2e-3 + 2e-3 * np.abs(raw)
                    if np.any(np.abs(candidate_values - raw) > allowed):
                        raise AssertionError(
                            f"{name} rows={rows} {format_name} {label}: drift exceeds tolerance",
                        )

                    main_call = partial(_visible, layer, x, _main_kernel)
                    candidate_call = partial(layer, x)
                    main_ms, candidate_ms = _measure_pair(
                        main_call, candidate_call, args.warmups, args.samples,
                    )
                    print(
                        f"{name} | {rows} | {format_name.upper()} | {label} | "
                        f"{main_ms:.4f} | {candidate_ms:.4f} | "
                        f"{main_ms / candidate_ms:.3f}x | {changed} | "
                        f"{arithmetic_error:.9g} | {rounded_error:.9g} | {raw_error:.9g}",
                        flush=True,
                    )
            del layer
            gc.collect()
            mx.clear_cache()


if __name__ == "__main__":
    main()
