# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark direct-dtype BitsAndBytes 4-bit output on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import (
    MlxBitsAndBytesLinear,
    _four_bit_kernel,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_bnb_output_dtype import _CODEBOOKS


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


def _main_internal(layer, x):
    rows = x.size // layer.in_features
    row_tile = 1 if rows == 1 else min(8, rows)
    threads = _threads(layer, x, rows)
    return _four_bit_kernel(mx.float32)(
        inputs=[x, layer.weight, layer.scales, layer.codebook, layer.bias],
        template=[
            ("K", layer.in_features), ("N", layer.out_features),
            ("BLOCK", layer.block_size), ("ROWS", rows),
            ("RTILE", row_tile), ("THREADS", threads),
            ("EVEN", layer.in_features % 2 == 0), ("GROUPS", threads // 32),
        ],
        grid=(threads, layer.out_features, (rows + row_tile - 1) // row_tile),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, layer.out_features)], output_dtypes=[mx.float32],
    )[0]


def _main(layer, x):
    return _main_internal(layer, x).reshape(*x.shape[:-1], layer.out_features).astype(x.dtype)


def _measure_pair(main_fn, candidate_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), candidate_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = time.perf_counter_ns()
            mx.eval((main_fn, candidate_fn)[path]())
            timings[path].append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(timings[0]), statistics.median(timings[1])


def run(rows_values, warmups, samples):
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        for format_name in ("nf4", "fp4"):
            layer, row_weight, bias = _fixture(format_name, out_features, in_features)
            for rows in rows_values:
                source = np.stack([
                    np.sin(positions * (0.013 + row * 0.0001)) * 0.08
                    + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
                    for row in range(rows)
                ])
                for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                    x = mx.array(source).astype(dtype)
                    mx.eval(x)

                    def main_fn(value=x, bnb_layer=layer):
                        return _main(bnb_layer, value)

                    def candidate_fn(value=x, bnb_layer=layer):
                        return bnb_layer(value)

                    main_ms, candidate_ms = _measure_pair(
                        main_fn, candidate_fn, warmups, samples,
                    )
                    main_output = main_fn()
                    candidate_output = candidate_fn()
                    internal = _main_internal(layer, x)
                    mx.eval(main_output, candidate_output, internal)
                    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
                    raw_row = (
                        torch_input @ torch.from_numpy(row_weight).double()
                    ).numpy()
                    raw = raw_row[:, None] + bias.astype(np.float64)[None, :]
                    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
                    rounded = torch.from_numpy(raw).to(torch_dtype).float().numpy()
                    main_visible = np.asarray(main_output.astype(mx.float32))
                    candidate_visible = np.asarray(candidate_output.astype(mx.float32))
                    direct = out_features <= 2048
                    records.append((
                        name, rows, format_name.upper(), dtype_name,
                        "direct" if direct else "unchanged",
                        main_ms, candidate_ms, main_ms / candidate_ms,
                        int(np.count_nonzero(candidate_visible != main_visible)),
                        float(np.max(np.abs(np.asarray(internal) - raw))),
                        float(np.max(np.abs(candidate_visible - rounded))),
                        float(np.max(np.abs(candidate_visible - raw))),
                    ))
                    record = records[-1]
                    print(
                        f"{name} rows={rows} {format_name.upper()} {dtype_name}: "
                        f"{main_ms:.4f} -> {candidate_ms:.4f} ms "
                        f"({record[7]:.3f}x, {record[4]}), changed {record[8]}, errors "
                        f"{record[9]:.8g}/{record[10]:.8g}/{record[11]:.8g}",
                        flush=True,
                    )
            del layer
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Rows | Format | Dtype | Path | Main ms | PR ms | Speedup | "
        "Changed | FP32/raw max abs | Visible/rounded max abs | Visible/raw max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]:.4f} | {row[6]:.4f} | {row[7]:.3f}x | {row[8]} | "
            f"{row[9]:.8g} | {row[10]:.8g} | {row[11]:.8g} |"
        )
    changed = [row for row in records if row[4] == "direct"]
    print(f"\nDirect minimum speedup: {min(row[7] for row in changed):.3f}x")
    print(f"Direct maximum speedup: {max(row[7] for row in changed):.3f}x")
    print(f"Maximum FP32/raw error: {max(row[9] for row in records):.8g}")
    print(f"Maximum visible/rounded error: {max(row[10] for row in records):.8g}")
    print(f"Maximum visible/raw error: {max(row[11] for row in records):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 16))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.rows, args.warmups, args.samples)
