# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Compare main and target-dtype BitsAndBytes kernels on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import (
    MlxBitsAndBytesLinear,
    _four_bit_kernel,
)
from gptqmodel.quantization.mlx_bitsandbytes import _FP4, _NF4
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _fixture(format_name, out_features, in_features):
    scale = np.float32(0.01875)
    codebook = np.array(_NF4 if format_name == "nf4" else _FP4, dtype=np.float32)
    codes = np.arange(in_features, dtype=np.uint8) & np.uint8(15)
    packed_row = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
    packed = np.tile(packed_row, out_features)
    scales = np.full(out_features * in_features // 64, scale, dtype=np.float32)
    row_weight = codebook[codes] * scale
    bias = np.linspace(-0.002, 0.002, out_features, dtype=np.float32).astype(np.float16)
    layer = MlxBitsAndBytesLinear(
        packed, scales, in_features=in_features, out_features=out_features,
        bits=4, block_size=64, codebook=codebook, bias=bias,
    )
    return layer, row_weight.astype(np.float64), bias


def _main_forward(layer, x):
    """Reproduce main's FP32 result allocation followed by an activation cast."""
    rows = x.size // layer.in_features
    row_tile = 1 if rows == 1 else min(8, rows)
    small_decode = (
        rows == 1 and layer.out_features <= 2048 and layer.in_features >= 4096
    )
    if small_decode and x.dtype == mx.float16:
        threads = 128
    elif small_decode and x.dtype == mx.bfloat16:
        threads = 256
    else:
        threads = 32
    output = _four_bit_kernel(mx.float32)(
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
    return output.astype(x.dtype)


def _measure_pair(main_fn, pr_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), pr_fn())
    timings = ([], [])
    functions = (main_fn, pr_fn)
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter_ns()
            mx.eval(functions[index]())
            timings[index].append((time.perf_counter_ns() - start) / 1e6)
    return tuple(statistics.median(values) for values in timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    results = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        for format_name in ("nf4", "fp4"):
            layer, row_weight, bias = _fixture(format_name, out_features, in_features)
            for rows in (1, 16):
                source = np.stack([
                    np.sin(positions * (0.013 + row * 0.0001)) * 0.08
                    for row in range(rows)
                ])
                for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                    x = mx.array(source).astype(dtype)
                    main_fn = partial(_main_forward, layer, x)
                    pr_fn = partial(layer, x)
                    main_ms, pr_ms = _measure_pair(
                        main_fn, pr_fn, args.warmups, args.samples,
                    )
                    main_output, pr_output = main_fn(), pr_fn()
                    mx.eval(main_output, pr_output)
                    input_values = torch.from_numpy(
                        np.asarray(x.astype(mx.float32)),
                    ).double()
                    raw_oracle = (
                        input_values @ torch.from_numpy(row_weight[:, None])
                    ).numpy() + bias.astype(np.float64)[None, :]
                    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
                    rounded = torch.from_numpy(raw_oracle).to(torch_dtype).float().numpy()
                    main_visible = np.asarray(main_output.astype(mx.float32))
                    pr_visible = np.asarray(pr_output.astype(mx.float32))
                    speedup = main_ms / pr_ms
                    results.append((
                        name, format_name.upper(), rows, dtype_name, main_ms, pr_ms,
                        speedup, float(np.max(np.abs(pr_visible - rounded))),
                        float(np.max(np.abs(pr_visible - main_visible))),
                    ))
                    print(
                        f"{name} {format_name.upper()} rows={rows} {dtype_name}: "
                        f"{main_ms:.4f} -> {pr_ms:.4f} ms ({speedup:.3f}x), "
                        f"PR error {results[-1][7]:.8g}, PR-main {results[-1][8]:.8g}",
                        flush=True,
                    )
            del layer
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Format | Rows | Dtype | Main ms | PR ms | Speedup | "
        "PR error | PR-main |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.4f} | "
            f"{row[5]:.4f} | {row[6]:.3f}x | {row[7]:.8g} | {row[8]:.8g} |",
        )
    for rows in (1, 16):
        values = [row[6] for row in results if row[2] == rows]
        print(f"Rows={rows} median speedup: {statistics.median(values):.3f}x")
    print(f"Overall median speedup: {statistics.median(row[6] for row in results):.3f}x")
    print(f"Maximum PR error: {max(row[7] for row in results):.8g}")
    print(f"Maximum PR-main difference: {max(row[8] for row in results):.8g}")


if __name__ == "__main__":
    main()
