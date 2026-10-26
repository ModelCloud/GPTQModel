# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# bitsandbytes: Tim Dettmers et al., MIT, https://github.com/bitsandbytes-foundation/bitsandbytes
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark shape-selected BNB 4-bit decode threads against current main."""

import argparse
import gc
from statistics import median
from time import perf_counter_ns

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_bitsandbytes import MlxBitsAndBytesLinear, _four_bit_kernel
from gptqmodel.quantization.mlx_bitsandbytes import _FP4, _NF4
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _fixture(format_name, out_features, in_features):
    codebook = np.array(_NF4 if format_name == "nf4" else _FP4, dtype=np.float32)
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


def _main_call(layer, x):
    if x.shape[-1] != layer.in_features:
        raise ValueError(f"expected input width {layer.in_features}, got {x.shape[-1]}")
    output_shape = (*x.shape[:-1], layer.out_features)
    if x.size == 0:
        return mx.zeros(output_shape, dtype=x.dtype)
    rows = x.size // layer.in_features
    row_tile = 1 if rows == 1 else min(8, rows)
    output = _four_bit_kernel()(
        inputs=[x, layer.weight, layer.scales, layer.codebook, layer.bias],
        template=[
            ("K", layer.in_features), ("N", layer.out_features),
            ("BLOCK", layer.block_size), ("ROWS", rows),
            ("RTILE", row_tile), ("THREADS", 32),
            ("EVEN", layer.in_features % 2 == 0), ("GROUPS", 1),
        ],
        grid=(32, layer.out_features, (rows + row_tile - 1) // row_tile),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, layer.out_features)],
        output_dtypes=[mx.float32],
    )[0]
    return output.reshape(output_shape).astype(x.dtype)


def _measure_pair(main_fn, candidate_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), candidate_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter_ns()
            mx.eval((main_fn, candidate_fn)[path]())
            timings[path].append((perf_counter_ns() - started) / 1e6)
    return median(timings[0]), median(timings[1])


def run(warmups, samples):
    rows_out = []
    for index, (name, out_features, in_features) in enumerate(QWEN38_27B_PROJECTIONS):
        positions = np.arange(in_features, dtype=np.float32)
        source = (np.sin(positions * 0.013) * 0.08 + np.cos(positions * 0.007) * 0.04)[None]
        for format_name in ("nf4", "fp4"):
            layer, row_weight, bias = _fixture(format_name, out_features, in_features)
            for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                x = mx.array(source).astype(dtype)
                mx.eval(x)

                def main_fn(value=x, bnb_layer=layer):
                    return _main_call(bnb_layer, value)

                def candidate_fn(value=x, bnb_layer=layer):
                    return bnb_layer(value)

                main_ms, candidate_ms = _measure_pair(
                    main_fn, candidate_fn, warmups, samples,
                )
                main_output, candidate_output = main_fn(), candidate_fn()
                mx.eval(main_output, candidate_output)
                input_values = np.asarray(x.astype(mx.float32)).astype(np.float64)
                raw_oracle = np.sum(input_values * row_weight[None, :], axis=1)
                raw_oracle = raw_oracle[:, None] + bias.astype(np.float64)[None, :]
                torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
                expected = torch.from_numpy(raw_oracle).to(torch_dtype).float().numpy()
                main_visible = np.asarray(main_output.astype(mx.float32))
                candidate_visible = np.asarray(candidate_output.astype(mx.float32))
                main_error = float(np.max(np.abs(main_visible - expected)))
                candidate_error = float(np.max(np.abs(candidate_visible - expected)))
                candidate_main = float(np.max(np.abs(candidate_visible - main_visible)))
                rows_out.append((
                    name, format_name.upper(), dtype_name, main_ms, candidate_ms,
                    main_ms / candidate_ms, main_error, candidate_error, candidate_main,
                ))
                print(
                    f"{name} {format_name.upper()} {dtype_name}: "
                    f"{main_ms:.4f} -> {candidate_ms:.4f} ms "
                    f"({main_ms / candidate_ms:.3f}x), errors "
                    f"{main_error:.8g}/{candidate_error:.8g}/{candidate_main:.8g}",
                    flush=True,
                )
            del layer
            mx.clear_cache()
            gc.collect()

    print(
        "\n| Projection | Format | Dtype | Main ms | PR ms | Speedup | "
        "Main error | PR error | PR-main |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | "
            f"{row[4]:.4f} | {row[5]:.3f}x | {row[6]:.8g} | "
            f"{row[7]:.8g} | {row[8]:.8g} |",
        )
    print(f"\nMedian speedup: {median(row[5] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in rows_out):.3f}x")
    print(f"Maximum PR error: {max(row[7] for row in rows_out):.8g}")
    print(f"Maximum PR-main difference: {max(row[8] for row in rows_out):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.warmups, args.samples)
