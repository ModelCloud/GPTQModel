# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark QQQ dynamic-quant reduction threads on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.mlx_qqq import (
    MlxQQQLinear,
    _dynamic_quant,
    _dynamic_quant_kernel,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _fixture(group_size, out_features, in_features):
    shape = (out_features, in_features // 128)
    if group_size == -1:
        bits = 4
        packed_word = 0x76543210
        weight = mx.full((out_features, in_features // 8), packed_word, dtype=mx.uint32)
        scales = mx.full(shape, 0.002, dtype=mx.float32)
        biases = mx.full(shape, -0.007, dtype=mx.float32)
        row_weight = (
            (np.arange(in_features, dtype=np.int32) & 7) * np.float32(0.002)
            - np.float32(0.007)
        )
    else:
        bits = 8
        packed_word = 0x03020100
        weight = mx.full((out_features, in_features // 4), packed_word, dtype=mx.uint32)
        scales = mx.full(shape, 0.001, dtype=mx.float32)
        biases = mx.full(shape, -0.0015, dtype=mx.float32)
        row_weight = (
            (np.arange(in_features, dtype=np.int32) & 3) * np.float32(0.001)
            - np.float32(0.0015)
        )
    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=128, bits=bits,
    )
    linear.load_weights([
        ("weight", weight), ("scales", scales), ("biases", biases),
    ])
    channel_scale = np.linspace(
        0.008, 0.012, out_features, dtype=np.float32,
    )[None, :]
    return MlxQQQLinear(linear, channel_scale), row_weight, channel_scale


def _dynamic_quant_256(x):
    width = x.shape[-1]
    rows = x.size // width
    return _dynamic_quant_kernel(256)(
        inputs=[x],
        template=[("K", width)],
        grid=(256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (*x.shape[:-1], 1)],
        output_dtypes=[mx.float32, mx.float32],
    )


def _forward(layer, x, dynamic_quant):
    original_dtype = x.dtype
    quantized, input_scale = dynamic_quant(x.astype(mx.float16))
    internal = layer.linear(quantized) * input_scale * layer.channel_scale
    return internal.astype(mx.float16).astype(original_dtype), internal, quantized, input_scale


def _measure_pair(main_fn, candidate_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), candidate_fn())
    timings = ([], [])
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            mx.eval((main_fn, candidate_fn)[path]())
            timings[path].append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(timings[0]), statistics.median(timings[1])


def run(rows_values, warmups, samples):
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        for group_size in (-1, 128):
            layer, row_weight, channel_scale = _fixture(
                group_size, out_features, in_features,
            )
            for rows in rows_values:
                source = np.stack([
                    np.sin(positions * (0.013 + row * 0.0001)) * 0.05
                    + np.cos(positions * (0.007 + row * 0.0001)) * 0.025
                    for row in range(rows)
                ])
                for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                    x = mx.array(source).astype(dtype)
                    mx.eval(x)

                    def main_fn(value=x, qqq_layer=layer):
                        return _forward(qqq_layer, value, _dynamic_quant_256)[0]

                    def candidate_fn(value=x, qqq_layer=layer):
                        return qqq_layer(value)

                    main_ms, candidate_ms = _measure_pair(
                        main_fn, candidate_fn, warmups, samples,
                    )
                    main_output, _, main_codes, main_scale = _forward(
                        layer, x, _dynamic_quant_256,
                    )
                    candidate_output, internal, codes, input_scale = _forward(
                        layer, x, _dynamic_quant,
                    )
                    mx.eval(
                        main_output, candidate_output, internal, main_codes,
                        main_scale, codes, input_scale,
                    )

                    half = x.astype(mx.float16)
                    torch_input = torch.from_numpy(np.asarray(half)).double()
                    torch_scale = (
                        torch_input.abs().amax(dim=-1, keepdim=True).half() / 127
                    ).double()
                    torch_codes = (
                        torch_input / torch_scale
                    ).round().clamp(-128, 127)
                    raw_row = (
                        torch_codes @ torch.from_numpy(row_weight.astype(np.float64))
                    )[:, None] * torch_scale
                    raw = raw_row.numpy() * channel_scale.astype(np.float64)
                    rounded = torch.from_numpy(raw).to(torch.float16)
                    if dtype == mx.bfloat16:
                        rounded = rounded.to(torch.bfloat16)
                    rounded = rounded.float().numpy()

                    main_visible = np.asarray(main_output.astype(mx.float32))
                    candidate_visible = np.asarray(candidate_output.astype(mx.float32))
                    assert candidate_output.dtype == dtype
                    np.testing.assert_array_equal(np.asarray(codes), torch_codes.numpy())
                    np.testing.assert_array_equal(np.asarray(input_scale), torch_scale.numpy())
                    np.testing.assert_array_equal(candidate_visible, main_visible)
                    np.testing.assert_allclose(
                        np.asarray(internal), raw, rtol=2e-3, atol=2e-3,
                    )
                    np.testing.assert_allclose(
                        candidate_visible, rounded, rtol=2e-3, atol=2e-3,
                    )
                    optimized = in_features == 17408
                    records.append((
                        name, rows, group_size, dtype_name,
                        "threads512" if optimized else "unchanged",
                        main_ms, candidate_ms, main_ms / candidate_ms,
                        int(np.count_nonzero(candidate_visible != main_visible)),
                        int(np.count_nonzero(np.asarray(codes) != torch_codes.numpy())),
                        float(np.max(np.abs(np.asarray(internal) - raw))),
                        float(np.max(np.abs(candidate_visible - rounded))),
                        float(np.max(np.abs(candidate_visible - raw))),
                    ))
                    row = records[-1]
                    print(
                        f"{name} rows={rows} group={group_size} {dtype_name}: "
                        f"{main_ms:.4f} -> {candidate_ms:.4f} ms "
                        f"({row[7]:.3f}x, {row[4]}), changed {row[8]}, "
                        f"code mismatches {row[9]}, errors "
                        f"{row[10]:.8g}/{row[11]:.8g}/{row[12]:.8g}",
                        flush=True,
                    )
            del layer
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Rows | Group | Dtype | Path | Main ms | PR ms | Speedup | "
        "Changed | Code mismatches | FP32/raw max abs | Visible/rounded max abs | "
        "Visible/raw max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]:.4f} | {row[6]:.4f} | {row[7]:.3f}x | {row[8]} | "
            f"{row[9]} | {row[10]:.8g} | {row[11]:.8g} | {row[12]:.8g} |"
        )
    changed = [row for row in records if row[4] == "threads512"]
    print(f"\n512-thread minimum speedup: {min(row[7] for row in changed):.3f}x")
    print(f"512-thread maximum speedup: {max(row[7] for row in changed):.3f}x")
    print(f"Maximum code mismatches: {max(row[9] for row in records)}")
    print(f"Maximum FP32/raw error: {max(row[10] for row in records):.8g}")
    print(f"Maximum visible/rounded error: {max(row[11] for row in records):.8g}")
    print(f"Maximum visible/raw error: {max(row[12] for row in records):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 16))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.rows, args.warmups, args.samples)
