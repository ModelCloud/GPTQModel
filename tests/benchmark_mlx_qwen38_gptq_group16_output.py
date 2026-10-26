# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ format and packing: ModelCloud.ai, Apache-2.0, https://github.com/ModelCloud/GPTQModel
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Compare main and target-dtype GPTQ group-16 decode on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import lru_cache, partial

import mlx.core as mx
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQGroup16Linear
from gptqmodel.utils.mlx_packing import _pack_rows
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_kernel():
    """Reproduce main's FP32 output buffer for paired measurements."""
    return mx.fast.metal_kernel(
        name="gptqmodel_gptq_group16_matmul_main_benchmark",
        input_names=[
            "x", "weight", "scales_even", "scales_odd",
            "biases_even", "biases_odd", "bias",
        ],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_threadgroup.x;
            uint column = threadgroup_position_in_grid.y;
            threadgroup float partials[8];
            float sum = 0.0f;
            uint weight_offset = column * (K * BITS / 32);

            for (uint group = lane; group < K / 16; group += THREADS) {
                uint start_bit = group * 16 * BITS;
                uint word_base = weight_offset + (start_bit >> 5);
                uint first_shift = start_bit & 31u;
                uint packed[4];
                for (uint word = 0; word < WORDS; ++word)
                    packed[word] = weight[word_base + word];

                uint scale_index = column * (K / 32) + (group >> 1);
                float scale = (group & 1u)
                    ? scales_odd[scale_index] : scales_even[scale_index];
                float offset = (group & 1u)
                    ? biases_odd[scale_index] : biases_even[scale_index];
                for (uint index = 0; index < 16; ++index) {
                    uint bit = first_shift + index * BITS;
                    uint word = bit >> 5;
                    uint shift = bit & 31u;
                    ulong window = ulong(packed[word]);
                    if (word + 1 < WORDS)
                        window |= ulong(packed[word + 1]) << 32;
                    uint code = uint((window >> shift) & MASK);
                    float value = metal::fma(float(code), scale, offset);
                    sum = metal::fma(float(x[group * 16 + index]), value, sum);
                }
            }

            float reduced = simd_sum(sum);
            if ((lane & 31u) == 0)
                partials[lane >> 5] = reduced;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduced = lane < GROUPS ? partials[lane] : 0.0f;
            reduced = simd_sum(reduced);
            if (lane == 0)
                output[column] = reduced + bias[column];
        """,
    )


def _fixture(out_features, in_features, source_bits, patterns=2):
    runtime_bits = 8 if source_bits == 7 else source_bits
    positions = np.arange(in_features, dtype=np.uint32)
    phases = np.arange(patterns, dtype=np.uint32)[:, None]
    codes = (
        positions[None, :] * (phases * 2 + 3) + phases * 7 + 1
    ) & ((1 << source_bits) - 1)
    codes = codes.astype(np.uint8)
    groups = np.arange(in_features // 16, dtype=np.float32)[None, :]
    scales = (0.0007 + ((groups + phases) % 11) * 0.00003).astype(np.float32)
    zeros = ((groups.astype(np.uint32) + phases) & ((1 << source_bits) - 1)).astype(np.float32)
    offsets = -zeros * scales
    output_pattern = np.arange(out_features) % patterns

    layer = MlxGPTQGroup16Linear(in_features, out_features, runtime_bits)
    layer.weight = mx.array(_pack_rows(codes, runtime_bits)[output_pattern])
    layer.scales_even = mx.array(scales[:, ::2][output_pattern])
    layer.scales_odd = mx.array(scales[:, 1::2][output_pattern])
    layer.biases_even = mx.array(offsets[:, ::2][output_pattern])
    layer.biases_odd = mx.array(offsets[:, 1::2][output_pattern])
    return layer


def _main_forward(layer, x):
    output_dims = layer.weight.shape[0]
    threads = 128 if layer.input_dims >= 8192 or output_dims >= 16384 else 64
    output = _main_kernel()(
        inputs=[
            x, layer.weight, layer.scales_even, layer.scales_odd,
            layer.biases_even, layer.biases_odd, layer.zero_bias[0],
        ],
        template=[
            ("K", layer.input_dims), ("BITS", layer.bits),
            ("MASK", (1 << layer.bits) - 1), ("WORDS", (layer.bits + 1) // 2),
            ("THREADS", threads), ("GROUPS", threads // 32),
        ],
        grid=(threads, output_dims, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(1, output_dims)], output_dtypes=[mx.float32],
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
        source = (np.sin(positions * 0.013) * 0.08)[None]
        for source_bits in (2, 3, 4, 5, 6, 7, 8):
            layer = _fixture(out_features, in_features, source_bits)
            for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                x = mx.array(source).astype(dtype)
                main_fn = partial(_main_forward, layer, x)
                pr_fn = partial(layer, x)
                main_ms, pr_ms = _measure_pair(main_fn, pr_fn, args.warmups, args.samples)
                main_output, pr_output = main_fn(), pr_fn()
                mx.eval(main_output, pr_output)
                difference = float(np.max(np.abs(
                    np.asarray(main_output.astype(mx.float32))
                    - np.asarray(pr_output.astype(mx.float32))
                )))
                speedup = main_ms / pr_ms
                results.append((name, source_bits, dtype_name, main_ms, pr_ms, speedup, difference))
                print(
                    f"{name} {source_bits}-bit {dtype_name}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                    f"({speedup:.3f}x), PR-main {difference:.8f}", flush=True,
                )
            gc.collect()
            mx.clear_cache()

    print("\n| Projection | Bits | Dtype | Main ms | PR ms | Speedup | PR-main |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {row[4]:.4f} | "
            f"{row[5]:.3f}x | {row[6]:.8f} |",
        )
    for dtype_name in ("FP16", "BF16"):
        values = [row[5] for row in results if row[2] == dtype_name]
        print(f"{dtype_name} median speedup: {statistics.median(values):.3f}x")
    print(f"Overall median speedup: {statistics.median(row[5] for row in results):.3f}x")
    print(f"Maximum PR-main difference: {max(row[6] for row in results):.8f}")


if __name__ == "__main__":
    main()
