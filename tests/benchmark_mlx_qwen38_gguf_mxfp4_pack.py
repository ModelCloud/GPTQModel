# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark main's signed MXFP4 search against magnitude-only search."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_gguf import gguf_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_mxfp4_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_mxfp4_pack_main_benchmark",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            uint maximum_bits = 0;
            for (uint k = 0; k < 32; ++k) {
                maximum_bits = metal::max(maximum_bits,
                    as_type<uint>(float(weights[block * 32 + k]))
                        & 0x7fffffffu);
            }
            int power = int(maximum_bits >> 23) - 127;
            if (maximum_bits == 0) {
                power = -125;
            } else if (maximum_bits < 0x00800000u) {
                uint leading = 31u - metal::clz(maximum_bits);
                power = int(leading) - 149;
                int next_power = power + 1;
                uint cutoff = next_power == -126 ? 22
                    : (next_power >= -128 ? 11
                    : (next_power == -129 ? 5
                    : (next_power == -130 ? 2
                    : (next_power == -131 ? 1 : 0))));
                uint next_bits = 1u << (leading + 1u);
                if (maximum_bits >= next_bits - cutoff) power = next_power;
            } else {
                int next_power = power + 1;
                uint cutoff = 0;
                if (next_power <= -64) cutoff = 44;
                else if (next_power <= -32) cutoff = 22;
                else if (next_power <= -16) cutoff = 11;
                else if (next_power <= -8) cutoff = 5;
                else if (next_power <= -4) cutoff = 2;
                else if (next_power <= -2) cutoff = 1;
                else if (next_power >= 65) cutoff = 44;
                else if (next_power >= 33) cutoff = 22;
                else if (next_power >= 17) cutoff = 11;
                else if (next_power >= 9) cutoff = 5;
                else if (next_power >= 5) cutoff = 2;
                else if (next_power >= 3) cutoff = 1;
                if (maximum_bits >= ((maximum_bits & 0x7f800000u)
                    + 0x00800000u - cutoff)) power = next_power;
            }
            uint exponent = uint(power + 125) & 255u;
            uint scale_bits = exponent < 2
                ? (0x00200000u << exponent) : ((exponent - 1) << 23);
            float scale = as_type<float>(scale_bits);
            float multiplier = exponent < 2 ? 0x1p126f : 1.0f;
            float comparison_scale = exponent < 2
                ? (exponent == 0 ? 0.25f : 0.5f) : scale;
            float fp4[16] = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
            };
            uchar codes[32];
            for (uint k = 0; k < 32; ++k) {
                float weight = float(weights[block * 32 + k]) * multiplier;
                uint best = 0;
                float best_error = metal::abs(weight);
                for (uint candidate = 1; candidate < 16; ++candidate) {
                    float error = metal::abs(
                        comparison_scale * fp4[candidate] - weight);
                    if (error < best_error) {
                        best_error = error;
                        best = candidate;
                    }
                }
                codes[k] = uchar(best);
            }
            uint offset = block * 17;
            packed[offset] = uchar(exponent);
            for (uint k = 0; k < 16; ++k) {
                packed[offset + 1 + k] = uchar(codes[k] | (codes[k + 16] << 4));
            }
        """,
    )


def _main_pack(weight):
    rows, columns = weight.shape
    blocks = rows * columns // 32
    return _main_mxfp4_kernel()(
        inputs=[mx.contiguous(weight)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // 32 * 17)],
        output_dtypes=[mx.uint8],
    )[0]


def _measure_pair(main_fn, direct_fn, samples):
    for _ in range(10):
        mx.eval(main_fn(), direct_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, direct_fn)[path]())
            timings[path].append((perf_counter() - start) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples):
    rows_out = []
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        mx.random.seed(380004 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.5
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_pack(value)

            def direct_fn(value=weight):
                return gguf_quantize_weight_mlx(value, "MXFP4")

            main, direct = main_fn(), direct_fn()
            mx.eval(main, direct)
            byte_error = int(
                mx.max(mx.abs(main.astype(mx.int16) - direct.astype(mx.int16))).item()
            )
            main_ms, direct_ms = _measure_pair(main_fn, direct_fn, samples)
            rows_out.append(
                (name, label, main_ms, direct_ms, main_ms / direct_ms, byte_error)
            )
            print(
                f"{name} {label}: {main_ms:.4f} -> {direct_ms:.4f} ms "
                f"({main_ms / direct_ms:.3f}x), byte error {byte_error}",
                flush=True,
            )
        del source, weight
        mx.clear_cache()
        gc.collect()

    print("\n| Projection | Dtype | Main ms | PR ms | Speedup | Maximum byte error |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Maximum byte error: {max(row[5] for row in rows_out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=101)
    run(parser.parse_args().samples)
