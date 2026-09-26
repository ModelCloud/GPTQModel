# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark main's FP32-temporary Q1_0 packer against direct MLX input."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_gguf import gguf_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_q1_0_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q1_0_pack_main_benchmark",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float sums[8];
            for (uint lane = 0; lane < 8; ++lane) {
                sums[lane] = metal::abs(weights[block * 128 + lane]);
            }
            for (uint k = 8; k < 128; k += 8) {
                for (uint lane = 0; lane < 8; ++lane) {
                    sums[lane] += metal::abs(weights[block * 128 + k + lane]);
                }
            }
            float absolute_sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
            ushort scale_bits = as_type<ushort>(half(absolute_sum / 128.0f));
            uint offset = block * 18;
            packed[offset] = uchar(scale_bits & 255);
            packed[offset + 1] = uchar(scale_bits >> 8);
            for (uint byte = 0; byte < 16; ++byte) {
                uchar bits = 0;
                for (uint bit = 0; bit < 8; ++bit) {
                    bits |= uchar(weights[block * 128 + byte * 8 + bit] >= 0.0f)
                        << bit;
                }
                packed[offset + 2 + byte] = bits;
            }
        """,
    )


def _main_pack(weight):
    rows, columns = weight.shape
    blocks = rows * columns // 128
    return _main_q1_0_kernel()(
        inputs=[weight.astype(mx.float32)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // 128 * 18)],
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
        mx.random.seed(380001 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.5
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_pack(value)

            def direct_fn(value=weight):
                return gguf_quantize_weight_mlx(value, "Q1_0")

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
