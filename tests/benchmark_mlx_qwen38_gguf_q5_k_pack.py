# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark main's serial Q5_K packer against parallel MLX block packing."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_gguf import gguf_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_q5_k_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q5_k_pack_main_benchmark",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float minima[8];
            float scales[8];
            float max_scale = 0.0f;
            float max_minimum = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                float low = weights[block * 256 + group * 32];
                float high = low;
                for (uint k = 1; k < 32; ++k) {
                    float value = weights[block * 256 + group * 32 + k];
                    low = metal::min(low, value);
                    high = metal::max(high, value);
                }
                float minimum = metal::max(-low, 0.0f);
                float scale = (high + minimum) / 31.0f;
                minima[group] = minimum;
                scales[group] = scale;
                max_scale = metal::max(max_scale, scale);
                max_minimum = metal::max(max_minimum, minimum);
            }
            float base = max_scale / 63.0f;
            float min_base = max_minimum / 63.0f;
            uint offset = block * 176;
            ushort d = as_type<ushort>(half(base));
            ushort dmin = as_type<ushort>(half(min_base));
            packed[offset] = uchar(d & 255);
            packed[offset + 1] = uchar(d >> 8);
            packed[offset + 2] = uchar(dmin & 255);
            packed[offset + 3] = uchar(dmin >> 8);
            uint scale_codes[8];
            uint min_codes[8];
            for (uint group = 0; group < 8; ++group) {
                scale_codes[group] = base > 0.0f
                    ? uint(metal::clamp(int(metal::rint(scales[group] / base)), 0, 63))
                    : 0;
                min_codes[group] = min_base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(minima[group] / min_base)), 0, 63))
                    : 0;
            }
            for (uint k = 0; k < 4; ++k) {
                packed[offset + 4 + k] = uchar(
                    (scale_codes[k] & 63) | ((scale_codes[k + 4] & 48) << 2));
                packed[offset + 8 + k] = uchar(
                    (min_codes[k] & 63) | ((min_codes[k + 4] & 48) << 2));
                packed[offset + 12 + k] = uchar(
                    (scale_codes[k + 4] & 15) | ((min_codes[k + 4] & 15) << 4));
            }
            for (uint k = 0; k < 32; ++k) packed[offset + 16 + k] = 0;
            for (uint group = 0; group < 8; group += 2) {
                float step0 = base * float(scale_codes[group]);
                float step1 = base * float(scale_codes[group + 1]);
                float bias0 = min_base * float(min_codes[group]);
                float bias1 = min_base * float(min_codes[group + 1]);
                for (uint k = 0; k < 32; ++k) {
                    float shifted0 = weights[block * 256 + group * 32 + k] + bias0;
                    float shifted1 = weights[
                        block * 256 + (group + 1) * 32 + k] + bias1;
                    uint code0 = step0 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted0 / step0)), 0, 31))
                        : 0;
                    uint code1 = step1 > 0.0f
                        ? uint(metal::clamp(int(metal::rint(shifted1 / step1)), 0, 31))
                        : 0;
                    packed[offset + 16 + k] |= uchar(
                        ((code0 >> 4) & 1) << group
                        | ((code1 >> 4) & 1) << (group + 1));
                    packed[offset + 48 + group * 16 + k] = uchar(
                        (code0 & 15) | ((code1 & 15) << 4));
                }
            }
        """,
    )


def _main_pack(weight):
    rows, columns = weight.shape
    blocks = rows * columns // 256
    return _main_q5_k_kernel()(
        inputs=[weight.astype(mx.float32)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // 256 * 176)],
        output_dtypes=[mx.uint8],
    )[0]


def _measure_pair(main_fn, parallel_fn, samples):
    for _ in range(10):
        mx.eval(main_fn(), parallel_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            mx.eval((main_fn, parallel_fn)[path]())
            timings[path].append((perf_counter() - start) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples):
    rows_out = []
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        mx.random.seed(380025 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.5
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_pack(value)

            def parallel_fn(value=weight):
                return gguf_quantize_weight_mlx(value, "Q5_K")

            main, parallel = main_fn(), parallel_fn()
            mx.eval(main, parallel)
            byte_error = int(mx.max(mx.abs(
                main.astype(mx.int16) - parallel.astype(mx.int16)
            )).item())
            main_ms, parallel_ms = _measure_pair(main_fn, parallel_fn, samples)
            rows_out.append((
                name, label, main_ms, parallel_ms, main_ms / parallel_ms, byte_error,
            ))
            print(
                f"{name} {label}: {main_ms:.4f} -> {parallel_ms:.4f} ms "
                f"({main_ms / parallel_ms:.3f}x), byte error {byte_error}",
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
