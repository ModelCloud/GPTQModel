# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark main's serial Q6_K packer against parallel MLX block packing."""

import argparse
import gc
from functools import lru_cache
from statistics import median
from time import perf_counter

import mlx.core as mx

from gptqmodel.quantization.mlx_gguf import gguf_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_q6_k_kernel():
    return mx.fast.metal_kernel(
        name="gptqmodel_gguf_q6_k_pack_main_benchmark",
        input_names=["weights"],
        output_names=["packed"],
        source="""
            uint block = thread_position_in_grid.x;
            float scales[16];
            float largest_scale = 0.0f;
            for (uint group = 0; group < 16; ++group) {
                float maximum = 0.0f;
                for (uint k = 0; k < 16; ++k) {
                    maximum = metal::max(maximum,
                        metal::abs(weights[block * 256 + group * 16 + k]));
                }
                scales[group] = maximum / 31.0f;
                largest_scale = metal::max(largest_scale, scales[group]);
            }
            float base = largest_scale / 127.0f;
            uint offset = block * 210;
            uint scale_codes[16];
            for (uint group = 0; group < 16; ++group) {
                uint code = base > 0.0f
                    ? uint(metal::clamp(
                        int(metal::rint(scales[group] / base)), 0, 127))
                    : 0;
                scale_codes[group] = code;
                packed[offset + 192 + group] = uchar(code);
            }
            ushort scale_bits = as_type<ushort>(half(base));
            packed[offset + 208] = uchar(scale_bits & 255);
            packed[offset + 209] = uchar(scale_bits >> 8);
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint k = 0; k < 32; ++k) {
                    uint raw[4];
                    for (uint lane = 0; lane < 4; ++lane) {
                        uint index = segment * 128 + lane * 32 + k;
                        uint group = index / 16;
                        float step = base * float(scale_codes[group]);
                        int code = step > 0.0f
                            ? metal::clamp(int(metal::rint(
                                weights[block * 256 + index] / step)), -32, 31)
                            : 0;
                        raw[lane] = uint(code + 32);
                    }
                    packed[offset + segment * 64 + k] = uchar(
                        (raw[0] & 15) | ((raw[2] & 15) << 4));
                    packed[offset + segment * 64 + 32 + k] = uchar(
                        (raw[1] & 15) | ((raw[3] & 15) << 4));
                    packed[offset + 128 + segment * 32 + k] = uchar(
                        ((raw[0] >> 4) & 3) | (((raw[1] >> 4) & 3) << 2)
                        | (((raw[2] >> 4) & 3) << 4)
                        | (((raw[3] >> 4) & 3) << 6));
                }
            }
        """,
    )


def _main_pack(weight):
    rows, columns = weight.shape
    blocks = rows * columns // 256
    return _main_q6_k_kernel()(
        inputs=[weight.astype(mx.float32)],
        grid=(blocks, 1, 1),
        threadgroup=(min(blocks, 256), 1, 1),
        output_shapes=[(rows, columns // 256 * 210)],
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
        mx.random.seed(380027 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.5
        for label, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = source.astype(dtype)
            mx.eval(weight)

            def main_fn(value=weight):
                return _main_pack(value)

            def parallel_fn(value=weight):
                return gguf_quantize_weight_mlx(value, "Q6_K")

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
