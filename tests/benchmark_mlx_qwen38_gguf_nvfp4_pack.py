# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Benchmark native MLX GGUF NVFP4 packing against the Torch oracle."""

import argparse
import gc
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_gguf import gguf_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _torch_nvfp4_pack(weight):
    rows, columns = weight.shape
    blocks = weight.reshape(-1, 4, 16).float()
    scale_source = (blocks.abs().amax(dim=-1) / 6.0).clamp(0.0, 448.0)
    bits = scale_source.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 120
    subnormal_mantissa = (scale_source * 512.0 + 0.5).to(torch.int32).clamp(0, 7)
    subnormal = torch.where(subnormal_mantissa >= 1, subnormal_mantissa, 0)
    mantissa = ((bits >> 20) & 7) + ((bits >> 19) & 1)
    overflow = mantissa > 7
    normal_mantissa = torch.where(overflow, 0, mantissa)
    normal_exponent = exponent + overflow.to(torch.int32)
    normal = torch.where(
        normal_exponent >= 15,
        0x7E,
        (normal_exponent << 3) | normal_mantissa,
    )
    encoded = torch.where(
        scale_source <= 0.0,
        0,
        torch.where(
            exponent <= 0,
            subnormal,
            torch.where(exponent >= 15, 0x7E, normal),
        ),
    ).to(torch.uint8)
    encoded_int = encoded.to(torch.int32)
    decoded_exponent = (encoded_int >> 3) & 15
    decoded_mantissa = (encoded_int & 7).float()
    decoded = torch.where(
        decoded_exponent == 0,
        decoded_mantissa * (2.0**-10),
        (1.0 + decoded_mantissa / 8.0)
        * torch.pow(2.0, decoded_exponent.float() - 8.0),
    )
    decoded = torch.where((encoded == 0) | (encoded == 0x7F), 0.0, decoded)
    codebook = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.float32,
    )
    codes = (
        decoded[..., None, None] * codebook - blocks[..., None]
    ).abs().argmin(dim=-1).to(torch.uint8)
    payload = codes[..., :8] | (codes[..., 8:] << 4)
    return torch.cat(
        (encoded.reshape(-1, 4), payload.reshape(-1, 32)), dim=-1
    ).reshape(rows, columns // 64 * 36)


def _measure_pair(torch_fn, mlx_fn, samples):
    for _ in range(10):
        torch_fn()
        mx.eval(mlx_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in ((0, 1) if sample % 2 == 0 else (1, 0)):
            start = perf_counter()
            if path == 0:
                torch_fn()
            else:
                mx.eval(mlx_fn())
            timings[path].append((perf_counter() - start) * 1000)
    return median(timings[0]), median(timings[1])


def run(samples):
    rows_out = []
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        mx.random.seed(380040 + rows + columns)
        source = mx.random.normal((rows, columns)) * 0.5
        for label, dtype, torch_dtype in (
            ("FP16", mx.float16, torch.float16),
            ("BF16", mx.bfloat16, torch.bfloat16),
        ):
            weight = source.astype(dtype)
            mx.eval(weight)
            torch_weight = torch.from_numpy(
                np.asarray(weight.astype(mx.float32))
            ).to(torch_dtype)

            def torch_fn(value=torch_weight):
                return _torch_nvfp4_pack(value)

            def mlx_fn(value=weight):
                return gguf_quantize_weight_mlx(value, "NVFP4")

            expected = torch_fn().numpy()
            actual = np.asarray(mlx_fn())
            mismatches = int(np.count_nonzero(expected != actual))
            maximum_byte_error = int(
                np.max(np.abs(expected.astype(np.int16) - actual.astype(np.int16)))
            )
            torch_ms, mlx_ms = _measure_pair(torch_fn, mlx_fn, samples)
            rows_out.append(
                (
                    name,
                    label,
                    torch_ms,
                    mlx_ms,
                    torch_ms / mlx_ms,
                    mismatches,
                    maximum_byte_error,
                )
            )
            print(
                f"{name} {label}: {torch_ms:.4f} -> {mlx_ms:.4f} ms "
                f"({torch_ms / mlx_ms:.3f}x), {mismatches} mismatches, "
                f"maximum byte error {maximum_byte_error}",
                flush=True,
            )
            del weight, torch_weight, expected, actual
            mx.clear_cache()
            gc.collect()
        del source

    print(
        "\n| Projection | Dtype | Torch oracle ms | MLX ms | Speedup | "
        "Byte mismatches | Maximum byte error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows_out:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
            f"{row[4]:.3f}x | {row[5]} | {row[6]} |"
        )
    print(f"\nMedian speedup: {median(row[4] for row in rows_out):.3f}x")
    print(f"Minimum speedup: {min(row[4] for row in rows_out):.3f}x")
    print(f"Maximum speedup: {max(row[4] for row in rows_out):.3f}x")
    print(f"Total byte mismatches: {sum(row[5] for row in rows_out)}")
    print(f"Maximum byte error: {max(row[6] for row in rows_out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=301)
    run(parser.parse_args().samples)
