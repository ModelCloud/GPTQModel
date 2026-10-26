# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MLX FP8 weight quantization against main's Torch arithmetic."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.nn_modules.qlinear.fp8 import quantize_fp8_weight
from gptqmodel.quantization.mlx_fp8 import quantize_fp8_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _median_ms(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--format", dest="formats", action="append")
    parser.add_argument("--scale-method", choices=("tensor", "row", "block"), default="row")
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    formats = args.formats or ["float8_e4m3fn"]
    block_size = (128, 128) if args.scale_method == "block" else None
    print("format,scale_method,projection,out,in,mlx_ms,torch_ms,speedup,max_scale_abs,byte_mismatches", flush=True)
    for fmt in formats:
        for index, (name, rows, cols) in enumerate(QWEN38_27B_PROJECTIONS):
            if args.projection and name not in args.projection:
                continue
            mx.random.seed(380027 + index)
            weight = mx.random.normal((rows, cols)).astype(mx.bfloat16)
            mx.eval(weight)
            torch_weight = torch.from_numpy(np.asarray(weight.astype(mx.float32))).to(torch.bfloat16)

            def run_mlx():
                result = quantize_fp8_weight_mlx(
                    weight, format=fmt, weight_scale_method=args.scale_method,
                    weight_block_size=block_size,
                )
                mx.eval(*result)
                return result

            def run_torch():
                return quantize_fp8_weight(
                    torch_weight, format=fmt, weight_scale_method=args.scale_method,
                    weight_block_size=block_size,
                )

            mlx_ms, mlx_result = _median_ms(run_mlx, args.repeats)
            torch_ms, torch_result = _median_ms(run_torch, args.repeats)
            bytes_mlx, scales_mlx = (np.asarray(value) for value in mlx_result)
            bytes_torch = torch_result[0].view(torch.uint8).numpy()
            scales_torch = torch_result[1].numpy()
            mismatches = int(np.count_nonzero(bytes_mlx != bytes_torch))
            max_scale_abs = float(np.max(np.abs(scales_mlx - scales_torch)))
            if mismatches or max_scale_abs > 1e-6:
                raise AssertionError(f"{fmt} {name}: {mismatches=} {max_scale_abs=}")
            print(
                f"{fmt},{args.scale_method},{name},{rows},{cols},{mlx_ms:.3f},{torch_ms:.3f},"
                f"{torch_ms / mlx_ms:.1f},{max_scale_abs:.3e},{mismatches}", flush=True,
            )
            del mlx_result, torch_result
            mx.clear_cache()
            gc.collect()


if __name__ == "__main__":
    main()
