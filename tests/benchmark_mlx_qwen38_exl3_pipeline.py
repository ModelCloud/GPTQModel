# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark calibrated EXL3 MLX quantization on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_pipeline import exl3_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_ldlq import _normalized_drift
from tests.test_mlx_exl3_pipeline import _pipeline_oracle


def _median_ms(function, samples):
    result = function()
    del result
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        result = function()
        timings.append((time.perf_counter() - start) * 1000)
        del result
    return statistics.median(timings)


def _calibrated_inputs(in_features, out_features, dtype_name, seed):
    mx.random.seed(seed)
    source = mx.random.normal((in_features, out_features)) * np.float32(0.2)
    checkpoint_dtype = mx.float16 if dtype_name == "FP16" else mx.bfloat16
    source = source.astype(checkpoint_dtype).astype(mx.float32)
    hessian = mx.eye(in_features, dtype=mx.float32) * np.float32(8.0)
    hessian = hessian + mx.eye(in_features, k=1, dtype=mx.float32) * np.float32(0.2)
    hessian = hessian + mx.eye(in_features, k=-1, dtype=mx.float32) * np.float32(0.2)
    generator = np.random.default_rng(seed)
    input_signs = mx.array(
        np.where(generator.integers(0, 2, in_features), -1.0, 1.0).astype(
            np.float32
        )
    )[:, None]
    output_signs = mx.array(
        np.where(generator.integers(0, 2, out_features), -1.0, 1.0).astype(
            np.float32
        )
    )[None, :]
    mx.eval(source, hessian, input_signs, output_signs)
    return source, hessian, input_signs, output_signs


def main(samples, projection, dtype_filter):
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    print(
        "projection | dtype | checkpoint shape | EXL3 shape | trellis mismatches | "
        "input-scale mismatches | output-scale mismatches | reconstructed outside | "
        "max reconstructed drift | normalized drift | proxy drift | "
        "MLX calibrated ms | main Apple baseline",
        flush=True,
    )
    rows = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if projection and name != projection:
            continue
        for dtype_name in ("FP16", "BF16"):
            if dtype_filter and dtype_name.lower() != dtype_filter.lower():
                continue
            checkpoint_dtype = (
                torch.float16 if dtype_name == "FP16" else torch.bfloat16
            )
            source_tensor = torch.full(
                (in_features, out_features), 0.2, dtype=checkpoint_dtype
            ).float()
            source = source_tensor.numpy()
            zero_hessian = np.zeros(
                (in_features, in_features), dtype=np.float32
            )
            input_signs = np.ones((in_features, 1), dtype=np.float32)
            output_signs = np.ones((1, out_features), dtype=np.float32)
            expected = _pipeline_oracle(
                source,
                zero_hessian,
                input_signs,
                output_signs,
                sample_count=1,
                bits=4,
                codebook="mcg",
                force_output_scales=False,
            )
            actual = exl3_quantize_weight_mlx(
                mx.array(source),
                mx.array(zero_hessian),
                mx.array(input_signs),
                mx.array(output_signs),
                sample_count=1,
                bits=4,
                codebook="mcg",
                force_output_scales=False,
            )
            actual_weight = np.asarray(actual[0])
            difference = np.abs(actual_weight - expected[0])
            allowed = 1e-6 + 1e-6 * np.abs(expected[0])
            accuracy = (
                int(np.count_nonzero(np.asarray(actual[2]) != expected[2])),
                int(np.count_nonzero(np.asarray(actual[3]) != expected[3])),
                int(np.count_nonzero(np.asarray(actual[4]) != expected[4])),
                int(np.count_nonzero(difference > allowed)),
                float(difference.max(initial=0.0)),
                _normalized_drift(actual_weight, expected[0]),
                abs(float(actual[1].item()) - expected[1]),
            )
            if any(accuracy[:4]) or accuracy[4] > 1e-6 or accuracy[5] > 1e-6:
                raise AssertionError(f"{name} {dtype_name}: {accuracy}")

            if samples:
                inputs = _calibrated_inputs(
                    in_features,
                    out_features,
                    dtype_name,
                    7331 + in_features + out_features,
                )

                def mlx_call(inputs=inputs):
                    result = exl3_quantize_weight_mlx(
                        *inputs,
                        sample_count=8,
                        bits=4,
                        codebook="mcg",
                        force_output_scales=False,
                    )
                    mx.eval(*result[:5])
                    return result

                mlx_ms = _median_ms(mlx_call, samples)
            else:
                inputs = None
                mlx_ms = None
            rows.append(
                (name, dtype_name, out_features, in_features, *accuracy, mlx_ms)
            )
            latency = "not measured" if mlx_ms is None else f"{mlx_ms:.3f}"
            print(
                f"{name} | {dtype_name} | {out_features}x{in_features} | "
                f"{in_features}x{out_features} | "
                f"{' | '.join(str(value) for value in accuracy[:4])} | "
                f"{accuracy[4]:.9g} | {accuracy[5]:.9g} | {accuracy[6]:.9g} | "
                f"{latency} | unsupported",
                flush=True,
            )
            del source_tensor, source, zero_hessian, input_signs, output_signs
            del expected, actual, actual_weight, difference, allowed, inputs
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Dtype | Checkpoint shape | EXL3 shape | "
        "Trellis mismatches | Input-scale mismatches | Output-scale mismatches | "
        "Reconstructed outside | Max reconstructed drift | Normalized drift | "
        "Proxy drift | MLX calibrated ms | Main Apple baseline |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        latency = "not measured" if row[11] is None else f"{row[11]:.3f}"
        print(
            f"| {row[0]} | {row[1]} | {row[2]}x{row[3]} | {row[3]}x{row[2]} | "
            f"{row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]:.9g} | "
            f"{row[9]:.9g} | {row[10]:.9g} | {latency} | unsupported |"
        )
    latencies = [row[11] for row in rows if row[11] is not None]
    if latencies:
        print(f"\nMedian MLX latency: {statistics.median(latencies):.3f} ms")
        print(f"Minimum MLX latency: {min(latencies):.3f} ms")
        print(f"Maximum MLX latency: {max(latencies):.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--projection")
    parser.add_argument("--dtype", choices=("fp16", "bf16"))
    args = parser.parse_args()
    main(args.samples, args.projection, args.dtype)
