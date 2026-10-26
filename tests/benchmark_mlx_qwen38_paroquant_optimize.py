# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native MLX ParoQuant module optimization against main's Torch path."""

import argparse
import gc
import statistics
import time

import mlx.core as mx
import torch

from gptqmodel.quantization.mlx_paroquant_optimize import optimize_paroquant_linear_mlx_to_torch
from gptqmodel.quantization.paroquant.optimization import (
    build_random_rotation_buffers,
    optimize_paroquant_linear,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _inputs(out_features, in_features, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn((out_features, in_features), generator=generator) * 0.02).to(dtype).float()
    activations = (torch.randn((4, in_features), generator=generator) * 0.02).to(dtype).float()
    pairs, mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=128,
        krot=1,
        pair_ratio=0.5,
        seed=seed,
        device=torch.device("cpu"),
    )
    return weight, activations, pairs, mask


def _common():
    return {
        "bits": 4,
        "group_size": 128,
        "train_rows": 2,
        "val_rows": 2,
        "batch_size": 2,
        "rotation_epochs": 1,
        "finetune_epochs": 1,
        "rotation_lr": 0.005,
        "weight_lr": 1e-5,
        "quantizer_lr": 1e-6,
        "optimizer_name": "adamw",
        "optimizer_weight_decay": 0.01,
        "optimizer_betas": (0.9, 0.95),
        "optimizer_eps": 1e-10,
        "optimizer_amsgrad": False,
        "sgd_momentum": 0.0,
        "sgd_dampening": 0.0,
        "sgd_nesterov": False,
        "best_state_dtype": "fp32",
        "scale_clamp_min": 0.01,
        "scale_clamp_max": 100.0,
    }


def _measure_pair(torch_call, mlx_call, samples):
    for function in (torch_call, mlx_call):
        result = function()
        del result
    timings = [[], []]
    functions = (torch_call, mlx_call)
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter()
            result = functions[index]()
            timings[index].append((time.perf_counter() - start) * 1000)
            del result
            gc.collect()
    return statistics.median(timings[0]), statistics.median(timings[1])


def _codes(result):
    weight = result.pack_weight.reshape(result.pack_weight.shape[0], result.q_scales.shape[1], 128)
    return torch.round(
        (weight + result.q_zeros[:, :, None] * result.q_scales[:, :, None])
        / result.q_scales[:, :, None]
    ).to(torch.int8)


def run(samples, projection, dtype_filter):
    if samples <= 0:
        raise ValueError("samples must be positive")
    rows = []
    common = _common()
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        if projection and name != projection:
            continue
        for dtype_name, dtype in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
            if dtype_filter and dtype_filter.lower() != dtype_name.lower():
                continue
            seed = 3235 + out_features + in_features
            weight, activations, pairs, mask = _inputs(out_features, in_features, dtype, seed)

            def mlx_call(
                weight=weight,
                activations=activations,
                pairs=pairs,
                mask=mask,
            ):
                return optimize_paroquant_linear_mlx_to_torch(
                    weight=weight,
                    bias=None,
                    inputs=activations,
                    pairs=pairs,
                    theta_mask=mask,
                    symmetric=False,
                    **common,
                )

            def torch_call(weight=weight, activations=activations):
                return optimize_paroquant_linear(
                    weight=weight,
                    bias=None,
                    inputs=activations,
                    sym=True,
                    krot=1,
                    pair_ratio=0.5,
                    seed=seed,
                    stage_impl="fast",
                    pair_impl="fast",
                    quantizer_impl="reference",
                    fused_rotation=False,
                    gradient_checkpointing=False,
                    stage_cudagraph=False,
                    **common,
                )

            main_result = torch_call()
            mlx_result = mlx_call()
            difference = (mlx_result.pseudo_weight - main_result.pseudo_weight).abs()
            normalized = float(
                torch.linalg.vector_norm(mlx_result.pseudo_weight - main_result.pseudo_weight)
                / torch.linalg.vector_norm(main_result.pseudo_weight).clamp_min(1e-30)
            )
            code_mismatches = int(torch.count_nonzero(_codes(mlx_result) != _codes(main_result)))
            target = torch.nn.functional.linear(activations, weight)
            main_output = torch.nn.functional.linear(activations, main_result.pseudo_weight)
            mlx_output = torch.nn.functional.linear(activations, mlx_result.pseudo_weight)
            main_error = float(torch.nn.functional.smooth_l1_loss(main_output, target))
            mlx_error = float(torch.nn.functional.smooth_l1_loss(mlx_output, target))
            loss_drift = abs(mlx_error - main_error)
            max_abs = float(difference.max())
            max_output_drift = float((mlx_output - main_output).abs().max())
            code_fraction = code_mismatches / weight.numel()
            del difference, target, main_output, mlx_output
            del main_result, mlx_result
            gc.collect()
            mx.clear_cache()

            main_ms, mlx_ms = _measure_pair(torch_call, mlx_call, samples)
            rows.append(
                (
                    name,
                    dtype_name,
                    out_features,
                    in_features,
                    code_mismatches,
                    code_fraction,
                    max_abs,
                    normalized,
                    max_output_drift,
                    main_error,
                    mlx_error,
                    loss_drift,
                    main_ms,
                    mlx_ms,
                    main_ms / mlx_ms,
                )
            )
            print(" | ".join(str(value) for value in rows[-1]), flush=True)
            del weight, activations, pairs, mask
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Dtype | Shape | Code mismatches | Code fraction | Max weight drift | "
        "Normalized drift | Max output drift | Main FP32-oracle loss | MLX FP32-oracle loss | "
        "Loss drift | Main ms | MLX ms | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row[0]} | {row[1]} | {row[2]}x{row[3]} | {row[4]} | "
            f"{row[5]:.9g} | {row[6]:.9g} | {row[7]:.9g} | {row[8]:.9g} | "
            f"{row[9]:.9g} | {row[10]:.9g} | {row[11]:.9g} | "
            f"{row[12]:.3f} | {row[13]:.3f} | {row[14]:.3f}x |"
        )
    print(f"\nMedian main latency: {statistics.median(row[12] for row in rows):.3f} ms")
    print(f"Median MLX latency: {statistics.median(row[13] for row in rows):.3f} ms")
    print(f"Median speedup: {statistics.median(row[14] for row in rows):.3f}x")
    print(f"Minimum speedup: {min(row[14] for row in rows):.3f}x")
    print(f"Maximum speedup: {max(row[14] for row in rows):.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--projection")
    parser.add_argument("--dtype", choices=("fp16", "bf16"))
    args = parser.parse_args()
    run(args.samples, args.projection, args.dtype)
