# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ParoQuant MLX packing against the existing Torch export path."""

import argparse
import statistics
import time

import mlx.core as mx
import torch

from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.mlx_paroquant import paroquant_pack_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print("projection,out_features,in_features,mlx_ms,torch_ms,speedup")
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        generator = torch.Generator().manual_seed(2718 + in_features + out_features)
        group_size = 64
        scales = torch.rand(
            (out_features, in_features // group_size), generator=generator,
            dtype=torch.float32,
        ).mul(0.02).add(0.01).to(torch.bfloat16)
        codes = torch.randint(
            -7, 8, (out_features, in_features), generator=generator, dtype=torch.int8,
        )
        weight = (codes.to(torch.bfloat16).reshape(out_features, -1, group_size)
                  * scales[:, :, None]).reshape(out_features, in_features)
        del codes
        linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
        linear.weight.data = weight
        existing = AwqTorchLinear(
            bits=4, group_size=group_size, sym=True, desc_act=False,
            in_features=in_features, out_features=out_features, register_buffers=False,
        )
        mlx_weight = mx.array(weight.float().numpy()).astype(mx.bfloat16)
        mlx_scales = mx.array(scales.float().numpy()).astype(mx.bfloat16)
        mx.eval(mlx_weight, mlx_scales)

        def run_mlx():
            packed = paroquant_pack_weight_mlx(
                mlx_weight, mlx_scales, group_size=group_size,
            )
            mx.eval(*packed)

        def run_torch():
            existing.pack(linear, scales, torch.full_like(scales, 8))

        run_mlx()
        run_torch()
        mlx_ms = _median_ms(run_mlx, args.repeats)
        torch_ms = _median_ms(run_torch, args.repeats)
        print(
            f"{name},{out_features},{in_features},"
            f"{mlx_ms:.3f},{torch_ms:.3f},{torch_ms / mlx_ms:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
