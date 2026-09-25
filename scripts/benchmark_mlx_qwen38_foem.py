# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native MLX FOEM against the existing Torch group-update math."""

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_foem import foem_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _torch_foem_group_update(weight, factor, group_size=128, beta=0.2):
    """Run main's alpha-zero, blocksize=group_size FOEM update on CPU."""
    rows, columns = weight.shape
    raw = weight.float()
    working = raw.clone()
    output = torch.empty_like(working)
    for start in range(0, columns, group_size):
        end = start + group_size
        group = working[:, start:end].clone()
        minimum = torch.minimum(group.amin(1), torch.zeros(rows))
        maximum = torch.maximum(group.amax(1), torch.zeros(rows))
        maximum = torch.maximum(minimum.abs(), maximum)
        minimum = torch.where(minimum < 0, -maximum, minimum)
        empty = (minimum == 0) & (maximum == 0)
        minimum = torch.where(empty, -1, minimum)
        maximum = torch.where(empty, 1, maximum)
        scale = (maximum - minimum) / 15
        zero = torch.full_like(scale, 8)
        errors = torch.empty_like(group)
        for offset in range(group_size):
            value = group[:, offset].clone()
            code = (torch.round(value / scale) + zero).clamp(0, 15)
            q = scale * (code - zero)
            error = ((value - q) - (value - raw[:, start + offset]) * beta) / factor[
                start + offset, start + offset
            ]
            output[:, start + offset] = q
            errors[:, offset] = error
            group[:, offset:] -= error[:, None] * factor[
                start + offset, start + offset:end
            ]
            if offset + 1 < group_size:
                group[:, offset + 1] -= beta * (
                    group[:, offset + 1] - raw[:, start + offset + 1]
                )
        if end < columns:
            working[:, end:] -= errors @ factor[start:end, end:]
    return output


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
    parser.add_argument("--projection", help="Only benchmark one projection role")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print("projection,out_features,in_features,mlx_ms,torch_ms,speedup", flush=True)
    for index, (name, out_features, in_features) in enumerate(QWEN38_27B_PROJECTIONS):
        if args.projection and args.projection != name:
            continue
        mx.random.seed(9000 + index)
        weight = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
        factor = mx.eye(in_features)
        mx.eval(weight, factor)
        torch_weight = torch.from_numpy(np.asarray(weight.astype(mx.float32)))
        torch_factor = torch.eye(in_features)

        def run_mlx():
            mx.eval(*foem_quantize_weight_mlx(weight, factor, group_size=128, beta=0.2))

        def run_torch():
            _torch_foem_group_update(torch_weight, torch_factor)

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
