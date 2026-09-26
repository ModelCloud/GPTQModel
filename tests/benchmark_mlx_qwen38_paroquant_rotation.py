# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation math: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark ParoQuant MLX rotations against main's Torch rotation path."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_paroquant_rotation import paroquant_rotate_mlx
from gptqmodel.utils.paroquant import apply_paroquant_rotation
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_paroquant_rotation import _pair_schedule


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _normalized_rms_drift(error, expected):
    squared_error = 0.0
    squared_expected = 0.0
    error_flat = error.reshape(-1)
    expected_flat = expected.reshape(-1)
    chunk_size = 1 << 20
    for start in range(0, error_flat.size, chunk_size):
        end = min(start + chunk_size, error_flat.size)
        error_chunk = error_flat[start:end].astype(np.float64)
        expected_chunk = expected_flat[start:end].astype(np.float64)
        squared_error += float(np.dot(error_chunk, error_chunk))
        squared_expected += float(np.dot(expected_chunk, expected_chunk))
    return float(np.sqrt(squared_error / squared_expected))


def _main_torch_rotation(weight, pairs, theta, scales, *, group_size, inverse):
    """Time main's ParoQuant rotation API on its Apple-Silicon CPU fallback."""
    rotated = apply_paroquant_rotation(
        weight,
        pairs,
        theta,
        scales=None if inverse else scales.reshape(1, -1),
        group_size=group_size,
    )
    return rotated / scales.reshape(1, -1) if inverse else rotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    group_size, krot = 128, 8
    print(
        "projection | direction | shape | outside 1e-6 | max abs drift | "
        "normalized RMS drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        seed = 8821 + rows + columns
        rng = np.random.default_rng(seed)
        weight = mx.array(
            rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
        ).astype(mx.bfloat16)
        theta = mx.array(
            rng.uniform(-0.35, 0.35, (krot, columns // 2)).astype(np.float32)
        )
        scales = mx.array(rng.uniform(0.7, 1.3, columns).astype(np.float32))
        mx.eval(weight, theta, scales)

        source = np.asarray(weight.astype(mx.float32))
        torch_source = torch.from_numpy(source)
        theta_source = np.asarray(theta)
        scales_source = np.asarray(scales)
        pairs = _pair_schedule(columns, group_size, krot, seed=seed + 17)
        torch_scales = torch.from_numpy(scales_source)

        for inverse in (False, True):
            torch_pairs = torch.from_numpy(pairs.copy())
            torch_theta = torch.from_numpy(theta_source.copy())
            if inverse:
                torch_pairs = torch_pairs.flip(0).contiguous()
                torch_theta = -torch_theta.flip(0).contiguous()
            torch_call = partial(
                _main_torch_rotation,
                torch_source,
                torch_pairs,
                torch_theta,
                torch_scales,
                group_size=group_size,
                inverse=inverse,
            )
            mlx_call = partial(
                paroquant_rotate_mlx,
                weight,
                pairs,
                theta,
                group_size=group_size,
                channel_scales=scales,
                inverse=inverse,
            )

            expected = torch_call().numpy()
            actual = np.asarray(mlx_call())
            error = np.abs(actual - expected)
            allowed = 1e-6 + 1e-6 * np.abs(expected)
            outside = int(np.count_nonzero(error > allowed))
            if outside:
                raise AssertionError(
                    f"{name} inverse={inverse}: {outside} outputs exceed 1e-6 tolerances"
                )
            max_abs = float(error.max())
            normalized_rms = _normalized_rms_drift(error, expected)
            torch_ms = _median_ms(torch_call, args.repeats)
            mlx_ms = _median_ms(mlx_call, args.repeats)
            print(
                f"{name} | {'inverse' if inverse else 'forward'} | "
                f"{rows}x{columns} | {outside} | {max_abs:.9g} | "
                f"{normalized_rms:.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | "
                f"{torch_ms / mlx_ms:.2f}x",
                flush=True,
            )
            del expected, actual, error, torch_call, mlx_call
            gc.collect()

        del (
            weight,
            theta,
            scales,
            source,
            torch_source,
            theta_source,
            scales_source,
            pairs,
        )
        del torch_scales
        gc.collect()


if __name__ == "__main__":
    main()
