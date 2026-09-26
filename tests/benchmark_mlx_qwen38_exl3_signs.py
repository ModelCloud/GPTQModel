# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 sign format: TurboDerp and ExLlamaV3 contributors.

"""Benchmark native MLX EXL3 sign packing on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3_signs import exl3_pack_signs_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_signs import _torch_pack_signs_oracle


def _median_ms(function, repeats, inner_loops):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(inner_loops):
            function()
        samples.append((time.perf_counter() - start) * 1000 / inner_loops)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-loops", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.inner_loops < 1:
        parser.error("--inner-loops must be positive")

    print(
        "projection | mode | checkpoint shape | signs | packed words | "
        "mismatched words | Torch CPU oracle ms | MLX median ms",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        for mode, channels in (("input", in_features), ("output", out_features)):
            rng = np.random.default_rng(93175 + out_features + in_features + len(mode))
            signs = rng.choice([-1.0, 1.0], size=channels).astype(np.float32)
            torch_signs = torch.from_numpy(signs).to(torch.bfloat16)
            mlx_signs = mx.array(signs).astype(mx.bfloat16)
            mx.eval(mlx_signs)
            torch_call = partial(_torch_pack_signs_oracle, torch_signs)
            mlx_call = partial(exl3_pack_signs_mlx, mlx_signs)

            expected = torch_call().numpy()
            actual = np.asarray(mlx_call())
            mismatches = int(np.count_nonzero(actual != expected))
            if mismatches:
                raise AssertionError(f"{name} {mode}: {mismatches} words differ")
            torch_ms = _median_ms(torch_call, args.repeats, args.inner_loops)
            mlx_ms = _median_ms(mlx_call, args.repeats, args.inner_loops)
            print(
                f"{name} | {mode} | {out_features}x{in_features} | "
                f"{channels} | {actual.size} | {mismatches} | "
                f"{torch_ms:.6f} | {mlx_ms:.6f}",
                flush=True,
            )
            del signs, torch_signs, mlx_signs, expected, actual
            del torch_call, mlx_call
            gc.collect()


if __name__ == "__main__":
    main()
