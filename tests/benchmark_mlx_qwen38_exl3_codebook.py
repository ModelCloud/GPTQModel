# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 codebooks: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark EXL3 MLX codebook reconstruction against exact Torch math."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_exl3 import exl3_decode_states_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_codebook import _torch_codebook_oracle


def _torch_call(encoded, lut):
    return lut[(encoded.to(torch.int64) & 0xFFFF)]


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    codebook = "mcg"
    lut = _torch_codebook_oracle(codebook)
    print(
        "projection | shape | decoded values | mismatched values | max abs drift | "
        "Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        shape = (in_features // 16, out_features // 16, 256)
        rng = np.random.default_rng(1187 + out_features + in_features)
        encoded = rng.integers(-32768, 32768, shape, dtype=np.int16)
        torch_encoded = torch.from_numpy(encoded)
        mlx_encoded = mx.array(encoded)
        mx.eval(mlx_encoded)
        torch_call = partial(_torch_call, torch_encoded, lut)
        mlx_call = partial(exl3_decode_states_mlx, mlx_encoded, codebook=codebook)

        expected = torch_call().numpy()
        actual = np.asarray(mlx_call())
        delta = np.abs(actual - expected)
        mismatches = int(np.count_nonzero(delta))
        if mismatches:
            raise AssertionError(f"{name}: {mismatches} decoded values differ")
        torch_ms = _median_ms(torch_call, args.repeats)
        mlx_ms = _median_ms(mlx_call, args.repeats)
        print(
            f"{name} | {out_features}x{in_features} | {actual.size} | {mismatches} | "
            f"{float(delta.max()):.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | "
            f"{torch_ms / mlx_ms:.2f}x",
            flush=True,
        )
        del encoded, torch_encoded, mlx_encoded, expected, actual, delta
        del torch_call, mlx_call
        gc.collect()


if __name__ == "__main__":
    main()
