# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx

"""Benchmark ParoQuant transformed-weight quantization against Torch."""

import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_paroquant_quant import (
    paroquant_quantize_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_paroquant_quantization import _torch_oracle


def _median_ms(function, repeats=3):
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def _torch_call(weight, scales, *, sym, zero_point_float):
    quantized = _torch_oracle(
        weight,
        scales,
        bits=4,
        group_size=128,
        sym=sym,
        zero_point_float=zero_point_float,
    )
    return torch.from_numpy(quantized).to(torch.bfloat16)


def main():
    print(
        "projection | sym | shape | changed outputs | max abs drift | max relative drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(8821 + rows + columns)
        weight = mx.array(
            rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
        ).astype(mx.bfloat16)
        mx.eval(weight)
        source = np.asarray(weight.astype(mx.float32))
        groups = columns // 128
        scales = rng.uniform(0.002, 0.06, (rows, groups)).astype(np.float32)
        resident_scales = mx.array(scales)
        for sym in (True, False):
            zero_float = (
                None if sym else rng.uniform(-12, 0, (rows, groups)).astype(np.float32)
            )
            resident_zeros = None if sym else mx.array(zero_float)
            torch_call = partial(
                _torch_call,
                source,
                scales,
                sym=sym,
                zero_point_float=zero_float,
            )
            mlx_call = partial(
                paroquant_quantize_weight_mlx,
                weight,
                resident_scales,
                bits=4,
                group_size=128,
                sym=sym,
                zero_point_float=resident_zeros,
            )
            expected = torch_call().float().numpy()
            actual = np.asarray(mlx_call().astype(mx.float32))
            delta = np.abs(actual - expected)
            relative = delta / np.maximum(np.abs(expected), 1e-12)
            torch_ms = _median_ms(torch_call)
            mlx_ms = _median_ms(mlx_call)
            print(
                f"{name} | {sym} | {rows}x{columns} | "
                f"{int(np.count_nonzero(delta))} | {float(delta.max()):.9g} | "
                f"{float(relative.max()):.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | "
                f"{torch_ms / mlx_ms:.2f}x",
                flush=True,
            )
            del expected, actual, delta, relative, torch_call, mlx_call
            gc.collect()
        del weight, source, scales, resident_scales
        gc.collect()


if __name__ == "__main__":
    main()
