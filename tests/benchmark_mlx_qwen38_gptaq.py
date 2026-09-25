# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx

"""Benchmark the GPTAQ update against equivalent resident Torch arithmetic."""

import gc
import statistics
import time
from decimal import Decimal
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_gptaq import gptaq_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_gptaq_quantization import (
    _codes,
    _exact_banded_tie_margin,
    _torch_oracle,
)


def _median_ms(function, repeats=3):
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def main():
    print(
        "projection | shape | changed codes | direct ties | tie descendants | max weight drift | max drift away from changed codes | max scale drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(3270 + rows + columns)
        weight = mx.array(
            rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
        ).astype(mx.bfloat16)
        mx.eval(weight)
        source = np.asarray(weight.astype(mx.float32))
        factor = np.eye(columns, dtype=np.float32)
        projection = np.zeros((columns, columns), dtype=np.float32)
        np.fill_diagonal(factor[:, 1:], 0.05)
        np.fill_diagonal(projection[:, 1:], 0.002)
        resident_factor = mx.array(factor)
        resident_projection = mx.array(projection)

        torch_call = partial(_torch_oracle, source, factor, projection, 4, 128, True)
        mlx_call = partial(
            gptaq_quantize_weight_mlx, weight, resident_factor, resident_projection
        )
        expected = torch_call()
        actual = tuple(np.asarray(value) for value in mlx_call())
        changed = _codes(actual, 128) != _codes(expected, 128)
        coordinates = np.argwhere(changed)
        direct = sum(
            _exact_banded_tie_margin(source[row], int(column)) < Decimal("1e-30")
            for row, column in coordinates
        )
        drift = np.abs(actual[0] - expected[0])
        max_weight = float(np.max(drift))
        max_away = float(np.max(drift[~changed]))
        max_scale = float(np.max(np.abs(actual[1] - expected[1])))
        torch_ms = _median_ms(torch_call)
        mlx_ms = _median_ms(mlx_call)
        print(
            f"{name} | {rows}x{columns} | {len(coordinates)} | {direct} | {len(coordinates) - direct} | {max_weight:.9g} | {max_away:.9g} | {max_scale:.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
            flush=True,
        )
        del (
            weight,
            source,
            factor,
            projection,
            resident_factor,
            resident_projection,
            expected,
            actual,
            torch_call,
            mlx_call,
        )
        gc.collect()


if __name__ == "__main__":
    main()
