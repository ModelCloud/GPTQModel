# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark resident QQQ quantization against equivalent Torch arithmetic."""

import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_qqq import qqq_quantize_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_qqq_quantization import (
    _boundary_margin,
    _codes,
    _torch_banded_oracle,
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
        "projection | mode | shape | changed codes | direct ties | tie descendants | max direct margin | max weight drift | max drift away from ties | max scale drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, rows, columns in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(744 + rows + columns)
        weight = mx.array(
            rng.normal(0, 0.2, (rows, columns)).astype(np.float32)
        ).astype(mx.bfloat16)
        mx.eval(weight)
        source = np.asarray(weight.astype(mx.float32))
        factor = np.eye(columns, dtype=np.float32)
        np.fill_diagonal(factor[:, 1:], 0.05)
        resident_factor = mx.array(factor)
        for group_size in (-1, 128):
            torch_call = partial(_torch_banded_oracle, source, group_size)
            mlx_call = partial(
                qqq_quantize_weight_mlx,
                weight,
                resident_factor,
                group_size=group_size,
            )
            expected = torch_call()
            actual = tuple(
                None if value is None else np.asarray(value) for value in mlx_call()
            )
            changed = _codes(actual, group_size) != _codes(expected, group_size)
            coordinates = np.argwhere(changed)
            margins = [
                float(
                    _boundary_margin(
                        source[row],
                        expected[0][row],
                        expected[1][row, 0 if group_size == -1 else column // 128],
                        column,
                    )
                )
                for row, column in coordinates
            ]
            direct_margins = [margin for margin in margins if margin < 0.0002]
            drift = np.abs(actual[0] - expected[0])
            torch_ms = _median_ms(torch_call)
            mlx_ms = _median_ms(mlx_call)
            print(
                f"{name} | {group_size} | {rows}x{columns} | {len(coordinates)} | "
                f"{len(direct_margins)} | {len(margins) - len(direct_margins)} | "
                f"{max(direct_margins, default=0):.9g} | "
                f"{float(np.max(drift)):.9g} | "
                f"{float(np.max(drift[~changed])):.9g} | "
                f"{float(np.max(np.abs(actual[1] - expected[1]))):.9g} | "
                f"{torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
                flush=True,
            )
            del expected, actual, torch_call, mlx_call
            gc.collect()
        del weight, source, factor, resident_factor
        gc.collect()


if __name__ == "__main__":
    main()
