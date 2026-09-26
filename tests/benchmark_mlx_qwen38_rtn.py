# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare resident MLX RTN weights with main's Torch RTN path."""

import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.config import RTNConfig
from gptqmodel.quantization.mlx_rtn import quantize_rtn_weight_mlx
from gptqmodel.quantization.rtn import RTN
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _median_ms(function, repeats=5):
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    print(
        "projection | shape | symmetric | changed weights | max weight drift | max scale drift | max zero drift | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for sym in (True, False):
        for name, rows, cols in QWEN38_27B_PROJECTIONS:
            _bench_shape(name, rows, cols, sym)


def _bench_shape(name, rows, cols, sym):
    generator = torch.Generator().manual_seed(380027 + rows + cols)
    source = (
        torch.randn((rows, cols), generator=generator, dtype=torch.bfloat16) * 0.025
    )
    layer = torch.nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
    layer.weight.data.copy_(source)
    config = RTNConfig(bits=4, group_size=128, sym=sym)
    oracle = RTN(layer, config)
    resident = mx.array(source.float().numpy()).astype(mx.bfloat16)

    expected = oracle.quantize()
    actual = quantize_rtn_weight_mlx(resident, bits=4, group_size=128, sym=sym)
    weight = np.asarray(actual[0].astype(mx.float32))
    expected_weight = expected[0].float().numpy()
    changed = int(np.count_nonzero(weight != expected_weight))
    weight_drift = float(np.max(np.abs(weight - expected_weight)))
    scale_drift = float(np.max(np.abs(np.asarray(actual[1]) - expected[1].numpy())))
    zero_drift = float(np.max(np.abs(np.asarray(actual[2]) - expected[2].numpy())))
    torch_ms = _median_ms(oracle.quantize)
    mlx_ms = _median_ms(
        partial(quantize_rtn_weight_mlx, resident, bits=4, group_size=128, sym=sym)
    )
    print(
        f"{name} | {rows}x{cols} | {sym} | {changed} | {weight_drift:.9g} | {scale_drift:.9g} | {zero_drift:.9g} | {torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
        flush=True,
    )
    del source, layer, oracle, resident, expected, actual, weight, expected_weight
    gc.collect()


if __name__ == "__main__":
    main()
