# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 trellis format: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark EXL3 MLX trellis packing against an exact Torch implementation."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_exl3 import exl3_pack_trellis_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_quantization import _torch_pack_trellis_oracle


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

    bits = 3
    print(
        "projection | shape | packed words | mismatched words | Torch ms | MLX ms | speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        shape = (in_features // 16, out_features // 16, 256)
        rng = np.random.default_rng(731 + out_features + in_features)
        encoded = rng.integers(-32768, 32768, shape, dtype=np.int16)
        resident = mx.array(encoded)
        mx.eval(resident)
        torch_call = partial(_torch_pack_trellis_oracle, encoded, bits)
        mlx_call = partial(exl3_pack_trellis_mlx, resident, bits=bits)

        expected = torch_call().numpy()
        actual = np.asarray(mlx_call())
        mismatches = int(np.count_nonzero(actual != expected))
        if mismatches:
            raise AssertionError(f"{name}: {mismatches} packed words differ")
        torch_ms = _median_ms(torch_call, args.repeats)
        mlx_ms = _median_ms(mlx_call, args.repeats)
        print(
            f"{name} | {out_features}x{in_features} | {actual.size} | {mismatches} | "
            f"{torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
            flush=True,
        )
        del encoded, resident, expected, actual, torch_call, mlx_call
        gc.collect()


if __name__ == "__main__":
    main()
