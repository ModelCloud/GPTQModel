# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 quantization formats: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Benchmark native EXL3 fallback quantization on Qwen3.8-27B shapes."""

import argparse
import gc
import statistics
import time
from functools import partial

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_exl3_fallback import exl3_fallback_quantize_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_exl3_fallback import (
    _torch_fallback_quantize_oracle,
    _torch_patterned_fallback_oracle,
)
from tests.test_mlx_exl3_gss import _patterned_weight


def _median_ms(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def _accuracy(actual_weight, actual_encoded, expected_weight, expected_encoded):
    actual_weight = np.asarray(actual_weight)
    actual_encoded = np.asarray(actual_encoded)
    weight_mismatches = int(
        np.count_nonzero(
            actual_weight.view(np.uint32) != expected_weight.view(np.uint32)
        )
    )
    state_mismatches = int(np.count_nonzero(actual_encoded != expected_encoded))
    max_abs_drift = float(np.max(np.abs(actual_weight - expected_weight)))
    return weight_mismatches, state_mismatches, max_abs_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    print(
        "projection | checkpoint shape | EXL3 matrix shape | tiles | "
        "weight bit mismatches | state mismatches | max abs drift | "
        "Torch oracle extrapolated ms | MLX measured ms | estimated speedup",
        flush=True,
    )
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        source = _patterned_weight(
            in_features,
            out_features,
            seed=43167 + out_features + in_features,
        )
        expected_weight, expected_encoded = _torch_patterned_fallback_oracle(
            source, bits=4, codebook="mcg"
        )
        mlx_weight = mx.array(source)
        mx.eval(mlx_weight)
        actual_weight, actual_encoded = exl3_fallback_quantize_mlx(mlx_weight, bits=4)
        accuracy = _accuracy(
            actual_weight, actual_encoded, expected_weight, expected_encoded
        )
        if accuracy[0] or accuracy[1]:
            raise AssertionError(f"{name}: accuracy={accuracy}")

        one_pattern = np.tile(source[:16, :16], (1, 8))
        torch_tile_ms = (
            _median_ms(
                partial(
                    _torch_fallback_quantize_oracle,
                    one_pattern,
                    bits=4,
                    codebook="mcg",
                ),
                args.repeats,
            )
            / 8
        )
        tile_count = (in_features // 16) * (out_features // 16)
        torch_ms = torch_tile_ms * tile_count
        mlx_ms = _median_ms(
            partial(exl3_fallback_quantize_mlx, mlx_weight, bits=4),
            args.repeats,
        )
        print(
            f"{name} | {out_features}x{in_features} | "
            f"{in_features}x{out_features} | {tile_count} | "
            f"{accuracy[0]} | {accuracy[1]} | {accuracy[2]:.9g} | "
            f"{torch_ms:.3f} | {mlx_ms:.3f} | {torch_ms / mlx_ms:.2f}x",
            flush=True,
        )

        del source, expected_weight, expected_encoded, mlx_weight
        del actual_weight, actual_encoded, one_pattern
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
