# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native MLX bitsandbytes weight quantization against CPU Torch."""

import argparse
import gc
import statistics
import time

import bitsandbytes as bnb
import mlx.core as mx
import numpy as np
import torch

from gptqmodel.quantization.mlx_bitsandbytes import (
    quantize_4bit_weight_mlx,
    quantize_int8_weight_mlx,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _median_ms(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--projection", action="append")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(
        "format,projection,out,in,mlx_ms,torch_ms,speedup,code_mismatches,"
        "non_tie_mismatches,nested_code_mismatches,max_scale_abs,offset_abs", flush=True,
    )
    for name, rows, cols in QWEN38_27B_PROJECTIONS:
        if args.projection and name not in args.projection:
            continue
        mx.random.seed(380027 + rows + cols)
        weight = mx.random.normal((rows, cols)).astype(mx.bfloat16)
        mx.eval(weight)
        source = torch.from_numpy(np.asarray(weight.astype(mx.float32))).to(torch.bfloat16)
        for fmt in ("nf4", "fp4", "int8"):
            if fmt == "int8":
                native_weight = weight.astype(mx.float16)
                oracle_weight = source.to(torch.float16)

                def native(native_weight=native_weight):
                    return quantize_int8_weight_mlx(native_weight)

                def oracle(oracle_weight=oracle_weight):
                    return bnb.functional.int8_vectorwise_quant(oracle_weight, threshold=0.0)

            else:
                def native(weight=weight, fmt=fmt):
                    return quantize_4bit_weight_mlx(weight, quant_type=fmt, block_size=64,
                                                     compress_statistics=True)

                def oracle(source=source, fmt=fmt):
                    return bnb.functional.quantize_4bit(source, quant_type=fmt, blocksize=64,
                                                        compress_statistics=True,
                                                        quant_storage=torch.uint8)

            native_ms = _median_ms(native, args.repeats)
            oracle_ms = _median_ms(oracle, args.repeats)
            actual, actual_scales = native()
            expected, expected_state, *_ = oracle()
            actual_codes = np.asarray(actual)
            expected_codes = expected.numpy()
            different = actual_codes != expected_codes
            mismatch_count = int(np.count_nonzero(different))
            if fmt == "int8":
                row, col = np.nonzero(different)
                if row.size:
                    value = oracle_weight.numpy()[row, col].astype(np.float64)
                    maximum = expected_state.numpy()[row].astype(np.float64)
                    exact = value * 127.0 / maximum
                    non_ties = int(np.count_nonzero(exact - np.floor(exact) != 0.5))
                else:
                    non_ties = 0
                nested_mismatches = 0
                scale_abs = float(np.max(np.abs(np.asarray(actual_scales) - expected_state.numpy())))
                offset_abs = 0.0
            else:
                non_ties = mismatch_count
                nested_mismatches = int(np.count_nonzero(
                    np.asarray(actual_scales["absmax"]) != expected_state.absmax.numpy(),
                ))
                scale_abs = float(np.max(np.abs(
                    np.asarray(actual_scales["nested_absmax"]) - expected_state.state2.absmax.numpy(),
                )))
                offset_abs = abs(float(np.asarray(actual_scales["offset"])) - float(expected_state.offset))
            print(
                f"{fmt},{name},{rows},{cols},{native_ms:.3f},{oracle_ms:.3f},"
                f"{oracle_ms/native_ms:.3f},{mismatch_count},{non_ties},"
                f"{nested_mismatches},{scale_abs:.9g},{offset_abs:.9g}", flush=True,
            )
        mx.clear_cache()
        gc.collect()


if __name__ == "__main__":
    main()
