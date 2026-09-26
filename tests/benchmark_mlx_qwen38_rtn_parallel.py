# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Benchmark main's serial RTN groups against parallel MLX groups on Qwen3.8-27B."""

import argparse
import gc
import statistics
import time
from functools import lru_cache

import mlx.core as mx
import numpy as np

from gptqmodel.quantization.mlx_rtn import quantize_rtn_weight_mlx
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


@lru_cache(maxsize=1)
def _main_kernel():
    """The exact serial group kernel from main before this change."""
    return mx.fast.metal_kernel(
        name="gptqmodel_rtn_group_main_benchmark",
        input_names=["weight"],
        output_names=["quantized", "scales", "zeros"],
        source="""
            uint group = thread_position_in_grid.x;
            if (group >= ROWS * GROUPS) return;
            uint row = group / GROUPS;
            uint start = (group % GROUPS) * GROUP_SIZE;
            uint end = min(start + uint(GROUP_SIZE), uint(COLS));
            float minimum = 0.0f;
            float maximum = 0.0f;
            for (uint col = start; col < end; ++col) {
                float value = weight[row * COLS + col];
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }
            if (SYMMETRIC) {
                maximum = max(-minimum, maximum);
                if (minimum < 0.0f) minimum = -maximum;
            }
            if (minimum == 0.0f && maximum == 0.0f) {
                minimum = -1.0f;
                maximum = 1.0f;
            }
            float scale = (maximum - minimum) / float(MAXQ);
            float zero = SYMMETRIC ? float((MAXQ + 1) / 2) :
                metal::rint(-minimum / scale);
            scales[group] = scale;
            zeros[group] = zero;
            for (uint col = start; col < end; ++col) {
                float value = weight[row * COLS + col];
                float code = metal::clamp(metal::rint(value / scale) + zero,
                                          0.0f, float(MAXQ));
                quantized[row * COLS + col] = scale * (code - zero);
            }
        """,
    )


def _main_quantize(weight, *, bits, group_size, sym):
    rows, cols = weight.shape
    effective = cols if group_size == -1 else group_size
    groups = (cols + effective - 1) // effective
    quantized, scales, zeros = _main_kernel()(
        inputs=[weight.astype(mx.float32)],
        template=[
            ("ROWS", rows), ("COLS", cols), ("GROUPS", groups),
            ("GROUP_SIZE", effective), ("MAXQ", (1 << bits) - 1),
            ("SYMMETRIC", int(sym)),
        ],
        grid=(rows * groups, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols), (rows, groups), (rows, groups)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    quantized = quantized.astype(weight.dtype)
    indices = mx.arange(cols, dtype=mx.int32) // effective
    mx.eval(quantized, scales, zeros, indices)
    return quantized, scales, zeros, indices


def _measure_pair(main_call, pr_call, warmups, samples):
    for _ in range(warmups):
        main_call()
        pr_call()
    timings = {"main": [], "pr": []}
    calls = (("main", main_call), ("pr", pr_call))
    for sample in range(samples):
        for label, call in (calls if sample % 2 == 0 else calls[::-1]):
            start = time.perf_counter_ns()
            call()
            timings[label].append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(timings["main"]), statistics.median(timings["pr"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=101)
    args = parser.parse_args()
    records = []
    for name, rows, cols in QWEN38_27B_PROJECTIONS:
        rng = np.random.default_rng(380027 + rows + cols)
        source = rng.normal(0, 0.025, (rows, cols)).astype(np.float32)
        for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
            weight = mx.array(source).astype(dtype)
            mx.eval(weight)
            for sym in (False, True):
                kwargs = {"bits": 4, "group_size": 128, "sym": sym}

                def main_call():
                    return _main_quantize(weight, **kwargs)

                def pr_call():
                    return quantize_rtn_weight_mlx(weight, **kwargs)

                main_ms, pr_ms = _measure_pair(main_call, pr_call, args.warmups, args.samples)
                baseline, candidate = main_call(), pr_call()
                errors = []
                for old, new in zip(baseline, candidate):
                    old_values = np.asarray(old.astype(mx.float32))
                    new_values = np.asarray(new.astype(mx.float32))
                    errors.append(float(np.max(np.abs(old_values - new_values))))
                records.append((name, dtype_name, sym, main_ms, pr_ms, main_ms / pr_ms, *errors))
                print(
                    f"{name} {dtype_name} sym={sym}: {main_ms:.4f} -> {pr_ms:.4f} ms "
                    f"({main_ms / pr_ms:.3f}x), errors {errors}",
                    flush=True,
                )
        del source
        gc.collect()
        mx.clear_cache()

    print("\n| Projection | Dtype | Sym | Main ms | PR ms | Speedup | Weight error | Scale error | Zero error | Index error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {row[4]:.4f} | "
            f"{row[5]:.3f}x | {row[6]:.8f} | {row[7]:.8f} | {row[8]:.8f} | {row[9]:.8f} |",
        )
    print(f"\nMedian speedup: {statistics.median(row[5] for row in records):.3f}x")
    print(f"Minimum speedup: {min(row[5] for row in records):.3f}x")
    print(f"Maximum speedup: {max(row[5] for row in records):.3f}x")
    print(f"Maximum output difference: {max(max(row[6:]) for row in records):.8f}")


if __name__ == "__main__":
    main()
