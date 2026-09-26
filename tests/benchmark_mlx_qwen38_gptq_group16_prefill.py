# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark shape-gated packed GPTQ group-16 prefill against current main."""

import argparse
import gc
from statistics import median
from time import perf_counter_ns

import mlx.core as mx
import numpy as np

from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
from gptqmodel.nn_modules.qlinear.mlx_gptq import _uses_packed_prefill
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS
from tests.test_mlx_gptq_group16 import SOURCE_BITS, _group16_fixture, _rounded_oracle


def _native_unrounded(layer, x):
    rows = x.size // layer.input_dims
    return layer._packed_prefill(
        x, rows, min(rows, 4), output_dtype=mx.float32,
    )


def _measure_pair(main_fn, candidate_fn, warmups, samples):
    for _ in range(warmups):
        mx.eval(main_fn(), candidate_fn())
    timings = [[], []]
    for sample in range(samples):
        for path in (0, 1) if sample % 2 == 0 else (1, 0):
            started = perf_counter_ns()
            mx.eval((main_fn, candidate_fn)[path]())
            timings[path].append((perf_counter_ns() - started) / 1e6)
    return median(timings[0]), median(timings[1])


def run(rows, warmups, samples):
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        source = np.stack([
            np.sin(positions * (0.011 + row * 0.0001)) * 0.08
            + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
            for row in range(rows)
        ])
        for source_bits in SOURCE_BITS:
            candidate, reference, output_pattern, layer_bias = _group16_fixture(
                out_features, in_features, source_bits,
            )
            baseline = MlxGroup16Linear(
                in_features, out_features, bits=candidate.bits, bias=True,
            )
            for parameter in (
                "weight", "scales_even", "scales_odd",
                "biases_even", "biases_odd", "bias",
            ):
                setattr(baseline, parameter, getattr(candidate, parameter))
            for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                x = mx.array(source).astype(dtype)
                mx.eval(x)

                def main_fn(layer=baseline, value=x):
                    return layer(value)

                def candidate_fn(layer=candidate, value=x):
                    return layer(value)

                main_ms, candidate_ms = _measure_pair(
                    main_fn, candidate_fn, warmups, samples,
                )
                main_output, candidate_output = main_fn(), candidate_fn()
                mx.eval(main_output, candidate_output)
                rounded, raw = _rounded_oracle(
                    x, reference, output_pattern, layer_bias, dtype,
                )
                visible = np.asarray(candidate_output.astype(mx.float32))
                main_visible = np.asarray(main_output.astype(mx.float32))
                native = _uses_packed_prefill(
                    dtype, rows, in_features, out_features,
                )
                if native:
                    internal = _native_unrounded(candidate, x)
                    mx.eval(internal)
                    internal_error = float(
                        np.max(np.abs(np.asarray(internal) - raw))
                    )
                else:
                    internal_error = float("nan")
                record = (
                    name, source_bits, dtype_name,
                    "native" if native else "unchanged",
                    main_ms, candidate_ms, main_ms / candidate_ms,
                    int(np.count_nonzero(main_visible != visible)),
                    internal_error,
                    float(np.max(np.abs(visible - rounded))),
                    float(np.max(np.abs(visible - raw))),
                )
                records.append(record)
                print(
                    f"{name} {source_bits}-bit {dtype_name} {record[3]}: "
                    f"{main_ms:.4f} -> {candidate_ms:.4f} ms ({record[6]:.3f}x), "
                    f"changed {record[7]}, errors "
                    f"{record[8]:.8g}/{record[9]:.8g}/{record[10]:.8g}",
                    flush=True,
                )
            del baseline, candidate
            mx.clear_cache()
            gc.collect()

    print(
        "\n| Projection | Bits | Dtype | Path | Main ms | PR ms | Speedup | "
        "Changed | FP32/raw max abs | Visible/rounded max abs | Visible/raw max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        internal = "—" if np.isnan(row[8]) else f"{row[8]:.8g}"
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.4f} | "
            f"{row[5]:.4f} | {row[6]:.3f}x | {row[7]} | {internal} | "
            f"{row[9]:.8g} | {row[10]:.8g} |"
        )
    changed = [row for row in records if row[3] == "native"]
    print(f"\nNative minimum speedup: {min(row[6] for row in changed):.3f}x")
    print(f"Native maximum speedup: {max(row[6] for row in changed):.3f}x")
    print(f"Maximum FP32/raw error: {max(row[8] for row in changed):.8g}")
    print(f"Maximum visible/rounded error: {max(row[9] for row in records):.8g}")
    print(f"Maximum visible/raw error: {max(row[10] for row in records):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.rows, args.warmups, args.samples)
