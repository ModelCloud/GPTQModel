# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the shape-gated packed Q6_K prefill path against current main."""

import argparse
import gc
from statistics import median
from time import perf_counter_ns

import mlx.core as mx
import numpy as np
import torch

from tests.benchmark_mlx_qwen38_gguf_q6_k import make_layers
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def _native_unrounded(layer, x):
    rows = x.size // layer.input_dims
    return layer._packed_matmul(
        x, rows, min(rows, 4), 32, output_dtype=mx.float32,
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


def _oracle(x, reference, output_pattern, dtype):
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    raw = (torch_input @ torch.from_numpy(reference).double().T)[
        :, torch.from_numpy(output_pattern)
    ]
    target = torch.float16 if dtype == mx.float16 else torch.bfloat16
    return raw.numpy(), raw.to(target).float().numpy()


def run(rows, warmups, samples):
    records = []
    for index, (name, out_features, in_features) in enumerate(QWEN38_27B_PROJECTIONS):
        baseline, candidate, reference, output_pattern = make_layers(
            out_features, in_features,
        )
        positions = np.arange(in_features, dtype=np.float32)
        source = np.stack([
            np.sin(positions * (0.011 + row * 0.0001)) * 0.08
            + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
            for row in range(rows)
        ])
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
            raw, rounded = _oracle(x, reference, output_pattern, dtype)
            visible = np.asarray(candidate_output.astype(mx.float32))
            main_visible = np.asarray(main_output.astype(mx.float32))
            native = dtype == mx.bfloat16 and 1 < rows <= 16 and out_features <= 2048
            if native:
                internal = _native_unrounded(candidate, x)
                mx.eval(internal)
                internal_error = float(np.max(np.abs(np.asarray(internal) - raw)))
            else:
                internal_error = float("nan")
            record = (
                name, dtype_name, "native" if native else "unchanged",
                main_ms, candidate_ms, main_ms / candidate_ms,
                int(np.count_nonzero(main_visible != visible)),
                internal_error,
                float(np.max(np.abs(visible - rounded))),
                float(np.max(np.abs(visible - raw))),
            )
            records.append(record)
            print(
                f"{name} {dtype_name} {record[2]}: {main_ms:.4f} -> "
                f"{candidate_ms:.4f} ms ({record[5]:.3f}x), changed "
                f"{record[6]}, errors {record[7]:.8g}/{record[8]:.8g}/{record[9]:.8g}",
                flush=True,
            )
        del baseline, candidate
        mx.clear_cache()
        gc.collect()

    print(
        "\n| Projection | Dtype | Path | Main ms | PR ms | Speedup | Changed | "
        "FP32/raw max abs | Visible/rounded max abs | Visible/raw max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        internal = "—" if np.isnan(row[7]) else f"{row[7]:.8g}"
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | "
            f"{row[4]:.4f} | {row[5]:.3f}x | {row[6]} | {internal} | "
            f"{row[8]:.8g} | {row[9]:.8g} |"
        )
    changed = [row for row in records if row[2] == "native"]
    print(f"\nNative minimum speedup: {min(row[5] for row in changed):.3f}x")
    print(f"Native maximum speedup: {max(row[5] for row in changed):.3f}x")
    print(f"Maximum FP32/raw error: {max(row[7] for row in changed):.8g}")
    print(f"Maximum visible/rounded error: {max(row[8] for row in records):.8g}")
    print(f"Maximum visible/raw error: {max(row[9] for row in records):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.rows, args.warmups, args.samples)
