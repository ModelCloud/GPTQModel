# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the 16-row BitsAndBytes prefill tile on Qwen3.8-27B shapes."""

import argparse
import gc

import mlx.core as mx
import numpy as np
import torch

from tests.benchmark_mlx_qwen38_bnb_output_dtype import (
    _fixture,
    _main,
    _main_internal,
    _measure_pair,
)
from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


def run(rows_values, warmups, samples):
    records = []
    for name, out_features, in_features in QWEN38_27B_PROJECTIONS:
        positions = np.arange(in_features, dtype=np.float32)
        for format_name in ("nf4", "fp4"):
            layer, row_weight, bias = _fixture(format_name, out_features, in_features)
            for rows in rows_values:
                source = np.stack([
                    np.sin(positions * (0.013 + row * 0.0001)) * 0.08
                    + np.cos(positions * (0.007 + row * 0.0001)) * 0.04
                    for row in range(rows)
                ])
                for dtype_name, dtype in (("FP16", mx.float16), ("BF16", mx.bfloat16)):
                    x = mx.array(source).astype(dtype)
                    mx.eval(x)

                    def main_fn(value=x, bnb_layer=layer):
                        return _main(bnb_layer, value)

                    def candidate_fn(value=x, bnb_layer=layer):
                        return bnb_layer(value)

                    main_ms, candidate_ms = _measure_pair(
                        main_fn, candidate_fn, warmups, samples,
                    )
                    main_output = main_fn()
                    candidate_output = candidate_fn()
                    internal = _main_internal(layer, x)
                    mx.eval(main_output, candidate_output, internal)
                    torch_input = torch.from_numpy(
                        np.asarray(x.astype(mx.float32)),
                    ).double()
                    raw_row = (
                        torch_input @ torch.from_numpy(row_weight).double()
                    ).numpy()
                    raw = raw_row[:, None] + bias.astype(np.float64)[None, :]
                    torch_dtype = torch.float16 if dtype == mx.float16 else torch.bfloat16
                    rounded = torch.from_numpy(raw).to(torch_dtype).float().numpy()
                    main_visible = np.asarray(main_output.astype(mx.float32))
                    candidate_visible = np.asarray(candidate_output.astype(mx.float32))
                    optimized = (
                        8 < rows <= 16
                        and in_features == 6144
                        and out_features == 5120
                    )
                    records.append((
                        name, rows, format_name.upper(), dtype_name,
                        "tile16" if optimized else "unchanged",
                        main_ms, candidate_ms, main_ms / candidate_ms,
                        int(np.count_nonzero(candidate_visible != main_visible)),
                        float(np.max(np.abs(np.asarray(internal) - raw))),
                        float(np.max(np.abs(candidate_visible - rounded))),
                        float(np.max(np.abs(candidate_visible - raw))),
                    ))
                    record = records[-1]
                    print(
                        f"{name} rows={rows} {format_name.upper()} {dtype_name}: "
                        f"{main_ms:.4f} -> {candidate_ms:.4f} ms "
                        f"({record[7]:.3f}x, {record[4]}), changed {record[8]}, "
                        f"errors {record[9]:.8g}/{record[10]:.8g}/{record[11]:.8g}",
                        flush=True,
                    )
            del layer
            gc.collect()
            mx.clear_cache()

    print(
        "\n| Projection | Rows | Format | Dtype | Path | Main ms | PR ms | Speedup | "
        "Changed | FP32/raw max abs | Visible/rounded max abs | Visible/raw max abs |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in records:
        print(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]:.4f} | {row[6]:.4f} | {row[7]:.3f}x | {row[8]} | "
            f"{row[9]:.8g} | {row[10]:.8g} | {row[11]:.8g} |"
        )
    changed = [row for row in records if row[4] == "tile16"]
    print(f"\nTile-16 minimum speedup: {min(row[7] for row in changed):.3f}x")
    print(f"Tile-16 maximum speedup: {max(row[7] for row in changed):.3f}x")
    print(f"Maximum FP32/raw error: {max(row[9] for row in records):.8g}")
    print(f"Maximum visible/rounded error: {max(row[10] for row in records):.8g}")
    print(f"Maximum visible/raw error: {max(row[11] for row in records):.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 16))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=301)
    args = parser.parse_args()
    run(args.rows, args.warmups, args.samples)
