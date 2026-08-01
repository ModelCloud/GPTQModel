#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare scale/g_idx distributions and quant_log across group-size / GAR checkpoints.

Example:
    python scripts/diag_group_size_scale_quant_log.py \
        --checkpoint g128_agaTrue=/tmp/llama32_agaTrue_g128 \
        --checkpoint g32_agaTrue=/tmp/llama32_agaTrue_g32 \
        --output /tmp/group_size_scale_quant_log.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def _stats(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    mean = float(np.mean(finite))
    return {
        "mean": mean,
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "cv": float(np.std(finite) / mean) if mean else float("nan"),
    }


def _load_quant_log(path):
    import pandas as pd

    df = pd.read_csv(path)
    return {
        "mean": float(df["loss"].mean()),
        "median": float(df["loss"].median()),
        "max": float(df["loss"].max()),
        "p95": float(df["loss"].quantile(0.95)),
        "p99": float(df["loss"].quantile(0.99)),
        "samples": int(df["samples"].sum()) if "samples" in df.columns else None,
    }


def _tensor_map(checkpoint_dir, suffix):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)["weight_map"]
    by_layer = {}
    files = set()
    for key, file in idx.items():
        if key.endswith(suffix):
            by_layer[key[: -len(suffix)]] = file
            files.add(file)
    loaded = {}
    for file in files:
        with safe_open(str(checkpoint_dir / file), framework="pt", device="cpu") as f:
            loaded[file] = {k: f.get_tensor(k) for k in f.keys()}
    return {layer: loaded[file][layer + suffix] for layer, file in by_layer.items()}


def _scale_stats(checkpoint_dir):
    scales = _tensor_map(checkpoint_dir, ".scales")
    all_vals, col_cv, col_ratio, col_range = [], [], [], []
    row_cv, row_ratio, row_range = [], [], []
    for s in scales.values():
        s = s.to(torch.float32).numpy()
        all_vals.append(s.flatten())
        col_min, col_max = s.min(axis=0), s.max(axis=0)
        col_mean, col_std = s.mean(axis=0), s.std(axis=0)
        col_ratio.append(np.where(col_min > 0, col_max / col_min, np.nan))
        col_range.append(np.where(col_mean > 0, (col_max - col_min) / col_mean, np.nan))
        col_cv.append(np.where(col_mean > 0, col_std / col_mean, np.nan))

        row_min, row_max = s.min(axis=1), s.max(axis=1)
        row_mean, row_std = s.mean(axis=1), s.std(axis=1)
        row_ratio.append(np.where(row_min > 0, row_max / row_min, np.nan))
        row_range.append(np.where(row_mean > 0, (row_max - row_min) / row_mean, np.nan))
        row_cv.append(np.where(row_mean > 0, row_std / row_mean, np.nan))

    all_vals = np.concatenate(all_vals)
    return {
        "global": _stats(all_vals),
        "per_output_col": {
            "cv": _stats(np.concatenate(col_cv)),
            "max_min_ratio": _stats(np.concatenate(col_ratio)),
            "relative_range": _stats(np.concatenate(col_range)),
        },
        "per_input_group": {
            "cv": _stats(np.concatenate(row_cv)),
            "max_min_ratio": _stats(np.concatenate(row_ratio)),
            "relative_range": _stats(np.concatenate(row_range)),
        },
    }


def _gidx_stats(checkpoint_dir):
    gidxs = _tensor_map(checkpoint_dir, ".g_idx")
    transitions = []
    run_lengths = []
    for gidx in gidxs.values():
        g = gidx.to(torch.int64).numpy()
        transitions.append(int(np.sum(g[1:] != g[:-1])))
        diffs = np.concatenate(([1], np.diff(g).astype(bool).astype(int)))
        starts = np.where(diffs)[0]
        ends = np.append(starts[1:], len(g))
        run_lengths.extend((ends - starts).tolist())
    return {
        "mean_transitions": float(np.mean(transitions)),
        "median_transitions": float(np.median(transitions)),
        "mean_run_length": float(np.mean(run_lengths)),
        "median_run_length": float(np.median(run_lengths)),
        "min_run_length": int(np.min(run_lengths)),
        "max_run_length": int(np.max(run_lengths)),
    }


def _read_group_size(checkpoint_dir):
    with open(Path(checkpoint_dir) / "quantize_config.json") as f:
        cfg = json.load(f)
    return cfg.get("group_size") or cfg.get("meta", {}).get("group_size")


def _summarize(label, checkpoint_dir):
    result = {
        "label": label,
        "group_size": _read_group_size(checkpoint_dir),
        "checkpoint": str(checkpoint_dir),
    }
    ql = Path(checkpoint_dir) / "quant_log.csv"
    if ql.exists():
        result["quant_log"] = _load_quant_log(ql)
    result["scales"] = _scale_stats(checkpoint_dir)
    result["g_idx"] = _gidx_stats(checkpoint_dir)
    return result


def _markdown_table(results):
    header = (
        "| label | group_size | q_mean | q_max | scale_mean | scale_cv | "
        "col_ratio_mean | col_ratio_p99 | grp_ratio_mean | grp_ratio_p99 | "
        "gidx_mean_run |\n"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = [header, sep]
    for r in results:
        ql = r.get("quant_log", {})
        sc = r["scales"]
        pc = sc["per_output_col"]["max_min_ratio"]
        pr = sc["per_input_group"]["max_min_ratio"]
        lines.append(
            f"| {r['label']} | {r['group_size']} | "
            f"{ql.get('mean', float('nan')):.4e} | {ql.get('max', float('nan')):.4e} | "
            f"{sc['global']['mean']:.5f} | {sc['global']['cv']:.3f} | "
            f"{pc['mean']:.3f} | {pc['p99']:.3f} | "
            f"{pr['mean']:.3f} | {pr['p99']:.3f} | "
            f"{r['g_idx']['mean_run_length']:.1f} |\n"
        )
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare GPTQ group-size checkpoints")
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path")
    parser.add_argument("--output", type=str, required=True, help="JSON output path")
    args = parser.parse_args()

    results = []
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        results.append(_summarize(label, path))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(_markdown_table(results))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
