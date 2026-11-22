#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Report the KV-cache classification for every supported model type.

The estimator resolves the cache layout per layer, so its correctness depends on
how each architecture declares that layout. This walks the registry and prints
what each model resolved to, which is the fastest way to spot a family that is
being described by the wrong formula - or not described at all.

    python scripts/check_kv_cache_models.py            # table
    python scripts/check_kv_cache_models.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

from transformers import AutoConfig  # noqa: E402

from gptqmodel.models.auto import SUPPORTED_MODELS  # noqa: E402
from gptqmodel.utils.kv_cache import estimate_kv_cache  # noqa: E402


def classify() -> list[dict]:
    rows: list[dict] = []
    for model_type in sorted(SUPPORTED_MODELS):
        row = {"model_type": model_type}
        try:
            config = AutoConfig.for_model(model_type)
        except Exception as exc:
            row.update(status="unmapped", reason=f"{type(exc).__name__}")
            rows.append(row)
            continue

        try:
            estimate = estimate_kv_cache(config)
        except Exception as exc:  # the sweep exists to surface these
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            continue

        if estimate is None:
            row.update(status="no_cache")
        else:
            row.update(
                status="ok",
                layout=estimate.layer_type,
                growing_layers=estimate.growing_layers,
                sliding_layers=estimate.sliding_layers,
                linear_layers=estimate.linear_layers,
                mtp_layers=estimate.mtp_layers,
                kv_bytes_per_token=estimate.kv_bytes_per_token,
                exact=estimate.is_exact,
                unclassified=list(estimate.unclassified),
            )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None, help="write the full table here")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    rows = classify()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    if not args.quiet:
        layout_counts: dict[str, int] = {}
        for row in rows:
            if row["status"] == "ok":
                layout_counts[row["layout"]] = layout_counts.get(row["layout"], 0) + 1
        print(f"{'model_type':28s} {'status':9s} {'layout':7s} {'layers':>6s} {'sliding':>7s} {'KiB/token':>10s}")
        for row in rows:
            if row["status"] != "ok":
                print(f"{row['model_type']:28s} {row['status']:9s} {'':7s} {'':>6s} {'':>7s} {'':>10s}  {row.get('reason', '')}")
                continue
            print(
                f"{row['model_type']:28s} {row['status']:9s} {row['layout']:7s} "
                f"{row['growing_layers']:6d} {row['sliding_layers']:7d} "
                f"{row['kv_bytes_per_token'] / 1024:10.1f}"
            )
        print()
        print("layouts:", dict(sorted(layout_counts.items())))

    print(f"models={len(rows)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
