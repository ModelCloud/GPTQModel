#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare two QVQ P32 benchmark JSON files using median kernel latency."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    return parser.parse_args()


def _key(row: dict) -> tuple:
    return (
        row.get("shape"),
        row["bits"],
        row["m"],
        row["k"],
        row["n"],
    )


def _summary(speedups: list[float]) -> dict[str, float | int]:
    return {
        "count": len(speedups),
        "minimum": min(speedups),
        "geometric_mean": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "maximum": max(speedups),
    }


def main() -> None:
    args = _parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    baseline_rows = {_key(row): row for row in baseline["rows"]}
    candidate_rows = {_key(row): row for row in candidate["rows"]}
    if baseline_rows.keys() != candidate_rows.keys():
        missing = sorted(baseline_rows.keys() ^ candidate_rows.keys(), key=str)
        raise ValueError(f"benchmark matrices differ: {missing[:8]}")

    rows = []
    by_m = defaultdict(list)
    by_shape = defaultdict(list)
    for key in sorted(baseline_rows, key=str):
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]
        baseline_ms = baseline_row["p32"]["median_ms"]
        candidate_ms = candidate_row["p32"]["median_ms"]
        speedup = baseline_ms / candidate_ms
        shape, bits, m, k, n = key
        rows.append(
            {
                "shape": shape,
                "bits": bits,
                "m": m,
                "k": k,
                "n": n,
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "speedup": speedup,
            }
        )
        by_m[str(m)].append(speedup)
        by_shape[str(shape)].append(speedup)

    speedups = [row["speedup"] for row in rows]
    result = {
        "schema": "qvq_p32_benchmark_comparison_v1",
        "benchmark_valid": baseline["benchmark_valid"] and candidate["benchmark_valid"],
        "baseline": {"commit": args.baseline_commit, "path": str(args.baseline)},
        "candidate": {"commit": args.candidate_commit, "path": str(args.candidate)},
        "summary": _summary(speedups),
        "by_m": {
            key: _summary(values)
            for key, values in sorted(by_m.items(), key=lambda item: int(item[0]))
        },
        "by_shape": {key: _summary(values) for key, values in sorted(by_shape.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
