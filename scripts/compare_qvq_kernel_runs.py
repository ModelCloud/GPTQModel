# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Print and optionally save a matched A/B QVQ inference-kernel comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = _args()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    baseline_rows = {
        (row["shape"], row["k"], row["n"], row["bits"], row["m"]): row
        for row in baseline["rows"]
        if row["path"].startswith("qvq_cuda")
    }
    candidate_rows = {
        (row["shape"], row["k"], row["n"], row["bits"], row["m"]): row
        for row in candidate["rows"]
        if row["path"].startswith("qvq_cuda")
    }
    keys = sorted(candidate_rows)
    rows = []
    for key in keys:
        if key not in baseline_rows:
            raise ValueError(f"candidate row has no baseline match: {key}")
        old = baseline_rows[key]
        new = candidate_rows[key]
        rows.append(
            {
                "shape": key[0],
                "bits": key[3],
                "m": key[4],
                "baseline_ms": old["median_ms"],
                "candidate_ms": new["median_ms"],
                "speedup": old["median_ms"] / new["median_ms"],
                "max_abs": new["max_abs"],
            }
        )

    headers = ("Shape", "Bits", "M", "Baseline ms", "Candidate ms", "Speedup", "Max abs")
    rendered = [
        (row["shape"], str(row["bits"]), str(row["m"]), f"{row['baseline_ms']:.4f}",
         f"{row['candidate_ms']:.4f}", f"{row['speedup']:.3f}x", f"{row['max_abs']:.6g}")
        for row in rows
    ]
    widths = [max(len(header), *(len(row[index]) for row in rendered)) for index, header in enumerate(headers)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    print("|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, widths)) + "|")
    print(border)
    for row in rendered:
        print("|" + "|".join(f" {value:<{width}} " for value, width in zip(row, widths)) + "|")
    print(border)

    if args.output_md:
        lines = [
            f"Baseline `{baseline['commit']}` vs candidate `{candidate['commit']}`.",
            "",
            "| Shape | Bits | M | Baseline ms | Candidate ms | Speedup | Max abs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines.extend(
            f"| {row['shape']} | W{row['bits']} | {row['m']} | {row['baseline_ms']:.4f} | "
            f"{row['candidate_ms']:.4f} | {row['speedup']:.3f}x | {row['max_abs']:.6g} |"
            for row in rows
        )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
