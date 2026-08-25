#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Build disjoint development and locked Divergence-300 prompt manifests.

The source snapshots must already exist locally.  This script deliberately
does not download data so benchmark construction cannot silently move to a
new Hub revision between runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SOURCE_COUNTS = {
    "terminal_bench_2_1": 40,
    "swe_bench_verified": 100,
    "matharena_2025_2026": 60,
    "multi_if_non_english": 50,
    "longbench_v2": 50,
}
SPLITS = ("development", "locked")
DEFAULT_SEED = "qvq-divergence300-v1-20260825"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser


def _revision(path: Path) -> str:
    trees = sorted((path / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(trees) != 1 or len(trees[0].stem) != 40:
        raise RuntimeError(f"expected exactly one pinned Hugging Face tree in {path}, got {trees}")
    return trees[0].stem


def _stable_order(rows: list[dict[str, Any]], *, seed: str, source: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        identity = str(row["source_id"])
        return hashlib.sha256(f"{seed}\0{source}\0{identity}".encode()).hexdigest()

    return sorted(rows, key=key)


def _terminal_rows(root: Path) -> list[dict[str, Any]]:
    source = root / "terminal-bench-2.1"
    revision = _revision(source)
    rows = []
    for path in sorted(source.glob("tasks/*/instruction.md")):
        task = path.parent.name
        rows.append(
            {
                "source_group": "terminal_bench_2_1",
                "source_repo": "harborframework/terminal-bench-2.1",
                "source_revision": revision,
                "source_id": task,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You are working as an autonomous coding agent in a terminal environment. "
                            "Complete the following task carefully.\n\n" + path.read_text(encoding="utf-8").strip()
                        ),
                    }
                ],
            }
        )
    return rows


def _swe_rows(root: Path) -> list[dict[str, Any]]:
    source = root / "swe-bench-verified"
    revision = _revision(source)
    parquet = next(source.glob("data/*.parquet"))
    rows = []
    for row in pq.read_table(parquet, columns=["instance_id", "repo", "problem_statement"]).to_pylist():
        rows.append(
            {
                "source_group": "swe_bench_verified",
                "source_repo": "princeton-nlp/SWE-bench_Verified",
                "source_revision": revision,
                "source_id": row["instance_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Repository: {row['repo']}\n\nResolve the following software issue. "
                            "Explain the necessary code changes before presenting the patch.\n\n"
                            f"{row['problem_statement'].strip()}"
                        ),
                    }
                ],
            }
        )
    return rows


def _math_rows(root: Path) -> list[dict[str, Any]]:
    datasets = (
        ("matharena-aime-2025", "MathArena/aime_2025"),
        ("matharena-aime-2026", "MathArena/aime_2026"),
        ("matharena-hmmt-feb-2025", "MathArena/hmmt_feb_2025"),
        ("matharena-hmmt-nov-2025", "MathArena/hmmt_nov_2025"),
    )
    rows = []
    for directory, repo in datasets:
        source = root / directory
        revision = _revision(source)
        parquet = next(source.glob("data/*.parquet"))
        for row in pq.read_table(parquet, columns=["problem_idx", "problem"]).to_pylist():
            rows.append(
                {
                    "source_group": "matharena_2025_2026",
                    "source_repo": repo,
                    "source_revision": revision,
                    "source_id": f"{repo}:{row['problem_idx']}",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Solve the following competition mathematics problem step by step, then give "
                                f"the final answer clearly.\n\n{row['problem'].strip()}"
                            ),
                        }
                    ],
                }
            )
    return rows


def _multi_if_rows(root: Path) -> list[dict[str, Any]]:
    source = root / "multi-if"
    revision = _revision(source)
    rows = []
    with (source / "multiIF_20241018.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["language"] == "English" or row["turn_index"] != "0":
                continue
            message = json.loads(row["turn_1_prompt"])
            rows.append(
                {
                    "source_group": "multi_if_non_english",
                    "source_repo": "facebook/Multi-IF",
                    "source_revision": revision,
                    "source_id": row["key"],
                    "language": row["language"],
                    "messages": [message],
                }
            )
    return rows


def _longbench_rows(root: Path) -> list[dict[str, Any]]:
    source = root / "longbench-v2"
    revision = _revision(source)
    payload = json.loads((source / "data.json").read_text(encoding="utf-8"))
    rows = []
    for row in payload:
        choices = "\n".join(f"{letter}. {row[f'choice_{letter}']}" for letter in "ABCD")
        rows.append(
            {
                "source_group": "longbench_v2",
                "source_repo": "zai-org/LongBench-v2",
                "source_revision": revision,
                "source_id": row["_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Read the context and answer the multiple-choice question. Explain your reasoning "
                            "and finish with the option letter.\n\n"
                            f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\n{choices}"
                        ),
                    }
                ],
            }
        )
    return rows


def _prompt_hash(row: dict[str, Any]) -> str:
    canonical = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_split(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = []
    lines = []
    for index, row in enumerate(rows):
        prompt_hash = _prompt_hash(row)
        hashes.append(prompt_hash)
        lines.append(json.dumps({"manifest_index": index, "prompt_sha256": prompt_hash, **row}, ensure_ascii=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    content_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "path": str(output.resolve()),
        "rows": len(rows),
        "sha256": content_hash,
        "prompt_sha256": hashes,
        "source_counts": dict(sorted(Counter(row["source_group"] for row in rows).items())),
    }


def main() -> int:
    args = build_parser().parse_args()
    root = args.source_root.expanduser().resolve()
    output_targets = [
        args.output_dir / f"divergence300-{split}.jsonl" for split in SPLITS
    ] + [args.output_dir / "divergence300-manifest.json"]
    existing_targets = [path for path in output_targets if path.exists()]
    if existing_targets:
        raise FileExistsError(f"refusing to overwrite existing Divergence-300 artifacts: {existing_targets}")
    builders = {
        "terminal_bench_2_1": _terminal_rows,
        "swe_bench_verified": _swe_rows,
        "matharena_2025_2026": _math_rows,
        "multi_if_non_english": _multi_if_rows,
        "longbench_v2": _longbench_rows,
    }
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_revisions: dict[str, set[str]] = {}
    for group, count in SOURCE_COUNTS.items():
        candidates = _stable_order(builders[group](root), seed=args.seed, source=group)
        required = count * len(SPLITS)
        if len(candidates) < required:
            raise RuntimeError(f"{group} has {len(candidates)} candidates; {required} are required")
        for split_index, split in enumerate(SPLITS):
            begin = split_index * count
            selected[split].extend(candidates[begin : begin + count])
        source_revisions[group] = {row["source_revision"] for row in candidates}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifests = {
        split: _write_split(args.output_dir / f"divergence300-{split}.jsonl", selected[split])
        for split in SPLITS
    }
    development_hashes = set(split_manifests["development"]["prompt_sha256"])
    locked_hashes = set(split_manifests["locked"]["prompt_sha256"])
    if development_hashes & locked_hashes:
        raise RuntimeError("development and locked prompt manifests are not content-disjoint")
    manifest = {
        "schema": "qvq.divergence300.prompt_manifest.v1",
        "seed": args.seed,
        "source_counts_per_split": SOURCE_COUNTS,
        "source_revisions": {key: sorted(value) for key, value in sorted(source_revisions.items())},
        "splits": split_manifests,
        "content_disjoint": True,
        "deep_swe_note": (
            "The Unsloth DeepSWE subset is not public; SWE-bench Verified is the pinned public software-issue proxy."
        ),
    }
    manifest_path = args.output_dir / "divergence300-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "manifest": str(manifest_path.resolve()),
        "splits": {
            split: {key: value for key, value in payload.items() if key != "prompt_sha256"}
            for split, payload in split_manifests.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
