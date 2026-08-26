#!/usr/bin/env python3
"""Strict calibration/evaluation contamination preflight.

The check is deliberately question-only: assistant answers and chat wrappers are
not allowed to hide a duplicate prompt.  It emits a JSON manifest and exits 1
when any calibration prompt occurs in an evaluation split (or twice inside the
calibration mix).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def question_text(messages: Any) -> str:
    """Return user turns only, preserving order and excluding assistant text."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages
    if not isinstance(messages, (list, tuple)):
        return _text(messages)
    return "\n".join(_text(m) for m in messages if isinstance(m, dict) and m.get("role") == "user")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text)
    # Punctuation is formatting for this audit; retain letters/digits and spaces.
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows_from_parquet(path: Path) -> Iterable[tuple[str, str]]:
    frame = pd.read_parquet(path)
    if "messages" not in frame:
        raise ValueError(f"{path} has no messages column")
    for i, row in frame.iterrows():
        yield f"{path}:{i}", question_text(row["messages"])


def rows_from_jsonl(path: Path) -> Iterable[tuple[str, str]]:
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            yield f"{path}:{i}", question_text(row.get("messages", row.get("question", "")))


def rows_from_gsm8k() -> Iterable[tuple[str, str]]:
    from datasets import load_dataset

    # The dataset is expected to be pre-fetched by the evaluation environment;
    # reuse its cache and never request a fresh copy during a quant run.
    ds = load_dataset("madrylab/gsm8k-platinum", "main", split="test", download_mode="reuse_cache_if_exists")
    for i, row in enumerate(ds):
        yield f"madrylab/gsm8k-platinum:test:{i}", str(row["question"])


def index(rows: Iterable[tuple[str, str]], label: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for source, text in rows:
        if not text.strip():
            continue
        item = {"source": source, "label": label, "raw_sha256": digest(text), "normalized_sha256": digest(normalize(text))}
        out.setdefault(item["normalized_sha256"], []).append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", action="append", required=True, type=Path, help="Calibration parquet/jsonl (repeatable)")
    ap.add_argument("--d300", type=Path, help="Divergence-300 evaluation JSONL")
    ap.add_argument("--gsm8k", action="store_true", help="Check local GSM8K Platinum test split")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cal: dict[str, list[dict[str, str]]] = {}
    for path in args.calibration:
        source_rows = rows_from_parquet(path) if path.suffix == ".parquet" else rows_from_jsonl(path)
        for key, values in index(source_rows, "calibration").items():
            cal.setdefault(key, []).extend(values)

    evaluations: dict[str, dict[str, list[dict[str, str]]]] = {}
    if args.d300:
        evaluations["d300"] = index(rows_from_jsonl(args.d300), "d300")
    if args.gsm8k:
        evaluations["gsm8k_platinum"] = index(rows_from_gsm8k(), "gsm8k_platinum")

    internal = {k: v for k, v in cal.items() if len(v) > 1}
    overlaps: dict[str, list[dict[str, str]]] = {}
    for name, ev in evaluations.items():
        for key in sorted(set(cal) & set(ev)):
            overlaps[name + ":normalized"] = cal[key] + ev[key]
    eval_names = list(evaluations)
    for i, left in enumerate(eval_names):
        for right in eval_names[i + 1 :]:
            for key in sorted(set(evaluations[left]) & set(evaluations[right])):
                overlaps[f"{left}_vs_{right}:normalized"] = evaluations[left][key] + evaluations[right][key]

    payload = {
        "schema": "qvq.calibration_disjointness.v1",
        "calibration_files": [str(p) for p in args.calibration],
        "evaluation_splits": list(evaluations),
        "calibration_rows": sum(len(v) for v in cal.values()),
        "internal_duplicate_groups": internal,
        "overlap_groups": overlaps,
        "status": "pass" if not internal and not overlaps else "fail",
        "method": "user-turn-only; NFKC + casefold + whitespace/punctuation normalization; SHA256",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "calibration_rows", "evaluation_splits")}, indent=2))
    if overlaps:
        print(json.dumps({"overlap_groups": overlaps}, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
