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


def file_binding(path: Path) -> dict[str, Any]:
    """Describe a local input cryptographically for a run-bound manifest."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input manifest/file not found: {resolved}")
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "path": str(resolved),
        "sha256": hasher.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        return len(pd.read_parquet(path, columns=["messages"]))
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def rows_from_parquet(path: Path, row_start: int = 0, rows: int | None = None) -> Iterable[tuple[str, str]]:
    frame = pd.read_parquet(path)
    if "messages" not in frame:
        raise ValueError(f"{path} has no messages column")
    stop = len(frame) if rows is None else row_start + rows
    if row_start < 0 or stop > len(frame):
        raise ValueError(f"calibration slice [{row_start}, {stop}) exceeds {path} length {len(frame)}")
    for i, row in frame.iloc[row_start:stop].iterrows():
        yield f"{path}:{i}", question_text(row["messages"])


def rows_from_jsonl(path: Path, row_start: int = 0, rows: int | None = None) -> Iterable[tuple[str, str]]:
    if row_start < 0 or (rows is not None and rows < 1):
        raise ValueError(f"invalid row slice for {path}: start={row_start}, rows={rows}")
    seen = 0
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i < row_start:
                continue
            if rows is not None and i >= row_start + rows:
                break
            row = json.loads(line)
            yield f"{path}:{i}", question_text(row.get("messages", row.get("question", "")))
            seen += 1
    if rows is not None and seen < rows:
        raise ValueError(f"calibration slice [{row_start}, {row_start + rows}) exceeds {path} length")


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
    ap.add_argument("--calibration", action="append", default=[], type=Path, help="Calibration parquet/jsonl (repeatable)")
    ap.add_argument(
        "--calibration-slice",
        action="append",
        default=[],
        metavar="PATH:ROW_START:ROWS",
        help="Audit only a selected local slice; repeat for every preparation stream.",
    )
    ap.add_argument("--d300", type=Path, help="Divergence-300 evaluation JSONL")
    ap.add_argument(
        "--d300-locked",
        type=Path,
        help="Locked D300 evaluation JSONL; protected alongside --d300 when supplied.",
    )
    ap.add_argument("--gsm8k", action="store_true", help="Check local GSM8K Platinum test split")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not args.calibration and not args.calibration_slice:
        ap.error("at least one --calibration or --calibration-slice is required")

    calibration_entries: list[tuple[Path, int, int | None]] = [
        (path, 0, None) for path in args.calibration
    ]
    for raw in args.calibration_slice:
        try:
            raw_path, raw_start, raw_rows = raw.rsplit(":", 2)
            path = Path(raw_path)
            start, rows = int(raw_start), int(raw_rows)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid --calibration-slice `{raw}`; expected PATH:ROW_START:ROWS") from exc
        if start < 0 or rows < 1:
            raise ValueError(f"invalid calibration slice bounds in `{raw}`")
        calibration_entries.append((path, start, rows))

    cal: dict[str, list[dict[str, str]]] = {}
    for path, row_start, rows in calibration_entries:
        source_rows = (
            rows_from_parquet(path, row_start, rows)
            if path.suffix == ".parquet"
            else rows_from_jsonl(path, row_start, rows)
        )
        for key, values in index(source_rows, "calibration").items():
            cal.setdefault(key, []).extend(values)

    evaluations: dict[str, dict[str, list[dict[str, str]]]] = {}
    if args.d300:
        evaluations["d300"] = index(rows_from_jsonl(args.d300), "d300")
    if args.d300_locked:
        evaluations["d300_locked"] = index(rows_from_jsonl(args.d300_locked), "d300_locked")
    if args.gsm8k:
        evaluations["gsm8k_platinum"] = index(rows_from_gsm8k(), "gsm8k_platinum")

    internal = {k: v for k, v in cal.items() if len(v) > 1}
    overlaps: dict[str, list[dict[str, str]]] = {}
    for name, ev in evaluations.items():
        for key in sorted(set(cal) & set(ev)):
            # Include the digest in the key; otherwise multiple collisions in
            # one evaluation silently overwrite each other in the audit JSON.
            overlaps[f"{name}:normalized:{key}"] = cal[key] + ev[key]
    eval_names = list(evaluations)
    for i, left in enumerate(eval_names):
        for right in eval_names[i + 1 :]:
            for key in sorted(set(evaluations[left]) & set(evaluations[right])):
                overlaps[f"{left}_vs_{right}:normalized:{key}"] = (
                    evaluations[left][key] + evaluations[right][key]
                )

    binding_slices: dict[str, list[dict[str, int | None]]] = {}
    for path, row_start, rows in calibration_entries:
        if rows is None:
            rows = row_count(path) - row_start
        binding_slices.setdefault(str(path.expanduser().resolve()), []).append(
            {"row_start": row_start, "rows": rows}
        )
    calibration_bindings = []
    for path in sorted({path for path, _, _ in calibration_entries}, key=lambda item: str(item)):
        binding = file_binding(path)
        binding["slices"] = binding_slices[str(path.expanduser().resolve())]
        calibration_bindings.append(binding)
    evaluation_bindings: dict[str, Any] = {}
    if args.d300:
        evaluation_bindings["d300"] = file_binding(args.d300)
    if args.d300_locked:
        evaluation_bindings["d300_locked"] = file_binding(args.d300_locked)
    if args.gsm8k:
        # The GSM8K Platinum split is fetched from a pinned dataset identity;
        # there is no single local file to hash here, so record the exact
        # dataset/config/split contract for the evaluator to bind.
        evaluation_bindings["gsm8k_platinum"] = {
            "dataset": "madrylab/gsm8k-platinum",
            "config": "main",
            "split": "test",
        }

    payload = {
        "schema": "qvq.calibration_disjointness.v2",
        "calibration_files": [str(p) for p in sorted({path for path, _, _ in calibration_entries}, key=lambda item: str(item))],
        "calibration_bindings": calibration_bindings,
        "evaluation_splits": list(evaluations),
        "evaluation_bindings": evaluation_bindings,
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
