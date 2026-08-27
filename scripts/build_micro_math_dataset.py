#!/usr/bin/env python3
"""Build the small, held-out math proxy set used by post-quantization metrics.

The source is the local GSM8K *train* split.  Rows are selected deterministically
after removing normalized user-question matches from the calibration streams,
both Divergence-300 manifests, and the local GSM8K test split.  The resulting
JSONL is an evaluation artifact, never a quantization input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_calibration_disjointness import normalize, question_text


DEFAULT_TRAIN = Path("/monster/data/model/dataset/gsm8k/main/train-00000-of-00001.parquet")
DEFAULT_TEST = Path("/monster/data/model/dataset/gsm8k/main/test-00000-of-00001.parquet")
DEFAULT_CALIBRATION = (
    Path("/monster/data/model/dataset/nm-calibration/llm.parquet"),
    Path("dataset/calibration_mix_500k_llama3.2_1b/calibration.parquet"),
)
DEFAULT_D300 = (
    Path("/root/qvq-data/divergence300-v1/divergence300-development.jsonl"),
    Path("/root/qvq-data/divergence300-v1/divergence300-locked.jsonl"),
)


def _iter_questions(path: Path) -> Iterable[str]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        if "question" in frame:
            for value in frame["question"]:
                yield str(value)
        elif "messages" in frame:
            for value in frame["messages"]:
                yield question_text(value)
        else:
            raise ValueError(f"{path} has neither question nor messages column")
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield question_text(row.get("messages", row.get("question", "")))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    excluded: set[str] = set()
    exclusion_files = list(args.calibration) + list(args.d300) + ([args.test] if args.test else [])
    for path in exclusion_files:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        excluded.update(normalize(text) for text in _iter_questions(path) if text.strip())

    frame = pd.read_parquet(source, columns=["question", "answer"])
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for source_row, row in frame.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])
        normalized = normalize(question)
        if not normalized or normalized in excluded or normalized in seen:
            skipped += 1
            continue
        seen.add(normalized)
        selected.append(
            {
                "id": f"gsm8k-main-train:{int(source_row)}",
                "source": str(source),
                "source_row": int(source_row),
                "question": question,
                "answer": answer,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "normalized_question_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            }
        )
        if len(selected) >= args.rows:
            break
    if len(selected) < args.rows:
        raise RuntimeError(f"only selected {len(selected)} rows, requested {args.rows}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "qvq.micro_math.v1",
        "dataset": {
            "path": str(output),
            "sha256": _file_sha256(output),
            "rows": len(selected),
            "source": "local mirror of openai/gsm8k main train split",
            "source_file": str(source),
            "source_sha256": _file_sha256(source),
        },
        "method": "question-only NFKC+casefold+whitespace/punctuation normalization; excluded calibration, D300 development+locked, and local GSM8K test rows",
        "excluded_files": [str(path.expanduser().resolve()) for path in exclusion_files],
        "skipped_source_rows": skipped,
        "status": "pass",
    }
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--calibration", type=Path, action="append", default=list(DEFAULT_CALIBRATION))
    parser.add_argument("--d300", type=Path, action="append", default=list(DEFAULT_D300))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("dataset/micro_math_llama3.2_1b.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/experiments/micro-math-disjointness.json"))
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
