#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an immutable, deduplicated NM + YAQA calibration artifact.

The artifact is deliberately row-addressable and carries enough provenance to
bind both the source examples and their tokenizer-prepared representations.
YAQA wins a duplicate group because its rows were search-selected for
quantization coverage.  The emitted disjointness manifest audits the deduped
union against both D300 manifests and GSM8K Platinum, while also binding every
raw source slice used by the calibration-ablation launchers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_calibration_disjointness import (
    file_binding,
    index,
    rows_from_gsm8k,
    rows_from_jsonl,
    rows_from_parquet,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _normalize_user_text(messages: list[dict[str, str]]) -> str:
    text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text)
    text = "".join(character for character in text if character.isalnum() or character.isspace())
    return " ".join(text.split())


def _messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("calibration row must contain a non-empty messages list")
    normalized = []
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("calibration message must be a mapping")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if not role:
            raise ValueError("calibration message role must be non-empty")
        normalized.append({"role": role, "content": content})
    return normalized


@dataclass(frozen=True)
class PreparedRow:
    source_name: str
    source_path: str
    source_row: int
    source_priority: int
    messages: list[dict[str, str]]
    raw_example_sha256: str
    normalized_user_sha256: str
    prepared_example_sha256: str
    valid_input_tokens: int
    valid_fisher_output_token_samples: int

    def manifest_record(self, union_index: int | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "source_name": self.source_name,
            "source_path": self.source_path,
            "source_row": self.source_row,
            "raw_example_sha256": self.raw_example_sha256,
            "normalized_user_sha256": self.normalized_user_sha256,
            "prepared_example_sha256": self.prepared_example_sha256,
            "valid_input_tokens": self.valid_input_tokens,
            "valid_fisher_output_token_samples": self.valid_fisher_output_token_samples,
        }
        if union_index is not None:
            record["union_index"] = union_index
        return record


def _prepare_row(
    *,
    tokenizer: Any,
    source_name: str,
    source_path: Path,
    source_row: int,
    source_priority: int,
    messages: list[dict[str, str]],
) -> PreparedRow:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=True, return_attention_mask=True)
    input_ids = [int(value) for value in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded.get("attention_mask", [1] * len(input_ids))]
    if len(input_ids) != len(attention_mask):
        raise ValueError("prepared input IDs and attention mask have different lengths")
    valid_tokens = sum(bool(value) for value in attention_mask)
    if valid_tokens <= 10:
        raise ValueError(f"prepared calibration row {source_name}:{source_row} has only {valid_tokens} valid tokens")
    normalized_user = _normalize_user_text(messages)
    if not normalized_user:
        raise ValueError(f"calibration row {source_name}:{source_row} has no user content")
    return PreparedRow(
        source_name=source_name,
        source_path=str(source_path.resolve()),
        source_row=source_row,
        source_priority=source_priority,
        messages=messages,
        raw_example_sha256=_sha256_bytes(_canonical_json(messages)),
        normalized_user_sha256=_sha256_bytes(normalized_user.encode("utf-8")),
        prepared_example_sha256=_sha256_bytes(
            _canonical_json({"attention_mask": attention_mask, "input_ids": input_ids})
        ),
        valid_input_tokens=valid_tokens,
        # YAQA's current `valid_output_samples` counter is the number of valid
        # masked positions participating in its real-Fisher loss.
        valid_fisher_output_token_samples=valid_tokens,
    )


def _load_rows(
    *,
    tokenizer: Any,
    source_name: str,
    source_path: Path,
    row_start: int,
    rows: int,
    source_priority: int,
) -> list[PreparedRow]:
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(source_path.resolve())},
        split="train",
    )
    row_stop = row_start + rows
    if row_start < 0 or rows < 1 or row_stop > len(dataset):
        raise ValueError(f"slice [{row_start}, {row_stop}) exceeds {source_path} length {len(dataset)}")
    return [
        _prepare_row(
            tokenizer=tokenizer,
            source_name=source_name,
            source_path=source_path,
            source_row=index_value,
            source_priority=source_priority,
            messages=_messages(dataset[index_value]["messages"]),
        )
        for index_value in range(row_start, row_stop)
    ]


def deduplicate_rows(rows: list[PreparedRow]) -> tuple[list[PreparedRow], list[dict[str, Any]]]:
    groups: dict[str, list[PreparedRow]] = {}
    for row in rows:
        groups.setdefault(row.normalized_user_sha256, []).append(row)
    winners: list[PreparedRow] = []
    duplicate_groups: list[dict[str, Any]] = []
    for digest, members in groups.items():
        ranked = sorted(members, key=lambda item: (item.source_priority, item.source_row, item.raw_example_sha256))
        winner = ranked[0]
        winners.append(winner)
        if len(ranked) > 1:
            duplicate_groups.append(
                {
                    "normalized_user_sha256": digest,
                    "winner": winner.manifest_record(),
                    "members": [member.manifest_record() for member in ranked],
                }
            )
    winners.sort(key=lambda item: (item.source_priority, item.source_row, item.raw_example_sha256))
    duplicate_groups.sort(key=lambda item: item["normalized_user_sha256"])
    return winners, duplicate_groups


def _tokenizer_bindings(model_path: Path) -> list[dict[str, Any]]:
    bindings = []
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        if path.is_file():
            bindings.append(file_binding(path))
    return bindings


def _write_parquet(rows: list[PreparedRow], output: Path) -> None:
    records = []
    for union_index, row in enumerate(rows):
        record = row.manifest_record(union_index)
        record["messages"] = row.messages
        records.append(record)
    message_type = pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())]))
    schema = pa.schema(
        [
            pa.field("messages", message_type),
            pa.field("union_index", pa.int64()),
            pa.field("source_name", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_row", pa.int64()),
            pa.field("raw_example_sha256", pa.string()),
            pa.field("normalized_user_sha256", pa.string()),
            pa.field("prepared_example_sha256", pa.string()),
            pa.field("valid_input_tokens", pa.int64()),
            pa.field("valid_fisher_output_token_samples", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd", version="2.6", write_statistics=True)


def _source_summary(rows: list[PreparedRow], kept: list[PreparedRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for source_name in sorted({row.source_name for row in rows}):
        source_rows = [row for row in rows if row.source_name == source_name]
        kept_rows = [row for row in kept if row.source_name == source_name]
        summary[source_name] = {
            "selected_rows": len(source_rows),
            "independent_sequences": len(source_rows),
            "valid_input_tokens": sum(row.valid_input_tokens for row in source_rows),
            "valid_fisher_output_token_samples": sum(
                row.valid_fisher_output_token_samples for row in source_rows
            ),
            "rows_kept_after_dedup": len(kept_rows),
            "kept_valid_input_tokens": sum(row.valid_input_tokens for row in kept_rows),
            "kept_valid_fisher_output_token_samples": sum(
                row.valid_fisher_output_token_samples for row in kept_rows
            ),
        }
    return summary


def _disjointness_payload(
    *,
    union_path: Path,
    nm_path: Path,
    nm_start: int,
    nm_rows: int,
    yaqa_path: Path,
    yaqa_start: int,
    yaqa_rows: int,
    union_rows: int,
    d300_path: Path,
    d300_locked_path: Path,
) -> dict[str, Any]:
    calibration = index(rows_from_parquet(union_path, 0, union_rows), "calibration_union_v1")
    evaluations = {
        "d300": index(rows_from_jsonl(d300_path), "d300"),
        "d300_locked": index(rows_from_jsonl(d300_locked_path), "d300_locked"),
        "gsm8k_platinum": index(rows_from_gsm8k(), "gsm8k_platinum"),
    }
    overlaps: dict[str, list[dict[str, str]]] = {}
    for name, evaluation in evaluations.items():
        for digest in sorted(set(calibration) & set(evaluation)):
            overlaps[f"{name}:normalized:{digest}"] = calibration[digest] + evaluation[digest]

    nm_binding = file_binding(nm_path)
    nm_binding["slices"] = [
        {"row_start": nm_start, "rows": 128},
        {"row_start": nm_start, "rows": nm_rows},
    ]
    yaqa_binding = file_binding(yaqa_path)
    yaqa_binding["slices"] = [{"row_start": yaqa_start, "rows": yaqa_rows}]
    union_binding = file_binding(union_path)
    union_binding["slices"] = [{"row_start": 0, "rows": union_rows}]
    return {
        "schema": "qvq.calibration_disjointness.v2",
        "status": "pass" if not overlaps else "fail",
        "method": "deduped calibration_union_v1 user-turn audit; NFKC + casefold + whitespace/punctuation normalization; SHA256",
        "stream_reuse_policy": (
            "Cross-estimator reuse is intentional in this ablation. Internal duplicate checks apply to the "
            "deduplicated union artifact itself, not to the lifecycle-versus-YAQA stream pairing."
        ),
        "calibration_files": [str(path.resolve()) for path in (nm_path, yaqa_path, union_path)],
        "calibration_bindings": [nm_binding, yaqa_binding, union_binding],
        "calibration_rows": union_rows,
        "internal_duplicate_groups": {},
        "overlap_groups": overlaps,
        "evaluation_splits": list(evaluations),
        "evaluation_bindings": {
            "d300": file_binding(d300_path),
            "d300_locked": file_binding(d300_locked_path),
            "gsm8k_platinum": {
                "dataset": "madrylab/gsm8k-platinum",
                "config": "main",
                "split": "test",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--nm-row-start", type=int, default=0)
    parser.add_argument("--nm-rows", type=int, default=512)
    parser.add_argument("--yaqa", type=Path, required=True)
    parser.add_argument("--yaqa-row-start", type=int, default=0)
    parser.add_argument("--yaqa-rows", type=int, default=182)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--disjointness-output", type=Path, required=True)
    parser.add_argument("--d300", type=Path, required=True)
    parser.add_argument("--d300-locked", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True)
    yaqa_rows = _load_rows(
        tokenizer=tokenizer,
        source_name="yaqa",
        source_path=args.yaqa,
        row_start=args.yaqa_row_start,
        rows=args.yaqa_rows,
        source_priority=0,
    )
    nm_rows = _load_rows(
        tokenizer=tokenizer,
        source_name="nm",
        source_path=args.nm,
        row_start=args.nm_row_start,
        rows=args.nm_rows,
        source_priority=1,
    )
    source_rows = yaqa_rows + nm_rows
    union_rows, duplicate_groups = deduplicate_rows(source_rows)
    _write_parquet(union_rows, args.output)

    ordered_rows = [row.manifest_record(index_value) for index_value, row in enumerate(union_rows)]
    ordered_manifest_sha256 = _sha256_bytes(_canonical_json(ordered_rows))
    manifest = {
        "schema": "qvq.calibration_union.v1",
        "artifact_name": "calibration_union_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": file_binding(args.output),
        "tokenizer": {
            "model_path": str(args.model.resolve()),
            "class": tokenizer.__class__.__name__,
            "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
            "files": _tokenizer_bindings(args.model.resolve()),
        },
        "sources": {
            "nm": {
                **file_binding(args.nm),
                "split": "train",
                "row_start": args.nm_row_start,
                "rows": args.nm_rows,
            },
            "yaqa": {
                **file_binding(args.yaqa),
                "split": "train",
                "row_start": args.yaqa_row_start,
                "rows": args.yaqa_rows,
            },
        },
        "deduplication": {
            "key": "normalized user-turn SHA256",
            "normalization": "NFKC + casefold + whitespace/punctuation normalization",
            "precedence": ["yaqa", "nm"],
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_groups": duplicate_groups,
        },
        "coverage": {
            "before_dedup_rows": len(source_rows),
            "after_dedup_rows": len(union_rows),
            "independent_sequences": len(union_rows),
            "valid_input_tokens": sum(row.valid_input_tokens for row in union_rows),
            "valid_fisher_output_token_samples": sum(
                row.valid_fisher_output_token_samples for row in union_rows
            ),
            "by_source": _source_summary(source_rows, union_rows),
        },
        "ordered_calibration_manifest_sha256": ordered_manifest_sha256,
        "ordered_rows": ordered_rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    disjointness = _disjointness_payload(
        union_path=args.output,
        nm_path=args.nm,
        nm_start=args.nm_row_start,
        nm_rows=args.nm_rows,
        yaqa_path=args.yaqa,
        yaqa_start=args.yaqa_row_start,
        yaqa_rows=args.yaqa_rows,
        union_rows=len(union_rows),
        d300_path=args.d300,
        d300_locked_path=args.d300_locked,
    )
    args.disjointness_output.parent.mkdir(parents=True, exist_ok=True)
    args.disjointness_output.write_text(
        json.dumps(disjointness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": manifest["artifact"],
                "coverage": manifest["coverage"],
                "deduplication": {
                    "duplicate_group_count": len(duplicate_groups),
                    "ordered_calibration_manifest_sha256": ordered_manifest_sha256,
                },
                "disjointness_status": disjointness["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if disjointness["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
