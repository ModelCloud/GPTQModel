#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build immutable YAQA+NM Fisher-scaling corpora and compact provenance.

The full ordered-row manifests stay beside the generated Parquet artifacts so
the Git report remains compact.  The committed registry binds every artifact,
full manifest, disjointness contract, selected source rows, and tokenizer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_calibration_union import (
    PreparedRow,
    _canonical_json,
    _disjointness_payload,
    _load_rows,
    _sha256_bytes,
    _source_summary,
    _tokenizer_bindings,
    _write_parquet,
    deduplicate_rows,
)
from scripts.check_calibration_disjointness import file_binding

PREFIX_COUNTS = (128, 256, 512, 1024, 10000)
RANDOM_SEED = 20260830


def select_random_token_matched_rows(
    rows: list[PreparedRow],
    *,
    target_tokens: int,
    seed: int,
) -> list[PreparedRow]:
    """Choose a deterministic random prefix closest to a token target."""

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected: list[PreparedRow] = []
    total = 0
    for index_value in indices:
        candidate = rows[index_value]
        candidate_total = total + candidate.valid_fisher_output_token_samples
        if total < target_tokens:
            if candidate_total >= target_tokens:
                if abs(candidate_total - target_tokens) <= abs(total - target_tokens):
                    selected.append(candidate)
                break
            selected.append(candidate)
            total = candidate_total
        else:
            break
    return sorted(selected, key=lambda row: row.source_row)


def _selected_rows_sha256(rows: list[PreparedRow]) -> str:
    return _sha256_bytes(_canonical_json([row.source_row for row in rows]))


def _write_config(
    *,
    base_config: dict[str, Any],
    output: Path,
    minimum_sequences: int,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    payload["yaqa"]["minimum_sequences"] = minimum_sequences
    payload["yaqa"]["seed"] = seed
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_binding(output)


def _build_one(
    *,
    key: str,
    yaqa_rows: list[PreparedRow],
    nm_rows: list[PreparedRow],
    selection: dict[str, Any],
    output_dir: Path,
    model: Path,
    tokenizer: Any,
    nm: Path,
    yaqa: Path,
    d300: Path,
    d300_locked: Path,
    base_config: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    source_rows = yaqa_rows + nm_rows
    union_rows, duplicate_groups = deduplicate_rows(source_rows)
    artifact = output_dir / f"{key}.parquet"
    manifest_path = output_dir / f"{key}.manifest.json"
    disjointness_path = output_dir / f"{key}.disjointness.json"
    _write_parquet(union_rows, artifact)

    ordered_rows = [row.manifest_record(index_value) for index_value, row in enumerate(union_rows)]
    ordered_manifest_sha256 = _sha256_bytes(_canonical_json(ordered_rows))
    manifest = {
        "schema": "qvq.calibration_fisher_scaling_corpus.v1",
        "artifact_name": key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": file_binding(artifact),
        "tokenizer": {
            "model_path": str(model.resolve()),
            "class": tokenizer.__class__.__name__,
            "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
            "files": _tokenizer_bindings(model.resolve()),
        },
        "sources": {
            "yaqa": {
                **file_binding(yaqa),
                "split": "train",
                "row_start": 0,
                "rows": len(yaqa_rows),
            },
            "nm": {
                **file_binding(nm),
                "split": "train",
                "selection": selection,
                "selected_rows": [row.source_row for row in nm_rows],
                "selected_rows_sha256": _selected_rows_sha256(nm_rows),
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
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    disjointness = _disjointness_payload(
        union_path=artifact,
        nm_path=nm,
        nm_start=0,
        nm_rows=len(nm_rows),
        yaqa_path=yaqa,
        yaqa_start=0,
        yaqa_rows=len(yaqa_rows),
        union_rows=len(union_rows),
        d300_path=d300,
        d300_locked_path=d300_locked,
    )
    disjointness["method"] = (
        f"{key} user-turn audit; NFKC + casefold + whitespace/punctuation normalization; SHA256"
    )
    for binding in disjointness["calibration_bindings"]:
        if binding.get("path") == str(nm.resolve()):
            binding["selected_rows_sha256"] = _selected_rows_sha256(nm_rows)
            binding["selection"] = selection
            # The runtime reads NM[0:128] only as the fixed lifecycle stream;
            # the selected Fisher rows are already bound inside the union.
            binding["slices"] = [{"row_start": 0, "rows": 128}]
    disjointness_path.write_text(json.dumps(disjointness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if disjointness["status"] != "pass":
        raise RuntimeError(f"disjointness failed for {key}: {disjointness_path}")

    config_path = config_dir / f"llama32_1b_fisher_scaling_{key}.json"
    config_binding = _write_config(
        base_config=base_config,
        output=config_path,
        minimum_sequences=len(union_rows),
        seed=0,
    )
    return {
        "key": key,
        "artifact": file_binding(artifact),
        "manifest": file_binding(manifest_path),
        "disjointness": file_binding(disjointness_path),
        "config": config_binding,
        "selection": selection,
        "selected_nm_rows": len(nm_rows),
        "selected_nm_rows_sha256": _selected_rows_sha256(nm_rows),
        "coverage": manifest["coverage"],
        "duplicate_group_count": len(duplicate_groups),
        "ordered_calibration_manifest_sha256": ordered_manifest_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--yaqa", type=Path, required=True)
    parser.add_argument("--yaqa-rows", type=int, default=182)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--d300", type=Path, required=True)
    parser.add_argument("--d300-locked", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True)
    yaqa_rows = _load_rows(
        tokenizer=tokenizer,
        source_name="yaqa",
        source_path=args.yaqa,
        row_start=0,
        rows=args.yaqa_rows,
        source_priority=0,
    )
    # NM-full is exactly the complete local train Parquet: 10,000 rows.
    nm_rows = _load_rows(
        tokenizer=tokenizer,
        source_name="nm",
        source_path=args.nm,
        row_start=0,
        rows=10000,
        source_priority=1,
    )
    target_tokens = sum(row.valid_fisher_output_token_samples for row in nm_rows[:512])
    random_rows = select_random_token_matched_rows(
        nm_rows,
        target_tokens=target_tokens,
        seed=RANDOM_SEED,
    )

    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for count in PREFIX_COUNTS:
        artifacts.append(
            _build_one(
                key=f"yaqa182_nm{count}",
                yaqa_rows=yaqa_rows,
                nm_rows=nm_rows[:count],
                selection={"strategy": "prefix", "row_start": 0, "rows": count},
                output_dir=args.output_dir,
                model=args.model,
                tokenizer=tokenizer,
                nm=args.nm,
                yaqa=args.yaqa,
                d300=args.d300,
                d300_locked=args.d300_locked,
                base_config=base_config,
                config_dir=args.config_dir,
            )
        )
    artifacts.append(
        _build_one(
            key=f"yaqa182_nmrandom_token{target_tokens}_seed{RANDOM_SEED}",
            yaqa_rows=yaqa_rows,
            nm_rows=random_rows,
            selection={
                "strategy": "deterministic_random_token_match",
                "candidate_rows": len(nm_rows),
                "seed": RANDOM_SEED,
                "target_tokens": target_tokens,
                "selected_tokens": sum(row.valid_fisher_output_token_samples for row in random_rows),
            },
            output_dir=args.output_dir,
            model=args.model,
            tokenizer=tokenizer,
            nm=args.nm,
            yaqa=args.yaqa,
            d300=args.d300,
            d300_locked=args.d300_locked,
            base_config=base_config,
            config_dir=args.config_dir,
        )
    )

    full = next(item for item in artifacts if item["key"] == "yaqa182_nm10000")
    seed1_path = args.config_dir / "llama32_1b_fisher_scaling_yaqa182_nm10000_seed1.json"
    seed1_binding = _write_config(
        base_config=base_config,
        output=seed1_path,
        minimum_sequences=full["coverage"]["independent_sequences"],
        seed=1,
    )
    registry = {
        "schema": "qvq.calibration_fisher_scaling_registry.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "nm_full_definition": {
            "path": str(args.nm.resolve()),
            "split": "train",
            "rows": 10000,
            "sha256": file_binding(args.nm)["sha256"],
        },
        "yaqa_definition": {
            "path": str(args.yaqa.resolve()),
            "split": "train",
            "rows": len(yaqa_rows),
            "sha256": file_binding(args.yaqa)["sha256"],
        },
        "random_token_match_target": target_tokens,
        "artifacts": artifacts,
        "full_replica_seed1_config": seed1_binding,
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
