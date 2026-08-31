#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build immutable NM2048/NM4096 corpora and weighted-Fisher configs.

All YAQA-weight and seed arms at a given NM coverage share one unique
deduplicated Parquet artifact. Importance weighting is config-driven and
applied to Sketch-B Gram contributions; no row is physically duplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_calibration_fisher_scaling import _build_one, _write_config
from scripts.build_calibration_union import _load_rows
from scripts.check_calibration_disjointness import file_binding

NM_COUNTS = (2048, 4096)
YAQA_WEIGHTS = (1.0, 2.0)
NM4096_FOLLOWUPS = (
    ("f13_yaqa182_nm4096_yaqa15x", 1.5, 0),
    ("f14_yaqa182_nm4096_yaqa3x", 3.0, 0),
    ("f15_yaqa182_nm4096_yaqa2x_seed1", 2.0, 1),
    ("f16_yaqa182_nm4096_yaqa2x_seed2", 2.0, 2),
)


def _weight_label(value: float) -> str:
    return f"{value:g}".replace(".", "")


def _weighted_config(base_config: dict[str, Any], *, yaqa_weight: float) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    payload["yaqa"]["source_weight_column"] = "source_name"
    payload["yaqa"]["source_weights"] = [["yaqa", yaqa_weight], ["nm", 1.0]]
    return payload


def _weighted_coverage(coverage: dict[str, Any], *, yaqa_weight: float) -> dict[str, Any]:
    by_source = coverage["by_source"]
    yaqa = by_source["yaqa"]
    nm = by_source["nm"]
    yaqa_sequences = int(yaqa["rows_kept_after_dedup"])
    nm_sequences = int(nm["rows_kept_after_dedup"])
    yaqa_tokens = int(yaqa["kept_valid_fisher_output_token_samples"])
    nm_tokens = int(nm["kept_valid_fisher_output_token_samples"])
    return {
        "unique_sequences": int(coverage["independent_sequences"]),
        "raw_valid_tokens": int(coverage["valid_fisher_output_token_samples"]),
        "effective_weighted_sequences": nm_sequences + yaqa_weight * yaqa_sequences,
        "effective_weighted_tokens": nm_tokens + yaqa_weight * yaqa_tokens,
        "YAQA_weight": yaqa_weight,
        "NM_weight": 1.0,
        "by_source": {
            "yaqa": {"unique_sequences": yaqa_sequences, "raw_valid_tokens": yaqa_tokens},
            "nm": {"unique_sequences": nm_sequences, "raw_valid_tokens": nm_tokens},
        },
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
    nm_rows = _load_rows(
        tokenizer=tokenizer,
        source_name="nm",
        source_path=args.nm,
        row_start=0,
        rows=max(NM_COUNTS),
        source_priority=1,
    )
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    corpora = []
    arms = []
    for nm_count in NM_COUNTS:
        key = f"yaqa182_nm{nm_count}"
        alpha1 = _weighted_config(base_config, yaqa_weight=1.0)
        corpus = _build_one(
            key=key,
            yaqa_rows=yaqa_rows,
            nm_rows=nm_rows[:nm_count],
            selection={"strategy": "prefix", "row_start": 0, "rows": nm_count},
            output_dir=args.output_dir,
            model=args.model,
            tokenizer=tokenizer,
            nm=args.nm,
            yaqa=args.yaqa,
            d300=args.d300,
            d300_locked=args.d300_locked,
            base_config=alpha1,
            config_dir=args.config_dir,
        )
        corpora.append(corpus)
        for yaqa_weight in YAQA_WEIGHTS:
            arm_id = f"f{9 + len(arms)}_yaqa182_nm{nm_count}_yaqa{int(yaqa_weight)}x"
            if yaqa_weight == 1.0:
                config = corpus["config"]
            else:
                config_path = (
                    args.config_dir
                    / f"llama32_1b_fisher_composition_{key}_yaqa{int(yaqa_weight)}x.json"
                )
                config = _write_config(
                    base_config=_weighted_config(base_config, yaqa_weight=yaqa_weight),
                    output=config_path,
                    minimum_sequences=corpus["coverage"]["independent_sequences"],
                    seed=0,
                )
            arms.append(
                {
                    "arm_id": arm_id,
                    "corpus_key": key,
                    "artifact": corpus["artifact"],
                    "manifest": corpus["manifest"],
                    "disjointness": corpus["disjointness"],
                    "config": config,
                    "source_weights": {"yaqa": yaqa_weight, "nm": 1.0},
                    "coverage": _weighted_coverage(corpus["coverage"], yaqa_weight=yaqa_weight),
                }
            )

    nm4096 = next(corpus for corpus in corpora if corpus["key"] == "yaqa182_nm4096")
    for arm_id, yaqa_weight, seed in NM4096_FOLLOWUPS:
        label = _weight_label(yaqa_weight)
        suffix = f"_seed{seed}" if seed else ""
        config_path = (
            args.config_dir
            / f"llama32_1b_fisher_composition_yaqa182_nm4096_yaqa{label}x{suffix}.json"
        )
        config = _write_config(
            base_config=_weighted_config(base_config, yaqa_weight=yaqa_weight),
            output=config_path,
            minimum_sequences=nm4096["coverage"]["independent_sequences"],
            seed=seed,
        )
        arms.append(
            {
                "arm_id": arm_id,
                "corpus_key": nm4096["key"],
                "artifact": nm4096["artifact"],
                "manifest": nm4096["manifest"],
                "disjointness": nm4096["disjointness"],
                "config": config,
                "source_weights": {"yaqa": yaqa_weight, "nm": 1.0},
                "seed": seed,
                "coverage": _weighted_coverage(
                    nm4096["coverage"], yaqa_weight=yaqa_weight
                ),
            }
        )

    registry = {
        "schema": "qvq.calibration_fisher_composition_registry.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "One immutable deduplicated corpus per NM coverage; source weights multiply each "
            "independent sequence's input/output Sketch-B Gram contribution without row duplication."
        ),
        "model": str(args.model.resolve()),
        "sources": {"yaqa": file_binding(args.yaqa), "nm": file_binding(args.nm)},
        "corpora": corpora,
        "arms": arms,
    }
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
