#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Sweep group size vs activation-ordering settings on Llama-3.2-1B.

Tests the hypothesis that act_group_aware (GAR) overfits the calibration
Hessian for small group sizes, causing gp64/gp32 to underperform gp128.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests/models"))

from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
from test_llama3_2 import TestLlama3_2  # noqa: E402

IMAXTRIX_URL = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw"
IMAXTRIX_SEP = "==========="
NM_CALIBRATION_PATH = "/monster/data/model/dataset/nm-calibration"
NM_CALIBRATION_NAME = "LLM"
OUTPUT_DIR = Path("/tmp/llama32_gar_sweep")


def _download_imatrix() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local = OUTPUT_DIR / "imatrix_v5.txt"
    if not local.exists():
        print(f"[data] Downloading imatrix v5 from {IMAXTRIX_URL} ...")
        urlretrieve(IMAXTRIX_URL, str(local))
    return local


def _build_imatrix_dataset() -> Dataset:
    local = _download_imatrix()
    parts = [p.strip() for p in local.read_text(encoding="utf-8").split(IMAXTRIX_SEP)]
    records = [{"messages": [{"role": "user", "content": text}]} for text in parts if text]
    return Dataset.from_list(records)


def _build_nm_dataset(start: int = 0, end: int | None = None) -> Dataset:
    nm_ds = load_dataset(NM_CALIBRATION_PATH, name=NM_CALIBRATION_NAME, split="train")
    if end is None:
        end = len(nm_ds)
    return nm_ds.select(range(start, min(end, len(nm_ds)))).remove_columns(["text"])


def _build_recipe() -> Dataset:
    imatrix = _build_imatrix_dataset()
    nm = _build_nm_dataset(0, 1024)
    return concatenate_datasets([imatrix, nm])


class _SweptTest(TestLlama3_2):
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 0
    _dataset: Dataset | None = None
    _recipe_label: str = ""
    _result: dict[str, Any] | None = None

    def load_dataset(self, tokenizer=None, rows: int = 0):
        if rows > 0:
            return self._dataset.select(range(min(rows, len(self._dataset))))
        return self._dataset

    def check_results(self, task_results: dict[str, Any]) -> None:
        self._result = dict(task_results)
        print(f"\n=== {self._recipe_label} task results ===")
        print(json.dumps(task_results, indent=2, sort_keys=True))


def _run_config(
    label: str,
    dataset: Dataset,
    group_size: int,
    act_group_aware: bool,
    desc_act: bool = False,
    static_groups: bool = False,
    damp_percent: float | None = None,
) -> dict[str, Any]:
    save_path = f"/tmp/llama32_gar_{label}_g{group_size}"

    class _RecipeTest(_SweptTest):
        GROUP_SIZE = group_size
        SAVE_PATH = save_path
        ACT_GROUP_AWARE = act_group_aware
        DESC_ACT = desc_act
        _dataset = dataset
        _recipe_label = label

    if damp_percent is not None:
        _RecipeTest.DAMP_PERCENT = damp_percent
    if static_groups:
        _RecipeTest.STATIC_GROUPS = static_groups  # type: ignore[attr-defined]

    test = _RecipeTest()
    try:
        test.quantize_and_evaluate()
    except Exception as exc:
        print(f"[error] config {label} failed: {exc}")
        traceback.print_exc()
        return {"label": label, "error": str(exc)}

    # Record the effective act_group_aware; the small-group safeguard may have
    # changed it from the requested value during quantization.
    effective_act_group_aware = act_group_aware
    if getattr(test, "model", None) is not None and getattr(test.model, "quantize_config", None) is not None:
        effective_act_group_aware = test.model.quantize_config.act_group_aware

    return {
        "label": label,
        "group_size": group_size,
        "act_group_aware": effective_act_group_aware,
        "act_group_aware_requested": act_group_aware,
        "desc_act": desc_act,
        "static_groups": static_groups,
        "damp_percent": damp_percent,
        "save_path": save_path,
        "num_rows": len(dataset),
        "results": test._result or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=[128, 32],
        help="GPTQ group sizes to test (default: 128 32)",
    )
    parser.add_argument(
        "--act-group-aware",
        nargs="+",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=[True, False],
        help="act_group_aware values to test (default: True False)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "gar_sweep_results.json"),
        help="Path for the JSON summary",
    )
    args = parser.parse_args()

    dataset = _build_recipe()
    summary = []
    for g in args.group_sizes:
        for aga in args.act_group_aware:
            label = f"imatrix22_nm1024_aga{aga}_g{g}"
            print(f"\n{'=' * 60}\n[recipe] {label}\n{'=' * 60}")
            result = _run_config(
                label=label,
                dataset=dataset,
                group_size=g,
                act_group_aware=aga,
            )
            summary.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
