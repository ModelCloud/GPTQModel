#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Sweep Llama-3.2-1B GPTQ calibration recipes and compare post-quant scores.

Recipes include:
- nm-calibration first 1024 rows
- Unsloth imatrix v5 22 rows
- imatrix + nm_1024
- imatrix + each 128-row chunk of nm_1024 (first, second, ... eighth)

Each variant is fully quantized and evaluated on gsm8k_platinum_cot and arc_challenge
(slow/full mode) so the scores are directly comparable to the previous 512-row runs.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict
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
OUTPUT_DIR = Path("/tmp/llama32_calibration_sweep")


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
    records = [
        {"messages": [{"role": "user", "content": text}]}
        for text in parts
        if text
    ]
    return Dataset.from_list(records)


def _build_nm_dataset(start: int = 0, end: int | None = None) -> Dataset:
    nm_ds = load_dataset(NM_CALIBRATION_PATH, name=NM_CALIBRATION_NAME, split="train")
    if end is None:
        end = len(nm_ds)
    return nm_ds.select(range(start, min(end, len(nm_ds)))).remove_columns(["text"])


def _combine(ds_a: Dataset, ds_b: Dataset) -> Dataset:
    return concatenate_datasets([ds_a, ds_b])


def _build_recipes() -> list[tuple[str, Dataset]]:
    imatrix = _build_imatrix_dataset()
    recipes: list[tuple[str, Dataset]] = [
        ("nm_1024", _build_nm_dataset(0, 1024)),
        ("imatrix_22", imatrix),
        ("imatrix_22_plus_nm_1024", _combine(imatrix, _build_nm_dataset(0, 1024))),
    ]
    for chunk_idx in range(8):
        start = chunk_idx * 128
        end = start + 128
        nm_chunk = _build_nm_dataset(start, end)
        recipes.append((f"imatrix_22_plus_nm_{start}_{end}", _combine(imatrix, nm_chunk)))
    return recipes


class _SweptTest(TestLlama3_2):
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 0
    _dataset: Dataset | None = None
    _recipe_label: str = ""
    _result: Dict[str, Any] | None = None

    def load_dataset(self, tokenizer=None, rows: int = 0):
        if rows > 0:
            return self._dataset.select(range(min(rows, len(self._dataset))))
        return self._dataset

    def check_results(self, task_results: Dict[str, Any]) -> None:
        self._result = dict(task_results)
        print(f"\n=== {self._recipe_label} task results ===")
        print(json.dumps(task_results, indent=2, sort_keys=True))


def _run_recipe(label: str, dataset: Dataset, group_size: int = 128) -> Dict[str, Any]:
    save_path = f"/tmp/llama32_sweep_{label}_g{group_size}_gptq"

    class _RecipeTest(_SweptTest):
        GROUP_SIZE = group_size
        SAVE_PATH = save_path
        _dataset = dataset
        _recipe_label = label

    test = _RecipeTest()
    try:
        test.quantize_and_evaluate()
    except Exception as exc:
        print(f"[error] recipe {label} failed: {exc}")
        traceback.print_exc()
        return {"label": label, "error": str(exc)}

    return {
        "label": label,
        "group_size": group_size,
        "save_path": save_path,
        "num_rows": len(dataset),
        "results": test._result or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipes",
        nargs="+",
        default=None,
        help="Run only the named recipes (default: all)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "sweep_results.json"),
        help="Path for the JSON summary",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GPTQ group size (default: 128)",
    )
    args = parser.parse_args()

    all_recipes = _build_recipes()
    if args.recipes:
        selected = {name: ds for name, ds in all_recipes if name in args.recipes}
        missing = set(args.recipes) - set(selected)
        if missing:
            print(f"[warn] Unknown recipes: {sorted(missing)}")
        recipes = [(name, selected[name]) for name in args.recipes if name in selected]
    else:
        recipes = all_recipes

    summary = []
    for label, dataset in recipes:
        print(f"\n{'=' * 60}\n[recipe] {label} g{args.group_size} ({len(dataset)} rows)\n{'=' * 60}")
        summary.append(_run_recipe(label, dataset, group_size=args.group_size))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== sweep summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
