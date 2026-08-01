#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""One-off benchmark: Llama-3.2-1B-Instruct GPTQ with imatrix v5 + nm-calibration mix.

Mixes the 22 Unsloth imatrix v5 rows with the first N rows of `nm-calibration:LLM`
and runs the same `test_llama3_2.py` quant/eval pipeline. Use `GPTQMODEL_MODEL_TEST_MODE=slow`
for a full-model quant run.
"""

import json
import sys
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
NM_ROWS = 512
OUTPUT_DIR = Path("/tmp/llama32_mixed_benchmark")


def _download_imatrix() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local = OUTPUT_DIR / "imatrix_v5.txt"
    if not local.exists():
        print(f"[data] Downloading imatrix v5 from {IMAXTRIX_URL} ...")
        urlretrieve(IMAXTRIX_URL, str(local))
    return local


def _build_mixed_dataset() -> Dataset:
    # imatrix v5: wrap each raw text snippet as a user turn.
    local = _download_imatrix()
    parts = [p.strip() for p in local.read_text(encoding="utf-8").split(IMAXTRIX_SEP)]
    imatrix_records = [
        {"messages": [{"role": "user", "content": text}]}
        for text in parts
        if text
    ]
    imatrix_ds = Dataset.from_list(imatrix_records)

    # nm-calibration: take first NM_ROWS rows and keep only the `messages` column
    # to match the imatrix schema.
    nm_ds = load_dataset(NM_CALIBRATION_PATH, name=NM_CALIBRATION_NAME, split="train")
    nm_ds = nm_ds.select(range(min(NM_ROWS, len(nm_ds))))
    if "messages" not in nm_ds.column_names:
        raise ValueError("nm-calibration dataset has no `messages` column")
    nm_ds = nm_ds.remove_columns([c for c in nm_ds.column_names if c != "messages"])

    return concatenate_datasets([imatrix_ds, nm_ds])


class TestLlama3_2Mixed(TestLlama3_2):
    """Run the Llama-3.2-1B gate with a mixed imatrix + nm-calibration dataset."""

    SAVE_PATH = "/tmp/llama3_2_mixed_gptq_saved_ckpt"
    DELETE_QUANTIZED_MODEL = False
    # Use all rows in the mixed dataset (overrides the default 512 cap).
    DATASET_SIZE = 0

    def load_dataset(self, tokenizer=None, rows: int = 0):
        ds = _build_mixed_dataset()
        if rows > 0:
            return ds.select(range(min(rows, len(ds))))
        return ds

    def check_results(self, task_results: Dict[str, Any]) -> None:
        summary_path = OUTPUT_DIR / "mixed_task_results.json"
        summary_path.write_text(json.dumps(task_results, indent=2, sort_keys=True), encoding="utf-8")
        print("\n=== mixed (imatrix 22 + nm 512) task results ===")
        print(json.dumps(task_results, indent=2, sort_keys=True))
        print(f"\nWrote results to {summary_path}")


def main() -> int:
    test = TestLlama3_2Mixed()
    test.quantize_and_evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
