#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""One-off benchmark: Llama-3.2-1B-Instruct GPTQ using the 22-row Unsloth imatrix v5 dataset.

Compares post-quant scores against the default `nm-calibration:LLM` run performed by
`tests/models/test_llama3_2.py`. Set `GPTQMODEL_MODEL_TEST_MODE=slow` for a full-model
quant run.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).parent.parent
# Bring the repo root (for `tests.*` package imports) and the model-test directory
# (for the `test_llama3_2` module) onto the path.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests/models"))

from datasets import Dataset  # noqa: E402
from test_llama3_2 import TestLlama3_2  # noqa: E402

IMAXTRIX_URL = "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw"
IMAXTRIX_SEP = "==========="
OUTPUT_DIR = Path("/tmp/llama32_imatrix_benchmark")


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
    # Wrap each raw text snippet as a single user turn so the model's chat template is applied
    # during calibration, matching the scanner's `--apply-chat-template` path.
    records = [
        {"messages": [{"role": "user", "content": text}]}
        for text in parts
        if text
    ]
    return Dataset.from_list(records)


class TestLlama3_2Imatrix(TestLlama3_2):
    """Run the Llama-3.2-1B gate with the Unsloth imatrix v5 calibration subset."""

    SAVE_PATH = "/tmp/llama3_2_imatrix_gptq_saved_ckpt"
    DELETE_QUANTIZED_MODEL = False

    def load_dataset(self, tokenizer=None, rows: int = 0):
        ds = _build_imatrix_dataset()
        if rows > 0:
            return ds.select(range(min(rows, len(ds))))
        return ds

    def check_results(self, task_results: Dict[str, Any]) -> None:
        """Print the raw scores and write them to disk without failing on threshold diffs."""

        summary_path = OUTPUT_DIR / "imatrix_task_results.json"
        summary_path.write_text(json.dumps(task_results, indent=2, sort_keys=True), encoding="utf-8")
        print("\n=== imatrix v5 task results ===")
        print(json.dumps(task_results, indent=2, sort_keys=True))
        print(f"\nWrote results to {summary_path}")


def main() -> int:
    test = TestLlama3_2Imatrix()
    test.quantize_and_evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
