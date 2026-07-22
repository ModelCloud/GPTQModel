#!/usr/bin/env python3
"""Evaluate the corrected GPU 6 INT4 EoRA adapter on ARC and GSM8K Platinum."""

from __future__ import annotations

import sys
from pathlib import Path


# Import the shared full-dataset Evalution workflow from this artifact.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from artifacts.qwen3_8b_dual_gptq_eora_20260722 import evaluate_gpu7_eora_fixed as runner  # noqa: E402


# Point every adapter and result artifact at the INT4 checkpoint on GPU 6.
MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512")
runner.MODEL = MODEL
runner.ADAPTER = MODEL / "eora-rank128-eighfix"
runner.OUTPUT_DIR = MODEL / "evalution_eora_eighfix"
runner.EXPECTED_UUID = "737e2423-874a-23a4-1126-dfbe3e77c294"
runner.EXPECTED_PCI_BUS = "00000000:DE:00.0"
runner.PHYSICAL_GPU = 6
runner.EVAL_BACKEND = runner.BACKEND.MARLIN


if __name__ == "__main__":
    runner.main()
