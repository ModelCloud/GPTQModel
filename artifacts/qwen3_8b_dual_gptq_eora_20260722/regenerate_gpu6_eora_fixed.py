#!/usr/bin/env python3
"""Regenerate the GPU 6 INT4 EoRA adapter through the stable eigensolver runner."""

from __future__ import annotations

import sys
from pathlib import Path


# Import the shared regeneration workflow from this reproducibility artifact.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from artifacts.qwen3_8b_dual_gptq_eora_20260722 import regenerate_gpu7_eora_fixed as runner  # noqa: E402


# Keep the original adapter untouched and write the corrected factors separately.
MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512")
runner.MODEL = MODEL
runner.OLD_ADAPTER = MODEL / "eora-rank128"
runner.FIXED_ADAPTER = MODEL / "eora-rank128-eighfix"
runner.OUTPUT = MODEL / "eora_eighfix_validation.json"
runner.EXPECTED_UUID = "737e2423-874a-23a4-1126-dfbe3e77c294"
runner.EXPECTED_PCI_BUS = "00000000:DE:00.0"
runner.PHYSICAL_GPU = 6


if __name__ == "__main__":
    runner.main()
