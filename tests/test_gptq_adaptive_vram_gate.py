# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest
import torch


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the adaptive VRAM gate"),
]

MAX_PEAK_REGRESSION_PCT = 5.0
BASELINE_PATH = Path(__file__).parent / "fixtures" / "pr211_adaptive_memory_sm80.json"
PROFILE_PATH = Path(__file__).parents[1] / "scripts" / "profile_adaptive_memory.py"


def test_adaptive_quantization_peak_vram_matches_sm80_baseline():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROFILE_PATH),
            "--baseline-json",
            str(BASELINE_PATH),
            "--max-peak-regression-pct",
            str(MAX_PEAK_REGRESSION_PCT),
        ],
        cwd=PROFILE_PATH.parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    assert completed.returncode == 0, (
        f"Adaptive quantization VRAM profile failed with exit code {completed.returncode}."
    )
