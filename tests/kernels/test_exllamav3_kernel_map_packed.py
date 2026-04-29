# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts/generate_exl3_kernel_map_packed.py"
PACKED_HEADER = REPO_ROOT / "gptqmodel_ext/exllamav3/quant/exl3_kernel_map_packed.cuh"


@pytest.mark.timeout(120)
def test_exllamav3_kernel_map_matches_exllamav3_original_legacy_header(tmp_path: Path):
    generated = tmp_path / "exl3_kernel_map_packed.cuh"
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--output", str(generated)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "validated legacy lookup from https://raw.githubusercontent.com/" in result.stdout
    assert generated.read_text() == PACKED_HEADER.read_text()
