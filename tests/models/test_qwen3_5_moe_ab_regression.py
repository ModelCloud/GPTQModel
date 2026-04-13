# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestQwen3_5MoeABRegression(unittest.TestCase):
    """Opt-in real-model A/B benchmark against a baseline worktree."""

    RUN_ENV = "GPTQMODEL_RUN_QWEN35_MOE_AB"
    BASELINE_ENV = "GPTQMODEL_BASELINE_ROOT"

    def test_qwen3_5_moe_two_layer_ab_benchmark(self) -> None:
        if os.environ.get(self.RUN_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
            self.skipTest(f"Set {self.RUN_ENV}=1 to run the real Qwen 3.5 MoE A/B benchmark.")

        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "benchmark_qwen35_moe_ab.py"
        baseline_root = Path(os.environ.get(self.BASELINE_ENV, "/root/gptqmodel-main")).resolve()
        if not baseline_root.exists():
            self.skipTest(f"Baseline repo root does not exist: {baseline_root}")

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
        env["PYTHON_GIL"] = "0"
        env["DEBUG"] = "1"

        with tempfile.TemporaryDirectory(prefix="qwen35_moe_ab_regression_") as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--current-root",
                    str(repo_root),
                    "--baseline-root",
                    str(baseline_root),
                    "--output-dir",
                    temp_dir,
                    "--current-vram-strategy",
                    "dense_home_moe_balanced",
                    "--baseline-vram-strategy",
                    "balanced",
                ],
                check=True,
                cwd=repo_root,
                env=env,
            )
