# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import pytest
import torch
from test_checkpoint_quantization import assert_identical_tensors, run_guarded_driver


def test_guard_preserves_output(tmp_path):
    result = run_guarded_driver(
        [sys.executable, "-S", "-c", "print('completed', flush=True)"],
        tmp_path=tmp_path,
        mode="normal",
        timeout=10,
    )
    assert result.returncode == 0 and result.stdout == b"completed\n"


def test_guard_preserves_timeout_diagnostics(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded_driver(
            [
                sys.executable,
                "-S",
                "-c",
                "import time; print('started', flush=True); time.sleep(20)",
            ],
            tmp_path=tmp_path,
            mode="timeout",
            timeout=1,
        )
    assert (tmp_path / "timeout.stdout.log").read_bytes() == b"started\n"


def test_exact_comparison_rejects_changed_signed_zero():
    with pytest.raises(AssertionError):
        assert_identical_tensors(
            {"weight": torch.tensor([0.0])}, {"weight": torch.tensor([-0.0])}
        )
