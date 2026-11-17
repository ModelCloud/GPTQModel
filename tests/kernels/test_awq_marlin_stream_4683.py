# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ Marlin determinism across CUDA streams (#4683).

``gptq_marlin.cu`` spins on, advances and resets the ``int32`` counters in the
workspace it is handed (``locks_off = blockIdx.x``). ``AwqMarlinLinear`` and
``MarlinLinear`` used to hand every launch the same per-module buffer, so two
streams serving one module consumed each other's counters: the launches either
deadlocked in ``barrier_acquire`` or folded a partial sum into the output. The
issue reports the visible half of that (an occasional wrong 64x64 tile and a
PPL that moves between runs); the deadlock is the same defect with no timing
luck. ``QQQLinear._split_k_locks`` allocates per call for this reason (#4665).

The two-stream check runs in a child process because the unfixed code
deadlocks inside the kernel, which cannot be torn down from the owning process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin_awq import marlin_import_exception
from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name("awq_marlin_two_stream_driver.py")


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the AWQ Marlin stream determinism test"
    if marlin_import_exception is not None:
        return f"AWQ Marlin kernel unavailable: {marlin_import_exception}"
    if not marlin_runtime_available(torch.float16):
        return marlin_runtime_error(torch.float16)
    return None


@pytest.mark.cuda
def test_two_streams_on_one_module_complete_and_agree():
    """Two streams driving one module must both finish with their own result."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep),
    }
    # The driver self-reports a deadlock after 120 s; the outer limit only
    # covers a wedged interpreter, and keeps pytest from hanging.
    try:
        result = subprocess.run(
            [sys.executable, str(DRIVER)],
            capture_output=True,
            text=True,
            timeout=420,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "the two-stream AWQ Marlin driver did not exit: the launches "
            f"deadlocked on the shared lock workspace (#4683)\n{exc.stdout or ''}"
        )

    if result.returncode != 0:
        pytest.fail(
            "concurrent AWQ Marlin launches on one module are not deterministic "
            f"(driver exit {result.returncode}) (#4683)\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}"
        )
    assert "OK" in result.stdout, result.stdout


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lock_workspace_is_private_per_stream():
    """Each stream gets its own zeroed counters; one stream keeps reusing its own."""

    from gptqmodel.utils.marlin import marlin_stream_workspace

    class _Module:
        def __init__(self, workspace):
            self.workspace = workspace

    module = _Module(torch.zeros(128, dtype=torch.int32, device="cuda"))
    default_workspace = marlin_stream_workspace(module)
    assert marlin_stream_workspace(module) is default_workspace

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        side_workspace = marlin_stream_workspace(module)
        assert side_workspace is not default_workspace
        assert side_workspace.data_ptr() != default_workspace.data_ptr()
        assert side_workspace.dtype is torch.int32
        assert side_workspace.device == default_workspace.device
        assert side_workspace.numel() == default_workspace.numel()
        assert not bool(side_workspace.any())
        # A second call on the same stream must not allocate again.
        assert marlin_stream_workspace(module) is side_workspace

    # Back on the default stream the original buffer is still the one in use.
    assert marlin_stream_workspace(module) is default_workspace
