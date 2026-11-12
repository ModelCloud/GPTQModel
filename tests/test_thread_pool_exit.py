# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Building the device thread pool must not abort the interpreter at exit.

The workers call into CUDA. If they are still alive when the interpreter
finalizes, the process aborts with `terminate called without an active
exception` after every assertion has already passed, which turns a green pytest
run into exit code 134. The check runs in a subprocess because the failure mode
is process teardown.
"""

import subprocess
import sys
import textwrap

import pytest
import torch


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pool workers use CUDA")


@needs_cuda
def test_process_exits_cleanly_after_building_the_device_thread_pool():
    script = textwrap.dedent(
        """
        from gptqmodel import DEVICE_THREAD_POOL

        snapshot = DEVICE_THREAD_POOL._collect_state_snapshot()
        assert snapshot["total_workers"] > 0, snapshot
        print("pool built")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "pool built" in result.stdout
