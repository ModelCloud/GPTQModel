# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A worker task error must propagate through its future, never kill the process."""

import os
import subprocess
import sys
import textwrap

import pytest


pytestmark = [pytest.mark.cpu]


_SCRIPT = textwrap.dedent(
    """
    import atexit
    import pathlib
    import sys

    import torch

    assert sys._is_gil_enabled() is False, "the GIL must be disabled for this test"

    from gptqmodel.utils.threadx import DeviceThreadPool, WarmupTask

    marker = pathlib.Path(sys.argv[1])
    atexit.register(lambda: marker.write_text("cleanup"))

    pool = DeviceThreadPool(
        devices=[torch.device("cpu")], workers={"cpu": 1}, empty_cache_every_n=0
    )

    def boom():
        raise ValueError("task failed")

    future = pool.submit("cpu", boom)
    try:
        future.result(timeout=10)
    except ValueError as exc:
        assert "task failed" in str(exc)
    else:
        raise SystemExit("task exception was not raised")

    # The worker and the pool remain usable after one task fails.
    assert pool.submit("cpu", lambda: 7).result(timeout=10) == 7
    pool.shutdown(wait=True)

    def warmup(device, ctx):
        raise RuntimeError("warmup failed")

    warm_pool = DeviceThreadPool(
        devices=[torch.device("cpu")],
        workers={"cpu": 1},
        warmups={"cpu": WarmupTask(warmup)},
        empty_cache_every_n=0,
    )
    warm_future = warm_pool.submit("cpu", lambda: 1)
    try:
        warm_future.result(timeout=10)
    except RuntimeError as exc:
        assert "warmup failed" in str(exc)
    else:
        raise SystemExit("warmup exception was not raised")

    # The shared warmup state caches the error; later tasks fail instead of
    # hanging or exiting the interpreter.
    cached_future = warm_pool.submit("cpu", lambda: 2)
    try:
        cached_future.result(timeout=10)
    except RuntimeError as exc:
        assert "warmup failed" in str(exc)
    else:
        raise SystemExit("cached warmup exception was not raised")
    warm_pool.shutdown(wait=True)

    print("SURVIVED")
    """
)


def test_task_and_warmup_failures_do_not_exit_the_process(tmp_path):
    if not hasattr(sys, "_is_gil_enabled"):
        pytest.skip("Python runtime does not expose _is_gil_enabled; free-threading build required")
    if sys._is_gil_enabled():
        pytest.skip("GIL is enabled - run with PYTHON_GIL=0 to exercise worker error propagation")

    marker = tmp_path / "cleanup"
    env = dict(os.environ)
    env["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(marker)],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )

    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "SURVIVED" in result.stdout
    # `os._exit` skips this handler; a normal unwind runs it.
    assert marker.read_text(encoding="utf-8") == "cleanup"
