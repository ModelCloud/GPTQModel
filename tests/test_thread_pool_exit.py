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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pool workers use CUDA")


@needs_cuda
def test_process_exits_cleanly_after_building_the_device_thread_pool():
    script = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from gptqmodel import get_device_thread_pool

        gate = Barrier(8)

        def get_pool(_):
            gate.wait(timeout=10)
            return get_device_thread_pool()

        with ThreadPoolExecutor(max_workers=8) as executor:
            pools = list(executor.map(get_pool, range(8)))
        assert all(pool is pools[0] for pool in pools)
        snapshot = pools[0]._collect_state_snapshot()
        assert snapshot["total_workers"] > 0, snapshot
        print("pool built")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "pool built" in result.stdout


def test_concurrent_first_call_builds_and_registers_one_pool(monkeypatch):
    import gptqmodel

    class FakePool:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, *, wait):
            self.shutdown_calls.append(wait)

    pool = FakePool()
    started = threading.Event()
    release = threading.Event()
    gate = threading.Barrier(8)
    builds = []
    registrations = []

    def build():
        builds.append(1)
        started.set()
        assert release.wait(timeout=5)
        return pool

    def call():
        gate.wait(timeout=5)
        return gptqmodel.get_device_thread_pool()

    monkeypatch.setattr(gptqmodel, "_DEVICE_THREAD_POOL", None)
    monkeypatch.setattr(gptqmodel, "_build_device_thread_pool", build)
    monkeypatch.setattr(gptqmodel.atexit, "register", lambda *args: registrations.append(args))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(call) for _ in range(8)]
        assert started.wait(timeout=5)
        # Give all first callers time to enter the builder or wait on its lock.
        time.sleep(0.05)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert all(result is pool for result in results)
    assert len(builds) == 1
    assert registrations == [(gptqmodel._shutdown_device_thread_pool, pool)]
    # Cleanup must target the registered pool even if the global is replaced.
    monkeypatch.setattr(gptqmodel, "_DEVICE_THREAD_POOL", None)
    callback, registered_pool = registrations[0]
    callback(registered_pool)
    assert pool.shutdown_calls == [True]
