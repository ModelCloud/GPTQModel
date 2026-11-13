# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool, WarmupTask


pytestmark = [pytest.mark.cpu]


def test_wait_ignores_warmup_noop_completion_for_running_task():
    """A completed warmup no-op must not let wait() return early.

    ``_kick_worker_warmups`` enqueues a no-op directly on each accelerator
    worker so the worker runs its pending warmup. That no-op still travels
    through the worker's normal completion path, so it has to be accounted
    for exactly like a public ``submit`` or its completion decrements the
    in-flight counter that belongs to real work.

    ``mps`` is used only as an accelerator family label: no tensor or device
    operation is performed, so this runs on hosts without that backend.
    """
    warmup_started = threading.Event()
    release_warmup = threading.Event()
    task_started = threading.Event()
    release_task = threading.Event()

    def warmup(device, ctx):
        warmup_started.set()
        assert release_warmup.wait(timeout=5)

    def task():
        task_started.set()
        assert release_task.wait(timeout=5)
        return 42

    pool = DeviceThreadPool(
        devices=[torch.device("mps")],
        workers={"mps": 1},
        warmups={"mps": WarmupTask(warmup)},
        empty_cache_every_n=0,
    )
    waiter = None
    try:
        future = pool.submit("mps", task)

        # The warmup no-op runs first; releasing it lets the worker finish the
        # no-op and move on to the real task.
        assert warmup_started.wait(timeout=2)
        release_warmup.set()
        assert task_started.wait(timeout=2)

        # The no-op has completed by now, but the real task is still blocked.
        assert not future.done()
        assert pool._inflight["mps"] == 1

        waiter = threading.Thread(target=pool.wait, args=("mps",), daemon=True)
        waiter.start()
        time.sleep(0.2)
        assert waiter.is_alive(), "wait() returned before the real task finished"
        assert not future.done()

        release_task.set()
        assert future.result(timeout=2) == 42
        waiter.join(timeout=2)
        assert not waiter.is_alive()
    finally:
        release_warmup.set()
        release_task.set()
        pool.shutdown(wait=True)
