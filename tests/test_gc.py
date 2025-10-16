import gc
import os
import threading
import traceback
from queue import Queue

import pytest

from gptqmodel.utils.safe import gc as safe_gc


torch = pytest.importorskip("torch", reason="requires PyTorch")


_THREAD_COUNT = min(4, max(2, (os.cpu_count() or 2)))
_ITERATIONS = 8
_ALLOCATION_BYTES = 32 * 1024
_BARRIER_TIMEOUT_S = 5
_JOIN_TIMEOUT_S = 30


def _worker(barrier: threading.Barrier, safe: bool, errors: Queue) -> None:
    try:
        barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        for _ in range(_ITERATIONS):
            t = torch.empty(_ALLOCATION_BYTES, dtype=torch.uint8)
            del t
            if safe:
                safe_gc.collect()
            else:
                gc.collect()
    except Exception:  # pragma: no cover - stress test safeguard
        # Preserve the traceback so the failing test shows context from worker threads.
        errors.put(traceback.format_exc())


@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_multithreaded_gc_collect_unsafe():
    """Stress test that repeated gc.collect calls do not crash under threading."""
    barrier = threading.Barrier(_THREAD_COUNT)
    errors: Queue = Queue()

    threads = [
        threading.Thread(target=_worker, args=(barrier, False, errors), name=f"gc-worker-{i}")
        for i in range(_THREAD_COUNT)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=_JOIN_TIMEOUT_S)
        if thread.is_alive():
            print("thread did not finish")
            pytest.fail(f"GC stress worker {thread.name} did not finish")

    if not errors.empty():
        failures = []
        while not errors.empty():
            failures.append(errors.get())
        pytest.fail("GC stress worker raised:\n" + "\n".join(failures))


@pytest.mark.timeout(30)
def test_multithreaded_gc_collect_safe():
    """Stress test that repeated gc.collect calls do not crash under threading."""
    barrier = threading.Barrier(_THREAD_COUNT)
    errors: Queue = Queue()

    threads = [
        threading.Thread(target=_worker, args=(barrier, True, errors), name=f"gc-worker-{i}")
        for i in range(_THREAD_COUNT)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=_JOIN_TIMEOUT_S)
        if thread.is_alive():
            pytest.fail(f"GC stress worker {thread.name} did not finish")

    if not errors.empty():
        failures = []
        while not errors.empty():
            failures.append(errors.get())
        pytest.fail("GC stress worker raised:\n" + "\n".join(failures))
