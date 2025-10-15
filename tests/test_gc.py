import gc
import threading
from queue import Queue

import pytest


torch = pytest.importorskip("torch", reason="requires PyTorch")


_THREAD_COUNT = 16
_ITERATIONS = 20_000
_ALLOCATION_BYTES = 1024 * 1024


def _worker(barrier: threading.Barrier, errors: Queue) -> None:
    try:
        barrier.wait()
        for _ in range(_ITERATIONS):
            tensor = torch.empty(_ALLOCATION_BYTES, dtype=torch.uint8)
            del tensor
            gc.collect()
    except Exception as exc:  # pragma: no cover - stress test safeguard
        errors.put(exc)


@pytest.mark.gc_stress
@pytest.mark.timeout(300)
def test_multithreaded_gc_collect():
    """Stress test that repeated gc.collect calls do not crash under threading."""
    barrier = threading.Barrier(_THREAD_COUNT)
    errors: Queue = Queue()

    threads = [
        threading.Thread(target=_worker, args=(barrier, errors), name=f"gc-worker-{i}")
        for i in range(_THREAD_COUNT)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    if not errors.empty():
        exc = errors.get()
        pytest.fail(f"GC stress worker raised: {exc}")
