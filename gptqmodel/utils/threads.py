import threading
import traceback
from concurrent.futures import ThreadPoolExecutor


class AsyncManager:
    """Single-worker async offloader. Submit only callables (fn or lambda)."""
    def __init__(self, name="asyncmanager-", threads: int = 1):
        assert threads > 0
        self._exec = ThreadPoolExecutor(max_workers=threads, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._last_future = None

    def submit(self, fn):
        """Submit a callable (function or lambda)."""
        if not callable(fn):
            raise TypeError("AsyncOffloader.submit expects a callable")

        def _runner():
            try:
                return fn()
            except Exception:
                traceback.print_exc()
                return None

        with self._lock:
            fut = self._exec.submit(_runner)
            self._last_future = fut
            return fut

    def join(self, timeout=None):
        """Wait for the last submitted task to complete."""
        with self._lock:
            fut = self._last_future
        if fut is not None:
            fut.result(timeout=timeout)

    def shutdown(self, wait=True):
        self._exec.shutdown(wait=wait)


