# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import concurrent.futures as cf
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


class AsyncManager:
    """Single-queue async offloader. Submit only callables (fn or lambda)."""
    def __init__(self, name="asyncmanager-", threads: int = 1):
        assert threads > 0
        self._exec = ThreadPoolExecutor(max_workers=threads, thread_name_prefix=name)
        self._lock = threading.Lock()
        self._futures = set()         # all in-flight futures
        self._last_future = None

    def _discard_future(self, f: cf.Future) -> None:
        with self._lock:
            self._futures.discard(f)

    def submit(self, fn):
        """Submit a callable (function or lambda). Returns a Future."""
        if not callable(fn):
            raise TypeError("AsyncOffloader.submit expects a callable")

        def _runner():
            try:
                return fn()
            except Exception:
                traceback.print_exc()
                raise  # propagate to Future so .result() fails

        fut = self._exec.submit(_runner)
        fut.add_done_callback(self._discard_future)
        with self._lock:
            self._futures.add(fut)
            self._last_future = fut
        return fut

    def join(self, timeout=None, future: cf.Future = None):
        """
        Wait for tasks to complete.
          - If `future` is given, wait for that specific one.
          - Else, wait for all currently in-flight tasks.
        Respects `timeout` as a total budget (seconds).
        """
        deadline = None if timeout is None else (time.time() + timeout)

        if future is not None:
            # Wait only for the specified future
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            future.result(timeout=remaining)
            return

        # Wait for all in-flight tasks
        while True:
            with self._lock:
                pending = {f for f in self._futures if not f.done()}
            if not pending:
                return
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            if remaining == 0.0:
                # Give a final short poll so we raise a TimeoutError consistently
                remaining = 0.0
            done, not_done = cf.wait(pending, timeout=remaining, return_when=cf.ALL_COMPLETED)
            if not_done:
                raise TimeoutError(f"{len(not_done)} task(s) not finished before timeout")

    def shutdown(self, wait=True, cancel_pending=False):
        """
        Shut down the executor.
          - If `cancel_pending` is True, attempts to cancel futures that haven't started.
          - If `wait` is True, waits for all in-flight tasks first.
        """
        if cancel_pending:
            with self._lock:
                for f in list(self._futures):
                    f.cancel()
        if wait:
            self.join()  # wait for remaining tasks
        self._exec.shutdown(wait=wait)


class SerialWorker:
    def __init__(self, name="serial-worker"):
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._loop, name=name, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            fn = self._q.get()
            if fn is None:
                break
            try:
                fn()
            except Exception:
                traceback.print_exc()
            finally:
                self._q.task_done()

    def submit(self, fn):
        if not callable(fn):
            raise TypeError("submit expects a callable")
        self._q.put(fn)

    def join(self, timeout=None):
        self._q.join()

    def shutdown(self):
        self._q.put(None)
        self._t.join()
