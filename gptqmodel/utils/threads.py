# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import concurrent.futures as cf
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .device_telemetry import capture_device_telemetry


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def default_thread_count() -> int:
    """Return the default intra-op pool size: ``min(8, cpu_count())``.

    A container can advertise hundreds of CPUs while sharing a much smaller
    physical host. Torch sizes its intra-op pool from that advertised count,
    so an uncapped process creates a thread per CPU and spends its time in the
    OpenMP barrier instead of doing work.
    """

    return min(8, os.cpu_count() or 1)


def resolve_thread_count(n: int | None = None) -> int:
    """Resolve the intra-op pool size, honouring an explicit ``OMP_NUM_THREADS``.

    An explicit argument always wins. Otherwise a positive integer
    ``OMP_NUM_THREADS`` is used, and the capped host default otherwise.
    """

    if n is not None:
        return max(1, int(n))

    explicit = os.environ.get("OMP_NUM_THREADS")
    if explicit is not None:
        try:
            parsed = int(explicit)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return default_thread_count()


def apply_thread_env(n: int) -> None:
    """Set the numerical-library thread environment defaults for ``n``.

    ``setdefault`` keeps a caller's explicit values: an explicit
    ``OMP_NUM_THREADS`` (or ``MKL_NUM_THREADS`` and friends) is never
    overwritten. The OpenMP/MKL/OpenBLAS runtimes read these variables when
    they are loaded, so callers that need them to win must call this before
    importing numpy or torch.
    """

    for name in _THREAD_ENV_VARS:
        os.environ.setdefault(name, str(n))
    os.environ.setdefault("OMP_NESTED", "FALSE")


def cap_torch_threads(n: int | None = None, *, interop: int | None = None) -> int:
    """Cap Torch's intra-op (and optionally inter-op) thread pools.

    This is an opt-in helper for tests, CLI tools and scripts; importing
    :mod:`gptqmodel` does not call it, because silently changing a user's
    global thread count on import would change library numerics and
    performance without consent.

    ``n`` defaults to an explicit ``OMP_NUM_THREADS`` when set, otherwise
    ``min(8, cpu_count())``. The environment defaults are applied with
    ``setdefault`` so an explicit caller value is preserved.

    Torch's own pool is set explicitly because the environment is not
    authoritative once the runtime has loaded. ``torch.set_num_interop_threads``
    raises if parallel work has already started, so it is best-effort; the
    intra-op cap is always applied. Returns the applied intra-op count.
    """

    threads = resolve_thread_count(n)
    apply_thread_env(threads)

    import torch

    torch.set_num_threads(threads)

    if interop is None:
        interop = min(2, threads)
    try:
        torch.set_num_interop_threads(max(1, int(interop)))
    except RuntimeError:
        # Inter-op threads cannot be changed after the first parallel region.
        pass

    return threads


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

        fut = self._exec.submit(capture_device_telemetry(_runner))
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
        self._q.put(capture_device_telemetry(fn))

    def join(self, timeout=None):
        self._q.join()

    def shutdown(self):
        self._q.put(None)
        self._t.join()
