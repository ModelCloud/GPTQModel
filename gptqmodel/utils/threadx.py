# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import threadpoolctl as tctl

from .. import DEBUG_ON
from ..utils.ctx import ctx
from ..utils.logger import setup_logger


log = setup_logger()

# Debug logging is very chatty and can alter timings subtly in tests.
# We gate all extra diagnostics behind the DEBUG env (1/true/yes/on).

# DeviceLike allows ergonomic call sites: 'cuda:0', 0, torch.device('cuda', 0), etc.
DeviceLike = Union[str, int, torch.device]


# --------------------------- Backend availability helpers ---------------------------
# We keep these helpers small and side-effect free—only feature checks—so we can
# query once and rely on final results without redundant availability if-ladders.

def _mps_available() -> bool:
    return (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


# --- HARD COPIES of original empty_cache callables (never auto-switched) ---
# IMPORTANT: Do NOT “optimize” these by directly calling torch.*.empty_cache.
# We intentionally capture a snapshot of the original functions to defend against
# later code mutating those attributes to a no-op. The janitor will prefer the
# *live* attribute if callable (so monkeypatching works), but falls back to these
# hard copies if the live attr is missing or non-callable.
TORCH_CUDA_EMPTY_CACHE: Optional[Callable[[], None]] = None
TORCH_XPU_EMPTY_CACHE: Optional[Callable[[], None]] = None
TORCH_MPS_EMPTY_CACHE: Optional[Callable[[], None]] = None

try:
    TORCH_CUDA_EMPTY_CACHE = getattr(torch.cuda, "empty_cache", None) if hasattr(torch, "cuda") else None
    if TORCH_CUDA_EMPTY_CACHE is not None and not callable(TORCH_CUDA_EMPTY_CACHE):
        TORCH_CUDA_EMPTY_CACHE = None
except Exception:
    # If introspection fails, we keep the hard copy as None.
    TORCH_CUDA_EMPTY_CACHE = None

try:
    TORCH_XPU_EMPTY_CACHE = getattr(torch.xpu, "empty_cache", None) if hasattr(torch, "xpu") else None
    if TORCH_XPU_EMPTY_CACHE is not None and not callable(TORCH_XPU_EMPTY_CACHE):
        TORCH_XPU_EMPTY_CACHE = None
except Exception:
    TORCH_XPU_EMPTY_CACHE = None

try:
    TORCH_MPS_EMPTY_CACHE = getattr(torch.mps, "empty_cache", None) if hasattr(torch, "mps") else None
    if TORCH_MPS_EMPTY_CACHE is not None and not callable(TORCH_MPS_EMPTY_CACHE):
        TORCH_MPS_EMPTY_CACHE = None
except Exception:
    TORCH_MPS_EMPTY_CACHE = None


# --------------------------- Device coercion & context helpers ---------------------------

def _coerce_device(d: DeviceLike) -> torch.device:
    """
    Convert a DeviceLike into a concrete torch.device. For integers, we
    interpret as accelerator indices if present, otherwise map to CPU/MPS.
    """
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        if torch.cuda.is_available():
            return torch.device("cuda", d)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu", d)
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    # Accept strings like 'cuda:0', 'xpu:1', 'cpu', 'mps'
    return torch.device(d)


@contextlib.contextmanager
def _device_ctx(dev: torch.device):
    """
    Set the caller thread’s *current* device while running a task so handles/streams
    line up correctly. For CUDA/XPU we set the per-thread current device; CPU/MPS
    do not require pinning here.
    """
    if dev.type == "cuda":
        with torch.cuda.device(dev.index):
            yield
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        with torch.xpu.device(dev.index):
            yield
    else:
        yield


def _activate_thread_device(dev: torch.device):
    """
    Pin the worker thread to its device once, before entering its main loop.
    CUDA/XPU require per-thread device activation for correct handle usage.
    """
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index)
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_device(dev.index)
    # mps/cpu: nothing to pin


def _pop_public_kwarg(kwargs: Dict[str, Any], public_name: str, private_name: str):
    if private_name in kwargs:
        raise TypeError(f"'{private_name}' is no longer supported; use '{public_name}'")
    if public_name in kwargs:
        return kwargs.pop(public_name)
    return None


# --------------------------- Read-Write Lock (writer-preference) ---------------------------
# We implement a writer-preference RWLock. Multiple readers may hold the lock,
# but when a writer is waiting we block new readers, ensuring GC (writer) can
# eventually acquire exclusivity even under task pressure.

class _RWLock:
    """
    Reader-writer lock with writer preference.

    - Multiple readers may hold the lock simultaneously.
    - A single writer holds exclusivity.
    - When a writer is waiting, new readers will block.
    - Writer is re-entrant for its owning thread.
    """
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer: Optional[int] = None  # thread id that owns write
        self._writer_depth = 0
        self._writers_waiting = 0

    # --- Write (exclusive) ---
    def acquire_write(self):
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:  # Re-entrant writer
                self._writer_depth += 1
                return
            self._writers_waiting += 1
            try:
                while self._writer is not None or self._readers > 0:
                    self._cond.wait()
                self._writer = me
                self._writer_depth = 1
            finally:
                self._writers_waiting -= 1

    def release_write(self):
        me = threading.get_ident()
        with self._cond:
            if self._writer != me:
                raise RuntimeError("release_write called by non-owner")
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer = None
                self._cond.notify_all()

    @contextlib.contextmanager
    def writer(self):
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    # --- Read (shared) ---
    def acquire_read(self):
        me = threading.get_ident()
        with self._cond:
            # The writer may re-enter as a reader; this keeps invariants simple
            # for code that wants to read while already holding write.
            if self._writer == me:
                self._readers += 1
                return
            # If a writer is waiting, block new readers to give it priority.
            while self._writer is not None or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            if self._readers <= 0:
                raise RuntimeError("release_read without acquire_read")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    @contextlib.contextmanager
    def reader(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()


class _LockGroup(contextlib.AbstractContextManager):
    """
    Acquire multiple device **write** locks in deterministic order to avoid deadlocks.
    Helpful for GC passes or any multi-device exclusive operation.
    """
    def __init__(self, ordered_pairs: List[tuple[str, _RWLock]]):
        self._pairs = ordered_pairs

    def __enter__(self):
        for name, lk in self._pairs:
            if DEBUG_ON: log.debug(f"_LockGroup: acquiring write lock for {name}")
            lk.acquire_write()
            if DEBUG_ON: log.debug(f"_LockGroup: acquired write lock for {name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, lk in reversed(self._pairs):
            if DEBUG_ON: log.debug(f"_LockGroup: releasing write lock for {name}")
            lk.release_write()
        return False


class _ReadLockGroup(contextlib.AbstractContextManager):
    """
    Acquire multiple device **read** locks in deterministic order.
    Useful for multi-device snapshots or read-only, consistent views.
    """
    def __init__(self, ordered_pairs: List[tuple[str, _RWLock]]):
        self._pairs = ordered_pairs

    def __enter__(self):
        for name, lk in self._pairs:
            if DEBUG_ON: log.debug(f"_ReadLockGroup: acquiring read lock for {name}")
            lk.acquire_read()
            if DEBUG_ON: log.debug(f"_ReadLockGroup: acquired read lock for {name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, lk in reversed(self._pairs):
            if DEBUG_ON: log.debug(f"_ReadLockGroup: releasing read lock for {name}")
            lk.release_read()
        return False


class _WaitAndLock(contextlib.AbstractContextManager):
    """
    Context manager returned by pool.wait(scope, lock=True).
    On enter: acquires writer locks over the scope in canonical order,
    which inherently waits for in-flight readers (tasks) to drain.
    On exit: releases locks.
    """
    def __init__(self, pairs: List[tuple[str, _RWLock]]):
        self._pairs = pairs
        self._group = _LockGroup(pairs)

    def __enter__(self):
        return self._group.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._group.__exit__(exc_type, exc, tb)


# --------------------------- Worker Thread ---------------------------
# Each worker is bound to a specific device and runs a single thread. Tasks are
# executed under the device’s read lock; GC acquires the writer lock to keep
# memory management steps from interleaving with tasks.

class _DeviceWorker:
    """
    Single worker thread bound to one device.

    Queue entries: (is_task: bool, fn, args, kwargs, future)
    - is_task=False is a sentinel to exit the thread loop.
    - Tasks run within a device-scoped reader lock to prevent interleaving
      with GC passes (which need a write lock).
    """
    def __init__(
        self,
        device: torch.device,
        rwlock: _RWLock,
        on_task_finished: Callable[[str], None],
        on_worker_exit: Callable[[str, "_DeviceWorker"], None],
        name: Optional[str] = None,
        inference_mode: bool = False,
    ):
        self.device = device
        self.rwlock = rwlock
        self._on_task_finished = on_task_finished
        self._on_worker_exit = on_worker_exit

        self.key = f"{device.type}:{device.index}" if device.index is not None else device.type
        self.name = name or f"DPWorker-{self.key}"
        self._q: "queue.Queue[Tuple[bool, Callable[..., Any], tuple, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()

        self._inference_mode = inference_mode
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        if DEBUG_ON: log.debug(f"Spawned worker '{self.name}' for {self.key}")

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        """
        Enqueue a callable and return a Future that resolves with its result/exception.
        """
        fut = Future()
        self._q.put((True, fn, args, kwargs, fut))
        if DEBUG_ON: log.debug(f"{self.name}: task enqueued; qsize={self._q.qsize()}")
        return fut

    def stop(self):
        """
        Request thread exit by setting a sentinel work item. The run loop exits
        ASAP after receiving it.
        """
        self._stop.set()
        self._q.put((False, lambda: None, (), {}, Future()))
        if DEBUG_ON: log.debug(f"{self.name}: stop requested; sentinel queued")

    def join(self):
        """
        Join the worker thread; for graceful shutdowns and tests.
        """
        if DEBUG_ON: log.debug(f"{self.name}: joining thread")
        self._thread.join()

    def _run(self):
        """
        Main loop: pull tasks, set device context, execute, mark completion, and
        fulfill or fail the future. Completion is accounted BEFORE resolving the
        future to make stats() deterministic even under test interleavings.

        Workers default to inference mode for throughput but individual tasks
        may override via `inference_mode`.
        """
        _activate_thread_device(self.device)
        with tctl.threadpool_limits(limits=1):
            while True:
                is_task, fn, args, kwargs, fut = self._q.get()
                try:
                    if not is_task:
                        if DEBUG_ON: log.debug(f"{self.name}: received sentinel; exiting")
                        break
                    if self._stop.is_set():
                        # Pool is stopping; skip executing queued work to allow fast shutdown.
                        if DEBUG_ON:
                            log.debug(f"{self.name}: dropping task during shutdown; qsize={self._q.qsize()}")
                        self._on_task_finished(self.key)
                        fut.cancel()
                        continue
                    if DEBUG_ON: log.debug(f"{self.name}: task begin; qsize={self._q.qsize()}")

                    stream = kwargs.pop("cuda_stream", None)
                    override_inference = _pop_public_kwarg(
                        kwargs, "inference_mode", "_threadx_inference_mode"
                    )
                    use_inference = self._inference_mode if override_inference is None else bool(override_inference)

                    # Tasks take a **read** lock so janitor's write lock can't interleave
                    with ctx(self.rwlock.reader(), _device_ctx(self.device)):
                        inference_ctx = torch.inference_mode() if use_inference else contextlib.nullcontext()
                        with inference_ctx:
                            if stream is not None and self.device.type == "cuda":
                                with torch.cuda.stream(stream):
                                    result = fn(*args, **kwargs)
                            else:
                                result = fn(*args, **kwargs)
                    # Counters must be updated before resolving futures to prevent
                    # tests reading stats mid-transition and seeing stale totals.
                    self._on_task_finished(self.key)
                    if not fut.cancelled():
                        fut.set_result(result)
                    if DEBUG_ON: log.debug(f"{self.name}: task done")
                except BaseException as exc:
                    # Even on exception we must decrement inflight and update totals.
                    self._on_task_finished(self.key)
                    if not fut.cancelled():
                        fut.set_exception(exc)
                    if DEBUG_ON: log.debug(f"{self.name}: task exception: {exc!r}")
                finally:
                    self._q.task_done()
        try:
            self._on_worker_exit(self.key, self)
        finally:
            if DEBUG_ON: log.debug(f"{self.name}: exited")


class _SyncWorker:
    """Fallback worker that executes tasks synchronously when threads are unsafe."""

    def __init__(
        self,
        *,
        key: str,
        device: torch.device,
        rwlock: _RWLock,
        on_task_finished: Callable[[str], None],
        on_worker_exit: Callable[[str, "_SyncWorker"], None],
        inference_mode: bool = False,
    ) -> None:
        self.key = key
        self.device = device
        self.rwlock = rwlock
        self._on_task_finished = on_task_finished
        self._on_worker_exit = on_worker_exit
        self._inference_mode = inference_mode
        self.name = f"DPWorker-{key}#sync"

    def submit(self, fn: Callable[..., Any], /, *args, cuda_stream=None, **kwargs) -> Future:
        fut = Future()
        try:
            stream = cuda_stream
            override_inference = _pop_public_kwarg(
                kwargs, "inference_mode", "_threadx_inference_mode"
            )
            use_inference = self._inference_mode if override_inference is None else bool(override_inference)
            with ctx(self.rwlock.reader(), _device_ctx(self.device)):
                with tctl.threadpool_limits(limits=1):
                    inference_ctx = torch.inference_mode() if use_inference else contextlib.nullcontext()
                    with inference_ctx:
                        if stream is not None and self.device.type == "cuda":
                            with torch.cuda.stream(stream):
                                result = fn(*args, **kwargs)
                        else:
                            result = fn(*args, **kwargs)
            self._on_task_finished(self.key)
            if not fut.cancelled():
                fut.set_result(result)
        except BaseException as exc:
            self._on_task_finished(self.key)
            if not fut.cancelled():
                fut.set_exception(exc)
        return fut

    def stop(self) -> None:
        self._on_worker_exit(self.key, self)

    def join(self) -> None:
        return


# --------------------------- Public Pool ---------------------------
# - Builds workers per device with per-device RWLocks
# - Tracks inflight counts (with condition vars) and completion counters
# - Provides wait() with optional lock=True for exclusive operations
# - Runs a janitor background thread that performs periodic empty_cache() under
#   exclusive writer locks, coalescing triggers via a short debounce window.

class DeviceThreadPool:
    """
    Multi-device thread pool with:
      - Eager discovery/creation of workers and locks for CUDA/XPU/MPS/CPU.
      - Configurable worker counts per device (default 1).
      - Correct per-thread device context.
      - submit()/do() for async/sync, with optional `cuda_stream` (CUDA only).
      - Per-device RWLocks + global lock and family/all read-locks.
      - wait(scope, lock=False/True) to drain tasks (optionally with exclusive locks).
      - Per-device/global completed counters and in-flight counters.
      - Janitor: triggers empty-cache after N completions on accelerator devices (per-device lock).
      - GC diagnostics helpers (snapshots and ANSI tables).
    """

    def __init__(
        self,
        devices: Optional[Iterable[DeviceLike]] = None,
        *,
        include_cuda: bool = True,
        include_xpu: bool = True,
        include_mps: bool = True,
        include_cpu: bool = True,
        inference_mode: bool = False,
        empty_cache_every_n: int = 50,     # <=0 disables janitor
        workers: Optional[Dict[str, int]] = None,  # e.g. {'cpu':4, 'cuda:per':1, 'cuda:0':3}
        gc_debounce_seconds: float = 0.02,  # absorb bursty triggers before GC
    ):
        """
        Args:
            devices: explicit list of devices. If None, auto-discover per include_* flags.
            workers: dict mapping worker-count policy:
                - 'cpu': N                -> N workers total for CPU
                - 'mps': N                -> N workers for MPS (single device)
                - 'cuda:per': N           -> N workers per CUDA index
                - 'xpu:per': N            -> N workers per XPU index
                - 'cuda:<i>': N           -> override for specific CUDA index
                - 'xpu:<i>': N            -> override for specific XPU index
              Unspecified devices default to 1 worker each.
            gc_debounce_seconds: short wait to coalesce multiple triggers.
        """
        # Default to threaded workers; allow explicit opt-in to synchronous mode for
        # environments where background threads are prohibited.
        self._sync_mode = os.environ.get("THREADX_FORCE_SYNC", "0") == "1"

        if devices is None:
            discovered: List[torch.device] = []
            if include_cuda and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    discovered.append(torch.device("cuda", i))
            if include_xpu and hasattr(torch, "xpu") and torch.xpu.is_available():
                for i in range(torch.xpu.device_count()):
                    discovered.append(torch.device("xpu", i))
            if include_mps and _mps_available():
                discovered.append(torch.device("mps"))
            if include_cpu:
                discovered.append(torch.device("cpu"))
            devices = discovered

        # Locks and device registry (keyed by "type[:index]" strings like 'cuda:0').
        self._locks: Dict[str, _RWLock] = {}
        self._devices_by_key: Dict[str, torch.device] = {}

        # Worker groups and RR dispatch bookkeeping.
        self._worker_groups: Dict[str, List[_DeviceWorker]] = {}
        self._dispatch_rr: Dict[str, int] = {}
        self._dispatch_lock = threading.Lock()
        self._serial_workers: Dict[str, _DeviceWorker] = {}

        # Stats / GC / inflight control
        self._stats_lock = threading.Lock()
        self._per_device_done: Dict[str, int] = {}
        self._total_done: int = 0

        self._empty_cache_every_n = int(empty_cache_every_n)
        self._gc_event = threading.Event()
        self._stop_event = threading.Event()
        self._janitor: Optional[threading.Thread] = None

        # Auto-GC disable tracking (allows latency-sensitive regions to pause janitor)
        self._auto_gc_disable_count = 0
        self._auto_gc_disable_cv = threading.Condition()

        # In-flight (scheduled but not finished) counters + per-device CVs.
        # Each device has a condition variable to let wait() callers block
        # until inflight hits zero for that device scope.
        self._inflight: Dict[str, int] = {}
        self._inflight_cv: Dict[str, threading.Condition] = {}

        # GC dedupe/coalesce: debounce window to absorb bursty triggers;
        # per-device "done" watermark to skip redundant GC passes.
        self._gc_debounce_s = float(gc_debounce_seconds)
        self._last_gc_done_per_device: Dict[str, int] = {}

        self._inference_mode = bool(inference_mode)

        workers = workers or {}

        # Eagerly build workers, locks, inflight tracking, and counters.
        for d in devices:
            dev = _coerce_device(d)
            if dev.type not in ("cuda", "xpu", "mps", "cpu"):
                continue
            key = self._key(dev)
            if key in self._devices_by_key:
                continue

            self._devices_by_key[key] = dev
            self._locks[key] = _RWLock()
            self._per_device_done[key] = 0
            self._inflight[key] = 0
            self._inflight_cv[key] = threading.Condition()
            self._last_gc_done_per_device[key] = 0

            n_workers = self._resolve_workers_for_device(dev, workers)
            group: List[_DeviceWorker] = []
            for wid in range(int(max(1, n_workers))):
                worker = self._spawn_worker(dev, name=f"DPWorker-{key}#{wid}")
                group.append(worker)
            self._worker_groups[key] = group
            self._dispatch_rr[key] = 0
            if group:
                self._serial_workers[key] = group[0]

        # A canonical ordering for multi-device lock acquisitions.
        self._ordered_keys = sorted(self._locks.keys())

        # GC diagnostics counters
        self._gc_passes = 0
        self._last_gc_ts: Optional[float] = None

        # Start janitor if enabled and there exists at least one accelerator.
        if self._empty_cache_every_n > 0 and any(
            self._devices_by_key[k].type in ("cuda", "xpu", "mps") for k in self._ordered_keys
        ):
            self._janitor = threading.Thread(
                target=self._janitor_loop, name="DP-Janitor", daemon=True
            )
            self._janitor.start()
            if DEBUG_ON:
                log.debug(f"DP-Janitor thread started (debounce={self._gc_debounce_s:.3f}s, threshold={self._empty_cache_every_n})")
        else:
            if DEBUG_ON:
                log.debug("DP-Janitor disabled (no accelerators or threshold <= 0)")

    # --------------- Worker management ---------------

    def _spawn_worker(self, dev: torch.device, name: Optional[str] = None):
        """
        Create and start a worker bound to the provided device.
        """
        key = self._key(dev)
        if self._sync_mode:
            w = _SyncWorker(
                key=key,
                device=dev,
                rwlock=self._locks[key],
                on_task_finished=self._on_task_finished,
                on_worker_exit=self._on_worker_exit,
                inference_mode=self._inference_mode,
            )
        else:
            w = _DeviceWorker(
                device=dev,
                rwlock=self._locks[key],
                on_task_finished=self._on_task_finished,
                on_worker_exit=self._on_worker_exit,
                name=name,
                inference_mode=self._inference_mode,
            )
        return w

    def _on_worker_exit(self, key: str, worker: _DeviceWorker) -> None:
        """
        Clean up worker bookkeeping after a thread exits.
        """
        with self._dispatch_lock:
            group = self._worker_groups.get(key, [])
            if worker in group:
                group.remove(worker)
                self._worker_groups[key] = group
                if group:
                    self._dispatch_rr[key] %= len(group)
                else:
                    self._dispatch_rr[key] = 0
            self._refresh_serial_worker_locked(key)
        if DEBUG_ON: log.debug(f"Worker '{worker.name}' exited for {key}")

    # --------------- Public Work API ---------------

    def submit(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Future:
        """
        Asynchronously schedule work on the given device; returns a Future.
        Optional (CUDA): pass `cuda_stream=` to launch into a specific stream.
        """
        dev = _coerce_device(device)
        key = self._key(dev)
        worker = self._pick_worker(key)
        if cuda_stream is not None and dev.type != "cuda":
            raise ValueError("cuda_stream is only valid for CUDA devices")

        if DEBUG_ON: log.debug(f"submit: device={key} fn={getattr(fn, '__name__', repr(fn))}")
        # Mark in-flight before enqueue to avoid races with wait().
        self._mark_scheduled(key)
        try:
            return worker.submit(fn, *args, cuda_stream=cuda_stream, **kwargs)
        except BaseException:
            # Roll back inflight if enqueue fails (rare)
            self._mark_finished(key)
            raise

    def submit_serial(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Future:
        """
        Schedule work that must execute sequentially on a device. Tasks are
        enqueued onto a dedicated worker so they run in submission order.
        """
        dev = _coerce_device(device)
        key = self._key(dev)
        if cuda_stream is not None and dev.type != "cuda":
            raise ValueError("cuda_stream is only valid for CUDA devices")

        with self._dispatch_lock:
            group = self._worker_groups.get(key)
            if group is None:
                group = []
                self._worker_groups[key] = group
            if key not in self._dispatch_rr:
                self._dispatch_rr[key] = 0
            if not group:
                fresh = self._spawn_worker(dev, name=f"DPWorker-{key}#0")
                group.append(fresh)
                self._dispatch_rr[key] = 0
            self._refresh_serial_worker_locked(key)
            worker = self._serial_workers.get(key)

        if worker is None:
            raise ValueError(f"No serial worker available for device '{key}'")

        if DEBUG_ON: log.debug(f"submit_serial: device={key} fn={getattr(fn, '__name__', repr(fn))}")
        self._mark_scheduled(key)
        try:
            return worker.submit(fn, *args, cuda_stream=cuda_stream, **kwargs)
        except BaseException:
            self._mark_finished(key)
            raise

    def do(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Any:
        """
        Synchronously schedule work and block for the result.
        """
        fut = self.submit(device, fn, *args, cuda_stream=cuda_stream, **kwargs)
        return fut.result()

    def shutdown(self, wait: bool = True):
        """
        Gracefully stop all workers and janitor.

        IMPORTANT: We snapshot groups before stopping/joining to avoid mutating
        the lists while iterating (workers remove themselves on exit).
        """
        self._stop_event.set()
        self._gc_event.set()  # wake janitor if waiting

        # Take stable snapshots under the dispatch lock.
        with self._dispatch_lock:
            group_snapshots: Dict[str, List[_DeviceWorker]] = {
                key: list(group) for key, group in self._worker_groups.items()
            }

        # Stop janitor first so it won't grab locks while workers wind down.
        if self._janitor is not None and wait:
            if DEBUG_ON: log.debug("Joining DP-Janitor thread…")
            self._janitor.join()

        # Issue stop to every worker from the snapshots (no mutation hazards).
        for key, snapshot in group_snapshots.items():
            for w in snapshot:
                w.stop()

        # Join everyone if requested.
        if wait:
            for key, snapshot in group_snapshots.items():
                for w in snapshot:
                    w.join()

        if DEBUG_ON: log.debug("DeviceThreadPool shutdown complete")

    @contextlib.contextmanager
    def no_auto_gc(self):
        """
        Temporarily disable automatic empty-cache passes. Useful for latency-sensitive
        critical sections (e.g., forwarding) where janitor interference is undesirable.
        """
        with self._auto_gc_disable_cv:
            self._auto_gc_disable_count += 1
        try:
            yield
        finally:
            should_signal = False
            with self._auto_gc_disable_cv:
                if self._auto_gc_disable_count > 0:
                    self._auto_gc_disable_count -= 1
                if self._auto_gc_disable_count == 0:
                    should_signal = True
                    self._auto_gc_disable_cv.notify_all()
            if should_signal:
                # Wake janitor in case a trigger is pending.
                self._gc_event.set()

    # --------------- Public Lock API ---------------

    def device_lock(self, device: DeviceLike):
        """
        Obtain an exclusive lock for a single device (blocks all its workers).
        """
        dev = _coerce_device(device)
        key = self._key(dev)
        lk = self._locks.get(key)
        if lk is None:
            raise ValueError(f"Unknown device for pool: {dev}")
        return lk.writer()

    def read_lock(self, device: DeviceLike | str):
        """
        Obtain a read/shared lock. Accepts:
          - Concrete device: torch.device('cuda:0'), 'cuda:1'
          - Family device:  'cuda', 'xpu', 'mps', 'cpu'
          - 'all' for every device in the pool
        Returns a context manager.
        """
        # Family string shortcut (e.g., "cuda" or "all")
        if isinstance(device, str):
            if device == "all":
                pairs = [(k, self._locks[k]) for k in self._ordered_keys]
                return _ReadLockGroup(pairs)
            if device in ("cuda", "xpu", "mps", "cpu"):
                keys = [k for k in self._ordered_keys if k.startswith(device)]
                if not keys:
                    raise ValueError(f"No devices of type '{device}' in pool")
                pairs = [(k, self._locks[k]) for k in keys]
                return _ReadLockGroup(pairs)

        # torch.device / int / 'cuda:0' etc.
        dev = _coerce_device(device)
        key = self._key(dev)

        # Family device with index=None -> all devices of that type
        if dev.index is None:
            fam = dev.type
            keys = [k for k in self._ordered_keys if k.startswith(fam)]
            if not keys:
                raise ValueError(f"No devices of type '{fam}' in pool")
            pairs = [(k, self._locks[k]) for k in keys]
            return _ReadLockGroup(pairs)

        # Concrete device
        lk = self._locks.get(key)
        if lk is None:
            raise ValueError(f"Unknown device for pool: {dev}")
        return lk.reader()

    def lock(self, devices: Optional[Iterable[DeviceLike]] = None):
        """
        Exclusive lock across multiple devices (default: all pool devices).
        Acquires each device's write lock in canonical order to avoid deadlocks.
        """
        if devices is None:
            pairs = [(k, self._locks[k]) for k in self._ordered_keys]
        else:
            keys = sorted(self._normalize_scope_to_keys(devices))
            pairs = [(k, self._locks[k]) for k in keys]
        return _LockGroup(pairs)

    # --------------- Public Wait API ---------------
    # The wait() primitive blocks until inflight work drains for a scope.
    # With lock=True, it returns a context manager that first drains, then
    # acquires writer locks—handy for "drain-and-free" sequences.

    def wait(
        self,
        scope: Optional[Union[str, DeviceLike, Iterable[DeviceLike]]] = None,
        *,
        lock: bool = False,
    ) -> None | _WaitAndLock:
        """
        Wait until in-flight tasks for `scope` drain to zero.

        scope:
          - None or 'all' -> all devices
          - 'cuda' | 'xpu' | 'mps' | 'cpu' -> all devices of that type
          - 'cuda:0' | 'xpu:1' -> specific device key
          - torch.device or iterable of the above

        lock:
          - False: block until drained.
          - True: return a context manager that drains then acquires writer locks.
        """
        keys = self._resolve_scope_to_keys(scope)

        if lock:
            pairs = [(k, self._locks[k]) for k in sorted(keys)]

            class _WaitThenLock(contextlib.AbstractContextManager):
                """
                Drain inflight for the given keys, then acquire writer locks
                in canonical order; release on exit.
                """
                def __init__(self, outer: DeviceThreadPool, pairs_local: List[tuple[str, _RWLock]], keys_local: List[str]):
                    self._outer = outer
                    self._pairs = pairs_local
                    self._keys = keys_local
                    self._group = _LockGroup(pairs_local)

                def __enter__(self):
                    if DEBUG_ON: log.debug(f"wait(lock=True) drain start: keys={self._keys}")
                    for kk in self._keys:
                        cv = self._outer._inflight_cv[kk]
                        with cv:
                            while self._outer._inflight[kk] > 0:
                                if DEBUG_ON: log.debug(f"wait(lock=True) blocking on inflight[{kk}]={self._outer._inflight[kk]}")
                                cv.wait()
                    if DEBUG_ON: log.debug(f"wait(lock=True) acquire writer locks: keys={self._keys}")
                    return self._group.__enter__()

                def __exit__(self, exc_type, exc, tb):
                    if DEBUG_ON: log.debug(f"wait(lock=True) releasing writer locks: keys={self._keys}")
                    return self._group.__exit__(exc_type, exc, tb)

            return _WaitThenLock(self, pairs, keys)

        # Simple drain (no locks)
        if DEBUG_ON: log.debug(f"wait(lock=False) drain start: keys={keys}")
        for k in keys:
            cv = self._inflight_cv[k]
            with cv:
                while self._inflight[k] > 0:
                    if DEBUG_ON: log.debug(f"wait(lock=False) blocking on inflight[{k}]={self._inflight[k]}")
                    cv.wait()
        if DEBUG_ON: log.debug(f"wait(lock=False) drain done: keys={keys}")
        return None

    # --------------- Public Stats API ---------------

    def stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of counters. Use under tests or ad-hoc diagnostics.
        """
        with self._stats_lock:
            return {
                "per_device": dict(self._per_device_done),
                "total": int(self._total_done),
                "threshold": int(self._empty_cache_every_n),
            }

    def device_completed(self, device: DeviceLike) -> int:
        """
        Convenience accessor for per-device completed count (atomic snapshot).
        """
        key = self._key(_coerce_device(device))
        with self._stats_lock:
            return int(self._per_device_done.get(key, 0))

    def total_completed(self) -> int:
        """
        Convenience accessor for global completed count (atomic snapshot).
        """
        with self._stats_lock:
            return int(self._total_done)

    # --------------- Internals ---------------

    def _key(self, dev: torch.device) -> str:
        idx = "" if dev.index is None else f":{dev.index}"
        return f"{dev.type}{idx}"

    def _pick_worker(self, key: str) -> _DeviceWorker:
        """
        Round-robin selection across available workers for a device key.
        If no workers exist (should not happen under normal init), we
        spawn one lazily for robustness.
        """
        with self._dispatch_lock:
            group = self._worker_groups.get(key)
            if not group:
                dev = self._devices_by_key[key]
                w = self._spawn_worker(dev, name=f"DPWorker-{key}#0")
                group = [w]
                self._worker_groups[key] = group
                self._dispatch_rr[key] = 0
                self._refresh_serial_worker_locked(key)
                return w

            n = len(group)
            idx = self._dispatch_rr[key] % n
            w = group[idx]
            self._dispatch_rr[key] = (idx + 1) % n
            self._refresh_serial_worker_locked(key)
            return w

    def _refresh_serial_worker_locked(self, key: str) -> None:
        """
        Ensure the serial worker mapping references a live worker.
        Caller must hold self._dispatch_lock.
        """
        group = self._worker_groups.get(key, [])
        if not group:
            self._serial_workers.pop(key, None)
            return

        current = self._serial_workers.get(key)
        if current in group:
            return

        self._serial_workers[key] = group[0]

    def _resolve_workers_for_device(self, dev: torch.device, table: Dict[str, int]) -> int:
        """
        Resolve worker count from policy table:
          - exact key (e.g. 'cuda:0') overrides
          - family-per (e.g. 'cuda:per') applies to all indices
          - family singletons ('cpu', 'mps') apply to that single device
        """
        key = self._key(dev)
        if key in table:
            return int(table[key])
        fam_key = f"{dev.type}:per"
        if fam_key in table:
            return int(table[fam_key])
        if dev.type in ("cpu", "mps") and dev.type in table:
            return int(table[dev.type])
        return 1

    def _normalize_scope_to_keys(self, scope: Iterable[DeviceLike]) -> List[str]:
        """
        Normalize a scope specification into a sorted list of device keys.
        Accepts strings ('cuda', 'cuda:0', 'all'), ints (device indices),
        and torch.device objects.
        """
        keys: List[str] = []
        for s in scope:
            if isinstance(s, str):
                if s in ("all",):
                    keys.extend(self._ordered_keys)
                elif ":" in s:
                    if s not in self._locks:
                        raise ValueError(f"Unknown device key in scope: {s}")
                    keys.append(s)
                else:
                    fam = s
                    fam_keys = [k for k in self._ordered_keys if k.startswith(fam)]
                    if not fam_keys:
                        raise ValueError(f"No devices of type '{fam}' in pool")
                    keys.extend(fam_keys)
            else:
                dev = _coerce_device(s)
                k = self._key(dev)
                if k not in self._locks:
                    raise ValueError(f"Device not in pool: {dev}")
                keys.append(k)
        return keys

    def _resolve_scope_to_keys(self, scope: Optional[Union[str, DeviceLike, Iterable[DeviceLike]]] = None) -> List[str]:
        """
        Helper for wait()/lock(): expand a scope into concrete device keys.
        """
        if scope is None or (isinstance(scope, str) and scope == "all"):
            return list(self._ordered_keys)
        if isinstance(scope, (str, torch.device, int)):
            return self._normalize_scope_to_keys([scope])
        return self._normalize_scope_to_keys(scope)

    # ---- inflight & completion accounting ----

    def _mark_scheduled(self, key: str) -> None:
        """
        Increment inflight for a device key and emit a debug breadcrumb if enabled.
        """
        cv = self._inflight_cv[key]
        with cv:
            self._inflight[key] = self._inflight.get(key, 0) + 1
            if DEBUG_ON: log.debug(f"inflight[{key}] ++ -> {self._inflight[key]}")

    def _mark_finished(self, key: str) -> None:
        """
        Decrement inflight for a device key, clamp at zero on underflow, and
        notify waiters when the device drains.
        """
        cv = self._inflight_cv[key]
        with cv:
            new_val = self._inflight.get(key, 0) - 1
            if new_val < 0:
                if DEBUG_ON: log.debug(f"WARNING: inflight[{key}] underflow ({new_val}); clamping to 0")
                new_val = 0
            self._inflight[key] = new_val
            if DEBUG_ON: log.debug(f"inflight[{key}] -- -> {self._inflight[key]}")
            if self._inflight[key] == 0:
                cv.notify_all()

    def _on_task_finished(self, key: str) -> None:
        """
        Called at the end of every task (success or failure). Updates counters
        and signals the janitor if the per-device threshold is reached.
        """
        self._mark_finished(key)

        trigger_gc = False
        with self._stats_lock:
            self._per_device_done[key] = self._per_device_done.get(key, 0) + 1
            self._total_done += 1
            dev_type = self._devices_by_key[key].type
            if self._empty_cache_every_n > 0 and dev_type in ("cuda", "xpu", "mps"):
                n = self._per_device_done[key]
                if n % self._empty_cache_every_n == 0:
                    trigger_gc = True
                    if DEBUG_ON:
                        log.debug(f"GC trigger set by {key}: per_device_done={n} threshold={self._empty_cache_every_n} total_done={self._total_done}")
        if trigger_gc:
            self._gc_event.set()

    # ---- ANSI table rendering for GC diagnostics ----

    def _ansi_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Render a simple ANSI/ASCII table with bold headers. Used only for
        human-readable diagnostics; not used in hot paths.
        """
        widths = [len(h) for h in headers]
        for r in rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))

        def hrule(sep_left="+", sep_mid="+", sep_right="+", h="-"):
            parts = [sep_left]
            for i, w in enumerate(widths):
                parts.append(h * (w + 2))
                parts.append(sep_mid if i < len(widths) - 1 else sep_right)
            return "".join(parts)

        def format_row(cols: List[str]):
            out = ["|"]
            for i, cell in enumerate(cols):
                out.append(" " + cell.ljust(widths[i]) + " ")
                out.append("|")
            return "".join(out)

        BOLD = "\x1b[1m"
        RESET = "\x1b[0m"

        top = hrule()
        mid = hrule(h="=")
        bot = hrule()

        lines = [top, format_row([BOLD + h + RESET for h in headers]), mid]
        for r in rows:
            lines.append(format_row(r))
        lines.append(bot)
        return "\n".join(lines)

    def _collect_state_snapshot(self) -> Dict[str, Any]:
        """
        Safely collect a snapshot of pool state for diagnostics and GC decisions.
        """
        with self._stats_lock:
            per_done = dict(self._per_device_done)
            total_done = int(self._total_done)
            threshold = int(self._empty_cache_every_n)

        inflight: Dict[str, int] = {}
        for k, cv in self._inflight_cv.items():
            with cv:
                inflight[k] = int(self._inflight[k])

        workers = {k: len(self._worker_groups.get(k, [])) for k in self._devices_by_key.keys()}

        meta: Dict[str, Dict[str, str]] = {}
        for k, dev in self._devices_by_key.items():
            idx = "" if dev.index is None else str(dev.index)
            meta[k] = {"type": dev.type, "index": idx}

        snap: Dict[str, Any] = {
            "devices": sorted(self._devices_by_key.keys()),
            "per_done": per_done,
            "total_done": total_done,
            "threshold": threshold,
            "inflight": inflight,
            "workers": workers,
            "meta": meta,
            "total_inflight": sum(inflight.values()),
            "total_workers": sum(workers.values()),
            "gc_passes": int(self._gc_passes),
            "last_gc_ts": self._last_gc_ts,
            "now": time.time(),
        }
        return snap

    def _render_gc_table(self, snap: Dict[str, Any]) -> str:
        """
        Pretty-print a GC state table; used only when someone wants to log it.
        """
        headers = [
            "Device", "Type", "Index", "Workers", "Inflight",
            "Done", "Threshold", "NextGC", "Accel"
        ]
        rows: List[List[str]] = []
        thr = snap["threshold"]
        for k in snap["devices"]:
            t = snap["meta"][k]["type"]
            idx = snap["meta"][k]["index"]
            w = snap["workers"].get(k, 0)
            infl = snap["inflight"].get(k, 0)
            done = snap["per_done"].get(k, 0)
            accel = "Y" if t in ("cuda", "xpu", "mps") else "N"
            if thr > 0 and t in ("cuda", "xpu", "mps"):
                rem = thr - (done % thr) if (done % thr) != 0 else 0
                nextgc = "now" if rem == 0 and done > 0 else str(rem)
            else:
                nextgc = "-"
            rows.append([k, t, idx, str(w), str(infl), str(done), str(thr) if thr > 0 else "-", nextgc, accel])

        table_main = self._ansi_table(headers, rows)

        totals_headers = ["Total Workers", "Total Inflight", "Total Done", "GC Passes", "Since Last GC (s)"]
        since = "-" if self._last_gc_ts is None else f"{time.time() - self._last_gc_ts:.3f}"
        totals_rows = [[
            str(sum(len(v) for v in self._worker_groups.values())),
            str(snap["total_inflight"]),
            str(snap["total_done"]),
            str(snap["gc_passes"]),
            since,
        ]]
        table_totals = self._ansi_table(totals_headers, totals_rows)
        return table_main + "\n" + table_totals

    # ---- janitor (per-device empty-cache under exclusive writer lock) ----
    # The janitor runs in the background. When a device completes N tasks, a trigger
    # is set. The janitor debounces triggers, takes a snapshot, and if at least one
    # accelerator device progressed by >= N tasks since the last pass, it iterates
    # devices, acquiring each device's writer lock before calling empty_cache().

    def _synchronize_all(self):
        """
        Optionally ensure devices are idle before empty_cache() to avoid races with
        outstanding kernels. Keeping this disabled by default for performance.
        """
        # CUDA
        for key in self._ordered_keys:
            dev = self._devices_by_key[key]
            if dev.type != "cuda":
                continue
            with torch.cuda.device(dev.index):
                torch.cuda.synchronize()
        # XPU
        for key in self._ordered_keys:
            dev = self._devices_by_key[key]
            if dev.type != "xpu":
                continue
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                with torch.xpu.device(dev.index):
                    torch.xpu.synchronize()
        # MPS
        has_mps_device = any(self._devices_by_key[k].type == "mps" for k in self._ordered_keys)
        if has_mps_device and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def _should_run_gc_from_snapshot(self, snap: Dict[str, Any]) -> bool:
        """
        Decide whether to run GC by comparing per-device progress since last GC.
        This deduplicates bursty triggers that occurred before the previous GC ran.
        """
        thr = snap["threshold"]
        if thr <= 0:
            return False
        for k in snap["devices"]:
            dev_type = snap["meta"][k]["type"]
            if dev_type not in ("cuda", "xpu", "mps"):
                continue
            done_now = snap["per_done"].get(k, 0)
            done_prev = self._last_gc_done_per_device.get(k, 0)
            if done_now - done_prev >= thr:
                return True
        return False

    def _update_gc_watermarks(self, snap_after: Dict[str, Any]) -> None:
        """
        Record 'done' counters as of a GC pass to require fresh progress
        before a subsequent pass is allowed.
        """
        for k in snap_after["devices"]:
            self._last_gc_done_per_device[k] = snap_after["per_done"].get(k, 0)

    def _janitor_loop(self):
        """
        Main janitor loop:
          - Waits on a trigger (with short timeout to honor shutdowns promptly).
          - Debounces additional triggers for a brief window.
          - Takes a snapshot and decides whether to run.
          - For each accelerator device, acquires its writer lock and calls
            empty_cache() using the LIVE attribute if callable, otherwise the
            HARD COPY captured at import time.
        """
        while True:
            if DEBUG_ON:
                log.debug("DP-Janitor: waiting for trigger…")

            self._gc_event.wait()
            self._gc_event.clear()

            if self._stop_event.is_set():
                if DEBUG_ON:
                    log.debug("DP-Janitor: stop event set; exiting")
                break

            # Debounce window: absorb additional triggers before deciding.
            if self._gc_debounce_s > 0:
                t_end = time.time() + self._gc_debounce_s
                if DEBUG_ON: log.debug(f"DP-Janitor: debounce window start ({self._gc_debounce_s:.3f}s)")
                while time.time() < t_end:
                    if self._stop_event.is_set():
                        if DEBUG_ON: log.debug("DP-Janitor: stop during debounce; exiting")
                        return
                    self._gc_event.wait(timeout=max(0.0, t_end - time.time()))
                    self._gc_event.clear()
                if DEBUG_ON: log.debug("DP-Janitor: debounce window end")

            with self._auto_gc_disable_cv:
                while self._auto_gc_disable_count > 0 and not self._stop_event.is_set():
                    if DEBUG_ON:
                        log.debug("DP-Janitor: auto-GC disabled; waiting…")
                    self._auto_gc_disable_cv.wait()
                if self._stop_event.is_set():
                    if DEBUG_ON: log.debug("DP-Janitor: stop event set during auto-GC wait; exiting")
                    break

            # Snapshot & decision
            try:
                pre = self._collect_state_snapshot()
                if DEBUG_ON:
                    log.debug(f"DP-Janitor: pre-snapshot taken: total_done={pre['total_done']}, threshold={pre['threshold']}, inflight={pre['inflight']}")
                    log.debug("GC trigger received; evaluating whether to run…")
            except Exception as e:
                # Fallback snapshot (unlikely path; logging should not crash janitor)
                try:
                    log.warn(f"Failed to render GC pre-snapshot: {e!r}")
                except Exception:
                    pass
                pre = {
                    "devices": list(self._devices_by_key.keys()),
                    "per_done": {k: self._per_device_done.get(k, 0) for k in self._devices_by_key.keys()},
                    "threshold": self._empty_cache_every_n,
                    "meta": {k: {"type": self._devices_by_key[k].type} for k in self._devices_by_key.keys()},
                    "inflight": dict.fromkeys(self._devices_by_key.keys(), 0),
                    "workers": {k: len(self._worker_groups.get(k, [])) for k in self._devices_by_key.keys()},
                    "total_inflight": 0,
                    "total_workers": sum(len(v) for v in self._worker_groups.values()),
                    "gc_passes": self._gc_passes,
                    "last_gc_ts": self._last_gc_ts,
                    "now": time.time(),
                    "total_done": self._total_done,
                }

            if not self._should_run_gc_from_snapshot(pre):
                if DEBUG_ON: log.debug("DP-Janitor: skip GC (no device progressed by threshold since last pass)")
                continue

            t0 = time.time()
            # Optionally synchronize devices; often too slow to be worthwhile:
            # self._synchronize_all()

            # Per-device exclusive: acquire write lock, then call empty_cache().
            for key in sorted(self._ordered_keys):
                dev = self._devices_by_key[key]
                if dev.type not in ("cuda", "xpu", "mps"):
                    continue

                lk = self._locks[key]
                if DEBUG_ON: log.debug(f"DP-Janitor: attempting writer lock for {key}")
                with lk.writer():
                    if DEBUG_ON: log.debug(f"DP-Janitor: acquired writer lock for {key}")

                    if dev.type == "cuda":
                        live = getattr(torch.cuda, "empty_cache", None) if hasattr(torch, "cuda") else None
                        use_fn = live if callable(live) else TORCH_CUDA_EMPTY_CACHE
                        if DEBUG_ON:
                            src = "live" if use_fn is live else "hardcopy"
                            log.debug(f"DP-Janitor: empty_cache(cuda) using {src} on {key}")
                        if use_fn is not None:
                            with torch.cuda.device(dev.index):
                                use_fn()

                    elif dev.type == "xpu":
                        live = getattr(torch.xpu, "empty_cache", None) if hasattr(torch, "xpu") else None
                        use_fn = live if callable(live) else TORCH_XPU_EMPTY_CACHE
                        if DEBUG_ON:
                            src = "live" if use_fn is live else "hardcopy"
                            log.debug(f"DP-Janitor: empty_cache(xpu) using {src} on {key}")
                        if use_fn is not None:
                            with torch.xpu.device(dev.index):
                                use_fn()

                    elif dev.type == "mps":
                        live = getattr(torch.mps, "empty_cache", None) if hasattr(torch, "mps") else None
                        use_fn = live if callable(live) else TORCH_MPS_EMPTY_CACHE
                        if DEBUG_ON:
                            src = "live" if use_fn is live else "hardcopy"
                            log.debug(f"DP-Janitor: empty_cache(mps) using {src}")
                        if use_fn is not None:
                            use_fn()

            t1 = time.time()
            self._gc_passes += 1
            self._last_gc_ts = t1

            # Post-pass accounting & watermarks.
            try:
                post = self._collect_state_snapshot()
                self._update_gc_watermarks(post)
                log.info(f"GC completed in {t1 - t0:.3f}s (pass #{self._gc_passes}).")
                if DEBUG_ON: log.debug(f"DP-Janitor: post-snapshot: inflight={post['inflight']} per_done={post['per_done']}")
            except Exception as e:
                try:
                    log.warn(f"Failed to render GC post-snapshot: {e!r}")
                except Exception:
                    pass

    # Legacy helper (not used by janitor). Kept for compatibility with any
    # external callers that previously expected a "clear everything" helper.
    def _empty_all_caches(self):
        """
        Call the captured originals if available. This does not consult the live
        attribute and therefore does not pick up monkeypatching. Prefer the janitor’s
        per-device logic for production use.
        """
        if TORCH_CUDA_EMPTY_CACHE is not None:
            for key in self._ordered_keys:
                dev = self._devices_by_key[key]
                if dev.type != "cuda":
                    continue
                with torch.cuda.device(dev.index):
                    TORCH_CUDA_EMPTY_CACHE()
        if TORCH_XPU_EMPTY_CACHE is not None:
            for key in self._ordered_keys:
                dev = self._devices_by_key[key]
                if dev.type != "xpu":
                    continue
                with torch.xpu.device(dev.index):
                    TORCH_XPU_EMPTY_CACHE()
        if TORCH_MPS_EMPTY_CACHE is not None:
            has_mps_device = any(self._devices_by_key[k].type == "mps" for k in self._ordered_keys)
            if has_mps_device:
                TORCH_MPS_EMPTY_CACHE()
