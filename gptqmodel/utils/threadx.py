# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
import inspect
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch


try:
    from device_smi import Device  # type: ignore
except Exception:  # pragma: no cover - defensive: optional dependency may be unavailable
    Device = None

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

_EMPTY_CACHE_SIGNATURE_CACHE: Dict[int, Tuple[bool, bool]] = {}


def _analyze_empty_cache_callable(fn: Callable[..., Any]) -> Tuple[bool, bool]:
    """
    Inspect an empty_cache callable and determine whether it accepts a `device`
    keyword argument or at least one positional argument. Results are memoized.
    """
    cache_key = id(fn)
    cached = _EMPTY_CACHE_SIGNATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    supports_kw = False
    supports_pos = False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        _EMPTY_CACHE_SIGNATURE_CACHE[cache_key] = (supports_kw, supports_pos)
        return supports_kw, supports_pos

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            supports_kw = True
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            supports_pos = True
        elif param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            supports_pos = True
            if param.name == "device":
                supports_kw = True
        elif param.kind == inspect.Parameter.KEYWORD_ONLY and param.name == "device":
            supports_kw = True

    _EMPTY_CACHE_SIGNATURE_CACHE[cache_key] = (supports_kw, supports_pos)
    return supports_kw, supports_pos

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
        target = dev if dev.index is not None else "cuda"
        with torch.cuda.device(target):
            yield
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        target = dev if dev.index is not None else "xpu"
        with torch.xpu.device(target):
            yield
    else:
        yield


def _activate_thread_device(dev: torch.device):
    """
    Pin the worker thread to its device once, before entering its main loop.
    CUDA/XPU require per-thread device activation for correct handle usage.
    """
    if dev.type == "cuda":
        target = dev if dev.index is not None else "cuda"
        torch.cuda.set_device(target)
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        target = dev if dev.index is not None else "xpu"
        torch.xpu.set_device(target)
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
        cpu_core: Optional[int] = None,
        warmup_fn: Optional[Callable[[torch.device], None]] = None,
        *,
        key_override: Optional[str] = None,
    ):
        self.device = device
        self.rwlock = rwlock
        self._on_task_finished = on_task_finished
        self._on_worker_exit = on_worker_exit
        self._warmup_fn = warmup_fn

        if key_override is not None:
            self.key = key_override
        else:
            self.key = f"{device.type}:{device.index}" if device.index is not None else device.type
        self.name = name or f"DPWorker-{self.key}"
        self._q: "queue.Queue[Tuple[bool, Callable[..., Any], tuple, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()

        self._inference_mode = inference_mode
        self._target_cpu_core = cpu_core
        self._affinity_applied = False
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

    def _apply_cpu_affinity(self) -> None:
        if self._affinity_applied or self._target_cpu_core is None:
            return
        if not hasattr(os, "sched_setaffinity"):
            log.warn(
                "Thread pinning unsupported on this platform; %s will use OS scheduling.",
                self.name,
            )
            self._affinity_applied = True
            return
        try:
            os.sched_setaffinity(0, {int(self._target_cpu_core)})
            self._affinity_applied = True
            #if DEBUG_ON:
            log.debug(f"{self.name}: pinned to CPU core {self._target_cpu_core}")
        except PermissionError as exc:
            log.warn(
                "Thread pinning permission denied for %s on core %s (%s); falling back to OS scheduling.",
                self.name,
                self._target_cpu_core,
                exc,
            )
            self._affinity_applied = True
        except OSError as exc:
            log.warn(
                "Thread pinning failed for %s on core %s (%s); falling back to OS scheduling.",
                self.name,
                self._target_cpu_core,
                exc,
            )
            self._affinity_applied = True

    def _run_warmup(self) -> None:
        warmup_fn = self._warmup_fn
        if warmup_fn is None:
            return
        try:
            with ctx(self.rwlock.reader(), _device_ctx(self.device)):
                warmup_fn(self.device)
        finally:
            self._warmup_fn = None

    def _run(self):
        """
        Main loop: pull tasks, set device context, execute, mark completion, and
        fulfill or fail the future. Completion is accounted BEFORE resolving the
        future to make stats() deterministic even under test interleavings.

        Workers default to inference mode for throughput but individual tasks
        may override via `inference_mode`.
        """
        self._apply_cpu_affinity()
        _activate_thread_device(self.device)
        try:
            self._run_warmup()
        except BaseException as exc:
            self._abort_process(exc)
            return
        while not self._stop.is_set():
            is_task, fn, args, kwargs, fut = self._q.get()
            try:
                if not is_task:
                    if DEBUG_ON: log.debug(f"{self.name}: received sentinel; exiting")
                    break
                if DEBUG_ON: log.debug(f"{self.name}: task begin; qsize={self._q.qsize()}")

                event = kwargs.pop("cuda_event", None)
                override_inference = _pop_public_kwarg(
                    kwargs, "inference_mode", "_threadx_inference_mode"
                )
                use_inference = self._inference_mode if override_inference is None else bool(override_inference)

                # Tasks take a **read** lock so janitor's write lock can't interleave
                with ctx(self.rwlock.reader(), _device_ctx(self.device)):
                    inference_ctx = torch.inference_mode() if use_inference else contextlib.nullcontext()
                    with inference_ctx:
                        if event is not None and self.device.type == "cuda":
                            event.synchronize()
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
                self._abort_process(exc)
            finally:
                self._q.task_done()
        try:
            self._on_worker_exit(self.key, self)
        finally:
            if DEBUG_ON: log.debug(f"{self.name}: exited")

    def _abort_process(self, exc: BaseException) -> None:
        """Dump the stack trace and terminate the interpreter without cleanup."""
        try:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os._exit(1)
        except Exception:
            # Last resort if os._exit is unavailable for some reason.
            os.system("kill -9 %d" % os.getpid())


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

    def submit(self, fn: Callable[..., Any], /, *args, cuda_event=None, **kwargs) -> Future:
        fut = Future()
        try:
            event = cuda_event
            override_inference = _pop_public_kwarg(
                kwargs, "inference_mode", "_threadx_inference_mode"
            )
            use_inference = self._inference_mode if override_inference is None else bool(override_inference)
            with ctx(self.rwlock.reader(), _device_ctx(self.device)):
                # with tctl.threadpool_limits(limits=1):
                inference_ctx = torch.inference_mode() if use_inference else contextlib.nullcontext()
                with inference_ctx:
                    if event is not None and self.device.type == "cuda":
                        event.synchronize()
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
      - submit()/do() for async/sync, with optional CUDA event synchronization.
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
        warmups: Optional[Dict[str, Callable[[torch.device], None]]] = None,
        empty_cache_every_n: int = 50,     # <=0 disables janitor
        workers: Optional[Dict[str, int]] = None,  # e.g. {'cpu':4, 'cuda:per':1, 'cuda:0':3}
        gc_debounce_seconds: float = 0.02,  # absorb bursty triggers before GC
        gc_min_interval_seconds: float = 1.0,  # throttle janitor passes
        pin_cpu_workers: bool = False,
        pin_accelerator_workers: bool = False,
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
                - '<alias>:<parent>': N   -> virtual pool sharing locks with parent device
                  (CUDA parents must include an explicit index, e.g. 'alias:cuda:0')
              Unspecified devices default to 1 worker each.
            gc_debounce_seconds: short wait to coalesce multiple triggers.
            warmups: optional mapping from device family (e.g. 'cuda') to a callable
                run once after the worker activates its device. A special key
                'default' applies when no family-specific warmup is found.
            gc_min_interval_seconds: minimum interval between GC passes. Values <= 0 disable throttling.
            pin_cpu_workers: bind CPU device workers to individual CPU cores when
                affinity APIs are available. Defaults to False so CPU tasks may
                float across cores unless explicitly opt-in.
            pin_accelerator_workers: bind CUDA/XPU/MPS workers to dedicated CPU
                cores when affinity APIs are available. Defaults to False so
                accelerator workers inherit the process CPU affinity mask.
        """
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
        self._gc_min_interval_s = max(0.0, float(gc_min_interval_seconds))
        self._last_gc_done_per_device: Dict[str, int] = {}
        # Physical-device GC bookkeeping (per accelerator index).
        self._gc_done_physical: Dict[str, int] = {}
        self._last_gc_done_physical: Dict[str, int] = {}
        self._gc_pending_physical: Dict[str, int] = {}
        self._physical_children: Dict[str, Set[str]] = {}

        # Device-SMI handles are created lazily for GC logging.
        self._device_smi_lock = threading.Lock()
        self._device_smi_handles: Dict[str, Any] = {}
        self._device_smi_failures: Set[str] = set()

        self._inference_mode = bool(inference_mode)
        self._worker_warmups = (
            {str(k).lower(): fn for k, fn in warmups.items()} if warmups else None
        )
        self._warmup_lock = threading.Lock()
        self._warmup_ran_keys: Set[str] = set()

        workers_cfg = workers or {}
        base_workers: Dict[str, int] = {}
        virtual_workers: Dict[str, Tuple[str, int]] = {}
        for raw_key, raw_count in workers_cfg.items():
            alias_info = self._parse_virtual_worker_key(raw_key)
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Worker count for '{raw_key}' must be an integer (got {raw_count!r})") from exc
            if alias_info is None:
                base_workers[raw_key] = count
            else:
                _, parent_key = alias_info
                if parent_key.startswith("cuda"):
                    parts = parent_key.split(":")
                    if len(parts) < 2 or not parts[1].isdigit():
                        raise ValueError(
                            f"Virtual pool '{raw_key}' must target a concrete CUDA index (e.g. 'alias:cuda:0'); got '{parent_key}'"
                        )
                virtual_workers[raw_key] = (parent_key, count)

        self._virtual_to_parent: Dict[str, str] = {
            v_key: parent for v_key, (parent, _) in virtual_workers.items()
        }

        # Pre-compute CPU core assignments (when opted in) so GPU/XPU workers
        # do not collide with CPU workers. When affinity APIs are not available
        # we silently fall back to OS scheduling.
        affinity_plan = self._plan_worker_affinity(
            devices,
            base_workers,
            pin_cpu_workers=pin_cpu_workers,
            pin_accelerator_workers=pin_accelerator_workers,
        )

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
            self._physical_children[key] = {key}
            if dev.type in ("cuda", "xpu", "mps"):
                self._gc_done_physical[key] = 0
                self._last_gc_done_physical[key] = 0

            n_workers = self._resolve_workers_for_device(dev, base_workers)
            group: List[_DeviceWorker] = []
            for wid in range(int(max(1, n_workers))):
                cpu_core = affinity_plan.get((key, wid))
                worker = self._spawn_worker(dev, key, name=f"DPWorker-{key}#{wid}", cpu_core=cpu_core)
                group.append(worker)
            self._worker_groups[key] = group
            self._dispatch_rr[key] = 0
            if group:
                self._serial_workers[key] = group[0]

        for v_key, (parent_key, limit) in virtual_workers.items():
            if limit < 1:
                raise ValueError(f"Virtual pool '{v_key}' requires at least one worker (got {limit})")
            parent_dev = self._devices_by_key.get(parent_key)
            if parent_dev is None:
                raise ValueError(f"Virtual pool '{v_key}' references unknown parent '{parent_key}'")
            if parent_dev.type == "cuda" and parent_dev.index is None:
                raise ValueError(
                    f"Virtual pool '{v_key}' requires an indexed CUDA parent device (e.g. '{v_key}:cuda:0'); got '{parent_key}'"
                )
            parent_group = self._worker_groups.get(parent_key, [])
            parent_budget = len(parent_group)
            if limit > parent_budget:
                raise ValueError(
                    f"Virtual pool '{v_key}' requests {limit} workers but parent '{parent_key}' only has {parent_budget}"
                )

            self._locks[v_key] = self._locks[parent_key]
            self._devices_by_key[v_key] = parent_dev
            self._per_device_done[v_key] = 0
            self._inflight[v_key] = 0
            self._inflight_cv[v_key] = threading.Condition()
            self._last_gc_done_per_device[v_key] = 0
            self._physical_children.setdefault(parent_key, set()).add(v_key)

            alias_group: List[_DeviceWorker] = []
            for wid in range(limit):
                worker = self._spawn_worker(parent_dev, v_key, name=f"DPWorker-{v_key}#{wid}")
                alias_group.append(worker)
            self._worker_groups[v_key] = alias_group
            self._dispatch_rr[v_key] = 0
            if alias_group:
                self._serial_workers[v_key] = alias_group[0]

        # A canonical ordering for multi-device lock acquisitions.
        self._ordered_keys = sorted(self._locks.keys())
        self._rebuild_family_keys()

        # GC diagnostics counters
        self._gc_passes = 0
        self._last_gc_ts: Optional[float] = None
        self._gc_generation: int = 0
        self._last_consumed_gc_generation: int = 0

        # Start janitor if enabled and there exists at least one accelerator.
        if self._empty_cache_every_n > 0 and any(
            self._devices_by_key[k].type in ("cuda", "xpu", "mps") for k in self._ordered_keys
        ):
            self._janitor = threading.Thread(
                target=self._janitor_loop, name="DP-Janitor", daemon=True
            )
            self._janitor.start()
            if DEBUG_ON:
                log.debug(
                    f"DP-Janitor thread started (debounce={self._gc_debounce_s:.3f}s, "
                    f"min_interval={self._gc_min_interval_s:.3f}s, threshold={self._empty_cache_every_n})"
                )
        else:
            if DEBUG_ON:
                log.debug("DP-Janitor disabled (no accelerators or threshold <= 0)")

    # --------------- Worker management ---------------

    @staticmethod
    def _parse_virtual_worker_key(key: str) -> Optional[Tuple[str, str]]:
        if not isinstance(key, str) or ":" not in key:
            return None
        head, tail = key.split(":", 1)
        if head in {"cuda", "xpu", "mps", "cpu"}:
            return None
        if not tail:
            return None
        return head, tail

    def _rebuild_family_keys(self) -> None:
        fam_map: Dict[str, List[str]] = {}
        for key in sorted(self._locks.keys()):
            fam = key.split(":", 1)[0]
            fam_map.setdefault(fam, []).append(key)

        for alias_key, parent_key in self._virtual_to_parent.items():
            parent_fam = parent_key.split(":", 1)[0]
            fam_map.setdefault(parent_fam, [])
            if alias_key not in fam_map[parent_fam]:
                fam_map[parent_fam].append(alias_key)

        for fam, keys in fam_map.items():
            keys.sort()

        self._family_keys = fam_map

    def _resolve_device_key(self, device: DeviceLike | str) -> str:
        if isinstance(device, str):
            if device == "all":
                raise ValueError("'all' is not a valid concrete device specification")
            if device in self._locks:
                return device
            fam_keys = self._family_keys.get(device)
            if fam_keys is not None:
                if len(fam_keys) == 1:
                    return fam_keys[0]
                raise ValueError(
                    f"Device specification '{device}' is ambiguous; choose one of {fam_keys}"
                )

        dev = _coerce_device(device)
        key = self._key(dev)
        if key not in self._locks:
            raise ValueError(f"Device not in pool: {device}")
        return key

    def _plan_worker_affinity(
        self,
        devices: Iterable[torch.device],
        worker_table: Dict[str, int],
        *,
        pin_cpu_workers: bool,
        pin_accelerator_workers: bool,
    ) -> Dict[Tuple[str, int], Optional[int]]:
        """Return per-worker CPU core targets to minimise contention."""
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            return {}

        try:
            available_cores = sorted(os.sched_getaffinity(0))
        except AttributeError:
            return {}
        except OSError as exc:
            log.warn(
                "Thread pinning disabled: unable to read CPU affinity mask (%s).",
                exc,
            )
            return {}

        if not available_cores:
            return {}

        worker_specs: List[Tuple[str, str, int]] = []
        for raw_dev in devices:
            dev = _coerce_device(raw_dev)
            if dev.type not in ("cuda", "xpu", "mps", "cpu"):
                continue
            if dev.type == "cpu" and not pin_cpu_workers:
                continue
            if dev.type in ("cuda", "xpu", "mps") and not pin_accelerator_workers:
                continue
            key = self._key(dev)
            n_workers = int(max(1, self._resolve_workers_for_device(dev, worker_table)))
            for wid in range(n_workers):
                worker_specs.append((dev.type, key, wid))

        if not worker_specs:
            return {}

        def _priority(dev_type: str) -> int:
            if dev_type in ("cuda", "xpu"):
                return 0
            if dev_type == "mps":
                return 1
            return 2  # cpu last so it yields remaining cores

        worker_specs.sort(key=lambda spec: (_priority(spec[0]), spec[1], spec[2]))

        core_iter = iter(available_cores)
        plan: Dict[Tuple[str, int], Optional[int]] = {}
        exhaustion_logged = False

        for dev_type, key, wid in worker_specs:
            try:
                core = next(core_iter)
            except StopIteration:
                core = None

            if core is None:
                if not exhaustion_logged:
                    log.warn(
                        "Thread pinning: insufficient CPU cores for %d workers; remaining workers will use OS scheduling.",
                        len(worker_specs) - len(plan),
                    )
                    exhaustion_logged = True
                plan[(key, wid)] = None
            else:
                plan[(key, wid)] = core

        return plan

    def _resolve_worker_warmup(self, dev: torch.device, key: str) -> Optional[Callable[[torch.device], None]]:
        mapping = self._worker_warmups
        if not mapping:
            return None
        family = dev.type.lower()
        warmup = mapping.get(family)
        primary_key = key.split(":", 1)[0].lower()
        if warmup is None and primary_key in mapping:
            warmup = mapping[primary_key]
        if warmup is None:
            warmup = mapping.get("default")
        if warmup is None:
            return None

        # Map virtual workers back to their parent key so warmup runs once per physical device.
        physical_key = self._virtual_to_parent.get(key, key)
        with self._warmup_lock:
            if physical_key in self._warmup_ran_keys:
                return None
            self._warmup_ran_keys.add(physical_key)
        return warmup

    def _spawn_worker(
        self,
        dev: torch.device,
        key: str,
        name: Optional[str] = None,
        cpu_core: Optional[int] = None,
    ) -> _DeviceWorker:
        """
        Create and start a worker bound to the provided device.
        """
        warmup_fn = self._resolve_worker_warmup(dev, key)
        w = _DeviceWorker(
            device=dev,
            rwlock=self._locks[key],
            on_task_finished=self._on_task_finished,
            on_worker_exit=self._on_worker_exit,
            name=name,
            inference_mode=self._inference_mode,
            cpu_core=cpu_core,
            warmup_fn=warmup_fn,
            key_override=key,
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
        cuda_event: Optional[torch.cuda.Event] = None,
        **kwargs,
    ) -> Future:
        """
        Asynchronously schedule work on the given device; returns a Future.
        Optional (CUDA): pass `cuda_event=` to ensure work waits for the recorded event.
        """
        key = self._resolve_device_key(device)
        dev = self._devices_by_key[key]
        worker = self._pick_worker(key)
        if cuda_event is not None and dev.type != "cuda":
            raise ValueError("cuda_event is only valid for CUDA devices")

        if DEBUG_ON: log.debug(f"submit: device={key} fn={getattr(fn, '__name__', repr(fn))}")
        # Mark in-flight before enqueue to avoid races with wait().
        self._mark_scheduled(key)
        try:
            return worker.submit(fn, *args, cuda_event=cuda_event, **kwargs)
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
        cuda_event: Optional[torch.cuda.Event] = None,
        **kwargs,
    ) -> Future:
        """
        Schedule work that must execute sequentially on a device. Tasks are
        enqueued onto a dedicated worker so they run in submission order.
        """
        key = self._resolve_device_key(device)
        dev = self._devices_by_key[key]
        if cuda_event is not None and dev.type != "cuda":
            raise ValueError("cuda_event is only valid for CUDA devices")

        with self._dispatch_lock:
            group = self._worker_groups.get(key)
            if group is None:
                group = []
                self._worker_groups[key] = group
            if key not in self._dispatch_rr:
                self._dispatch_rr[key] = 0
            if not group:
                fresh = self._spawn_worker(dev, key, name=f"DPWorker-{key}#0")
                group.append(fresh)
                self._dispatch_rr[key] = 0
            self._refresh_serial_worker_locked(key)
            worker = self._serial_workers.get(key)

        if worker is None:
            raise ValueError(f"No serial worker available for device '{key}'")

        if DEBUG_ON: log.debug(f"submit_serial: device={key} fn={getattr(fn, '__name__', repr(fn))}")
        self._mark_scheduled(key)
        try:
            return worker.submit(fn, *args, cuda_event=cuda_event, **kwargs)
        except BaseException:
            self._mark_finished(key)
            raise

    def do(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        cuda_event: Optional[torch.cuda.Event] = None,
        **kwargs,
    ) -> Any:
        """
        Synchronously schedule work and block for the result.
        """
        fut = self.submit(device, fn, *args, cuda_event=cuda_event, **kwargs)
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

        if hasattr(self, "_device_smi_handles"):
            with self._device_smi_lock:
                for handle in list(self._device_smi_handles.values()):
                    try:
                        handle.close()
                    except Exception:
                        pass
                self._device_smi_handles.clear()

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
        key = self._resolve_device_key(device)
        lk = self._locks.get(key)
        if lk is None:
            raise ValueError(f"Unknown device for pool: {device}")
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
            fam_keys = self._family_keys.get(device)
            if fam_keys is not None:
                pairs = [(k, self._locks[k]) for k in fam_keys]
                return _ReadLockGroup(pairs)
            if device in self._locks:
                return self._locks[device].reader()
            raise ValueError(f"Unknown device for pool: {device}")

        # torch.device / int / 'cuda:0' etc.
        dev = _coerce_device(device)
        key = self._key(dev)

        # Family device with index=None -> all devices of that type
        if dev.index is None:
            fam_keys = self._family_keys.get(key)
            if not fam_keys:
                raise ValueError(f"No devices of type '{dev.type}' in pool")
            pairs = [(k, self._locks[k]) for k in fam_keys]
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
        key = self._resolve_device_key(device)
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
                w = self._spawn_worker(dev, key, name=f"DPWorker-{key}#0")
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
        seen: set[str] = set()
        for s in scope:
            if isinstance(s, str):
                if s == "all":
                    for key in self._ordered_keys:
                        if key not in seen:
                            keys.append(key)
                            seen.add(key)
                    continue

                fam_keys = self._family_keys.get(s)
                if fam_keys is not None:
                    for key in fam_keys:
                        if key not in seen:
                            keys.append(key)
                            seen.add(key)
                    continue

                if s not in self._locks:
                    raise ValueError(f"Unknown device key in scope: {s}")
                if s not in seen:
                    keys.append(s)
                    seen.add(s)
            else:
                dev = _coerce_device(s)
                k = self._key(dev)
                if dev.index is None:
                    fam_keys = self._family_keys.get(k)
                    if not fam_keys:
                        raise ValueError(f"No devices of type '{dev.type}' in pool")
                    for key in fam_keys:
                        if key not in seen:
                            keys.append(key)
                            seen.add(key)
                else:
                    if k not in self._locks:
                        raise ValueError(f"Device not in pool: {dev}")
                    if k not in seen:
                        keys.append(k)
                        seen.add(k)
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

    def _physical_key(self, key: str) -> str:
        """
        Map a logical worker/alias key back to its physical device key.
        """
        return getattr(self, "_virtual_to_parent", {}).get(key, key)

    def _invoke_empty_cache(self, fn: Callable[..., Any], dev: torch.device) -> None:
        """
        Call an empty_cache-like callable, preferring a `device` argument when
        supported and falling back to positional or zero-arg variants.
        """
        supports_kw, supports_pos = _analyze_empty_cache_callable(fn)
        if supports_kw:
            try:
                fn(device=dev)
                return
            except TypeError:
                if DEBUG_ON:
                    log.debug("empty_cache callable rejected keyword arg; retrying positional (%s)", fn)
        if supports_pos:
            try:
                fn(dev)
                return
            except TypeError:
                if DEBUG_ON:
                    log.debug("empty_cache callable rejected positional arg; retrying no-arg (%s)", fn)
        fn()

    def _run_empty_cache_for_device(self, key: str, dev: torch.device) -> Optional[float]:
        """
        Execute an empty_cache call for the given device. Returns execution time in seconds.
        """
        start = time.time()
        if dev.type == "cuda":
            live = getattr(torch.cuda, "empty_cache", None) if hasattr(torch, "cuda") else None
            use_fn = live if callable(live) else TORCH_CUDA_EMPTY_CACHE
            if use_fn is None:
                if DEBUG_ON:
                    log.debug("DP-Janitor: no empty_cache callable available for %s", key)
                return None
            target = dev if dev.index is not None else "cuda"
            with torch.cuda.device(target):
                self._invoke_empty_cache(use_fn, dev)
            return time.time() - start

        if dev.type == "xpu" and hasattr(torch, "xpu"):
            live = getattr(torch.xpu, "empty_cache", None)
            use_fn = live if callable(live) else TORCH_XPU_EMPTY_CACHE
            if use_fn is None:
                if DEBUG_ON:
                    log.debug("DP-Janitor: no empty_cache callable available for %s", key)
                return None
            target = dev if dev.index is not None else "xpu"
            with torch.xpu.device(target):
                self._invoke_empty_cache(use_fn, dev)
            return time.time() - start

        if dev.type == "mps" and hasattr(torch, "mps"):
            live = getattr(torch.mps, "empty_cache", None)
            use_fn = live if callable(live) else TORCH_MPS_EMPTY_CACHE
            if use_fn is None:
                if DEBUG_ON:
                    log.debug("DP-Janitor: no empty_cache callable available for %s", key)
                return None
            self._invoke_empty_cache(use_fn, dev)
            return time.time() - start

        if DEBUG_ON:
            log.debug("DP-Janitor: unsupported device type '%s' for key %s", dev.type, key)
        return None

    @staticmethod
    def _format_gib_value(value: float) -> str:
        text = f"{value:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}G"

    def _device_smi_identifier(self, dev: torch.device) -> Optional[str]:
        if Device is None:
            return None
        if dev.type == "cuda":
            idx = dev.index
            if idx is None:
                return None
            prefix = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            return f"{prefix}:{idx}"
        if dev.type == "xpu":
            idx = dev.index
            if idx is None:
                return None
            return f"xpu:{idx}"
        return None

    def _query_device_vram_gib(self, key: str) -> Optional[float]:
        if Device is None:
            return None
        if not hasattr(self, "_device_smi_lock"):
            self._device_smi_lock = threading.Lock()
            self._device_smi_handles = {}
            self._device_smi_failures = set()
        dev = self._devices_by_key.get(key)
        if dev is None:
            return None
        identifier = self._device_smi_identifier(dev)
        if identifier is None:
            return None

        with self._device_smi_lock:
            if identifier in self._device_smi_failures:
                return None
            handle = self._device_smi_handles.get(identifier)
            if handle is None:
                try:
                    handle = Device(identifier)
                except Exception:
                    self._device_smi_failures.add(identifier)
                    return None
                self._device_smi_handles[identifier] = handle

        try:
            metrics = handle.metrics(fast=True)
        except Exception:
            with self._device_smi_lock:
                self._device_smi_failures.add(identifier)
                stored = self._device_smi_handles.pop(identifier, None)
                if stored is not None:
                    try:
                        stored.close()
                    except Exception:
                        pass
            return None

        memory_used = getattr(metrics, "memory_used", None)
        if memory_used is None:
            return None
        return float(memory_used) / (1024 ** 3)

    def _format_vram_summary(self, physical_keys: Iterable[str]) -> str:
        readings: List[str] = []
        for key in physical_keys:
            value = self._query_device_vram_gib(key)
            if value is None:
                continue
            readings.append(f"{key}={self._format_gib_value(value)}")
        return ", ".join(readings) if readings else "n/a"

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
        if not hasattr(self, "_gc_done_physical"):
            self._gc_done_physical = {}
        if not hasattr(self, "_gc_pending_physical"):
            self._gc_pending_physical = {}
        elif not isinstance(self._gc_pending_physical, dict):
            self._gc_pending_physical = {k: 1 for k in self._gc_pending_physical}
        if not hasattr(self, "_last_gc_done_physical"):
            self._last_gc_done_physical = {}
        if not hasattr(self, "_physical_children"):
            self._physical_children = {}

        self._mark_finished(key)

        trigger_gc = False
        with self._stats_lock:
            self._per_device_done[key] = self._per_device_done.get(key, 0) + 1
            self._total_done += 1
            dev = self._devices_by_key.get(key)
            if (
                dev is not None
                and self._empty_cache_every_n > 0
                and dev.type in ("cuda", "xpu", "mps")
            ):
                physical_key = self._physical_key(key)
                current = self._gc_done_physical.get(physical_key, 0) + 1
                self._gc_done_physical[physical_key] = current
                if current % self._empty_cache_every_n == 0:
                    pending_map = self._gc_pending_physical
                    pending_map[physical_key] = pending_map.get(physical_key, 0) + 1
                    self._gc_generation += 1
                    trigger_gc = True
                    if DEBUG_ON:
                        log.debug(
                            "GC trigger set by %s (physical=%s): per_physical_done=%d threshold=%d total_done=%d pending=%s",
                            key,
                            physical_key,
                            current,
                            self._empty_cache_every_n,
                            self._total_done,
                            sorted(k for k, count in pending_map.items() if count > 0),
                        )
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

        physical_children = getattr(self, "_physical_children", {})
        per_done_physical: Dict[str, int] = {}
        for phys_key, members in physical_children.items():
            per_done_physical[phys_key] = sum(per_done.get(member, 0) for member in members)

        raw_pending = getattr(self, "_gc_pending_physical", {})
        if isinstance(raw_pending, dict):
            pending_gc = sorted(k for k, count in raw_pending.items() if count > 0)
        else:
            pending_gc = sorted(raw_pending)

        snap: Dict[str, Any] = {
            "devices": sorted(self._devices_by_key.keys()),
            "per_done": per_done,
            "per_done_physical": per_done_physical,
            "total_done": total_done,
            "threshold": threshold,
            "inflight": inflight,
            "workers": workers,
            "meta": meta,
            "total_inflight": sum(inflight.values()),
            "total_workers": sum(workers.values()),
            "gc_passes": int(self._gc_passes),
            "gc_generation": int(self._gc_generation),
            "gc_generation_consumed": int(self._last_consumed_gc_generation),
            "last_gc_ts": self._last_gc_ts,
            "now": time.time(),
            "pending_gc": pending_gc,
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
        pending = snap.get("pending_gc") or []
        if pending:
            return True
        per_done_physical = snap.get("per_done_physical") or {}
        last_done_physical = getattr(self, "_last_gc_done_physical", {})
        for phys_key, done_now in per_done_physical.items():
            done_prev = last_done_physical.get(phys_key, 0)
            if done_now - done_prev >= thr:
                return True
        return False

    def _update_gc_watermarks(self, snap_after: Dict[str, Any]) -> None:
        """
        Record 'done' counters as of a GC pass to require fresh progress
        before a subsequent pass is allowed.
        """
        threshold = int(self._empty_cache_every_n)
        per_done_physical = snap_after.get("per_done_physical") or {}
        per_done = snap_after.get("per_done") or {}
        meta = snap_after.get("meta") or {}
        processed = snap_after.get("_gc_processed_devices")
        if processed is None:
            processed_iter = per_done_physical.keys()
        else:
            processed_iter = processed

        for phys_key in processed_iter:
            done_phys = per_done_physical.get(phys_key)
            if done_phys is None:
                continue
            if threshold <= 0:
                self._last_gc_done_physical[phys_key] = done_phys
            else:
                self._last_gc_done_physical[phys_key] = done_phys - (done_phys % threshold)

            members = self._physical_children.get(phys_key, {phys_key})
            for member in members:
                done_member = per_done.get(member)
                if done_member is None:
                    continue
                dev_type = meta.get(member, {}).get("type")
                if threshold <= 0 or dev_type not in ("cuda", "xpu", "mps"):
                    self._last_gc_done_per_device[member] = done_member
                else:
                    self._last_gc_done_per_device[member] = done_member - (done_member % threshold)

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

            with self._stats_lock:
                current_generation = self._gc_generation
                last_generation = self._last_consumed_gc_generation
                last_gc_ts = self._last_gc_ts

            if current_generation == last_generation:
                if DEBUG_ON:
                    log.debug("DP-Janitor: trigger generation already consumed; skipping")
                continue

            min_interval = self._gc_min_interval_s
            if min_interval > 0.0 and last_gc_ts is not None:
                elapsed = time.time() - last_gc_ts
                if elapsed < min_interval:
                    wait_for = min_interval - elapsed
                    if DEBUG_ON:
                        log.debug(
                            f"DP-Janitor: last pass {elapsed * 1000:.1f}ms ago; waiting {wait_for * 1000:.1f}ms to honor min interval"
                        )
                    if self._stop_event.wait(timeout=wait_for):
                        if DEBUG_ON:
                            log.debug("DP-Janitor: stop event set during min-interval wait; exiting")
                        break
                    if self._stop_event.is_set():
                        if DEBUG_ON:
                            log.debug("DP-Janitor: stop event observed after min-interval wait; exiting")
                        break
                    with self._stats_lock:
                        current_generation = self._gc_generation
                        last_generation = self._last_consumed_gc_generation
                    if current_generation == last_generation:
                        if DEBUG_ON:
                            log.debug("DP-Janitor: no pending GC generation after min-interval wait; skipping")
                        continue

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
                with self._stats_lock:
                    self._last_consumed_gc_generation = current_generation
                continue

            t0 = time.time()

            try:
                pre = self._collect_state_snapshot()
                if DEBUG_ON:
                    log.debug(
                        "DP-Janitor: pre-snapshot taken: total_done=%s threshold=%s inflight=%s pending=%s",
                        pre["total_done"],
                        pre["threshold"],
                        pre["inflight"],
                        pre.get("pending_gc"),
                    )
                    log.debug("GC trigger received; evaluating whether to run…")
                pending_targets = [k for k in pre.get("pending_gc", []) if k in self._locks]
            except Exception as e:
                try:
                    log.warn(f"Failed to render GC pre-snapshot: {e!r}")
                except Exception:
                    pass
                raw_pending = getattr(self, "_gc_pending_physical", {})
                if isinstance(raw_pending, dict):
                    pending_targets = sorted(k for k, count in raw_pending.items() if count > 0)
                else:
                    pending_targets = sorted(raw_pending)

            if not pending_targets:
                if DEBUG_ON:
                    log.debug("DP-Janitor: no pending devices after snapshot; marking generation %d consumed", current_generation)
                with self._stats_lock:
                    self._last_consumed_gc_generation = max(self._last_consumed_gc_generation, current_generation)
                continue

            processed_devices: List[str] = []
            skipped_devices: List[str] = []
            per_device_durations: Dict[str, float] = {}

            for key in pending_targets:
                dev = self._devices_by_key.get(key)
                if dev is None or dev.type not in ("cuda", "xpu", "mps"):
                    skipped_devices.append(key)
                    continue
                lk = self._locks.get(key)
                if lk is None:
                    skipped_devices.append(key)
                    continue
                if DEBUG_ON:
                    log.debug("DP-Janitor: attempting writer lock for %s", key)
                with lk.writer():
                    if DEBUG_ON:
                        log.debug("DP-Janitor: acquired writer lock for %s", key)
                    duration = self._run_empty_cache_for_device(key, dev)
                    if duration is not None:
                        per_device_durations[key] = duration
                processed_devices.append(key)

            if not processed_devices and DEBUG_ON:
                log.debug("DP-Janitor: no eligible accelerator devices found in pending=%s", pending_targets)

            t1 = time.time()
            prev_gc_ts = self._last_gc_ts
            if processed_devices:
                self._gc_passes += 1
                self._last_gc_ts = t1
            gc_timestamp = datetime.fromtimestamp(t1, tz=timezone.utc).isoformat()
            if prev_gc_ts is None:
                since_last_gc = "since last GC: n/a"
            else:
                delta_s = t1 - prev_gc_ts
                since_last_gc = f"since last GC: {delta_s:.3f}s ({delta_s * 1000:.1f}ms)"

            if processed_devices:
                vram_summary = self._format_vram_summary(processed_devices)
                try:
                    post = self._collect_state_snapshot()
                    post["_gc_processed_devices"] = processed_devices
                    self._update_gc_watermarks(post)
                    devices_clause = ", ".join(processed_devices)
                    log.info(
                        f"GC completed in {t1 - t0:.3f}s (pass #{self._gc_passes}) at {gc_timestamp}; devices={devices_clause}; VRAM {vram_summary}; {since_last_gc}."
                    )
                    if DEBUG_ON:
                        log.debug(
                            "DP-Janitor: post-snapshot inflight=%s per_done=%s per_done_physical=%s durations=%s",
                            post["inflight"],
                            post["per_done"],
                            post.get("per_done_physical"),
                            per_device_durations,
                        )
                except Exception as e:
                    try:
                        log.warn(f"Failed to render GC post-snapshot: {e!r}")
                    except Exception:
                        pass

            with self._stats_lock:
                pending_map = self._gc_pending_physical
                if not isinstance(pending_map, dict):
                    pending_map = {k: 1 for k in pending_map}
                    self._gc_pending_physical = pending_map
                for key in processed_devices:
                    pending_map.pop(key, None)
                    self._last_gc_done_physical[key] = self._gc_done_physical.get(key, 0)
                for key in skipped_devices:
                    pending_map.pop(key, None)
                self._last_consumed_gc_generation = max(self._last_consumed_gc_generation, current_generation)
                if any(count > 0 for count in pending_map.values()):
                    self._gc_event.set()

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
