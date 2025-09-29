# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch  # hard requirement

from ..utils.logger import setup_logger

log = setup_logger()


# Assumption: a logbar-style logger named `log` is already imported elsewhere.
# We will call log.info / log.debug / log.warn / log.error directly.


DeviceLike = Union[str, int, torch.device]


def _mps_available() -> bool:
    return (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _coerce_device(d: DeviceLike) -> torch.device:
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        if torch.cuda.is_available():
            return torch.device("cuda", d)
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return torch.device("xpu", d)
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


@contextlib.contextmanager
def _device_ctx(dev: torch.device):
    """Set the caller thread’s current device for CUDA/XPU so library handles match."""
    if dev.type == "cuda":
        with torch.cuda.device(dev.index):
            yield
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        with torch.xpu.device(dev.index):  # type: ignore[attr-defined]
            yield
    else:
        yield


def _activate_thread_device(dev: torch.device):
    """Pin the worker thread to the device."""
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index)
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_device(dev.index)  # type: ignore[attr-defined]
    # mps/cpu: nothing to pin


# --------------------------- Read-Write Lock (writer-preference) ---------------------------

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
            if self._writer == me:  # re-entrant
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
            # writer can re-enter as reader
            if self._writer == me:
                self._readers += 1
                return
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
    """
    def __init__(self, ordered_pairs: List[tuple[str, _RWLock]]):
        self._pairs = ordered_pairs

    def __enter__(self):
        for _, lk in self._pairs:
            lk.acquire_write()
        return self

    def __exit__(self, exc_type, exc, tb):
        for _, lk in reversed(self._pairs):
            lk.release_write()
        return False


class _ReadLockGroup(contextlib.AbstractContextManager):
    """
    Acquire multiple device **read** locks in deterministic order.
    """
    def __init__(self, ordered_pairs: List[tuple[str, _RWLock]]):
        self._pairs = ordered_pairs

    def __enter__(self):
        for _, lk in self._pairs:
            lk.acquire_read()
        return self

    def __exit__(self, exc_type, exc, tb):
        for _, lk in reversed(self._pairs):
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
        self._group = _LockGroup(pairs)

    def __enter__(self):
        return self._group.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._group.__exit__(exc_type, exc, tb)


# --------------------------- Worker Thread ---------------------------

class _DeviceWorker:
    """
    Single worker thread bound to one device.
    Queue entries: (is_task: bool, fn, args, kwargs, future)
    """
    def __init__(
        self,
        device: torch.device,
        rwlock: _RWLock,
        on_task_finished: Callable[[str], None],
        name: Optional[str] = None,
        inference_mode: bool = False,
    ):
        self.device = device
        self.rwlock = rwlock
        self._on_task_finished = on_task_finished
        self.key = f"{device.type}:{device.index}" if device.index is not None else device.type
        self.name = name or f"DPWorker-{self.key}"
        self._q: "queue.Queue[Tuple[bool, Callable[..., Any], tuple, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._inference_mode = inference_mode
        self._thread.start()

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        fut = Future()
        self._q.put((True, fn, args, kwargs, fut))
        return fut

    def stop(self):
        self._stop.set()
        self._q.put((False, lambda: None, (), {}, Future()))  # sentinel

    def join(self):
        self._thread.join()

    def _run(self):
        _activate_thread_device(self.device)
        maybe_inference = torch.inference_mode() if self._inference_mode else contextlib.nullcontext()
        with maybe_inference:
            while not self._stop.is_set():
                is_task, fn, args, kwargs, fut = self._q.get()
                try:
                    if not is_task:
                        break  # sentinel -> exit
                    # Tasks take a **read** lock so GC's writer lock can't interleave
                    with self.rwlock.reader():
                        stream = kwargs.pop("_cuda_stream", None)
                        with _device_ctx(self.device):
                            if stream is not None and self.device.type == "cuda":
                                with torch.cuda.stream(stream):
                                    result = fn(*args, **kwargs)
                            else:
                                result = fn(*args, **kwargs)
                    if not fut.cancelled():
                        fut.set_result(result)
                except BaseException as exc:
                    if not fut.cancelled():
                        fut.set_exception(exc)
                finally:
                    if is_task:
                        self._on_task_finished(self.key)
                    self._q.task_done()


# --------------------------- Public Pool ---------------------------

class DeviceThreadPool:
    """
    Multi-device thread pool with:
      - Eager discovery/creation of workers and locks for CUDA/XPU/MPS/CPU.
      - **Configurable worker counts per device** (default 1).
      - Correct per-thread device context.
      - submit()/do() for async/sync, with optional `_cuda_stream` (CUDA only).
      - Per-device **RWLocks** + global lock and family/all read-locks.
      - **wait(scope, lock=False/True)** to drain tasks (optionally with exclusive locks).
      - Per-device/global **completed** counters and **in-flight** counters.
      - Janitor: triggers empty-cache after N completions on accelerator devices, under a global lock.
      - GC diagnostics: logs before/after snapshots as ANSI tables via `log.info`.
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
        """
        if devices is None:
            discovered: List[torch.device] = []
            if include_cuda and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    discovered.append(torch.device("cuda", i))
            if include_xpu and hasattr(torch, "xpu") and torch.xpu.is_available():
                for i in range(torch.xpu.device_count()):  # type: ignore[attr-defined]
                    discovered.append(torch.device("xpu", i))
            if include_mps and _mps_available():
                discovered.append(torch.device("mps"))
            if include_cpu:
                discovered.append(torch.device("cpu"))
            devices = discovered

        self._locks: Dict[str, _RWLock] = {}
        self._devices_by_key: Dict[str, torch.device] = {}

        # Worker groups: key -> List[_DeviceWorker]
        self._worker_groups: Dict[str, List[_DeviceWorker]] = {}
        self._dispatch_rr: Dict[str, int] = {}   # round-robin index per key
        self._dispatch_lock = threading.Lock()

        # Stats / GC / inflight control
        self._stats_lock = threading.Lock()
        self._per_device_done: Dict[str, int] = {}
        self._total_done: int = 0

        self._empty_cache_every_n = int(empty_cache_every_n)
        self._gc_event = threading.Event()
        self._stop_event = threading.Event()
        self._janitor: Optional[threading.Thread] = None

        # in-flight (scheduled but not finished) counters + per-device CVs
        self._inflight: Dict[str, int] = {}
        self._inflight_cv: Dict[str, threading.Condition] = {}

        workers = workers or {}

        # Build locks, inflight structs, and workers eagerly
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

            n_workers = self._resolve_workers_for_device(dev, workers)
            group: List[_DeviceWorker] = []
            for wid in range(int(max(1, n_workers))):
                worker = _DeviceWorker(
                    device=dev,
                    rwlock=self._locks[key],
                    on_task_finished=self._on_task_finished,
                    name=f"DPWorker-{key}#{wid}",
                    inference_mode=inference_mode,
                )
                group.append(worker)
            self._worker_groups[key] = group
            self._dispatch_rr[key] = 0

        # Canonical lock order
        self._ordered_keys = sorted(self._locks.keys())

        # GC diagnostics counters
        self._gc_passes = 0
        self._last_gc_ts: Optional[float] = None

        # Start janitor if enabled and accelerators exist
        if self._empty_cache_every_n > 0 and any(
            self._devices_by_key[k].type in ("cuda", "xpu", "mps") for k in self._ordered_keys
        ):
            self._janitor = threading.Thread(
                target=self._janitor_loop, name="DP-Janitor", daemon=True
            )
            self._janitor.start()

    # --------------- Public Work API ---------------

    def submit(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        _cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Future:
        """
        Asynchronously schedule work on the given device; returns a Future.
        Optional (CUDA): pass `_cuda_stream=` to launch into a specific stream.
        """
        dev = _coerce_device(device)
        key = self._key(dev)
        worker = self._pick_worker(key)
        if _cuda_stream is not None and dev.type != "cuda":
            raise ValueError("_cuda_stream is only valid for CUDA devices")

        # mark in-flight before enqueue to avoid races with wait()
        self._mark_scheduled(key)
        try:
            return worker.submit(fn, *args, _cuda_stream=_cuda_stream, **kwargs)
        except BaseException:
            # roll back inflight if enqueue fails (rare)
            self._mark_finished(key)
            raise

    def do(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        _cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Any:
        """Synchronously schedule work and block for the result."""
        fut = self.submit(device, fn, *args, _cuda_stream=_cuda_stream, **kwargs)
        return fut.result()

    def shutdown(self, wait: bool = True):
        """Gracefully stop all workers and janitor."""
        self._stop_event.set()
        self._gc_event.set()  # wake janitor
        if self._janitor is not None and wait:
            self._janitor.join()

        for group in self._worker_groups.values():
            for w in group:
                w.stop()
        if wait:
            for group in self._worker_groups.values():
                for w in group:
                    w.join()

    # --------------- Public Lock API ---------------

    def device_lock(self, device: DeviceLike):
        """Exclusive lock for a single device (blocks all its workers)."""
        dev = _coerce_device(device)
        key = self._key(dev)
        lk = self._locks.get(key)
        if lk is None:
            raise ValueError(f"Unknown device for pool: {dev}")
        return lk.writer()

    def read_lock(self, device: DeviceLike | str):
        """
        Shared/read lock. Accepts:
          - concrete device: torch.device('cuda:0'), 'cuda:1'
          - family device:  torch.device('cuda'), 'cuda', 'xpu', 'mps', 'cpu'
          - 'all' for every device in the pool
        Returns a context manager.
        """
        # family string shortcut
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

        # family device with index=None -> all devices of that type
        if dev.index is None:
            fam = dev.type
            keys = [k for k in self._ordered_keys if k.startswith(fam)]
            if not keys:
                raise ValueError(f"No devices of type '{fam}' in pool")
            pairs = [(k, self._locks[k]) for k in keys]
            return _ReadLockGroup(pairs)

        # concrete device
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

    def wait(self, scope: Optional[Union[str, DeviceLike, Iterable[DeviceLike]]] = None, *, lock: bool = False):
        """
        Wait until in-flight tasks for `scope` drain to zero.

        scope:
          - None or 'all' -> all devices
          - 'cuda' | 'xpu' | 'mps' | 'cpu' -> all devices of that type
          - 'cuda:0' | 'xpu:1' -> specific device key
          - torch.device or iterable of the above
        lock:
          - False (default): block until drained, then return None.
          - True: return a context manager that **waits for drain AND acquires
                  exclusive write locks** over the scope. Usage:
                  `with pool.wait("cuda", lock=True): ...`
        """
        keys = self._resolve_scope_to_keys(scope)
        if lock:
            pairs = [(k, self._locks[k]) for k in sorted(keys)]
            return _WaitAndLock(pairs)

        # Pure wait without lock: wait for inflight to reach zero for each key.
        for k in keys:
            cv = self._inflight_cv[k]
            with cv:
                while self._inflight[k] > 0:
                    cv.wait()
        return None

    # --------------- Public Stats API ---------------

    def stats(self) -> Dict[str, Any]:
        """Return counters snapshot: per-device and global."""
        with self._stats_lock:
            return {
                "per_device": dict(self._per_device_done),
                "total": int(self._total_done),
                "threshold": int(self._empty_cache_every_n),
            }

    def device_completed(self, device: DeviceLike) -> int:
        key = self._key(_coerce_device(device))
        with self._stats_lock:
            return int(self._per_device_done.get(key, 0))

    def total_completed(self) -> int:
        with self._stats_lock:
            return int(self._total_done)

    # --------------- Internals ---------------

    def _key(self, dev: torch.device) -> str:
        idx = "" if dev.index is None else f":{dev.index}"
        return f"{dev.type}{idx}"

    def _pick_worker(self, key: str) -> _DeviceWorker:
        group = self._worker_groups.get(key)
        if not group:
            raise ValueError(f"Device {key} not part of this pool.")
        if len(group) == 1:
            return group[0]
        with self._dispatch_lock:
            idx = self._dispatch_rr[key]
            self._dispatch_rr[key] = (idx + 1) % len(group)
            return group[idx]

    def _resolve_workers_for_device(self, dev: torch.device, table: Dict[str, int]) -> int:
        key = self._key(dev)
        # exact override
        if key in table:
            return int(table[key])
        # per family
        fam_key = f"{dev.type}:per"
        if fam_key in table:
            return int(table[fam_key])
        # single device-type entries for cpu/mps
        if dev.type in ("cpu", "mps") and dev.type in table:
            return int(table[dev.type])
        return 1

    def _normalize_scope_to_keys(self, scope: Iterable[DeviceLike]) -> List[str]:
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
                    # family: cuda/xpu/mps/cpu
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

    def _resolve_scope_to_keys(self, scope: Optional[Union[str, DeviceLike, Iterable[DeviceLike]]]) -> List[str]:
        if scope is None or (isinstance(scope, str) and scope == "all"):
            return list(self._ordered_keys)
        if isinstance(scope, (str, torch.device, int)):
            return self._normalize_scope_to_keys([scope])
        return self._normalize_scope_to_keys(scope)

    # ---- inflight & completion accounting ----

    def _mark_scheduled(self, key: str) -> None:
        cv = self._inflight_cv[key]
        with cv:
            self._inflight[key] += 1

    def _mark_finished(self, key: str) -> None:
        cv = self._inflight_cv[key]
        with cv:
            self._inflight[key] -= 1
            if self._inflight[key] == 0:
                cv.notify_all()

    def _on_task_finished(self, key: str) -> None:
        # inflight decrement + counters + potential GC trigger
        self._mark_finished(key)

        trigger_gc = False
        with self._stats_lock:
            self._per_device_done[key] += 1
            self._total_done += 1
            dev_type = self._devices_by_key[key].type
            if self._empty_cache_every_n > 0 and dev_type in ("cuda", "xpu", "mps"):
                n = self._per_device_done[key]
                if n % self._empty_cache_every_n == 0:
                    trigger_gc = True
        if trigger_gc:
            self._gc_event.set()

    # ---- ANSI table rendering for GC diagnostics ----

    def _ansi_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render a simple ANSI/ASCII table with bold headers."""
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
        """Safely collect a snapshot of pool state for diagnostics."""
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
        """Build the ANSI table for the current snapshot."""
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
            rows.append([
                k, t, idx, str(w), str(infl),
                str(done), str(thr) if thr > 0 else "-",
                nextgc, accel
            ])

        table_main = self._ansi_table(headers, rows)

        totals_headers = ["Total Workers", "Total Inflight", "Total Done", "GC Passes", "Since Last GC (s)"]
        if snap["last_gc_ts"] is None:
            since = "-"
        else:
            since = f"{snap['now'] - snap['last_gc_ts']:.3f}"
        totals_rows = [[
            str(snap["total_workers"]),
            str(snap["total_inflight"]),
            str(snap["total_done"]),
            str(snap["gc_passes"]),
            since,
        ]]
        table_totals = self._ansi_table(totals_headers, totals_rows)
        return table_main + "\n" + table_totals

    # ---- janitor (global empty-cache under lock) ----

    def _synchronize_all(self):
        """
        Ensure devices are idle before empty_cache() to avoid races with outstanding kernels.
        """
        # CUDA
        try:
            if torch.cuda.is_available():
                for key in self._ordered_keys:
                    dev = self._devices_by_key[key]
                    if dev.type != "cuda":
                        continue
                    with torch.cuda.device(dev.index):
                        torch.cuda.synchronize()
        except Exception:
            pass

        # XPU
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
                torch.xpu.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass

        # MPS
        try:
            if _mps_available():
                torch.mps.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _janitor_loop(self):
        while True:
            self._gc_event.wait()
            if self._stop_event.is_set():
                break
            self._gc_event.clear()

            # PRE-GC snapshot & banner
            try:
                pre = self._collect_state_snapshot()
                # log.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
                # log.info("┃ DeviceThreadPool: GC pass starting               ┃")
                # log.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
                log.debug("GC trigger received; acquiring global exclusive lock…")
                #log.info(self._render_gc_table(pre))
            except Exception as e:
                try:
                    log.warn(f"Failed to render GC pre-snapshot: {e!r}")
                except Exception:
                    pass

            with self.lock():  # writer lock across ALL devices
                t0 = time.time()
                # Ensure all devices are idle before freeing cached blocks
                # self._synchronize_all() <-- too slow
                try:
                    self._empty_all_caches()
                except Exception as e:
                    try:
                        log.error(f"GC pass encountered an error: {e!r}")
                    except Exception:
                        pass
                t1 = time.time()

                # POST-GC snapshot & banner
                self._gc_passes += 1
                self._last_gc_ts = t1
                try:
                    # post = self._collect_state_snapshot()
                    log.info(f"GC completed in {t1 - t0:.3f}s (pass #{self._gc_passes}).")
                    # log.info(self._render_gc_table(post))
                    # log.info("── GC pass finished ──")
                except Exception as e:
                    try:
                        log.warn(f"Failed to render GC post-snapshot: {e!r}")
                    except Exception:
                        pass

    def _empty_all_caches(self):
        # CUDA
        if torch.cuda.is_available():
            for key in self._ordered_keys:
                dev = self._devices_by_key[key]
                if dev.type != "cuda":
                    continue
                with torch.cuda.device(dev.index):
                    torch.cuda.empty_cache()

        # XPU (if available)
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            for key in self._ordered_keys:
                dev = self._devices_by_key[key]
                if dev.type != "xpu":
                    continue
                with torch.xpu.device(dev.index):  # type: ignore[attr-defined]
                    torch.xpu.empty_cache()  # type: ignore[attr-defined]

        # MPS (if available)
        if _mps_available():
            torch.mps.empty_cache()  # type: ignore[attr-defined]
