# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import threading
import queue
import contextlib
from concurrent.futures import Future
from typing import Callable, Any, Dict, Tuple, Optional, Iterable, Union, List

import torch  # hard requirement


DeviceLike = Union[str, int, torch.device]


def _mps_available() -> bool:
    return hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _coerce_device(d: DeviceLike) -> torch.device:
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        if torch.cuda.is_available():
            return torch.device("cuda", d)
        if hasattr(torch, "xpu") and getattr(torch, "xpu").is_available():  # type: ignore[attr-defined]
            return torch.device("xpu", d)
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


@contextlib.contextmanager
def _device_ctx(dev: torch.device):
    """Per-call guard matching the backend."""
    if dev.type == "cuda":
        with torch.cuda.device(dev.index):
            yield
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        with torch.xpu.device(dev.index):  # type: ignore[attr-defined]
            yield
    else:
        # cpu/mps/meta -> no special context manager
        yield


def _activate_thread_device(dev: torch.device):
    """Pin the *thread* to the device where supported (CUDA/XPU)."""
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index)
    elif dev.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_device(dev.index)  # type: ignore[attr-defined]
    # MPS/CPU/META: nothing to do


# --------------------------- Read-Write Lock ---------------------------

class _RWLock:
    """
    Simple reader-writer lock:
      - Multiple readers may hold the lock.
      - Writers are exclusive.
      - Writer is re-entrant for its owning thread.
    """
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = None  # thread id
        self._writer_depth = 0

    # --- Write (exclusive) ---
    def acquire_write(self):
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._writer_depth += 1
                return
            while self._writer is not None or self._readers > 0:
                self._cond.wait()
            self._writer = me
            self._writer_depth = 1

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
            # Writer can acquire read re-entrantly
            if self._writer == me:
                self._readers += 1
                return
            while self._writer is not None:
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
    Acquire multiple device **write** locks in a deterministic order (to avoid deadlocks).
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


# --------------------------- Worker Thread ---------------------------

class _DeviceWorker:
    """Single dedicated thread that executes tasks for one device only."""
    def __init__(
        self,
        device: torch.device,
        rwlock: _RWLock,
        on_task_done: Callable[[str], None],
        name: Optional[str] = None,
        inference_mode: bool = False,
    ):
        self.device = device
        self.rwlock = rwlock
        self._on_task_done = on_task_done
        self.key = f"{device.type}:{device.index}" if device.index is not None else device.type
        self.name = name or f"DPWorker-{self.key}"
        # queue entries: (fn, args, kwargs, fut)
        self._q: "queue.Queue[Tuple[Callable[..., Any], tuple, dict, Future]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._inference_mode = inference_mode
        self._thread.start()

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        fut = Future()
        self._q.put((fn, args, kwargs, fut))
        return fut

    def stop(self):
        self._stop.set()
        self._q.put((lambda: None, tuple(), {}, Future()))  # poison pill

    def join(self):
        self._thread.join()

    def _run(self):
        # Pin this thread to the device once (CUDA/XPU; no-op for CPU/MPS)
        _activate_thread_device(self.device)

        maybe_inference = (
            torch.inference_mode() if self._inference_mode else contextlib.nullcontext()
        )
        with maybe_inference:
            while not self._stop.is_set():
                fn, args, kwargs, fut = self._q.get()
                if self._stop.is_set():
                    break
                if fut.cancelled():
                    self._q.task_done()
                    continue
                try:
                    # NORMAL TASKS TAKE A **READ** LOCK:
                    with self.rwlock.reader():
                        # Optional: launch into a user-provided CUDA stream for overlap
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
                    try:
                        self._on_task_done(self.key)
                    finally:
                        self._q.task_done()


# --------------------------- Public Pool ---------------------------

class DeviceThreadPool:
    """
    Multi-device thread pool:
      - Eagerly discovers & creates one worker per CUDA/XPU/MPS/CPU device at init.
      - Workers are pinned where applicable and run under correct device ctx per task.
      - submit() returns a Future; do() blocks for a one-liner await.
      - Per-device **RWLocks** exposed via API.
        * Workers take **read** locks per task.
        * You can take **write** locks to idle a device or all devices.
      - Optional `_cuda_stream` parameter for CUDA stream assignment.
      - **Counters** per device and global total; **janitor thread** triggers
        empty-cache after N completed tasks per device (CUDA/XPU/MPS).
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
        empty_cache_every_n: int = 50,  # set <=0 to disable janitor
    ):
        """
        Args:
            devices: explicit list of devices. If None, auto-discovers.
            include_cuda/xpu/mps/cpu: used only if devices is None.
            inference_mode: run all tasks under torch.inference_mode() in workers.
            empty_cache_every_n: when per-device completed-task count is a positive
                multiple of this value, schedule a global empty-cache pass.
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
                # single MPS device (no index)
                discovered.append(torch.device("mps"))
            if include_cpu:
                discovered.append(torch.device("cpu"))
            devices = discovered

        # EAGER CONSTRUCTION: locks, workers, device maps
        self._locks: Dict[str, _RWLock] = {}
        self._workers: Dict[str, _DeviceWorker] = {}
        self._devices_by_key: Dict[str, torch.device] = {}

        # Stats / GC control
        self._stats_lock = threading.Lock()
        self._per_device_done: Dict[str, int] = {}
        self._total_done: int = 0
        self._empty_cache_every_n = int(empty_cache_every_n)
        self._gc_event = threading.Event()
        self._stop_event = threading.Event()
        self._janitor: Optional[threading.Thread] = None

        for d in devices:
            dev = _coerce_device(d)
            # Now include cuda/xpu/mps/cpu
            if dev.type not in ("cuda", "xpu", "mps", "cpu"):
                continue
            key = self._key(dev)
            if key in self._workers:
                continue

            rw = _RWLock()
            worker = _DeviceWorker(
                device=dev,
                rwlock=rw,
                on_task_done=self._on_task_done,
                name=f"DPWorker-{key}",
                inference_mode=inference_mode,
            )
            self._locks[key] = rw
            self._workers[key] = worker
            self._devices_by_key[key] = dev
            self._per_device_done[key] = 0

        # Immutable ordering for global lock
        self._ordered_keys = sorted(self._locks.keys())

        # Start janitor if enabled and we have any accelerator devices
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
        worker = self._get_worker(dev)
        if _cuda_stream is not None and dev.type != "cuda":
            raise ValueError("_cuda_stream is only valid for CUDA devices")
        return worker.submit(fn, *args, _cuda_stream=_cuda_stream, **kwargs)

    def do(
        self,
        device: DeviceLike,
        fn: Callable[..., Any],
        /,
        *args,
        _cuda_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Any:
        """
        Synchronously schedule work on the given device and block for the result.
        """
        fut = self.submit(device, fn, *args, _cuda_stream=_cuda_stream, **kwargs)
        return fut.result()

    def shutdown(self, wait: bool = True):
        """Gracefully stop all workers and janitor."""
        self._stop_event.set()
        self._gc_event.set()  # wake janitor
        if self._janitor is not None and wait:
            self._janitor.join()

        for w in self._workers.values():
            w.stop()
        if wait:
            for w in self._workers.values():
                w.join()

    # --------------- Public Lock API ---------------

    def device_lock(self, device: DeviceLike):
        """
        **Exclusive** lock for a single device. While held the device worker blocks.
        """
        dev = _coerce_device(device)
        key = self._key(dev)
        lk = self._locks.get(key)
        if lk is None:
            raise ValueError(f"Unknown device for pool: {dev}")
        return lk.writer()

    def lock(self, devices: Optional[Iterable[DeviceLike]] = None):
        """
        **Exclusive** lock across multiple devices (default: all pool devices).
        Acquires each device's write lock in a canonical order to avoid deadlocks.
        """
        if devices is None:
            pairs = [(k, self._locks[k]) for k in self._ordered_keys]
        else:
            keys = [self._key(_coerce_device(d)) for d in devices]
            keys = sorted(keys)
            pairs = [(k, self._locks[k]) for k in keys]
        return _LockGroup(pairs)

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
        """Get completed-task count for a specific device."""
        key = self._key(_coerce_device(device))
        with self._stats_lock:
            return int(self._per_device_done.get(key, 0))

    def total_completed(self) -> int:
        """Get global completed-task count."""
        with self._stats_lock:
            return int(self._total_done)

    # --------------- Internals ---------------

    def _key(self, dev: torch.device) -> str:
        idx = "" if dev.index is None else f":{dev.index}"
        return f"{dev.type}{idx}"

    def _get_worker(self, dev: torch.device) -> _DeviceWorker:
        key = self._key(dev)
        worker = self._workers.get(key)
        if worker is None:
            raise ValueError(
                f"Device {dev} not part of this pool. Provide it in `devices=` at init."
            )
        return worker

    def _on_task_done(self, key: str) -> None:
        trigger_gc = False
        with self._stats_lock:
            self._per_device_done[key] += 1
            self._total_done += 1
            # Trigger GC on accelerator devices (CUDA/XPU/MPS)
            dev_type = self._devices_by_key[key].type
            if self._empty_cache_every_n > 0 and dev_type in ("cuda", "xpu", "mps"):
                n = self._per_device_done[key]
                if n % self._empty_cache_every_n == 0:
                    trigger_gc = True
        if trigger_gc:
            self._gc_event.set()

    def _janitor_loop(self):
        # Runs until shutdown. Coalesces multiple triggers.
        while True:
            self._gc_event.wait()
            if self._stop_event.is_set():
                break
            self._gc_event.clear()

            # Acquire global exclusive lock to pause all device workers.
            with self.lock():
                self._empty_all_caches()

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
            # Single device; no device context manager required
            torch.mps.empty_cache()  # type: ignore[attr-defined]
