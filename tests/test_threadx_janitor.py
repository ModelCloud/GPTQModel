import contextlib
import threading
import time

import pytest
import torch

from gptqmodel.utils import threadx as threadx_mod


DeviceThreadPool = threadx_mod.DeviceThreadPool


class _DummyLock:
    @contextlib.contextmanager
    def writer(self):
        yield


def _make_pool():
    pool = DeviceThreadPool.__new__(DeviceThreadPool)
    pool._gc_event = threading.Event()
    pool._stop_event = threading.Event()
    pool._auto_gc_disable_cv = threading.Condition()
    pool._auto_gc_disable_count = 0
    pool._gc_debounce_s = 0.0
    pool._gc_min_interval_s = 0.0
    pool._stats_lock = threading.Lock()
    pool._per_device_done = {}
    pool._total_done = 0
    pool._empty_cache_every_n = 3
    pool._devices_by_key = {}
    pool._locks = {}
    pool._ordered_keys = []
    pool._worker_groups = {}
    pool._inflight = {}
    pool._inflight_cv = {}
    pool._last_gc_done_per_device = {}
    pool._gc_passes = 0
    pool._last_gc_ts = None
    pool._gc_generation = 0
    pool._last_consumed_gc_generation = 0
    pool._synchronize_all = lambda: None
    pool._virtual_to_parent = {}
    pool._family_keys = {}
    pool._dispatch_lock = threading.Lock()
    pool._warmup_lock = threading.Lock()
    pool._warmup_ran_keys = set()
    pool._worker_warmups = {}
    pool._serial_workers = {}
    pool._ordered_keys = []
    # Bind instance methods that rely on self
    pool._collect_state_snapshot = DeviceThreadPool._collect_state_snapshot.__get__(pool, DeviceThreadPool)
    pool._should_run_gc_from_snapshot = DeviceThreadPool._should_run_gc_from_snapshot.__get__(pool, DeviceThreadPool)
    pool._update_gc_watermarks = DeviceThreadPool._update_gc_watermarks.__get__(pool, DeviceThreadPool)
    pool._mark_finished = DeviceThreadPool._mark_finished.__get__(pool, DeviceThreadPool)
    pool._on_task_finished = DeviceThreadPool._on_task_finished.__get__(pool, DeviceThreadPool)
    return pool


@pytest.mark.parametrize("threshold_triggers", [3])
def test_janitor_coalesces_pending_triggers(monkeypatch, threshold_triggers):
    pool = _make_pool()
    pool._empty_cache_every_n = threshold_triggers

    key = "cuda:0"
    dev = torch.device("cuda", 0)
    pool._devices_by_key[key] = dev
    pool._locks[key] = _DummyLock()
    pool._ordered_keys = [key]
    pool._worker_groups[key] = []
    pool._inflight[key] = 0
    pool._inflight_cv[key] = threading.Condition()
    pool._last_gc_done_per_device[key] = 0
    pool._per_device_done[key] = 0

    calls = {"count": 0}

    def fake_empty_cache():
        calls["count"] += 1

    monkeypatch.setattr(threadx_mod.torch.cuda, "empty_cache", fake_empty_cache, raising=False)
    monkeypatch.setattr(threadx_mod, "TORCH_CUDA_EMPTY_CACHE", fake_empty_cache, raising=False)

    @contextlib.contextmanager
    def fake_cuda_device(index):
        yield

    monkeypatch.setattr(threadx_mod.torch.cuda, "device", fake_cuda_device, raising=False)

    # Simulate multiple threshold triggers before janitor runs.
    for _ in range(threshold_triggers * 3):
        pool._inflight[key] = pool._inflight.get(key, 0) + 1
        pool._on_task_finished(key)

    assert pool._gc_generation == 3
    assert pool._gc_event.is_set()

    janitor = threading.Thread(target=pool._janitor_loop, daemon=True)
    janitor.start()

    start = time.time()
    while calls["count"] < 1 and time.time() - start < 1.0:
        time.sleep(0.01)

    # Allow janitor time to spin in case extra passes would occur.
    time.sleep(0.1)

    pool._stop_event.set()
    pool._gc_event.set()
    janitor.join(timeout=1.0)

    assert calls["count"] == 1
    assert pool._gc_passes == 1
    assert pool._last_consumed_gc_generation == pool._gc_generation
