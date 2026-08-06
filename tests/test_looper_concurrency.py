# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Concurrency coverage for shared looper state and worker completion."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
import torch

from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.loop_processor import LoopProcessor, _ThreadSafeDict, _ThreadSafeInputCache
from gptqmodel.looper.stage_subset import _collect_worker_results


class _RecordingFuture:
    def __init__(self, events, label, *, result=None, error=None):
        self.events = events
        self.label = label
        self.value = result
        self.error = error

    def result(self):
        self.events.append(self.label)
        if self.error is not None:
            raise self.error
        return self.value


def _make_processor() -> LoopProcessor:
    processor = LoopProcessor.__new__(LoopProcessor)
    processor.lock = threading.Lock()
    processor._results_lock = threading.Lock()
    processor._progress_lock = threading.Lock()
    processor._fwd_time_lock = threading.Lock()
    processor._device_smi_lock = threading.RLock()
    processor._input_cache_lock = threading.RLock()
    processor._results = {}
    processor.tasks = _ThreadSafeDict()
    processor.inputs_cache = _ThreadSafeInputCache(InputCache([], [], [], []))
    processor.pb = None
    processor.fwd_time = None
    processor._device_smi_handles = {}
    processor._cpu_device_smi = None
    processor._device_metric_failures = set()
    return processor


def test_loop_processor_initializes_synchronized_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(LoopProcessor, "_init_device_smi_handles", lambda _self: {})
    monkeypatch.setattr(LoopProcessor, "_init_cpu_device_handle", lambda _self: None)

    processor = LoopProcessor(tokenizer=None, qcfg=MagicMock(), calibration=None)

    assert isinstance(processor.tasks, _ThreadSafeDict)
    assert isinstance(processor.inputs_cache, _ThreadSafeInputCache)
    assert processor.inputs_cache.unwrap().layer_inputs == []


def test_collect_worker_results_preserves_submission_order():
    events = []
    futures = [
        _RecordingFuture(events, "first", result=("a", 1)),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    assert _collect_worker_results(futures) == [("a", 1), ("b", 2)]
    assert events == ["first", "second"]


def test_collect_worker_results_drains_after_failure_and_raises_first_error():
    events = []
    first_error = RuntimeError("first worker failed")
    futures = [
        _RecordingFuture(events, "first", error=first_error),
        _RecordingFuture(events, "second", result=("b", 2)),
        _RecordingFuture(events, "third", error=ValueError("later worker failed")),
    ]

    with pytest.raises(RuntimeError, match="first worker failed") as exc_info:
        _collect_worker_results(futures)

    assert exc_info.value is first_error
    assert events == ["first", "second", "third"]


def test_collect_worker_results_drains_before_raising_submission_error():
    events = []
    submission_error = RuntimeError("submission failed")
    futures = [
        _RecordingFuture(events, "first", error=ValueError("worker failed")),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    with pytest.raises(RuntimeError, match="submission failed") as exc_info:
        _collect_worker_results(futures, prior_error=submission_error)

    assert exc_info.value is submission_error
    assert events == ["first", "second"]


def test_thread_safe_dict_uses_snapshots_during_concurrent_mutation():
    tasks = _ThreadSafeDict()
    barrier = threading.Barrier(8)

    def worker(worker_index):
        barrier.wait(timeout=5)
        for iteration in range(100):
            key = f"{worker_index}:{iteration}"
            tasks[key] = iteration
            assert key in tasks
            assert tasks.get(key) == iteration
            list(tasks)
            tasks.keys()
            tasks.values()
            tasks.items()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(8)]
        for future in futures:
            future.result()

    assert len(tasks) == 800
    assert len(tasks.items()) == 800


def test_thread_safe_dict_mutation_api():
    tasks = _ThreadSafeDict()
    tasks.update({"a": 1, "b": 2})
    assert tasks.setdefault("a", 3) == 1
    assert tasks.setdefault("c", 3) == 3
    del tasks["b"]
    assert tasks.pop("missing", None) is None
    key, value = tasks.popitem()
    assert (key, value) in {("a", 1), ("c", 3)}
    tasks.clear()
    assert not tasks


def test_thread_safe_input_cache_replacement_and_attribute_access():
    cache = _ThreadSafeInputCache(InputCache([], [], [], []))
    cache.layer_inputs = [[torch.tensor([1.0])]]
    assert len(cache.layer_inputs) == 1

    replacement = InputCache([[torch.tensor([2.0])]], [], [], [])
    cache.set_cache(replacement)
    assert cache.unwrap() is replacement
    assert torch.equal(cache.layer_inputs[0][0], torch.tensor([2.0]))
    cache.transient = True
    del cache.transient
    assert not hasattr(cache.unwrap(), "transient")

    proxy = _ThreadSafeInputCache(InputCache([], [], [], []))
    proxy.set_cache(cache)
    assert proxy.unwrap() is replacement


def test_loop_processor_shared_accessors_are_synchronized():
    processor = _make_processor()
    processor.pb = MagicMock()
    processor.pb.title.return_value = processor.pb
    processor.pb.subtitle.return_value = processor.pb

    def worker(worker_index):
        processor.result_save(str(worker_index), worker_index)
        processor.set_fwd_time(float(worker_index))
        processor.draw_progress(str(worker_index))
        processor.receive_layer_inputs([[torch.tensor([float(worker_index)])]])

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(50)]
        for future in futures:
            future.result()

    snapshot = processor.results()
    snapshot["external"] = True
    assert "external" not in processor.results()
    assert len(processor.results()) == 50
    assert processor.formatted_fwd_time().endswith(".000")
    assert processor.pb.draw.call_count == 50
    assert len(processor.inputs_cache.layer_inputs) == 1

    processor.receive_input_cache(InputCache([], [], [], []))
    processor.tasks["task"] = object()
    processor.clear_cache_data()
    assert processor.inputs_cache.layer_inputs == []
    assert not processor.tasks


def test_device_metric_handles_are_serialized_and_closed():
    class _Metrics:
        memory_used = 2 * 1024**3

    class _Handle:
        def __init__(self):
            self.closed = False

        def metrics(self, fast=True):
            assert fast is True
            assert not self.closed
            return _Metrics()

        def close(self):
            self.closed = True

    processor = _make_processor()
    cuda_handle = _Handle()
    cpu_handle = _Handle()
    processor._device_smi_handles = {"cuda:0": cuda_handle}
    processor._cpu_device_smi = cpu_handle

    report = processor.device_memory_report()
    assert report.startswith("cuda ")
    assert "2" in report
    assert processor._snapshot_cpu_memory_gib() == 2.0
    processor._close_device_smi_handles()

    assert cuda_handle.closed is True
    assert cpu_handle.closed is True
    assert processor._device_smi_handles == {}
    assert processor._cpu_device_smi is None
