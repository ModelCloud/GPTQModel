# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Concurrency coverage for shared looper state and worker completion."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.looper.eora_processor import EoraProcessor
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
    assert isinstance(processor._input_cache_lock, type(threading.RLock()))
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


@pytest.mark.parametrize("control_error", [KeyboardInterrupt("stop"), SystemExit("stop")])
def test_collect_worker_results_drains_before_raising_control_error(control_error):
    events = []
    futures = [
        _RecordingFuture(events, "first", error=control_error),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    with pytest.raises(type(control_error)) as exc_info:
        _collect_worker_results(futures)

    assert exc_info.value is control_error
    assert events == ["first", "second"]


def test_collect_worker_results_drains_before_raising_submission_control_error():
    events = []
    submission_error = KeyboardInterrupt("submission interrupted")
    futures = [
        _RecordingFuture(events, "first", error=ValueError("worker failed")),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    with pytest.raises(KeyboardInterrupt, match="submission interrupted") as exc_info:
        _collect_worker_results(futures, prior_error=submission_error)

    assert exc_info.value is submission_error
    assert events == ["first", "second"]


class _WorkerAbort(BaseException):
    """Custom non-Exception BaseException used for drain regression tests."""


def test_collect_worker_results_drains_direct_baseexception_subclass():
    events = []
    error = _WorkerAbort("worker abort")
    futures = [
        _RecordingFuture(events, "first", error=error),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    with pytest.raises(_WorkerAbort, match="worker abort") as exc_info:
        _collect_worker_results(futures)

    assert exc_info.value is error
    assert events == ["first", "second"]


def test_collect_worker_results_drains_before_raising_custom_submission_error():
    events = []
    submission_error = _WorkerAbort("submission abort")
    futures = [
        _RecordingFuture(events, "first", result=("a", 1)),
        _RecordingFuture(events, "second", result=("b", 2)),
    ]

    with pytest.raises(_WorkerAbort, match="submission abort") as exc_info:
        _collect_worker_results(futures, prior_error=submission_error)

    assert exc_info.value is submission_error
    assert events == ["first", "second"]


def test_thread_safe_dict_iteration_snapshots_during_concurrent_mutation():
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
    missing_value = tasks.pop("missing", None)
    assert missing_value is None
    key, value = tasks.popitem()
    assert (key, value) in {("a", 1), ("c", 3)}
    tasks.clear()
    assert not tasks


def test_thread_safe_dict_uses_builtin_dict_equality():
    tasks = _ThreadSafeDict({"a": 1, "b": 2})
    matching = _ThreadSafeDict({"a": 1, "b": 2})

    assert _ThreadSafeDict.__eq__ is dict.__eq__
    assert _ThreadSafeDict.__ne__ is dict.__ne__
    assert tasks == matching
    assert tasks == {"a": 1, "b": 2}
    different = _ThreadSafeDict({"a": 1, "b": 2, "c": 3})
    assert tasks != different
    assert tasks != {"a": 2}


def test_thread_safe_dict_builtin_equality_during_concurrent_mutation():
    tasks = _ThreadSafeDict()
    barrier = threading.Barrier(8)

    def writer(worker_index):
        barrier.wait(timeout=5)
        for iteration in range(1_000):
            tasks[f"{worker_index}:{iteration}"] = iteration

    def comparer(_worker_index):
        barrier.wait(timeout=5)
        for _ in range(5_000):
            snapshot = dict(tasks.items())
            assert isinstance(tasks == snapshot, bool)
            assert isinstance(tasks != {}, bool)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(writer, worker_index) for worker_index in range(4)]
        futures.extend(executor.submit(comparer, worker_index) for worker_index in range(4))
        for future in futures:
            future.result()

    assert len(tasks) == 4_000


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


def test_receive_input_cache_rewraps_missing_and_raw_cache_state():
    processor = LoopProcessor.__new__(LoopProcessor)
    processor._input_cache_lock = threading.RLock()
    first = InputCache([[torch.tensor([1.0])]], [], [], [])

    processor.receive_input_cache(first)

    assert isinstance(processor.inputs_cache, _ThreadSafeInputCache)
    assert processor.inputs_cache.unwrap() is first

    processor.inputs_cache = InputCache([], [], [], [])
    replacement = InputCache([[torch.tensor([2.0])]], [], [], [])
    processor.receive_input_cache(replacement)

    assert isinstance(processor.inputs_cache, _ThreadSafeInputCache)
    assert processor.inputs_cache.unwrap() is replacement


def test_eora_progress_uses_synchronized_draw_helper():
    class _ProgressObserved(Exception):
        pass

    processor = EoraProcessor.__new__(EoraProcessor)
    processor.draw_progress = MagicMock(side_effect=_ProgressObserved)
    module = MagicMock()
    module.adapter_cfg = object.__new__(Lora)
    module.name = "layer.proj"
    module.module_dtype = torch.float16

    with pytest.raises(_ProgressObserved):
        processor.process(module)

    processor.draw_progress.assert_called_once_with("EoRA: Processing layer.proj (torch.float16) in layer")


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


def test_device_metric_close_failures_are_logged_and_suppressed(monkeypatch):
    class _FailingHandle:
        def close(self):
            raise RuntimeError("close failed")

    debug = MagicMock()
    monkeypatch.setattr("gptqmodel.looper.loop_processor.log.debug", debug)
    processor = _make_processor()
    processor._device_smi_handles = {"cuda:0": _FailingHandle()}
    processor._cpu_device_smi = _FailingHandle()

    processor._close_device_smi_handles()

    assert debug.call_count == 2
    assert all(call.kwargs == {"exc_info": True} for call in debug.call_args_list)
    assert processor._device_smi_handles == {}
    assert processor._cpu_device_smi is None
