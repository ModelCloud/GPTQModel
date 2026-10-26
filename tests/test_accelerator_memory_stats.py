# SPDX-License-Identifier: Apache-2.0
"""Allocator telemetry must follow the active accelerator, never torch.cuda."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.looper import eora_processor, loop_processor
from gptqmodel.models.writer import PROCESS_USED_MEMORY
from gptqmodel.utils import torch as gm_torch


@pytest.fixture(autouse=True)
def clear_allocator_cache():
    resolver = gm_torch.resolve_memory_stats_callable
    resolver.cache_clear()
    yield
    resolver.cache_clear()


@pytest.mark.parametrize("device_type", ["cuda", "xpu", "npu"])
def test_resolver_dispatches_to_requested_backend(monkeypatch, device_type):
    device = SimpleNamespace(type=device_type, index=1)
    expected = {"active_bytes.all.current": 7, "active_bytes.all.peak": 11}
    allocators = {}
    for backend in ("cuda", "xpu", "npu"):
        allocator = Mock(return_value=expected)
        allocators[backend] = allocator
        monkeypatch.setattr(gm_torch, f"HAS_{backend.upper()}", True)
        monkeypatch.setattr(torch, backend, SimpleNamespace(memory_stats=allocator), raising=False)

    assert gm_torch.memory_stats_for_device(device) == expected
    for backend, allocator in allocators.items():
        if backend == device_type:
            allocator.assert_called_once_with(device)
        else:
            allocator.assert_not_called()


@pytest.mark.parametrize("device_type", ["cuda", "xpu", "npu"])
@pytest.mark.parametrize("unavailable", ["disabled", "missing_namespace", "missing_method", "not_callable"])
def test_resolver_handles_unavailable_backends(monkeypatch, device_type, unavailable):
    allocator = Mock()
    namespace = SimpleNamespace(memory_stats=allocator)
    monkeypatch.setattr(gm_torch, f"HAS_{device_type.upper()}", unavailable != "disabled")
    monkeypatch.setattr(torch, device_type, namespace, raising=False)
    if unavailable == "missing_namespace":
        monkeypatch.delattr(torch, device_type)
    elif unavailable == "missing_method":
        del namespace.memory_stats
    elif unavailable == "not_callable":
        namespace.memory_stats = None

    assert gm_torch.memory_stats_for_device(SimpleNamespace(type=device_type, index=0)) == {}
    allocator.assert_not_called()


def test_cpu_reports_no_allocator_stats():
    """A device without an accelerator allocator stays empty instead of raising."""

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}


def test_cpu_has_no_allocator_stats_callable():
    assert gm_torch.resolve_memory_stats_callable("cpu") is None


def test_missing_allocator_reports_empty_stats(monkeypatch):
    """When the device type has no usable allocator the query returns an empty mapping."""

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: None)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}


def test_stats_come_from_the_requested_device_allocator(monkeypatch):
    """The active accelerator's allocator is queried with the requested device."""

    queried = []

    def fake_allocator(device):
        queried.append(device)
        return {"active_bytes.all.peak": 7}

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: fake_allocator)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {"active_bytes.all.peak": 7}
    assert queried == [torch.device("cpu")]


def test_allocator_failures_do_not_break_telemetry(monkeypatch):
    """Telemetry is best effort: a failing allocator query yields an empty mapping."""

    def failing_allocator(device):
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: failing_allocator)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}


def _eora_memory_row():
    processor = object.__new__(eora_processor.EoraProcessor)
    processor.lock = threading.Lock()
    processor.durations, processor.module_names, processor.log = [], [], []
    processor.qcfg = SimpleNamespace(dynamic=None)
    for name in ("draw_progress", "module_feature_summary", "module_dtype_size_summary", "log_new_row"):
        setattr(processor, name, Mock())
    processor.formatted_fwd_time = Mock(return_value="0")
    processor._finalize_eigen_scaling_matrix = Mock(return_value=None)
    processor.eora_compute_lora = Mock(return_value=(
        torch.zeros(1, 2, dtype=torch.float16),
        torch.zeros(2, 1, dtype=torch.float16),
    ))
    module = SimpleNamespace(
        name="proj",
        layer_index=0,
        adapter_cfg=Lora(rank=1),
        module_dtype=torch.float16,
        weight=torch.zeros(2, 2, dtype=torch.float16),
        state={
            "w_wq_diff": torch.zeros(2, 2),
            "wq": torch.zeros(2, 2, dtype=torch.float16),
        },
    )
    processor.process(module)
    processor.log_new_row.assert_called_once_with(processor.log[0])
    return processor.log[0][PROCESS_USED_MEMORY]


@pytest.mark.parametrize("device_type", ["cuda", "xpu", "npu"])
@pytest.mark.parametrize("second_device", ["same", "populated", "empty", "both_empty", "error"])
@pytest.mark.parametrize("peak", [False, True], ids=["loop_current", "eora_peak"])
def test_memory_rows_keep_device_slots(monkeypatch, device_type, second_device, peak):
    devices = [SimpleNamespace(type=device_type, index=index) for index in range(2)]
    first_stats = {"active_bytes.all.current": 1.25 * 1024 ** 2, "active_bytes.all.peak": 3.5 * 1024 ** 2}
    second_stats = {"active_bytes.all.current": 2.5 * 1024 ** 2, "active_bytes.all.peak": 7 * 1024 ** 2}
    results = [first_stats]
    if second_device == "populated":
        results.append(second_stats)
    elif second_device == "empty":
        results.append({})
    elif second_device == "both_empty":
        results = [{}, {}]
    elif second_device == "error":
        results.append(RuntimeError("allocator unavailable"))
    allocator = Mock(side_effect=results)
    monkeypatch.setattr(gm_torch, f"HAS_{device_type.upper()}", True)
    namespace = getattr(torch, device_type, None)
    if namespace is None:
        namespace = SimpleNamespace()
        monkeypatch.setattr(torch, device_type, namespace, raising=False)
    monkeypatch.setattr(namespace, "memory_stats", allocator, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", Mock(return_value=0))
    target = eora_processor if peak else loop_processor
    monkeypatch.setattr(target, "DEVICE_0", devices[0])
    monkeypatch.setattr(target, "DEVICE_1", devices[0] if second_device == "same" else devices[1])

    expected = "3.50MB" if peak else "1.25MB"
    if second_device == "both_empty":
        expected = "0.00MB"
    if second_device == "populated":
        expected += ", 7.00MB" if peak else ", 2.50MB"
    elif second_device != "same":
        expected += ", 0.00MB"
    assert (_eora_memory_row() if peak else loop_processor.get_max_memory()) == expected
    assert [call.args[0] for call in allocator.call_args_list] == (
        devices[:1] if second_device == "same" else devices
    )
    torch.cuda.device_count.assert_not_called()


@pytest.mark.parametrize("peak", [False, True], ids=["loop_current", "eora_peak"])
def test_cpu_memory_row_stays_zero(monkeypatch, peak):
    target = eora_processor if peak else loop_processor
    monkeypatch.setattr(target, "DEVICE_0", torch.device("cpu"))
    monkeypatch.setattr(target, "DEVICE_1", torch.device("cpu"))
    cuda_stats = Mock(side_effect=AssertionError("CPU telemetry must not query CUDA"))
    monkeypatch.setattr(torch.cuda, "memory_stats", cuda_stats)

    assert (_eora_memory_row() if peak else loop_processor.get_max_memory()) == "0.00MB"
    cuda_stats.assert_not_called()
