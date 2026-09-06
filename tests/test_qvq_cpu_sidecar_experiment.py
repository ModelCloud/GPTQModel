"""Lifecycle and fallback checks for the explicit Fisher sidecar experiment."""

import io
import threading
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from scripts.experiments.qvq_cpu_sidecar import collector, sidecar


@pytest.fixture
def supported_call(monkeypatch):
    monkeypatch.setattr(collector, "_BASELINE_COMPATIBLE", True)
    monkeypatch.setattr(collector.sys, "_is_gil_enabled", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=9,
            minor=0,
            multi_processor_count=132,
            name="NVIDIA H200",
        ),
    )
    monkeypatch.setattr(
        sidecar,
        "cpu_runtime_inventory",
        lambda: {
            "numa_allowed_cpus": {"node0": [0], "node1": [24]},
        },
    )
    monkeypatch.setattr(sidecar, "capture_lock", threading.Lock())
    monkeypatch.setattr(collector, "_capture_plan", None)
    for name in ("_group_graph_cache", "_group_pools", "_group_streams"):
        monkeypatch.setattr(collector, name, {})
    monkeypatch.setenv("QVQ_CPU_FINALIZE", "1")
    modules = {
        str(i): SimpleNamespace(
            weight=SimpleNamespace(dtype=torch.bfloat16),
            in_features=5120,
            out_features=17408,
        )
        for i in range(400)
    }
    batches = [{"input_ids": torch.empty(2, 64, dtype=torch.long)}]
    kwargs = {
        "device": torch.device("cuda:0"),
        "accumulator_device": torch.device("cuda:0"),
        "gram_strategy": "streaming_projected",
        "gram_projection_rank": 256,
    }
    return object(), batches, modules, kwargs


def test_cpu_device_uses_reference_without_workers(monkeypatch):
    expected = object()
    monkeypatch.setattr(collector._baseline, "capture_yaqa_sketch_b", lambda *args, **kwargs: expected)
    monkeypatch.setattr(sidecar, "cpu_runtime_inventory", lambda: pytest.fail("CPU sidecar must not start"))
    assert collector.capture_yaqa_sketch_b(None, [], {}, device=torch.device("cpu")) is expected


def test_changed_reference_uses_fallback_before_cuda_probe(monkeypatch):
    expected = object()
    monkeypatch.setattr(collector, "_BASELINE_COMPATIBLE", False)
    monkeypatch.setattr(collector._baseline, "capture_yaqa_sketch_b", lambda *args, **kwargs: expected)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: pytest.fail("Unexpected CUDA probe"))
    assert collector.capture_yaqa_sketch_b(None, [], {}, device=torch.device("cuda:0")) is expected


def test_concurrent_capture_uses_reference(monkeypatch, supported_call):
    model, batches, modules, kwargs = supported_call
    expected = object()
    monkeypatch.setattr(collector._baseline, "capture_yaqa_sketch_b", lambda *args, **kwargs: expected)
    with sidecar.capture_lock:
        assert collector.capture_yaqa_sketch_b(model, batches, modules, **kwargs) is expected


def test_capture_exception_releases_plan_lock(monkeypatch, supported_call):
    model, batches, modules, kwargs = supported_call

    def fail(*args, **kwargs):
        raise RuntimeError("injected capture error")

    monkeypatch.setattr(collector, "_capture_hybrid_impl", fail)
    with pytest.raises(RuntimeError, match="injected capture error"):
        collector.capture_yaqa_sketch_b(model, batches, modules, **kwargs)
    assert not sidecar.capture_lock.locked()


def test_geometry_change_invalidates_cached_graphs(monkeypatch, supported_call):
    model, batches, modules, kwargs = supported_call
    monkeypatch.setattr(collector, "_capture_hybrid_impl", lambda *args, **kwargs: ({}, {}, {}))
    collector.capture_yaqa_sketch_b(model, batches, modules, **kwargs)
    collector._group_graph_cache["old geometry"] = object()
    modules["0"].in_features = 6144
    collector.capture_yaqa_sketch_b(model, batches, modules, **kwargs)
    assert not collector._group_graph_cache


def test_finish_drains_successful_jobs_after_an_error():
    worker = sidecar.FactorSidecar.__new__(sidecar.FactorSidecar)
    failed, completed = Future(), Future()
    failed.set_exception(RuntimeError("first job failed"))
    completed.set_result([("completed", ("input", "output"))])
    worker.jobs = deque([failed, completed])
    worker.factors = {}
    with pytest.raises(RuntimeError, match="first job failed"):
        worker.finish()
    assert not worker.jobs
    assert worker.factors == {"completed": ("input", "output")}


def test_clear_cache_releases_owned_resources(monkeypatch):
    monkeypatch.setattr(sidecar, "capture_lock", threading.Lock())
    for name in ("_group_graph_cache", "_group_pools", "_group_streams"):
        monkeypatch.setattr(collector, name, {"owned": object()})
    monkeypatch.setattr(collector, "_capture_plan", ("old plan",))
    closed = []
    monkeypatch.setattr(sidecar, "close_workers", lambda: closed.append(True))
    collector.clear_cache()
    assert collector._capture_plan is None
    assert all(not getattr(collector, name) for name in ("_group_graph_cache", "_group_pools", "_group_streams"))
    assert closed == [True]


def test_worker_factors_keep_canonical_serialization():
    worker = sidecar.FactorSidecar.__new__(sidecar.FactorSidecar)
    worker.local = threading.local()
    worker.local.ready = True
    worker.modules = {"linear": SimpleNamespace(in_features=64, out_features=128)}
    worker.sequences, worker.rank, worker.seed = 16.0, 256, 17
    worker.cls = collector.YaqaGramSketch
    worker.records = []
    # Synthetic fixture checks serialization and event ordering, not model quality.
    tensors = [
        torch.ones(1, 64, 256),
        torch.ones(1, 128, 256),
        torch.zeros(1, 64),
        torch.zeros(1, 128),
        torch.full((1, 64), 32.0),
        torch.full((1, 128), 16.0),
    ]
    events = []

    def data_ready():
        events.append("ready")
        tensors[2].fill_(256.0)
        tensors[3].fill_(256.0)

    result = worker.run(0, ("linear",), tensors, SimpleNamespace(synchronize=data_ready))
    assert events == ["ready"]
    pair = result[0][1]
    assert all(type(factor) is collector._baseline.YaqaGramSketch for factor in pair)
    buffer = io.BytesIO()
    torch.save(pair, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    for factor, saved in zip(pair, restored, strict=True):
        assert saved.normalizer == factor.normalizer
        assert saved.seed == factor.seed
        assert torch.equal(saved.source, factor.source)
        assert torch.equal(saved.source_diagonal, factor.source_diagonal)
        assert torch.equal(saved.diagonal, torch.full_like(saved.diagonal, 0.015625))
