# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4378: ggml-cuda entry must be serialized process-wide.

`_GGMLBridge.cuda_backend_lock` used to hand out one lock per CUDA device, so
forwards on cuda:0 and cuda:1 could be inside ggml-cuda at the same time.
ggml-cuda keeps cuBLAS handles and memory pools in process-global state, so
overlapping entry aborts the process (`CUDA error: an internal operation
failed ... cublasGemmEx`). These tests drive the real locking code in
`gguf_cpp` through a mocked bridge, because the ggml CUDA extension is not
built in CI here, and record concurrent entries in the backend critical
section.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel, _GGMLBridge


class _CudaTensor:
    """A CPU tensor that advertises a CUDA device to `GGUFCudaKernel.forward`."""

    def __init__(self, tensor: torch.Tensor, device: torch.device):
        self._tensor = tensor
        self.device = device
        self.dtype = tensor.dtype
        self.shape = tensor.shape

    def reshape(self, *shape):
        return _CudaTensor(self._tensor.reshape(*shape), self.device)


class _Concurrency:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.entries = 0

    @contextmanager
    def track(self):
        with self._lock:
            self.active += 1
            self.entries += 1
            self.max_active = max(self.max_active, self.active)
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1


class _FakePlan:
    def __init__(self, rows: int, out_features: int):
        self.rows = rows
        self.out_features = out_features

    def close(self) -> None:
        pass


class _FakeWeight:
    def close(self) -> None:
        pass


def _make_bare_bridge() -> _GGMLBridge:
    """Create a bridge without importing llama-cpp-python.

    ``_GGMLBridge.__init__`` loads the llama-cpp shared libraries, which the
    CPU CI does not have, so recreate only the lock state it owns. The pre-fix
    per-device map is populated as well so this regression can be run against
    the old tree for an A/B check (where it fails).
    """

    bridge = object.__new__(_GGMLBridge)
    bridge._cuda_backend_locks = {}
    bridge._cuda_backend_locks_guard = threading.Lock()
    bridge._cuda_backend_lock = threading.RLock()
    bridge.cuda_scratch_generation = lambda device_index: 0
    return bridge


def _make_layer(in_features: int, out_features: int) -> GGUFCudaKernel:
    layer = object.__new__(GGUFCudaKernel)
    object.__setattr__(layer, "qweight", torch.arange(16, dtype=torch.uint8))
    object.__setattr__(layer, "padded_in_features", in_features)
    object.__setattr__(layer, "out_features", out_features)
    object.__setattr__(layer, "adapter", None)
    object.__setattr__(layer, "gguf_tensor_qtype", "Q4_K")
    object.__setattr__(layer, "_ggml_cuda_plans", OrderedDict())
    object.__setattr__(layer, "_ggml_cuda_weight", None)
    object.__setattr__(layer, "_ggml_cuda_weight_key", None)
    object.__setattr__(layer, "_ggml_cuda_plan_generation", -1)
    return layer


def _drop_layer(layer: GGUFCudaKernel) -> None:
    # Keep `__del__` from touching the real bridge during GC.
    object.__setattr__(layer, "_ggml_cuda_weight_key", None)
    layer._ggml_cuda_plans.clear()


def _install_bridge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    weight_matrix: torch.Tensor,
    tracker: _Concurrency,
    run_delay: float = 0.02,
) -> _GGMLBridge:
    bridge = _make_bare_bridge()

    def fake_run(*, x, gguf_tensor_qtype, padded_in_features, out_features, plan=None, weight=None):
        with tracker.track():
            time.sleep(run_delay)
            output = (x._tensor.float() @ weight_matrix).to(torch.float16)
        if plan is None:
            plan = _FakePlan(rows=x.shape[0], out_features=out_features)
        return output, plan

    def fake_build_cuda_weight(**_kwargs):
        return _FakeWeight()

    bridge.quantized_matmul_cuda = fake_run
    bridge.build_cuda_weight = fake_build_cuda_weight
    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", lambda: bridge)
    monkeypatch.setattr(
        gguf_cpp,
        "_finish_ggml_output",
        lambda module, x_flat, output: output,
    )
    return bridge


def test_cuda_backend_lock_is_process_wide():
    bridge = _make_bare_bridge()
    lock0 = bridge.cuda_backend_lock(0)
    lock1 = bridge.cuda_backend_lock(1)

    # #4378: ggml-cuda state (cuBLAS handles, memory pools) is shared across
    # devices, so both devices must enter the same critical section.
    assert lock0 is lock1
    # The lock must stay re-entrant: `forward` takes it and `_cuda_weight` /
    # `_get_cuda_backend` take it again on the same thread.
    with lock0:
        with lock1:
            pass


def test_cross_device_forwards_never_overlap_in_backend(monkeypatch):
    tracker = _Concurrency()
    weight = torch.randn(8, 4)
    _install_bridge(monkeypatch, weight_matrix=weight, tracker=tracker)

    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    layers = [_make_layer(8, 4) for _ in devices]
    inputs = [_CudaTensor(torch.randn(5, 8), device) for device in devices]

    serial = [layer.forward(x) for layer, x in zip(layers, inputs)]
    barrier = threading.Barrier(len(layers))
    errors: list[Exception] = []
    outputs: list[tuple[int, torch.Tensor]] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(25):
                outputs.append((index, layers[index].forward(inputs[index])))
        except Exception as exc:  # re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(layers))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors, errors
        assert tracker.entries >= 50
        # The bug: cuda:0 and cuda:1 forwards are inside ggml-cuda together.
        assert tracker.max_active == 1
        for index, output in outputs:
            assert torch.equal(output, serial[index])
    finally:
        for layer in layers:
            _drop_layer(layer)


def test_cross_device_lock_entries_serialize():
    bridge = _make_bare_bridge()
    tracker = _Concurrency()
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(device_index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(25):
                with bridge.cuda_backend_lock(device_index):
                    with tracker.track():
                        time.sleep(0.01)
        except Exception as exc:  # re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert tracker.entries == 50
    assert tracker.max_active == 1
