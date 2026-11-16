# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4314: `GGUFCudaKernel.forward` must be thread-safe.

The ggml CUDA extension is not built in CI here, so these tests drive the real
locking code in `gguf_cpp` through a mocked bridge. The mock `run` records
concurrent entries; without the per-device lock two threads overlap.
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

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.device = torch.device("cuda:0")
        self.dtype = tensor.dtype
        self.shape = tensor.shape

    def reshape(self, *shape):
        return _CudaTensor(self._tensor.reshape(*shape))


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
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWeight:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


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
    run_entered: threading.Event | None = None,
    build_delay: float = 0.0,
) -> _GGMLBridge:
    bridge = object.__new__(_GGMLBridge)
    bridge._cuda_backend_lock = threading.RLock()

    def fake_run(*, x, gguf_tensor_qtype, padded_in_features, out_features, plan=None, weight=None):
        with tracker.track():
            if run_entered is not None:
                run_entered.set()
            time.sleep(run_delay)
            output = (x._tensor.float() @ weight_matrix).to(torch.float16)
        if plan is None:
            plan = _FakePlan(rows=x.shape[0], out_features=out_features)
        return output, plan

    def fake_build_cuda_weight(**_kwargs):
        with tracker.track():
            time.sleep(build_delay)
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


@pytest.mark.parametrize("distinct_layers", [False, True])
def test_cuda_forward_serializes_and_matches_serial_output(monkeypatch, distinct_layers):
    tracker = _Concurrency()
    weight = torch.randn(8, 4)
    _install_bridge(monkeypatch, weight_matrix=weight, tracker=tracker)
    layers = [_make_layer(8, 4), _make_layer(8, 4)]
    x = _CudaTensor(torch.randn(5, 8))

    serial = [layer.forward(x) for layer in layers]
    targets = [layers[0], layers[1]] if distinct_layers else [layers[0], layers[0]]
    barrier = threading.Barrier(len(targets))
    errors: list[BaseException] = []
    outputs: list[tuple[int, torch.Tensor]] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(25):
                outputs.append((index, targets[index].forward(x)))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(targets))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in threads)
        assert not errors, errors
        assert tracker.entries >= 50
        # The bug: concurrent calls interleave in the shared backend.
        assert tracker.max_active == 1
        for index, output in outputs:
            expected = serial[index] if distinct_layers else serial[0]
            assert torch.equal(output, expected)
    finally:
        for layer in layers:
            _drop_layer(layer)


def test_cuda_weight_swap_is_serialized_against_in_flight_forward(monkeypatch):
    tracker = _Concurrency()
    weight = torch.randn(8, 4)
    run_entered = threading.Event()
    _install_bridge(
        monkeypatch,
        weight_matrix=weight,
        tracker=tracker,
        run_delay=0.02,
        run_entered=run_entered,
        build_delay=0.05,
    )
    layer = _make_layer(8, 4)
    x = _CudaTensor(torch.randn(5, 8))

    reference = layer.forward(x)
    errors: list[BaseException] = []

    def forward_worker() -> None:
        try:
            for _ in range(5):
                assert torch.equal(layer.forward(x), reference)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    thread = threading.Thread(target=forward_worker)
    try:
        thread.start()
        assert run_entered.wait(timeout=5), "forward never entered the mocked backend"
        # Force a new weight key while a forward is mid-run. `_cuda_weight`
        # closes the old plans and frees the old weight, which must wait.
        object.__setattr__(layer, "qweight", torch.arange(16, dtype=torch.uint8) + 1)
        layer._cuda_weight(x)
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert not errors, errors
        assert tracker.entries >= 2
        assert tracker.max_active == 1
    finally:
        _drop_layer(layer)
