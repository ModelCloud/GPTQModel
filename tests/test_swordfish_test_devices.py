# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Check the test harness independently of Swordfish hardware support."""

import importlib
import unittest
from contextlib import nullcontext

import pytest
import torch

from tests.accelerator_gates import cuda_only_skip_reason


_SUITES = [
    ("tests.kernels.test_swordfish", "TestSwordfishKernel"),
    ("tests.kernels.test_swordfish_speed", "TestSwordfishSpeed"),
    ("tests.kernels.test_swordfish_speed_deepseek_v4", "TestSwordfishSpeedDeepSeekV4"),
]


@pytest.mark.parametrize("module_name,class_name", _SUITES)
def test_setup_checks_selected_device(monkeypatch, module_name, class_name):
    module = importlib.import_module(module_name)
    target = torch.device("cuda:1")
    monkeypatch.setattr(module, "swordfish_test_device", lambda: target)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(module, "prewarm_swordfish_extension", lambda: True)
    monkeypatch.setattr(module, "swordfish_runtime_available", lambda device=None: device == target)
    monkeypatch.setattr(module, "swordfish_runtime_error", lambda device=None: "wrong runtime target")

    try:
        getattr(module, class_name).setUpClass()
    except unittest.SkipTest as exc:
        pytest.fail(f"a supported target must not be skipped: {exc}")


@pytest.mark.parametrize("module_name,class_name", _SUITES)
def test_setup_does_not_hide_build_failures(monkeypatch, module_name, class_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "swordfish_test_device", lambda: torch.device("cuda:1"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(module, "swordfish_runtime_error", lambda device=None: "compiler regression")

    def broken_build():
        raise RuntimeError("compiler regression")

    monkeypatch.setattr(module, "prewarm_swordfish_extension", broken_build)
    try:
        with pytest.raises(RuntimeError, match="compiler regression"):
            getattr(module, class_name).setUpClass()
    except unittest.SkipTest as exc:
        pytest.fail(f"a build regression must not be skipped: {exc}")


def _two_cuda_devices():
    reason = cuda_only_skip_reason("CUDA harness regression")
    if reason:
        pytest.skip(reason)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")


@pytest.mark.parametrize("module_name,class_name", _SUITES[1:])
def test_timing_uses_input_device_and_restores_current_device(module_name, class_name):
    _two_cuda_devices()
    module = importlib.import_module(module_name)
    target = torch.device("cuda:1")
    x = torch.ones(16, device=target)
    observed = []

    def operation(value):
        observed.append(torch.cuda.current_device())
        return value + 1

    with torch.cuda.device(0):
        if class_name == "TestSwordfishSpeed":
            elapsed = getattr(module, class_name)()._measure(operation, x, warmup=1, iters=2, blocks=1)
        else:
            elapsed = module._measure(operation, x, warmup=1, iters=2)
        assert torch.cuda.current_device() == 0
    assert elapsed > 0
    assert observed and set(observed) == {target.index}


def test_graph_capture_uses_selected_device(monkeypatch):
    _two_cuda_devices()
    module = importlib.import_module("tests.kernels.test_swordfish")
    target = torch.device("cuda:1")
    monkeypatch.setattr(module, "swordfish_test_device", lambda: target)
    # A preceding test may have initialized the process-wide default on a
    # different device. The harness must supply its own capture stream.
    default_stream = torch.cuda.Stream(device=0)
    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", default_stream)
    graphs = []
    graph_type = torch.cuda.CUDAGraph

    def graph_factory():
        graph = graph_type()
        graphs.append(graph)
        return graph

    class Operation(torch.nn.Module):
        def forward(self, value):
            # No Swordfish kernel required to exercise the harness's graph scope.
            assert torch.cuda.current_device() == target.index
            return value + 1

    monkeypatch.setattr(module, "_make_pair", lambda **kwargs: (None, Operation()))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_factory)
    with torch.cuda.device(0):
        module.TestSwordfishKernel().test_swordfish_cuda_graph_capture_and_replay()
        assert torch.cuda.current_device() == 0
    assert len(graphs) == 1
    assert torch.cuda.graph.default_capture_stream is default_stream
