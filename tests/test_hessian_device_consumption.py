# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Quantization must consume a Hessian on the layer's own device.

`materialize_hessian(target_device)` may have materialized the cached statistics
on another device (an explicit inspection on GPU, say). The quantization pass
must therefore use the device-corrected value it asks for, not whatever
`self.H` still points at.
"""

import pytest
import torch

import gptqmodel.quantization.qqq as qqq_module
from gptqmodel.quantization.config import QQQConfig, VPTQConfig
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.vptq import VPTQ


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP device")


def _configure_qqq_quantizer(task: QQQ) -> None:
    task.quantizer.configure(bits=4, perchannel=True, groupsize=-1, sym=True)


def test_qqq_quantize_uses_the_layer_device_after_a_foreign_materialization(monkeypatch):
    """A GPU inspection must not leave the CPU quantization pass consuming GPU data."""

    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 8, bias=False)  # stays on CPU
    task = QQQ(layer, QQQConfig())
    _configure_qqq_quantizer(task)
    task.add_batch(torch.randn(4, 8, 16), None)

    inspected = task.materialize_hessian(torch.device("cuda:0"))
    assert inspected.device.type == "cuda"
    assert task.H.device.type == "cuda"

    consumed = {}
    original = qqq_module._neutralize_dead_columns

    def record(H, W):
        consumed["H"] = H.device
        consumed["W"] = W.device
        return original(H, W)

    monkeypatch.setattr(qqq_module, "_neutralize_dead_columns", record)

    task.quantize()

    assert consumed["H"].type == "cpu", f"consumed a Hessian on {consumed['H']}"
    assert consumed["H"] == consumed["W"]


def test_qqq_quantize_keeps_the_device_when_the_capture_was_inspected_here(monkeypatch):
    """The same-device path must stay a plain reuse, not a copy of a stale tensor."""

    torch.manual_seed(1)
    layer = torch.nn.Linear(16, 8, bias=False)
    task = QQQ(layer, QQQConfig())
    _configure_qqq_quantizer(task)
    task.add_batch(torch.randn(2, 8, 16), None)
    task.materialize_hessian(torch.device("cpu"))

    consumed = {}
    original = qqq_module._neutralize_dead_columns

    def record(H, W):
        consumed["H"] = H.device
        return original(H, W)

    monkeypatch.setattr(qqq_module, "_neutralize_dead_columns", record)

    task.quantize()

    assert consumed["H"].type == "cpu"


def test_vptq_quantize_uses_the_layer_device_after_a_foreign_materialization(monkeypatch):
    """VPTQ consumes the return value; pin that it is the device-corrected one."""

    torch.manual_seed(2)
    layer = torch.nn.Linear(16, 8, bias=False)
    task = VPTQ(layer, VPTQConfig(num_centroids=[-1, 16], num_res_centroids=[-1, -1], vector_lens=[-1, 4]))
    task.fallback = None
    task.add_batch(torch.randn(4, 8, 16), None)

    inspected = task.materialize_hessian(torch.device("cuda:0"))
    assert inspected.device.type == "cuda"

    seen = {}
    original = task.quantize_weight

    def record(weight, hessian, *args, **kwargs):
        seen["weight"] = weight.device
        seen["hessian"] = hessian.device
        return original(weight, hessian, *args, **kwargs)

    monkeypatch.setattr(task, "quantize_weight", record)

    task.quantize()

    assert seen["hessian"].type == "cpu"
    assert seen["hessian"] == seen["weight"]
