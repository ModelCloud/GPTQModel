# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`pre_quantize` must stage layers on the filtered quantization device.

`quantize_config.device` carries only the accelerator family, so resolving it
directly yields the index-less family device, i.e. cuda:0. When the loader
narrows the run through `compute_device_filter` (``device=("cuda:1",)``), every
staged layer landed on an excluded GPU and OOM'd when that GPU was busy
(issue #4233). These tests pin the device `pre_quantize` hands to
`shell_module_materialize` / `move_to` without needing a GPU.
"""

from types import SimpleNamespace

import torch
from torch import nn

import gptqmodel.models.base as base_mod
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.torch import as_torch_device


def _bare_model(device, compute_device_filter=None):
    model = BaseQModel.__new__(BaseQModel)
    model.quantize_config = SimpleNamespace(
        device=device,
        compute_device_filter=compute_device_filter,
    )
    return model


def _pool(*indexes):
    return [torch.device("cuda", index) for index in indexes]


def _meta_layer():
    return nn.Linear(4, 4, bias=True, device="meta")


def _canonical(device):
    """Collapse an index-less accelerator family to its device-0 form."""

    device = as_torch_device(device)
    if device.type in {"cuda", "xpu", "npu"} and device.index is None:
        return torch.device(device.type, 0)
    return device


def _exclude_first(devices):
    return [device for device in devices if device.index != 0]


def _stage_device(monkeypatch, model, layer):
    """Return the device `pre_quantize` staged `layer` on."""

    staged = []

    def fake_materialize(self, module, device):
        staged.append(device)
        return True

    monkeypatch.setattr(BaseQModel, "_materialize_floatx_layer_shell", fake_materialize)
    assert model.pre_quantize(layer) is layer
    assert len(staged) == 1
    return _canonical(staged[0])


def test_meta_layer_stages_on_filtered_device(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1), raising=False)
    model = _bare_model(DEVICE.CUDA, _exclude_first)

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cuda:1")


def test_shell_materialize_gets_filtered_device(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1), raising=False)
    monkeypatch.setattr(
        BaseQModel,
        "_materialize_floatx_layer_shell",
        lambda self, module, device: False,
    )
    seen = []

    def fake_shell(self, target_submodule, device, **kwargs):
        seen.append(_canonical(device))
        return target_submodule

    monkeypatch.setattr(BaseQModel, "shell_module_materialize", fake_shell)

    model = _bare_model(DEVICE.CUDA, _exclude_first)
    layer = _meta_layer()

    assert model.pre_quantize(layer) is layer
    assert seen == [torch.device("cuda:1")]


def test_cpu_layer_moves_to_filtered_device(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1), raising=False)
    moved = []

    def fake_move_to(module, device, dtype=None):
        moved.append(_canonical(device))
        return module

    monkeypatch.setattr(base_mod, "move_to", fake_move_to)

    model = _bare_model(DEVICE.CUDA, lambda devices: [devices[1]])
    layer = nn.Linear(4, 4)

    assert model.pre_quantize(layer) is layer
    assert moved == [torch.device("cuda:1")]


def test_filter_sees_the_forward_device_pool(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1, 2), raising=False)
    pools = []

    def keep_last(devices):
        pools.append(list(devices))
        return [devices[-1]]

    model = _bare_model(DEVICE.CUDA, keep_last)

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cuda:2")
    assert pools == [_pool(0, 1, 2)]


def test_without_filter_keeps_the_family_default(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1), raising=False)
    model = _bare_model(DEVICE.CUDA)

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cuda:0")


def test_filter_rejecting_every_device_keeps_the_full_pool(monkeypatch):
    monkeypatch.setattr(base_mod, "select_forward_devices", lambda base: _pool(0, 1), raising=False)
    model = _bare_model(DEVICE.CUDA, lambda devices: [])

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cuda:0")


def test_cpu_run_stages_on_cpu(monkeypatch):
    model = _bare_model(DEVICE.CPU)

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cpu")

def test_indexed_config_device_is_staged_directly(monkeypatch):
    # Direct config users pass an explicit ordinal; keep honoring it.
    model = _bare_model("cuda:1")

    assert _stage_device(monkeypatch, model, _meta_layer()) == torch.device("cuda:1")
