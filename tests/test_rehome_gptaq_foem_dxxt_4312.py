# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4312: GPTAQ/FOEM ``dXXT`` must follow ``H`` when a task is rehomed.

``_rehome_processor_task`` used to move only ``H``/``module_copy``. With two
GPUs, a module whose batches all arrived from one device keeps its live
``H``/``dXXT`` buffers, and the quantization target can be the other GPU. The
rehome then split ``H`` from ``dXXT`` and ``_neutralize_dead_columns`` mixed
devices. Both paths are covered here without requiring a second GPU.
"""

import types

import pytest
import torch
from torch import nn

import gptqmodel.looper.module_looper as module_looper
import gptqmodel.quantization.foem as foem_module
import gptqmodel.quantization.gptaq as gptaq_module
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


_CLASSES = {"gptaq": GPTAQ, "foem": FOEM}
_MODULES = {"gptaq": gptaq_module, "foem": foem_module}


def _named(linear):
    return NamedModule(
        linear, name="proj", full_name="model.layers.0.proj", layer_index=0
    )


def _qcfg(kind):
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig(alpha=0.25))
    return QuantizeConfig(
        bits=4, group_size=8, foem=FOEMConfig(alpha=0.25, beta=0.2)
    )


class _ForeignDeviceTensor:
    """Report another device while wrapping a real CPU tensor.

    Only ``.device`` is inspected before ``safe_move_to`` is mocked, so this
    models a ``dXXT`` that stayed on the calibration GPU while ``H`` was
    rehomed to the quantization target.
    """

    def __init__(self, tensor, device):
        self.tensor = tensor
        self.device = device

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def test_rehome_processor_task_moves_hessian_companions(monkeypatch):
    named = _named(nn.Linear(4, 4, bias=False))
    target = torch.device("cpu")
    originals = {
        "H": torch.zeros(4, 4),
        "module_copy": torch.ones(4, 4),
        "dXXT": torch.full((4, 4), 2.0),
    }
    task = types.SimpleNamespace(**originals, dev=torch.device("cpu"))
    processor = types.SimpleNamespace(tasks={named.name: task})

    moved = []

    def fake_safe_move_to(tensor, device, **_kwargs):
        moved.append((tensor, torch.device(device)))
        return tensor.clone()

    monkeypatch.setattr(module_looper, "safe_move_to", fake_safe_move_to)

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = types.SimpleNamespace(model=None)
    looper._rehome_processor_task(processor, named, target)

    assert [device for _tensor, device in moved] == [target, target, target]
    assert [tensor for tensor, _device in moved] == [
        originals["H"],
        originals["module_copy"],
        originals["dXXT"],
    ]
    # Every companion was replaced by the rehomed tensor, not left behind.
    for name in originals:
        assert getattr(task, name) is not originals[name]
        torch.testing.assert_close(getattr(task, name), originals[name])
    assert task.dev == target


def test_rehome_processor_task_moves_foem_hessian_pair(monkeypatch):
    """FOEM keeps ``H`` + its ``dXXT`` equivalent, and only present attrs move."""

    named = _named(nn.Linear(4, 4, bias=False))
    target = torch.device("cpu")
    task = types.SimpleNamespace(
        H=torch.zeros(4, 4), dXXT=torch.ones(4, 4), dev=torch.device("cpu")
    )
    processor = types.SimpleNamespace(tasks={named.name: task})

    moved = []

    def fake_safe_move_to(tensor, device, **_kwargs):
        moved.append(torch.device(device))
        return tensor.clone()

    monkeypatch.setattr(module_looper, "safe_move_to", fake_safe_move_to)

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = types.SimpleNamespace(model=None)
    looper._rehome_processor_task(processor, named, target)

    assert moved == [target, target]
    assert not hasattr(task, "module_copy")


@pytest.mark.parametrize("kind", ("gptaq", "foem"))
def test_quantize_aligns_dxxt_with_rehomed_hessian(kind, monkeypatch):
    """The single-device finalize early-return must not leave ``dXXT`` behind."""

    torch.manual_seed(0)
    named = _named(nn.Linear(8, 4, bias=False))
    named.state[NATIVE_INPUTS_STATE_KEY] = [(0, torch.randn(3, 8))]
    task = _CLASSES[kind](named, _qcfg(kind))
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.randn(1, 3, 8), None, batch_index=0)

    # All batches came from one device, so finalize returns early and the live
    # H/dXXT pair survives. Simulate H already rehomed to CPU (the target)
    # while dXXT stayed on the calibration GPU.
    assert getattr(task, f"_{kind}_device_hessian") == {}
    stale_dxxt = _ForeignDeviceTensor(task.dXXT, torch.device("cuda:1"))
    stale_weight = _ForeignDeviceTensor(
        task.clone_module(), torch.device("cuda:1")
    )
    task.dXXT = stale_dxxt
    task.module_copy = stale_weight

    moved = []

    def fake_safe_move_to(tensor, device, **_kwargs):
        moved.append((tensor, torch.device(device)))
        assert isinstance(tensor, _ForeignDeviceTensor)
        return tensor.tensor.clone()

    monkeypatch.setattr(_MODULES[kind], "safe_move_to", fake_safe_move_to)

    quantized = task.quantize()[0]

    assert moved == [
        (stale_weight, torch.device("cpu")),
        (stale_dxxt, torch.device("cpu")),
    ]
    assert torch.isfinite(quantized).all()
