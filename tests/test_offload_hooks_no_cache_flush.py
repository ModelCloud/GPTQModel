# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel's offload hooks never flush the device cache, without patching Accelerate.

Accelerate's ``AlignDevicesHook`` moves tensors with ``clear_cache=True``, which
empties the accelerator cache on every offloaded forward; that is unsafe and
slow while GPT-QModel drives offloaded modules from several threads. GPT-QModel
used to replace ``accelerate.utils.modeling.clear_device_cache`` for the whole
process. Its hooks are now ``GPTQModelAlignDevicesHook`` and Accelerate stays as
shipped.
"""

import accelerate.utils.modeling as accelerate_modeling
import torch
from accelerate.hooks import (
    AlignDevicesHook,
    SequentialHook,
    add_hook_to_module,
    attach_align_device_hook,
    remove_hook_from_module,
    remove_hook_from_submodules,
)
from accelerate.utils.memory import clear_device_cache
from torch import nn

from gptqmodel.utils import offload_hooks
from gptqmodel.utils.offload_hooks import (
    GPTQModelAlignDevicesHook,
    adopt_align_device_hooks,
    disk_offload,
)


def _record_moves(monkeypatch):
    calls = []
    original = offload_hooks.set_module_tensor_to_device

    def recorder(module, name, device, *args, **kwargs):
        calls.append(kwargs.get("clear_cache", True))
        return original(module, name, device, *args, **kwargs)

    monkeypatch.setattr(offload_hooks, "set_module_tensor_to_device", recorder)
    return calls


def _tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)).eval()


def test_importing_gptqmodel_leaves_accelerate_clear_device_cache_alone():
    import gptqmodel  # noqa: F401
    import gptqmodel.utils.offload  # noqa: F401

    assert accelerate_modeling.clear_device_cache is clear_device_cache
    assert not hasattr(gptqmodel.utils.offload, "ACCELERATE_CLEAR_DEVICE_CACHE")


def test_gptqmodel_hook_moves_every_tensor_without_clearing_the_cache(monkeypatch):
    calls = _record_moves(monkeypatch)
    model = _tiny()
    x = torch.randn(2, 4)
    reference = model(x)

    offload_hooks.attach_align_device_hook(model, execution_device=torch.device("cpu"), offload=True)
    assert model[0].weight.device.type == "meta"
    assert torch.equal(model(x), reference)
    remove_hook_from_submodules(model)
    assert model[0].weight.device.type == "cpu"
    assert torch.equal(model(x), reference)

    # init (to meta), pre_forward (onload), detach (restore) all went through
    # GPT-QModel's hook, and none asked Accelerate to flush the device cache.
    assert len(calls) >= 3 * 4
    assert not any(calls)


def test_disk_offload_uses_the_gptqmodel_hook(tmp_path, monkeypatch):
    calls = _record_moves(monkeypatch)
    model = _tiny()
    x = torch.randn(2, 4)
    reference = model(x)

    disk_offload(model, str(tmp_path), execution_device=torch.device("cpu"), offload_buffers=True)
    hooks = [m._hf_hook for m in model.modules() if hasattr(m, "_hf_hook")]
    flat = [inner for hook in hooks for inner in (hook.hooks if isinstance(hook, SequentialHook) else (hook,))]
    assert flat and all(type(hook) is GPTQModelAlignDevicesHook for hook in flat)
    assert torch.equal(model(x), reference)
    assert calls and not any(calls)


def test_adopt_replaces_stock_hooks_on_this_model_only(monkeypatch):
    model = _tiny()
    other = _tiny()
    x = torch.randn(2, 4)
    reference = model(x)

    # What Transformers' `from_pretrained(device_map=...)` dispatch attaches.
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(model, execution_device=torch.device("cpu"), offload=True)
    attach_align_device_hook(other, execution_device=torch.device("cpu"), offload=True)

    adopted = adopt_align_device_hooks(model)
    assert adopted == 3  # root io hook + the two Linear leaves (ReLU has no tensors)
    flat = [
        inner for m in model.modules() if hasattr(m, "_hf_hook")
        for inner in (m._hf_hook.hooks if isinstance(m._hf_hook, SequentialHook) else (m._hf_hook,))
    ]
    assert all(type(hook) is GPTQModelAlignDevicesHook for hook in flat)
    # Another model's hooks and Accelerate's class are untouched.
    assert type(other[0]._hf_hook) is AlignDevicesHook
    assert AlignDevicesHook.pre_forward.__module__ == "accelerate.hooks"

    calls = _record_moves(monkeypatch)
    assert torch.equal(model(x), reference)
    assert calls and not any(calls)
    remove_hook_from_module(model, recurse=True)
    assert model[0].weight.device.type == "cpu"
    assert torch.equal(model(x), reference)
    assert adopt_align_device_hooks(model) == 0
