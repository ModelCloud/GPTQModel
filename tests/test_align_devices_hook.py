# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the inference-side AlignDevicesHook routing.

``accelerate.hooks.AlignDevicesHook.pre_forward`` moves activations with a
plain ``Tensor.to``. On an asymmetric ROCm peer topology that call segfaults
inside ``hipMemcpyPeerAsync``, so GPT-QModel installs a subclass that routes
the same moves through ``safe_move_to``.
"""

import accelerate.hooks as accelerate_hooks
import pytest
import torch
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
)

from gptqmodel.utils import accelerate_hooks as hooks_mod
from gptqmodel.utils import p2p
from gptqmodel.utils.accelerate_hooks import (
    GPTQModelAlignDevicesHook,
    install_safe_align_devices_hooks,
    safe_send_to_device,
)
from gptqmodel.utils.model import simple_dispatch_model


def _require_two_gpus() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA/HIP devices")


def _force_asymmetric_direction(monkeypatch, src: int = 0, dst: int = 1) -> None:
    """Report ``src -> dst`` as unusable while keeping the reverse direct."""

    def can_transfer_directly(src_device, dst_device, status=None):
        src_index = torch.device(src_device).index
        dst_index = torch.device(dst_device).index
        if src_index is None or dst_index is None or src_index == dst_index:
            return True
        return (src_index, dst_index) != (src, dst)

    monkeypatch.setattr(p2p, "can_transfer_directly", can_transfer_directly)


def test_safe_send_to_device_recurses_and_preserves_container_types(monkeypatch):
    moved = []

    def fake_move(tensor, device, **kwargs):
        moved.append((tuple(tensor.shape), torch.device(device)))
        return tensor.clone()

    monkeypatch.setattr(hooks_mod, "safe_move_to", fake_move)

    first = torch.ones(2)
    second = torch.zeros(3)
    skipped = torch.ones(1)
    payload = {
        "list": [first],
        "tuple": (second,),
        "skipped": skipped,
        "scalar": 7,
    }

    result = safe_send_to_device(payload, torch.device("cpu"), skip_keys=["skipped"])

    assert type(result["list"]) is list
    assert type(result["tuple"]) is tuple
    assert result["skipped"] is skipped
    assert result["scalar"] == 7
    assert moved == [((2,), torch.device("cpu")), ((3,), torch.device("cpu"))]
    assert result["list"][0] is not first
    assert result["tuple"][0] is not second


def test_pre_forward_routes_inputs_and_keeps_caller_device(monkeypatch):
    moved = []

    def fake_move(tensor, device, **kwargs):
        moved.append((tuple(tensor.shape), torch.device(device)))
        return tensor.clone()

    monkeypatch.setattr(hooks_mod, "safe_move_to", fake_move)
    # Upstream's own move must be a no-op for this unit test: the fake move
    # returns CPU tensors even though the hook targets cuda:1.
    monkeypatch.setattr(accelerate_hooks, "send_to_device", lambda tensor, device, **kwargs: tensor)
    monkeypatch.setattr(hooks_mod, "find_device", lambda tensors: torch.device("cuda:0"))

    hook = GPTQModelAlignDevicesHook(execution_device="cuda:1", io_same_device=True)
    hook.skip_keys = ["skip"]

    args, kwargs = hook.pre_forward(
        torch.nn.Identity(),
        torch.ones(2, 3),
        cache=torch.ones(1),
        skip=torch.ones(1),
    )

    assert moved == [((2, 3), torch.device("cuda:1")), ((1,), torch.device("cuda:1"))]
    assert hook.input_device == torch.device("cuda:0")
    assert "skip" in kwargs
    assert args[0].shape == (2, 3)


def test_post_forward_returns_output_through_safe_move(monkeypatch):
    moved = []

    def fake_move(tensor, device, **kwargs):
        moved.append((tuple(tensor.shape), torch.device(device)))
        return tensor.clone()

    monkeypatch.setattr(hooks_mod, "safe_move_to", fake_move)
    monkeypatch.setattr(accelerate_hooks, "send_to_device", lambda tensor, device, **kwargs: tensor)

    hook = GPTQModelAlignDevicesHook(execution_device="cuda:1", io_same_device=True)
    hook.input_device = torch.device("cuda:0")
    output = hook.post_forward(torch.nn.Identity(), torch.ones(2, 4))

    assert moved == [((2, 4), torch.device("cuda:0"))]
    assert output.shape == (2, 4)


def test_simple_dispatch_model_installs_safe_hook(monkeypatch):
    installed = []

    monkeypatch.setattr(
        accelerate_hooks, "add_hook_to_module", lambda module, hook: installed.append((module, hook))
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    simple_dispatch_model(model, {"": "cuda:0", "1": "cuda:1"})

    assert len(installed) == 1
    module, hook = installed[0]
    assert module is model[1]
    assert isinstance(hook, GPTQModelAlignDevicesHook)
    assert hook.execution_device == torch.device("cuda:1")


def test_install_safe_align_devices_hooks_upgrades_attached_and_nested_hooks():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    plain = AlignDevicesHook(execution_device=torch.device("cpu"))
    nested_plain = AlignDevicesHook(execution_device=torch.device("cpu"))
    nested = SequentialHook(ModelHook(), SequentialHook(nested_plain))
    module[0]._hf_hook = plain
    module[1]._hf_hook = nested

    assert install_safe_align_devices_hooks(module) == 2
    assert isinstance(module[0]._hf_hook, GPTQModelAlignDevicesHook)
    assert isinstance(nested.hooks[1].hooks[0], GPTQModelAlignDevicesHook)
    assert isinstance(nested.hooks[0], ModelHook)


def _two_layer_model():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    expected = model(torch.ones(2, 8))
    model[0].to("cuda:0")
    model[1].to("cuda:1")
    return model, expected


def test_device_map_inference_stages_unsupported_direction(monkeypatch):
    _require_two_gpus()
    _force_asymmetric_direction(monkeypatch, 0, 1)

    staged = []
    real_stage = p2p._stage_through_host

    def spy_stage(tensor, device, **kwargs):
        staged.append((tensor.device, torch.device(device)))
        return real_stage(tensor, device, **kwargs)

    monkeypatch.setattr(p2p, "_stage_through_host", spy_stage)

    model, expected = _two_layer_model()
    model = simple_dispatch_model(model, {"": "cuda:0", "1": "cuda:1"})

    output = model(torch.ones(2, 8, device="cuda:0"))

    assert output.device == torch.device("cuda:0")
    assert staged == [(torch.device("cuda:0"), torch.device("cuda:1"))]
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-4)


def test_real_asymmetric_pair_inference_does_not_crash():
    _require_two_gpus()
    if p2p.can_transfer_directly("cuda:0", "cuda:1") and p2p.can_transfer_directly(
        "cuda:1", "cuda:0"
    ):
        pytest.skip("host P2P is symmetric; the forced-direction test covers the routing")

    model, expected = _two_layer_model()
    model = simple_dispatch_model(model, {"": "cuda:0", "1": "cuda:1"})

    output = model(torch.ones(2, 8, device="cuda:0"))

    torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-4)
