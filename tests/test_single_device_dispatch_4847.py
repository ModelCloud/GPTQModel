# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""A layer-wise map that names one device must not install accelerate hooks.

``add_hook_to_module`` makes ``module.forward = partial(new_forward, module)`` and
stores the old bound method on ``module._old_forward``, which turns the model into a
reference cycle.  For a GPU load that means the VRAM is only released by a full
cyclic GC, not when the last user reference goes away (#4847).
"""

import gc
import weakref

import pytest
import torch

from gptqmodel.utils.model import simple_dispatch_model


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(device: str = "cuda:0") -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 4),
    ).to(device)


def _hooked_modules(model: torch.nn.Module):
    return [name for name, module in model.named_modules() if hasattr(module, "_hf_hook")]


def test_single_device_map_installs_no_hooks_and_keeps_forward():
    model = _model()
    inputs = torch.randn(3, 4, device="cuda:0")
    expected = model(inputs)

    simple_dispatch_model(model, {"0": "cuda:0", "2": "cuda:0"})

    assert _hooked_modules(model) == []
    assert model.hf_device_map == {"0": "cuda:0", "2": "cuda:0"}
    torch.testing.assert_close(model(inputs), expected)


def test_single_device_map_with_integer_device_installs_no_hooks():
    model = _model()

    simple_dispatch_model(model, {"0": 0, "2": 0})

    assert _hooked_modules(model) == []


def test_single_device_map_is_not_a_reference_cycle():
    gc.collect()
    gc.disable()
    try:
        def build() -> weakref.ref:
            model = _model()
            simple_dispatch_model(model, {"0": "cuda:0", "2": "cuda:0"})
            # Each hooked layer would be its own cycle; watch the first layer.
            return weakref.ref(model[0])

        ref = build()
        # Without hooks the last reference is the only one, so the model dies by
        # refcount and its VRAM is returned immediately (no cyclic GC needed).
        assert ref() is None
    finally:
        gc.enable()


def test_accelerate_hook_is_a_reference_cycle():
    """Guard for the test above: the hook really is what pins the model."""

    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    gc.collect()
    gc.disable()
    try:
        def build() -> weakref.ref:
            model = _model()
            add_hook_to_module(model[0], AlignDevicesHook(torch.device("cuda:0"), io_same_device=True))
            return weakref.ref(model[0])

        ref = build()
        assert ref() is not None
        gc.collect()
        assert ref() is None
    finally:
        gc.enable()


def test_multi_device_map_still_installs_hooks():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)).to("cuda:0")
    inputs = torch.randn(3, 4, device="cuda:0")
    expected = model(inputs)

    simple_dispatch_model(model, {"0": "cuda:0", "1": "cpu"})

    assert _hooked_modules(model) != []
    torch.testing.assert_close(model(inputs).cpu(), expected.cpu())


def test_unmaterialised_tensor_keeps_hooks():
    """A map over meta tensors still needs the hooks to place the submodules."""

    from gptqmodel.utils.model import _single_resident_device

    model = torch.nn.Linear(4, 4).to("meta")
    assert _single_resident_device(model, {"0": "cuda:0", "1": "cuda:0"}) is None
    assert _single_resident_device(model, {"0": "cuda:0", "1": "cuda:1"}) is None


def test_all_cpu_map_is_unchanged():
    model = _model("cpu")

    simple_dispatch_model(model, {"0": "cpu", "2": "cpu"})

    assert _hooked_modules(model) == []
    assert model.hf_device_map == {"0": "cpu", "2": "cpu"}


class _RotaryLike(torch.nn.Module):
    """A shell module whose cache buffer is built on the CPU and never saved.

    Remote-code rotary modules index ``cos_cached``/``sin_cached`` directly
    instead of moving ``inv_freq`` to the input device, so the buffer has to
    follow its layer (#5232).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "inv_freq", torch.arange(4, dtype=torch.float32), persistent=False
        )
        self.linear = torch.nn.Linear(4, 4)


def test_single_device_map_places_non_persistent_buffers():
    # The checkpoint loader only fills tensors the checkpoint provides, so a
    # non-persistent buffer stays where the shell built it. Skipping the hooks
    # must not skip that placement as well.
    model = torch.nn.Sequential(_RotaryLike())
    model[0].linear.to("cuda:0")

    simple_dispatch_model(model, {"0": "cuda:0"})

    assert model[0].inv_freq.device.type == "cuda"
    assert model[0].linear.weight.device.type == "cuda"


def test_single_device_map_keeps_a_shared_buffer_shared():
    shared = torch.arange(4, dtype=torch.float32)
    first, second = _RotaryLike(), _RotaryLike()
    first._buffers["inv_freq"] = shared
    second._buffers["inv_freq"] = shared
    first.linear.to("cuda:0")
    second.linear.to("cuda:0")
    model = torch.nn.Sequential(first, second)

    simple_dispatch_model(model, {"0": "cuda:0", "1": "cuda:0"})

    assert model[0].inv_freq.data_ptr() == model[1].inv_freq.data_ptr()


def test_single_resident_shortcut_calls_buffer_placement(monkeypatch):
    # CPU-only guard for the wiring: the shortcut must invoke the placement
    # helper with the map's device instead of returning silently.
    import gptqmodel.utils.model as model_utils

    calls = []
    monkeypatch.setattr(
        model_utils, "_single_resident_device", lambda model, device_map: torch.device("cpu")
    )
    monkeypatch.setattr(
        model_utils,
        "_place_non_persistent_buffers",
        lambda model, device: calls.append(device),
    )

    # A non-root entry so the ``single_root`` fast path is not the one taken.
    model_utils.simple_dispatch_model(torch.nn.Linear(2, 2), {"0": "cuda:0"})

    assert calls == [torch.device("cpu")]


def test_single_resident_device_rejects_offload_and_spanning_maps():
    from gptqmodel.utils.model import _single_resident_device

    model = _model()
    assert _single_resident_device(model, {"0": "cuda:0", "2": "cuda:0"}) == torch.device("cuda:0")
    assert _single_resident_device(model, {"0": "cuda:0", "2": "cpu"}) is None
    assert _single_resident_device(model, {"0": "cpu"}) is None
    assert _single_resident_device(model, {"0": "disk"}) is None
    assert _single_resident_device(model, {"0": "cuda:1"}) is None
