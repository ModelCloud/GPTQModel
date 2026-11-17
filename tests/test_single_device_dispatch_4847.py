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


def test_single_resident_device_rejects_offload_and_spanning_maps():
    from gptqmodel.utils.model import _single_resident_device

    model = _model()
    assert _single_resident_device(model, {"0": "cuda:0", "2": "cuda:0"}) == torch.device("cuda:0")
    assert _single_resident_device(model, {"0": "cuda:0", "2": "cpu"}) is None
    assert _single_resident_device(model, {"0": "cpu"}) is None
    assert _single_resident_device(model, {"0": "disk"}) is None
    assert _single_resident_device(model, {"0": "cuda:1"}) is None
