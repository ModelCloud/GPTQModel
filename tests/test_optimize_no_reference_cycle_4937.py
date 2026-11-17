# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4937: ``optimize()`` must not leave the model in a reference cycle.

``BaseQModel.optimize()`` installs a compiled ``forward`` on the model so
``generate()`` enters the compiled path (#4669). Compiling the bound method
stored that bound method inside the compiled wrapper, so
``model.__dict__['forward']`` closed a ``model -> forward -> model`` cycle: the
model, and all of its VRAM, survived the last user reference until a full
cyclic ``gc.collect()``. The forward is now compiled as the unbound class
function and reaches the model through a weak reference, the same pattern
``TorchLinear.optimize()`` uses for ``dequantize_weight`` (#4871).
"""

from __future__ import annotations

import gc
import types
import weakref

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel


class _TinyLM(nn.Module):
    """A ``generate()`` that calls ``self(...)`` exactly like transformers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)

    def generate(self, x):
        return self(x)


class _Harness(BaseQModel):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.quantized = True


def _compiled_forward_installed(harness) -> bool:
    """Whether ``optimize()`` actually compiled (vs. a build without torch.compile)."""

    return not isinstance(harness.model.forward, types.MethodType)


def _build_optimized():
    torch.manual_seed(0)
    model = _TinyLM()
    harness = _Harness(model)
    harness.optimize(backend="inductor")
    with torch.inference_mode():
        harness.model.generate(torch.randn(4, 8))
    return harness, weakref.ref(model)


def test_optimize_does_not_leave_a_model_cycle():
    """The model must die with cyclic GC disabled: no instance-attribute cycle."""

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        harness, model_ref = _build_optimized()
        if not _compiled_forward_installed(harness):
            pytest.skip("torch.compile is unavailable on this build")
        del harness
        assert model_ref() is None, (
            "the model survived the last reference with cyclic GC disabled: "
            "optimize() left a model -> forward -> model cycle (#4937)"
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def test_optimize_generate_still_compiles_and_matches_eager():
    torch.manual_seed(0)
    model = _TinyLM()
    harness = _Harness(model)
    x = torch.randn(4, 8)
    with torch.inference_mode():
        eager = model(x).clone()
    harness.optimize(backend="inductor")
    if not _compiled_forward_installed(harness):
        pytest.skip("torch.compile is unavailable on this build")

    from torch._dynamo.utils import counters

    counters.clear()
    with torch.inference_mode():
        out = harness.model.generate(x)

    assert counters["stats"].get("unique_graphs", 0) > 0, (
        "generate() ran the eager forward after optimize() (#4669 regression)"
    )
    assert torch.allclose(out, eager, atol=1e-5)


def test_repeated_optimize_does_not_stack_or_cycle():
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        harness, model_ref = _build_optimized()
        if not _compiled_forward_installed(harness):
            pytest.skip("torch.compile is unavailable on this build")

        harness.optimize(backend="inductor")

        inner = getattr(harness.model.forward, "_torchdynamo_orig_callable", None)
        assert inner is not None, "forward is not a compile proxy"
        assert getattr(inner, "_torchdynamo_orig_callable", None) is None, (
            "repeated optimize() nested a second compile wrapper"
        )
        # `inner` is the eager bound forward, which holds the model; drop it
        # before the liveness assertion.
        del inner
        with torch.inference_mode():
            harness.model.generate(torch.randn(4, 8))
        del harness
        assert model_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
