# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4669 item 2: ``optimize()`` must compile the forward ``generate()`` runs.

``GPTQModel.generate()`` calls ``self.model.generate``. Transformers'
``generate`` calls ``self(...)``, i.e. the module's own ``forward``. Wrapping
``self.model`` in a ``torch.compile`` ``OptimizedModule`` therefore left
generation on the original eager ``forward``: ``OptimizedModule.__getattr__``
hands ``generate`` back to the unwrapped module. ``optimize()`` now compiles
``forward`` in place, so a direct call and ``generate()`` both enter the
compiled path.

These tests use a real ``torch.compile`` on a tiny CPU module. They fail on the
parent tree because the compiled wrapper is bypassed and dynamo compiles zero
frames.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel


class _TinyLM(nn.Module):
    """A ``generate()`` that calls ``self(...)`` exactly like transformers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.generate_calls = 0

    def forward(self, x):
        return self.linear(x)

    def generate(self, x):
        self.generate_calls += 1
        return self(x)


class _Harness(BaseQModel):
    """``BaseQModel.optimize`` over a tiny model, without a full load."""

    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.quantized = True


def _optimize_tiny():
    torch.manual_seed(0)
    model = _TinyLM()
    harness = _Harness(model)
    x = torch.randn(4, 8)
    with torch.inference_mode():
        eager = model(x).clone()
    original_forward = model.forward
    harness.optimize(backend="inductor")
    if harness.model.forward is original_forward:
        pytest.skip("torch.compile is unavailable on this build")
    return harness, original_forward, x, eager


def _stats():
    from torch._dynamo.utils import counters

    return dict(counters["stats"])


def test_optimize_compiles_the_forward_generate_runs():
    harness, _, x, _ = _optimize_tiny()

    from torch._dynamo.utils import counters

    counters.clear()
    with torch.inference_mode():
        harness.model.generate(x)

    assert counters["stats"].get("unique_graphs", 0) > 0, (
        "generate() ran the eager forward: optimize() compiled a wrapper "
        "that generate() never enters (#4669)"
    )


def test_optimize_keeps_a_direct_forward_correct():
    harness, original_forward, x, eager = _optimize_tiny()

    with torch.inference_mode():
        compiled = harness.model(x)

    assert torch.allclose(compiled, eager, atol=1e-5)
    # The compiled forward is installed on the module generate() calls.
    assert harness.model.forward is not original_forward


def test_repeated_optimize_does_not_stack_compile_wrappers():
    harness, original_forward, x, _ = _optimize_tiny()

    harness.optimize(backend="inductor")

    inner = getattr(harness.model.forward, "_torchdynamo_orig_callable", None)
    assert inner is not None, "forward is not a compile-fallback proxy"
    assert getattr(inner, "_torchdynamo_orig_callable", None) is None, (
        "repeated optimize() nested a second compile wrapper"
    )
    assert inner == original_forward
    with torch.inference_mode():
        harness.model.generate(x)


def test_hooks_and_dtype_moves_survive_the_compiled_forward():
    harness, _, x, _ = _optimize_tiny()

    seen = []
    handle = harness.model.register_forward_pre_hook(lambda module, args: seen.append(1))
    try:
        with torch.inference_mode():
            harness.model(x)
        assert len(seen) == 1
    finally:
        handle.remove()

    harness.model.to(torch.float64)
    with torch.inference_mode():
        moved = harness.model(x.to(torch.float64))
    assert moved.dtype == torch.float64
