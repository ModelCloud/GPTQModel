# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A compile failure must not replay a partially executed forward (#5119).

With ``fullgraph=False`` dynamo runs a forward as compiled fragments plus resume
frames, so a compiler failure can arrive after earlier frames already mutated
state (the KV cache in ``generate()``, Mamba/SSM state, offload hooks). The old
fallback caught that failure and ran the original forward again from the start,
applying the side effect twice: the first ``generate()`` after ``optimize()``
appended every KV entry a second time and died on
``"... (19) must match the existing size (38)"``.

torch.compile cannot be asked "did the previous frame already run", so the fix
keeps dynamo in charge: the wrapper enables
``torch._dynamo.config.suppress_errors`` for the call, which makes dynamo skip
only the failing frame and run that one frame eagerly. Nothing executes twice,
and the previous flag value is restored afterwards.

These tests make a *resume* frame fail (the first graph succeeds, the second
raises), which is the case the old replay got wrong. They are red before the fix
because the side-effect counter reads 2 instead of 1.
"""

import pytest
import torch

import gptqmodel.utils.torch as torch_utils


def _require_dynamo():
    return pytest.importorskip("torch._dynamo")


def _resume_frame_failure():
    """Backend that compiles the first graph, then fails on the resume frame."""
    from torch._dynamo.backends.registry import lookup_backend

    state = {"compiles": 0}

    def backend(gm, example_inputs):
        state["compiles"] += 1
        if state["compiles"] == 2:
            raise RuntimeError("Attempting to use a closed pool")
        return lookup_backend("eager")(gm, example_inputs)

    return backend, state


def _broken_forward(runs):
    """Forward with a side effect in graph 1 and a graph break into a resume frame."""

    def forward(x):
        runs.append(1)
        y = x * 2
        if (y > 0).any():
            z = y + 1
        else:
            z = y - 1
        return z.sum()

    return forward


def test_resume_frame_failure_runs_the_forward_once():
    _require_dynamo()
    backend, state = _resume_frame_failure()
    runs = []
    compiled = torch_utils.torch_compile(_broken_forward(runs), backend=backend)

    assert float(compiled(torch.ones(2))) == 6.0
    assert state["compiles"] == 2, "the resume frame did not fail"
    assert runs == [1], "the partially executed forward was replayed"


def test_suppression_flag_is_restored_after_the_call():
    dynamo = _require_dynamo()
    backend, _state = _resume_frame_failure()
    runs = []
    compiled = torch_utils.torch_compile(_broken_forward(runs), backend=backend)

    before = dynamo.config.suppress_errors
    assert float(compiled(torch.ones(2))) == 6.0
    assert dynamo.config.suppress_errors is before


def test_fullgraph_still_fails_closed_and_does_not_enable_suppression():
    dynamo = _require_dynamo()

    def backend(_gm, _example_inputs):
        from torch._dynamo.exc import BackendCompilerFailed

        raise BackendCompilerFailed(
            "inductor", RuntimeError("Attempting to use a closed pool"), None
        )

    runs = []

    def forward(x):
        runs.append(1)
        return x * 2

    compiled = torch_utils.torch_compile(forward, backend=backend, fullgraph=True)
    before = dynamo.config.suppress_errors
    # fullgraph executes a single graph, so the eager retry is exact here.
    assert compiled(1) is not None
    assert runs == [1]
    assert dynamo.config.suppress_errors is before


def test_traced_user_error_still_propagates():
    _require_dynamo()
    runs = []

    def body(_value):
        runs.append(1)
        raise ValueError("user body failed")

    compiled = torch_utils.torch_compile(body, backend="eager")
    with pytest.raises(ValueError, match="user body failed"):
        compiled(1)
    assert runs == [1], "a user error must not be retried eagerly"


def test_module_forward_is_not_replayed_on_a_resume_frame_failure():
    _require_dynamo()
    backend, state = _resume_frame_failure()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.runs = []

        def forward(self, value):
            self.runs.append(1)
            y = value * 2
            if (y > 0).any():
                z = y + 1
            else:
                z = y - 1
            return z.sum()

    module = Model()
    compiled = torch_utils.torch_compile(module, backend=backend)

    assert isinstance(compiled, torch.nn.Module)
    assert float(compiled(torch.ones(2))) == 6.0
    assert state["compiles"] == 2
    assert module.runs == [1], "the module forward was replayed"
