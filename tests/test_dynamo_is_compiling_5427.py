# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``dynamo_is_compiling()`` must answer for this frame, not the process (#5427).

``torch.compiler.is_compiling()`` reports the module-level compile-session
flag, so while any thread runs ``torch.compile`` every eager thread also sees
``True``. Code that gated a trace-only path on it then took that path eagerly:
NF12 forward raised, ``optimize()``d models silently ran the eager body, and the
AWQ/Paro/TorchLinear fast paths were bypassed. ``torch.compiler.
is_dynamo_compiling()`` is a constant ``False`` in eager code that dynamo
folds to ``True`` inside the traced frame, so ``dynamo_is_compiling()`` now
prefers it and falls back only on torch builds that predate it.
"""

import threading

import pytest
import torch

import gptqmodel.utils.torch as torch_utils


pytestmark = [pytest.mark.cpu]


def test_other_threads_compile_session_does_not_flip_the_eager_caller(monkeypatch):
    """The process-global flag reads ``True`` while an eager caller runs (#5427)."""

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.compiler, "is_dynamo_compiling", lambda: False)

    assert torch_utils.dynamo_is_compiling() is False


def test_a_background_compile_session_stays_out_of_the_eager_caller(monkeypatch):
    """The reported interleaving: thread A compiles, thread B stays eager."""

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.compiler, "is_dynamo_compiling", lambda: False)

    seen = []
    started = threading.Event()
    release = threading.Event()

    def compiles():
        started.set()
        release.wait(5.0)
        seen.append(torch_utils.dynamo_is_compiling())

    thread = threading.Thread(target=compiles)
    thread.start()
    try:
        assert started.wait(5.0)
        # Thread B: eager, and must not see thread A's session.
        assert torch_utils.dynamo_is_compiling() is False
    finally:
        release.set()
        thread.join(5.0)
    assert seen == [False]


def test_predicate_folds_to_true_inside_the_traced_frame():
    """Inside the graph dynamo folds the constant, so trace-only paths run."""

    pytest.importorskip("torch._dynamo")

    def body(t):
        return t + 1, torch_utils.dynamo_is_compiling()

    compiled = torch.compile(body, backend="eager", dynamic=False)
    eager_out, eager_flag = body(torch.ones(1))
    compiled_out, compiled_flag = compiled(torch.ones(1))

    assert eager_flag is False
    assert compiled_flag is True
    assert torch.equal(eager_out, compiled_out)


def test_missing_dynamo_predicate_falls_back_to_is_compiling(monkeypatch):
    """Older torch keeps the historical answer rather than a silent ``False``."""

    monkeypatch.delattr(torch.compiler, "is_dynamo_compiling", raising=False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert torch_utils.dynamo_is_compiling() is True


def test_no_compiler_attr_reports_eager(monkeypatch):
    monkeypatch.delattr(torch, "compiler", raising=False)

    assert torch_utils.dynamo_is_compiling() is False
