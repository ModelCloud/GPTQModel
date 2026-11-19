# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#5427: "am I being traced" must be per-thread.

``torch.compiler.is_compiling()`` reads a process-global flag that stays set for
the whole compile session, so every eager thread sees ``True`` while any other
thread compiles. ``dynamo_is_compiling()`` is built on the constant
``torch.compiler.is_dynamo_compiling()`` and must stay ``False`` in eager code,
even while a foreign thread holds a compile session open.
"""

import threading

import pytest
import torch

from gptqmodel.utils.torch import dynamo_is_compiling


class _HeldBackend:
    """Dynamo backend that keeps the compile session open until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, gm, example_inputs):
        self.entered.set()
        assert self.release.wait(30), "test never released the compile session"
        return gm.forward


def _branch(x):
    return x * 0 if dynamo_is_compiling() else x + 1


def _run_with_foreign_compile(check):
    backend = _HeldBackend()
    compiled = torch.compile(lambda x: x * 3, backend=backend, fullgraph=True)
    errors = []

    def worker():
        try:
            compiled(torch.ones(4))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert backend.entered.wait(60), "background compile never started"
        result = check()
    finally:
        backend.release.set()
        thread.join(60)
    assert not errors, errors
    return result


@pytest.mark.skipif(not hasattr(torch.compiler, "is_dynamo_compiling"), reason="needs the constant predicate")
def test_foreign_compile_does_not_leak_into_eager_thread():
    def check():
        # The process-global flag really is set, which is the bug.
        assert torch.compiler.is_compiling() is True
        return dynamo_is_compiling()

    assert _run_with_foreign_compile(check) is False


@pytest.mark.skipif(not hasattr(torch.compiler, "is_dynamo_compiling"), reason="needs the constant predicate")
def test_eager_branch_survives_a_foreign_compile():
    x = torch.ones(4)
    assert _run_with_foreign_compile(lambda: _branch(x)).tolist() == [2.0, 2.0, 2.0, 2.0]


@pytest.mark.skipif(not hasattr(torch.compiler, "is_dynamo_compiling"), reason="needs the constant predicate")
def test_traced_frame_still_sees_true():
    compiled = torch.compile(_branch, fullgraph=True)
    assert compiled(torch.ones(4)).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_fresh_thread_is_not_compiling():
    seen = []
    thread = threading.Thread(target=lambda: seen.append(dynamo_is_compiling()))
    thread.start()
    thread.join(30)
    assert seen == [False]
