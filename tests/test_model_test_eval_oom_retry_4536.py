# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Regression tests for ModelTest eval OOM retries (#4536).

Before the fix, a task that pinned ``evalution_batch_size`` to ``"auto"``
re-resolved the same size on every retry, so ``evaluate_model`` recursed
forever after an eval OOM. The retry now feeds the halved size back into the
failing task and is bounded by ``EVAL_OOM_MAX_RETRIES``.
"""

import model_test as model_test_module
import pytest
import torch
from model_test import ModelTest


_TASK = "arc_challenge"
_RESULT = {"tests": [{"name": _TASK, "metrics": {"acc": 0.5}}]}


class _OomHarness(ModelTest):
    """Minimal ModelTest that never loads or evaluates a real model."""

    EVAL_BATCH_SIZE = "auto"
    MODEL_MAX_LEN = 4096
    EVAL_OOM_MAX_RETRIES = 6

    def runTest(self):
        pass

    def _normalize_task_list(self):
        return [_TASK]


def _make_harness(task_batch_size, *, eval_batch_size="auto", max_retries=6, max_len=4096):
    harness = _OomHarness("runTest")
    harness.EVAL_BATCH_SIZE = eval_batch_size
    harness.MODEL_MAX_LEN = max_len
    harness.EVAL_OOM_MAX_RETRIES = max_retries
    harness._task_evalution_batch_size = {_TASK: task_batch_size}
    return harness


def _patch_resolve(monkeypatch):
    def fake_resolve(batch_size, **_kwargs):
        # Simulate the auto chooser resolving to a fixed, too-large size.
        return 64 if batch_size == "auto" else batch_size

    monkeypatch.setattr(model_test_module, "resolve_eval_batch_size", fake_resolve)


def _patch_eval(monkeypatch, attempts, oom_while, *, result=_RESULT, limit=64):
    """Patch ``evaluate`` to record attempts and raise OOM while ``oom_while``."""

    def fake_evaluate(*, batch_size, **_kwargs):
        attempts.append(batch_size)
        if len(attempts) > limit:
            raise AssertionError(f"eval OOM retry loop did not terminate: {attempts}")
        if oom_while(batch_size):
            raise torch.OutOfMemoryError(f"simulated OOM at batch {batch_size}")
        return result

    monkeypatch.setattr(model_test_module, "evaluate", fake_evaluate)
    _patch_resolve(monkeypatch)


def test_per_task_auto_oom_shrinks_every_retry_and_terminates(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: True)
    harness = _make_harness("auto")

    with pytest.raises(torch.OutOfMemoryError):
        harness.evaluate_model(model="fake-model")

    assert attempts == [64, 32, 16, 8, 4, 2, 1]
    assert all(later < earlier for earlier, later in zip(attempts, attempts[1:]))
    # Initial attempt plus one per configured retry; never unbounded.
    assert len(attempts) == harness.EVAL_OOM_MAX_RETRIES + 1
    assert harness.EVAL_BATCH_SIZE == "0"


def test_per_task_auto_oom_recovers_at_smaller_batch(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: batch > 8)
    harness = _make_harness("auto")

    results = harness.evaluate_model(model="fake-model")

    assert attempts == [64, 32, 16, 8]
    assert results == {_TASK: {"acc": 0.5}}
    assert harness.EVAL_BATCH_SIZE == "8"


def test_per_task_explicit_batch_size_can_still_be_halved(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: batch > 4)
    harness = _make_harness(16)

    results = harness.evaluate_model(model="fake-model")

    assert attempts == [16, 8, 4]
    assert results == {_TASK: {"acc": 0.5}}
    # Legacy behavior is preserved: the explicit per-task size is halved on
    # retry instead of being re-read from the task config.
    assert harness.EVAL_BATCH_SIZE == "4"


def test_global_explicit_batch_size_still_halves(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: batch > 2)
    harness = _make_harness(None, eval_batch_size=8)

    results = harness.evaluate_model(model="fake-model")

    assert attempts == [8, 4, 2]
    assert results == {_TASK: {"acc": 0.5}}
    assert harness.EVAL_BATCH_SIZE == "2"
    # Unchanged legacy behavior: each retry also trims the max sequence length.
    assert harness.MODEL_MAX_LEN == 2048


def test_retry_budget_is_bounded(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: True)
    harness = _make_harness("auto", max_retries=2)

    with pytest.raises(torch.OutOfMemoryError):
        harness.evaluate_model(model="fake-model")

    assert attempts == [64, 32, 16]
    assert len(attempts) == harness.EVAL_OOM_MAX_RETRIES + 1


def test_non_oom_error_is_not_retried(monkeypatch):
    attempts = []

    def fake_evaluate(*, batch_size, **_kwargs):
        attempts.append(batch_size)
        raise ValueError("eval failed")

    monkeypatch.setattr(model_test_module, "evaluate", fake_evaluate)
    _patch_resolve(monkeypatch)
    harness = _make_harness("auto")

    with pytest.raises(ValueError, match="eval failed"):
        harness.evaluate_model(model="fake-model")

    assert attempts == [64]


def test_retry_state_is_reset_between_top_level_calls(monkeypatch):
    attempts = []
    _patch_eval(monkeypatch, attempts, lambda batch: batch > 8)
    harness = _make_harness("auto")

    harness.evaluate_model(model="fake-model")
    assert attempts == [64, 32, 16, 8]

    # A second call starts from the configured per-task "auto" again instead
    # of inheriting the reduced batch from the previous run.
    attempts.clear()
    harness.evaluate_model(model="fake-model")
    assert attempts == [64, 32, 16, 8]
