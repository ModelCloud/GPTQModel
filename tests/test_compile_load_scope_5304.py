# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A dense load must not stall optimized forwards outside the dtype window.

#5291/#5301 took the model-global-state writer around whole transformers
``from_pretrained``/``from_config`` calls. Only the ``local_torch_dtype``
window inside those calls is process-global state; the rest is weight I/O,
dtype casts, device placement, packing and kernel post-init. Holding the writer
for the whole call froze every ``optimize()``d forward in the process for the
entire load -- 23-36 s for a 7B checkpoint on the reported host (#5304).

These tests pin the two halves of the contract against a fake loader whose body
stands in for weight I/O and whose construction opens a real
``transformers.modeling_utils.local_torch_dtype`` window:

* a compiled forward (reader) runs while a dense load is in its body, and
* it still waits while that load is inside the dtype window.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch
from transformers import modeling_utils

from gptqmodel.models import loader as loader_module
from gptqmodel.utils import torch as torch_utils


pytest.importorskip("torch._dynamo")

BODY_SECONDS = 0.4


class _SlowLoadLoader:
    """``AutoModelForCausalLM`` stand-in with a slow, state-free load body."""

    def __init__(self):
        self.body_started = threading.Event()
        self.window_opened = threading.Event()
        self.release_window = threading.Event()

    def from_pretrained(self, path, **kwargs):
        config = kwargs.get("config")
        dtype = getattr(config, "dtype", None) or torch.float16
        # The load body before construction: no process-global state changes.
        self.body_started.set()
        time.sleep(BODY_SECONDS)
        with modeling_utils.local_torch_dtype(dtype, model_class_name="Fake"):
            self.window_opened.set()
            assert self.release_window.wait(timeout=30), "the test never released the window"
        return "model"


def _reader_waits_for(lock, limit=0.05):
    """Whether a reader can enter the lock within ``limit`` seconds."""

    entered = threading.Event()

    def reader():
        with lock.reader():
            entered.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    blocked = not entered.wait(timeout=limit)
    return blocked, entered, thread


def test_dense_load_body_does_not_block_compiled_forwards(monkeypatch):
    """The writer must not cover the weight I/O of a dense load (#5304)."""

    loader = _SlowLoadLoader()
    lock = loader_module.model_global_state_lock()
    expected_dtype = torch.float16
    load_errors = []

    def load():
        try:
            loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
                loader,
                "/tmp/model",
                config=SimpleNamespace(dtype=expected_dtype),
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            load_errors.append(exc)

    load_thread = threading.Thread(target=load, daemon=True)
    load_thread.start()
    try:
        assert loader.body_started.wait(timeout=30), "the load body never started"
        blocked, entered, reader_thread = _reader_waits_for(lock)
        if blocked:
            # Let the (blocking) reader finish so the test does not leak it.
            loader.release_window.set()
            load_thread.join(timeout=30)
            reader_thread.join(timeout=30)
            pytest.fail(
                "a compiled forward waited for a dense load's weight I/O; only "
                "transformers' default-dtype window may exclude it (#5304)"
            )
        assert entered.wait(timeout=30), "the reader never ran"
        reader_thread.join(timeout=30)
    finally:
        loader.release_window.set()
        load_thread.join(timeout=30)
        if loader.window_opened.is_set() is False:
            pytest.skip("the fake load never reached construction")
    assert not load_errors, f"the load failed: {load_errors!r}"
    assert loader.window_opened.is_set()
    assert lock._writer is None


def test_dense_load_window_still_excludes_compiled_forwards():
    """The dtype window itself must remain exclusive (no #5127/#5209 relapse)."""

    loader = _SlowLoadLoader()
    lock = loader_module.model_global_state_lock()
    load_errors = []

    def load():
        try:
            loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
                loader,
                "/tmp/model",
                config=SimpleNamespace(dtype=torch.float16),
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            load_errors.append(exc)

    load_thread = threading.Thread(target=load, daemon=True)
    load_thread.start()
    assert loader.window_opened.wait(timeout=30), "the load never opened its dtype window"
    try:
        in_window_dtype = torch.get_default_dtype()
        blocked = threading.Event()

        def reader():
            with lock.reader():
                blocked.set()

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        assert not blocked.wait(timeout=0.2), (
            "a compiled forward entered the lock while the default-dtype window "
            "was open; it could capture and restore that dtype (#5127, #5209)"
        )
        assert in_window_dtype is torch.float16
    finally:
        loader.release_window.set()
        load_thread.join(timeout=30)
    if "reader_thread" in locals():
        reader_thread.join(timeout=30)
    assert not load_errors, f"the load failed: {load_errors!r}"
    assert torch.get_default_dtype() is torch.float32
    assert lock._writer is None


def test_dtype_window_guard_is_the_only_writer():
    """Every site's exclusive hold must be the window, never a whole load."""

    assert torch_utils.model_global_state_dtype_guard_installed()
    assert getattr(modeling_utils.local_torch_dtype, "_gptqmodel_global_state_guarded", False)

    # The helper itself must not re-widen the writer around the load body.
    loader = _SlowLoadLoader()
    lock = loader_module.model_global_state_lock()

    def load():
        loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
            loader, "/tmp/model", config=SimpleNamespace(dtype=torch.float16)
        )

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    assert loader.body_started.wait(timeout=30)
    assert lock._writer is None, (
        "the dense-load helper holds the global-state writer during weight I/O"
    )
    loader.release_window.set()
    thread.join(timeout=30)

