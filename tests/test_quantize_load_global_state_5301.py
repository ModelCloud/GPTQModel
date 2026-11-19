# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The global-state writer belongs to transformers' dtype window, not to a load.

transformers runs ``from_pretrained``/``from_config`` construction inside
``local_torch_dtype``, which sets the process-global default dtype. Dynamo's
``preserve_global_state`` restores that dtype around every trace, so a compiled
forward re-tracing inside the window can leave bf16 installed after the load
restored fp32 (#5127/#5209).

#5291 made the window's owner the construction *call*: every
``loader.from_pretrained``/``from_config`` site took the writer for its whole
duration. For a dense load that is the entire weight I/O, dtype casts, device
placement, packing and kernel post-init, so every optimized forward in the
process stalled for the whole load (#5304: 23-36 s for a 7B model).

The writer now belongs to the window itself. These tests pin that contract:
the helper no longer wraps the call, transformers' own ``local_torch_dtype``
takes the writer while it is open, and no construction site re-widens it.
"""

from __future__ import annotations

import ast
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from gptqmodel.models import loader as loader_module
from gptqmodel.utils import torch as torch_utils


HELPER = "_hf_loader_from_pretrained_with_dynamic_module_retry"


class _FakeLoader:
    """Stand-in for ``transformers.AutoModelForCausalLM``."""

    def __init__(self, *, fail_first=False, error=None, body_seconds=0.0):
        self.fail_first = fail_first
        self.error = error
        self.body_seconds = body_seconds
        self.calls = []

    def from_pretrained(self, path, **kwargs):
        lock = loader_module.model_global_state_lock()
        self.calls.append(
            {
                "path": path,
                "force_download": kwargs.get("force_download"),
                "writer": lock._writer,
                "ident": threading.get_ident(),
            }
        )
        if self.body_seconds:
            # Stands in for weight I/O: the process-global state is untouched.
            time.sleep(self.body_seconds)
            self.calls[-1]["writer_after_body"] = lock._writer
        if self.fail_first and len(self.calls) == 1:
            raise self.error
        return "model"


def _broken_dynamic_module_error() -> FileNotFoundError:
    return FileNotFoundError(
        2, "missing", "/tmp/transformers_modules/qwen/model.py"
    )


def test_helper_does_not_hold_the_writer_for_the_load_body():
    """The weight I/O around construction must stay concurrent (#5304)."""

    loader = _FakeLoader(body_seconds=0.2)

    assert loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
        loader, "/tmp/model"
    ) == "model"

    assert len(loader.calls) == 1
    assert loader.calls[0]["writer"] is None, (
        "the helper widened the writer to the whole from_pretrained call; a "
        "compiled forward would stall for the load's weight I/O (#5304)"
    )
    assert loader.calls[0]["writer_after_body"] is None
    assert loader_module.model_global_state_lock()._writer is None


def test_helper_still_retries_a_broken_dynamic_module_cache():
    loader = _FakeLoader(fail_first=True, error=_broken_dynamic_module_error())

    assert loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
        loader, "/tmp/model"
    ) == "model"

    assert len(loader.calls) == 2
    assert loader.calls[1]["force_download"] is True
    assert all(call["writer"] is None for call in loader.calls)


def test_helper_releases_the_writer_when_the_load_fails():
    lock = loader_module.model_global_state_lock()
    loader = _FakeLoader(fail_first=True, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
            loader, "/tmp/model"
        )

    assert lock._writer is None
    assert len(loader.calls) == 1


def test_dtype_guard_is_installed_on_the_transformers_window():
    """Package import must wrap ``local_torch_dtype`` once with the writer."""

    from transformers import modeling_utils

    assert torch_utils.model_global_state_dtype_guard_installed()
    guarded = modeling_utils.local_torch_dtype
    assert getattr(guarded, "_gptqmodel_global_state_guarded", False)
    assert getattr(guarded, "_gptqmodel_original", None) is not None


def test_dtype_window_takes_the_writer_and_nothing_else_does():
    """Inside the window the writer is held; after it exits nothing is."""

    from transformers import modeling_utils

    lock = loader_module.model_global_state_lock()
    observed = {}

    with modeling_utils.local_torch_dtype(torch_utils.torch.bfloat16):
        observed["in_window_writer"] = lock._writer
        observed["in_window_dtype"] = torch_utils.torch.get_default_dtype()
        observed["in_window_ident"] = threading.get_ident()

    observed["after_writer"] = lock._writer
    observed["after_dtype"] = torch_utils.torch.get_default_dtype()

    assert observed["in_window_writer"] == observed["in_window_ident"], (
        "transformers opened its default-dtype window without the writer"
    )
    assert observed["in_window_dtype"] is torch_utils.torch.bfloat16
    assert observed["after_writer"] is None
    assert observed["after_dtype"] is torch_utils.torch.float32


def test_dtype_window_excludes_a_reader():
    """A compiled forward must still wait for the dtype window (not the load)."""

    from transformers import modeling_utils

    lock = loader_module.model_global_state_lock()
    in_window = threading.Event()
    release = threading.Event()
    reader_done = threading.Event()
    reader_errors = []

    def holder():
        with modeling_utils.local_torch_dtype(torch_utils.torch.bfloat16):
            in_window.set()
            release.wait(timeout=30)

    def reader():
        try:
            with lock.reader():
                reader_done.set()
        except BaseException as exc:  # noqa: BLE001 - reported below
            reader_errors.append(exc)

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    assert in_window.wait(timeout=30), "the dtype window never opened"

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    try:
        assert not reader_done.wait(timeout=0.3), (
            "a compiled call overlapped transformers' default-dtype window"
        )
    finally:
        release.set()
        holder_thread.join(timeout=30)
        reader_thread.join(timeout=30)

    assert reader_done.is_set(), "the reader never resumed after the window closed"
    assert not reader_errors, f"reader failed: {reader_errors!r}"


@contextmanager
def _empty_mapping():
    yield


def test_every_construction_site_uses_the_guarded_window():
    """No construction site may re-widen the writer around the load body."""

    path = Path(loader_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "model_global_state_lock":
                ranges.append((node.lineno, node.end_lineno))

    def _inside(line: int) -> bool:
        return any(start <= line <= end for start, end in ranges)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in (
            "from_pretrained",
            "from_config",
        ):
            continue
        receiver = ast.unparse(func.value)
        if receiver != "loader" and not receiver.endswith(".loader"):
            continue
        if _inside(node.lineno):
            offenders.append((node.lineno, receiver, func.attr))

    assert not offenders, (
        "these construction sites still hold the global-state writer around the "
        f"whole load instead of transformers' dtype window (#5304): {offenders}"
    )

    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == HELPER
    ]
    assert len(helper_calls) >= 3
