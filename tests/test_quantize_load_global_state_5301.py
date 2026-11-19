# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every transformers construction site must hold the global-state writer.

transformers runs ``from_pretrained``/``from_config`` construction inside
``local_torch_dtype``, which sets the process-global default dtype. Dynamo's
``preserve_global_state`` restores that dtype around every trace, so a compiled
forward re-tracing inside the window can leave bf16 installed after the load
restored fp32 (#5127/#5209).

#5291 split the old load lock into readers (compiled calls) and a writer
(construction), which made the coverage per-site: the quantize-path load with
``offload_to_disk=False`` kept calling ``loader.from_pretrained`` without the
writer and reintroduced the corruption (#5301). The writer now lives inside
``_hf_loader_from_pretrained_with_dynamic_module_retry`` so no call site can
miss it.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from gptqmodel.models import loader as loader_module


HELPER = "_hf_loader_from_pretrained_with_dynamic_module_retry"


class _FakeLoader:
    """Stand-in for ``transformers.AutoModelForCausalLM``."""

    def __init__(self, *, fail_first=False, error=None):
        self.fail_first = fail_first
        self.error = error
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
        if self.fail_first and len(self.calls) == 1:
            raise self.error
        return "model"


def _broken_dynamic_module_error() -> FileNotFoundError:
    return FileNotFoundError(
        2, "missing", "/tmp/transformers_modules/qwen/model.py"
    )


def test_helper_holds_the_writer_for_every_attempt():
    loader = _FakeLoader(fail_first=True, error=_broken_dynamic_module_error())

    assert loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
        loader, "/tmp/model"
    ) == "model"

    assert len(loader.calls) == 2
    # Both the first construction and the force-download retry are exclusive.
    assert all(call["writer"] == call["ident"] for call in loader.calls)
    assert loader.calls[1]["force_download"] is True
    assert loader_module.model_global_state_lock()._writer is None


def test_helper_releases_the_writer_when_the_load_fails():
    lock = loader_module.model_global_state_lock()
    loader = _FakeLoader(fail_first=True, error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        loader_module._hf_loader_from_pretrained_with_dynamic_module_retry(
            loader, "/tmp/model"
        )

    assert lock._writer is None
    assert len(loader.calls) == 1


def _write_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Line ranges covered by ``with model_global_state_lock():`` blocks."""

    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "model_global_state_lock":
                ranges.append((node.lineno, node.end_lineno))
    return ranges


def test_every_model_construction_site_is_guarded():
    """A new ``loader.from_pretrained``/``from_config`` site must not bypass it."""

    path = Path(loader_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded = _write_ranges(tree)

    def _inside(line: int, ranges) -> bool:
        return any(start <= line <= end for start, end in ranges)

    sites = []
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
        # ``loader.from_pretrained`` (the helper) and ``cls.loader.from_config``.
        if receiver != "loader" and not receiver.endswith(".loader"):
            continue
        sites.append((node.lineno, receiver, func.attr))

    assert sites, "no construction sites found; the guard is not looking at the right file"
    for line, receiver, attr in sites:
        assert _inside(line, guarded), (
            f"{receiver}.{attr} at line {line} builds a model without the "
            "model-global-state writer (#5301)"
        )

    # The helper itself is used by the construction paths, so the guard above
    # cannot be satisfied by leaving a raw call behind.
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == HELPER
    ]
    assert len(helper_calls) >= 3
