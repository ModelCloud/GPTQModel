# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for exception-handling visibility.

Two guarantees are covered:

1. Optional-dependency / compatibility-shim failures are reported instead of
   vanishing (issue: "静默吞异常面过大").
2. ``KeyboardInterrupt``/``SystemExit`` are never swallowed by fallbacks that
   exist only to survive a *failed* operation.
"""

import contextlib
import io
import threading

import pytest
import torch

import gptqmodel
import gptqmodel.utils.looper_helpers as looper_helpers
import gptqmodel.utils.torch as torch_utils
from gptqmodel.looper.module_looper import ModuleLooper


def _captured_stderr(fn) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        fn()
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 1. Compatibility shims must not fail silently
# ---------------------------------------------------------------------------


def test_transformers_device_map_compat_reports_upstream_api_change(monkeypatch):
    """An upstream rename must produce a visible warning, not a silent no-op."""
    import transformers.quantizers.quantizer_gptq as quantizer_gptq

    monkeypatch.delattr(quantizer_gptq, "GptqHfQuantizer")

    stderr = _captured_stderr(gptqmodel._patch_transformers_gptq_device_map_compat)

    assert "device-map compat is inactive" in stderr


def test_paroquant_quantizer_compat_reports_upstream_api_change(monkeypatch):
    # Simulate upstream moving GPTQConfig: the shim's import then fails even
    # though transformers itself is still installed.
    import transformers.utils.quantization_config as quantization_config

    monkeypatch.delattr(quantization_config, "GPTQConfig")

    stderr = _captured_stderr(gptqmodel._patch_transformers_paroquant_quantizer_compat)

    assert "ParoQuant quantizer compat is inactive" in stderr


def test_openvino_compat_stays_quiet_when_dependency_is_absent(monkeypatch):
    """A genuinely optional dependency must not spam warnings when missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("openvino"):
            raise ImportError("openvino not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stderr = _captured_stderr(gptqmodel._patch_openvino_gptqmodel_compat)

    assert "compat is inactive" not in stderr


def test_triton_patch_failure_is_reported_by_import_guard(monkeypatch):
    """The import-time guard must log instead of using a bare ``pass``."""
    import ast
    from pathlib import Path

    init_py = Path(gptqmodel.__file__)
    tree = ast.parse(init_py.read_text(encoding="utf-8"))

    triton_guard = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        names = [ast.unparse(handler.type) for handler in node.handlers if handler.type is not None]
        if "Exception" not in names:
            continue
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "TritonPatch"
            for stmt in node.body
            for call in ast.walk(stmt)
        ):
            triton_guard = node
            break

    assert triton_guard is not None, "TritonPatch.apply() import guard not found"
    handler = triton_guard.handlers[0]
    assert not any(isinstance(stmt, ast.Pass) for stmt in handler.body), (
        "TritonPatch.apply() failures are swallowed by a bare `pass`"
    )
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"warning", "warn", "error", "exception"}
        for stmt in handler.body
        for call in ast.walk(stmt)
    ), "TritonPatch.apply() failures are not logged"


# ---------------------------------------------------------------------------
# 2. KeyboardInterrupt / SystemExit must survive fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_torch_compile_fallback_propagates_control_flow_exceptions(monkeypatch, control_error):
    def fake_compile(*_args, **_kwargs):
        raise control_error("interrupted")

    monkeypatch.setattr(torch_utils.torch, "compile", fake_compile)

    with pytest.raises(control_error):
        torch_utils.torch_compile(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_torch_replicate_device_sync_propagates_control_flow_exceptions(monkeypatch, control_error):
    class _PassthroughPool:
        """Stand-in pool: no device/rwlock bookkeeping, no GPU required."""

        def lock(self, _scope):
            return contextlib.nullcontext()

    def fake_sync(_device):
        raise control_error("interrupted")

    monkeypatch.setattr(looper_helpers, "DEVICE_THREAD_POOL", _PassthroughPool())
    monkeypatch.setattr(looper_helpers, "torch_sync", fake_sync)

    with pytest.raises(control_error):
        looper_helpers.torch_replicate(torch.nn.Linear(2, 2), [torch.device("cuda", 0)])


def test_mlx_probe_does_not_catch_base_exception():
    """The MLX probe is optional, but must not mask Ctrl-C during import."""
    import inspect

    source = inspect.getsource(torch_utils)
    probe_start = source.index("import mlx.core")
    probe_end = source.index("BACKENDS_HAS_FP32_PRECISION", probe_start)

    assert "except BaseException" not in source[probe_start:probe_end]


def test_subset_callback_failure_is_logged(monkeypatch):
    """A failing subset callback must be recorded *and* reported."""
    logged = {}

    class FakeLogger:
        def warning(self, msg, *args, **kwargs):
            logged["msg"] = msg
            logged["exc_info"] = kwargs.get("exc_info")

    looper = object.__new__(ModuleLooper)
    looper.lock = threading.Lock()
    looper._loop_stop_event = threading.Event()
    looper._loop_stop_exc = None
    looper._loop_stop_waited = False

    def failing_handler(**_kwargs):
        raise RuntimeError("callback bug")

    monkeypatch.setattr(ModuleLooper, "_resolve_subset_callback", lambda self: failing_handler)
    monkeypatch.setattr("gptqmodel.looper.module_looper.log", FakeLogger())

    looper._emit_subset_event(
        stage="process",
        layer_idx=0,
        subset_index=0,
        subset_total=1,
        module_names=[],
        processor="test",
        raise_in_place=False,
    )

    assert logged.get("msg", "").startswith("Subset callback raised")
    assert logged.get("exc_info") is True
    assert isinstance(looper._loop_stop_exc, RuntimeError)
