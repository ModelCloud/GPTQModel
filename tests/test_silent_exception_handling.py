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


def _make_looper():
    looper = object.__new__(ModuleLooper)
    looper.lock = threading.Lock()
    looper._loop_stop_event = threading.Event()
    looper._loop_stop_exc = None
    looper._loop_stop_waited = False
    return looper


def _emit_subset(looper, *, raise_in_place):
    looper._emit_subset_event(
        stage="process",
        layer_idx=0,
        subset_index=0,
        subset_total=1,
        module_names=[],
        processor="test",
        raise_in_place=raise_in_place,
    )


def test_subset_callback_failure_is_logged_when_swallowed(monkeypatch):
    """The swallowed path must leave a record instead of only latching."""

    class FakeLogger:
        def __init__(self):
            self.calls = []

        def warning(self, msg, *args, **kwargs):
            self.calls.append((msg, kwargs.get("exc_info")))

    fake_log = FakeLogger()
    looper = _make_looper()

    def failing_handler(**_kwargs):
        raise RuntimeError("callback bug")

    monkeypatch.setattr(ModuleLooper, "_resolve_subset_callback", lambda self: failing_handler)
    monkeypatch.setattr("gptqmodel.looper.module_looper.log", fake_log)

    _emit_subset(looper, raise_in_place=False)

    assert len(fake_log.calls) == 1
    msg, exc_info = fake_log.calls[0]
    assert msg.startswith("Subset callback raised")
    assert exc_info is True
    assert isinstance(looper._loop_stop_exc, RuntimeError)


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit, RuntimeError])
def test_subset_callback_in_place_propagates_without_spurious_warning(monkeypatch, control_error):
    """Ctrl-C must propagate cleanly, not be reported as a callback bug."""

    class ExplodingLogger:
        def __init__(self):
            self.calls = 0

        def warning(self, *_args, **_kwargs):
            self.calls += 1

    fake_log = ExplodingLogger()
    looper = _make_looper()

    def failing_handler(**_kwargs):
        raise control_error("raised by handler")

    monkeypatch.setattr(ModuleLooper, "_resolve_subset_callback", lambda self: failing_handler)
    monkeypatch.setattr("gptqmodel.looper.module_looper.log", fake_log)

    with pytest.raises(control_error):
        _emit_subset(looper, raise_in_place=True)

    assert fake_log.calls == 0, "in-place propagation must not log a spurious warning"


# ---------------------------------------------------------------------------
# 3. Degradations that hide real failures must be reported
# ---------------------------------------------------------------------------


def test_unreadable_config_json_is_reported(tmp_path, monkeypatch):
    """A corrupt config.json must not silently read as "not DeepSeek-VL2"."""
    from gptqmodel.utils import hf as hf_utils

    model_dir = tmp_path / "broken-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{ this is not json", encoding="utf-8")

    calls = []

    class FakeLogger:
        def debug(self, msg, *args, **kwargs):
            calls.append((msg, kwargs.get("exc_info")))

    monkeypatch.setattr(hf_utils, "log", FakeLogger())

    assert hf_utils._is_deepseek_vl_v2_model_path(str(model_dir)) is False
    assert calls, "a corrupt config.json was silently ignored"
    assert calls[0][1] is True


def test_generation_config_is_loaded_not_silently_skipped(tmp_path):
    """Regression: `get_config_dict` vanished in transformers 5, so this path
    always raised AttributeError and the caller silently used defaults."""
    from gptqmodel.utils import hf as hf_utils

    (tmp_path / "generation_config.json").write_text(
        '{"max_new_tokens": 7, "do_sample": false}', encoding="utf-8"
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))

    assert cfg is not None, "generation config silently failed to load"
    assert cfg.max_new_tokens == 7
    # _sanitize_generation_config forces sampling on.
    assert cfg.do_sample is True


def test_unreadable_generation_config_is_reported(tmp_path, monkeypatch):
    from transformers import GenerationConfig

    from gptqmodel.utils import hf as hf_utils

    calls = []

    class FakeLogger:
        def debug(self, msg, *args, **kwargs):
            calls.append((msg, kwargs.get("exc_info")))

    def broken_from_pretrained(*_args, **_kwargs):
        raise ValueError("corrupt generation config")

    monkeypatch.setattr(hf_utils, "log", FakeLogger())
    monkeypatch.setattr(GenerationConfig, "from_pretrained", staticmethod(broken_from_pretrained))

    assert hf_utils._load_sanitized_generation_config(str(tmp_path)) is None
    assert calls, "an unreadable generation config was silently ignored"
    assert calls[0][1] is True


def test_failed_device_move_is_reported(monkeypatch):
    """A buffer that cannot reach the target device is a correctness problem."""
    calls = []

    class FakeLogger:
        def warning(self, msg, *args, **kwargs):
            calls.append((msg, kwargs.get("exc_info")))

    class StubbornTensor(torch.Tensor):
        def to(self, *_args, **_kwargs):
            raise RuntimeError("device move failed")

    stubborn = torch.zeros(2).as_subclass(StubbornTensor)

    module = torch.nn.Module()
    module.register_buffer("stubborn", torch.zeros(2))
    module._buffers["stubborn"] = stubborn

    monkeypatch.setattr(looper_helpers, "log", FakeLogger())

    looper_helpers.rehome_module_to_device(
        module, torch.device("cpu"), only_mismatched=False
    )

    assert calls, "a failed buffer move was silently swallowed"
    assert calls[0][1] is True
