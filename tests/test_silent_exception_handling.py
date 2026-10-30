# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for exception-handling visibility.

Two guarantees are covered:

1. Optional-dependency / compatibility-shim failures are reported instead of
   vanishing.
2. ``KeyboardInterrupt``/``SystemExit`` are never swallowed by fallbacks that
   exist only to survive a *failed* operation.
"""

import ast
import contextlib
import inspect
import io
import textwrap
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

    recording = _RecordingLogger()
    monkeypatch.setattr(gptqmodel, "log", recording)
    monkeypatch.delattr(quantizer_gptq, "GptqHfQuantizer")

    gptqmodel._patch_transformers_gptq_device_map_compat()

    assert recording.warning_calls, "an upstream rename was reported at no level at all"
    msg, exc_info = recording.warning_calls[0]
    assert "device-map compat is inactive" in msg
    assert exc_info is True, "the warning must carry the upstream traceback"


def test_paroquant_quantizer_compat_reports_upstream_api_change(monkeypatch):
    """Same guarantee for the ParoQuant shim."""
    # Simulate upstream moving GPTQConfig: the shim's import then fails even
    # though transformers itself is still installed.
    import transformers.utils.quantization_config as quantization_config

    recording = _RecordingLogger()
    monkeypatch.setattr(gptqmodel, "log", recording)
    monkeypatch.delattr(quantization_config, "GPTQConfig")

    gptqmodel._patch_transformers_paroquant_quantizer_compat()

    assert recording.warning_calls, "an upstream rename was reported at no level at all"
    msg, exc_info = recording.warning_calls[0]
    assert "ParoQuant quantizer compat is inactive" in msg
    assert exc_info is True, "the warning must carry the upstream traceback"


class _RecordingLogger:
    """Minimal stand-in for the project logger that records call levels."""

    def __init__(self):
        self.debug_calls = []
        self.warning_calls = []

    def debug(self, msg, *args, **kwargs):
        self.debug_calls.append((msg, kwargs.get("exc_info")))

    def warning(self, msg, *args, **kwargs):
        self.warning_calls.append((msg, kwargs.get("exc_info")))


def _block_imports(monkeypatch, prefix, exc_factory):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + "."):
            raise exc_factory()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_openvino_compat_stays_quiet_when_dependency_is_absent(monkeypatch):
    """A genuinely optional dependency must not spam warnings when missing."""
    recording = _RecordingLogger()
    monkeypatch.setattr(gptqmodel, "log", recording)
    _block_imports(monkeypatch, "openvino", lambda: ImportError("openvino not installed"))

    gptqmodel._patch_openvino_gptqmodel_compat()

    assert not recording.warning_calls, (
        f"a missing optional dependency must not log at warning level: {recording.warning_calls}"
    )
    assert any("OpenVINO" in msg for msg, _ in recording.debug_calls), (
        f"the skipped OpenVINO shim was not recorded at debug level: {recording.debug_calls}"
    )


def test_openvino_compat_reports_unexpected_probe_failure(monkeypatch):
    """A broken (not merely missing) OpenVINO install must be visible."""
    recording = _RecordingLogger()
    monkeypatch.setattr(gptqmodel, "log", recording)
    _block_imports(
        monkeypatch,
        "openvino",
        lambda: RuntimeError("openvino native library is corrupt"),
    )

    gptqmodel._patch_openvino_gptqmodel_compat()

    assert recording.warning_calls, "an unexpected OpenVINO probe failure was silently dropped"
    assert recording.warning_calls[0][1] is True, "the warning must carry the traceback"


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
    log_calls = [
        call
        for stmt in handler.body
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"warning", "warn", "error", "exception"}
    ]
    assert log_calls, "TritonPatch.apply() failures are not logged"

    # The traceback is the only way to diagnose which triton API moved.
    assert any(
        any(
            kw.arg == "exc_info"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in call.keywords
        )
        for call in log_calls
    ), "the TritonPatch.apply() warning must pass exc_info=True"


def test_causal_conv1d_compat_reports_upstream_api_change(monkeypatch):
    """A renamed transformers integration must not fail silently."""
    real_import = __import__

    def renamed_integration_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Break only the shim's own `from transformers.integrations import
        # hub_kernels`; every other import must keep working.
        if name == "transformers.integrations" and fromlist and "hub_kernels" in fromlist:
            raise ImportError("cannot import name 'hub_kernels' from 'transformers.integrations'")
        return real_import(name, globals, locals, fromlist, level)

    recording = _RecordingLogger()
    monkeypatch.setattr(gptqmodel, "log", recording)
    monkeypatch.setattr("builtins.__import__", renamed_integration_import)

    gptqmodel._patch_transformers_causal_conv1d_hub_kernel_compat()

    assert recording.warning_calls, "a broken causal-conv1d shim was silently ignored"
    msg, exc_info = recording.warning_calls[0]
    assert "causal-conv1d" in msg
    assert "inactive" in msg, (
        f"the shim must report that it is inert, not a generic probe failure: {msg!r}"
    )
    assert exc_info is True


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
    tree = ast.parse(textwrap.dedent(inspect.getsource(torch_utils._detect_mlx)))

    caught = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught.add("BaseException" if node.type is None else ast.unparse(node.type))

    assert caught, "the MLX probe no longer guards its import at all"
    assert not any("BaseException" in entry for entry in caught), (
        f"the MLX probe swallows control-flow exceptions: caught {sorted(caught)}"
    )


def test_mlx_probe_reports_unavailable_when_import_fails(monkeypatch):
    """MLX is optional, so a failed probe must resolve to "unavailable"."""
    _block_imports(monkeypatch, "mlx", lambda: ImportError("mlx not installed"))

    assert torch_utils._detect_mlx() is False


def test_mlx_and_has_mlx_are_updated_together(monkeypatch):
    """`HAS_MLX` and `mlx` must never disagree.

    Callers are written as `if HAS_MLX: mlx.core.clear_cache()`, so a probe
    that flips only one of them turns a supported path into an AttributeError.
    """
    import builtins
    import types

    real_import = builtins.__import__

    pkg = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: None
    core.metal = types.ModuleType("mlx.core.metal")
    pkg.core = core

    state = {"fail": False}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # CPython's `__import__` returns the *top-level* package for a dotted
        # name, and `import a.b.c` binds that return value to `a`. Returning the
        # submodule here would bind `mlx` to `mlx.core.metal` and make the
        # invariant assertion check something production never does.
        if name in {"mlx.core", "mlx.core.metal"}:
            if name == "mlx.core" and state["fail"]:
                raise ImportError("mlx broke after working once")
            return pkg
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert torch_utils._detect_mlx() is True
    assert torch_utils.HAS_MLX is True and torch_utils.mlx is pkg

    # A later probe fails: both halves must flip, not just one.
    state["fail"] = True
    assert torch_utils._detect_mlx() is False
    assert torch_utils.HAS_MLX is False and torch_utils.mlx is None


def test_partial_mlx_install_does_not_leave_module_bound(monkeypatch):
    """A half-installed MLX must not leave ``mlx`` set while ``HAS_MLX`` is False.

    ``import mlx.core`` binds the top-level name before ``mlx.core.metal`` can
    fail, so a naive probe leaves a dangling module reference behind.
    """
    import builtins
    import types

    real_import = builtins.__import__

    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx.core":
            # Mimic CPython: importing a submodule binds the parent package.
            globals["mlx"] = fake_mlx
            return fake_mlx_core
        if name == "mlx.core.metal":
            raise ModuleNotFoundError("No module named 'mlx.core.metal'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(torch_utils, "mlx", "sentinel")

    assert torch_utils._detect_mlx() is False
    assert torch_utils.mlx is None, "a partial MLX install left a dangling module reference"


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_mlx_probe_propagates_control_flow_exceptions(monkeypatch, control_error):
    """A Ctrl-C raised during the MLX probe must reach the caller."""
    _block_imports(monkeypatch, "mlx", control_error)

    with pytest.raises(control_error):
        torch_utils._detect_mlx()


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


def test_generation_config_enables_sampling_before_validation(tmp_path):
    """A `do_sample: false` config must not trigger transformers' flag warning.

    The loader validates temperature/top_p/top_k against the file's own
    `do_sample`; forcing the bit afterwards makes every such model warn that
    those flags "are not valid and may be ignored".
    """
    from gptqmodel.utils import hf as hf_utils

    (tmp_path / "generation_config.json").write_text(
        '{"max_new_tokens": 11, "do_sample": false,'
        ' "temperature": 0.5, "top_p": 0.9, "top_k": 12}',
        encoding="utf-8",
    )

    stderr = _captured_stderr(
        lambda: hf_utils._load_sanitized_generation_config(str(tmp_path))
    )

    assert "not valid and may be ignored" not in stderr, (
        "do_sample was forced after validation, so sampling flags were discarded"
    )

    cfg = hf_utils._load_sanitized_generation_config(str(tmp_path))
    assert cfg.do_sample is True
    assert cfg.max_new_tokens == 11
    assert cfg.temperature == 0.5, "sampling fields must survive the load"


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


def _stubborn_tensor():
    class StubbornTensor(torch.Tensor):
        def to(self, *_args, **_kwargs):
            raise RuntimeError("device move failed")

    return torch.zeros(2).as_subclass(StubbornTensor)


def test_failed_device_moves_do_not_flood_the_log(monkeypatch):
    """Failure reporting must be bounded: one traceback plus one summary line.

    ``rehome_module_to_device`` runs under a global lock and batch workers call
    it repeatedly, so a per-tensor warning would repeat a full traceback for
    every tensor of every module.
    """
    calls = []

    class FakeLogger:
        def warning(self, msg, *args, **kwargs):
            calls.append((msg % args if args else msg, kwargs.get("exc_info")))

    model = torch.nn.Module()
    for index in range(10):
        layer = torch.nn.Module()
        for name in ("a", "b", "c"):
            layer.register_buffer(name, torch.zeros(2))
            layer._buffers[name] = _stubborn_tensor()
        model.add_module(f"layer{index}", layer)

    monkeypatch.setattr(looper_helpers, "log", FakeLogger())
    monkeypatch.setattr(looper_helpers, "_REPORTED_MOVE_FAILURES", set())

    looper_helpers.rehome_module_to_device(model, torch.device("cpu"), only_mismatched=False)

    assert len(calls) == 2, f"expected 1 traceback + 1 summary, got {len(calls)}: {calls}"
    assert calls[0][1] is True, "the first failure must carry the traceback"
    assert not calls[1][1], "the summary needs no traceback"
    assert "30 tensor(s)" in calls[1][0], f"summary must count every failure: {calls[1][0]}"


def test_repeated_rehome_calls_do_not_repeat_the_traceback(monkeypatch):
    """Batch workers re-run rehome per batch; the traceback must not repeat.

    ``forward_batch_worker`` calls ``rehome_module_to_device`` once per batch,
    so a per-call budget alone still produces `2 * batches` lines for a tensor
    that can never move.
    """
    calls = []

    class FakeLogger:
        def warning(self, msg, *args, **kwargs):
            calls.append((msg % args if args else msg, kwargs.get("exc_info")))

    module = torch.nn.Module()
    module.register_buffer("stubborn", torch.zeros(2))
    module._buffers["stubborn"] = _stubborn_tensor()

    monkeypatch.setattr(looper_helpers, "log", FakeLogger())
    monkeypatch.setattr(looper_helpers, "_REPORTED_MOVE_FAILURES", set())

    for _ in range(8):
        looper_helpers.rehome_module_to_device(module, torch.device("cpu"), only_mismatched=False)

    tracebacks = [c for c in calls if c[1] is True]
    assert len(tracebacks) == 1, (
        f"the same failure was reported with a traceback {len(tracebacks)} times"
    )
    assert len(calls) == 9, f"expected one summary line per call, got {len(calls)}"


def test_single_failed_device_move_still_reports_a_count(monkeypatch):
    """One failure gets its traceback plus a one-line count."""
    calls = []

    class FakeLogger:
        def warning(self, msg, *args, **kwargs):
            calls.append((msg % args if args else msg, kwargs.get("exc_info")))

    module = torch.nn.Module()
    module.register_buffer("stubborn", torch.zeros(2))
    module._buffers["stubborn"] = _stubborn_tensor()

    monkeypatch.setattr(looper_helpers, "log", FakeLogger())
    monkeypatch.setattr(looper_helpers, "_REPORTED_MOVE_FAILURES", set())

    looper_helpers.rehome_module_to_device(module, torch.device("cpu"), only_mismatched=False)

    assert len(calls) == 2, f"expected traceback + count, got {calls}"
    assert calls[0][1] is True
    assert "1 tensor(s)" in calls[1][0]
