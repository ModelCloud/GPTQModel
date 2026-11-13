# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A deferred ``torch.compile`` failure must degrade to eager, not abort inference.

``torch_compile`` used to guard only the ``torch.compile(...)`` constructor.
Inductor compiles on the first invocation, so a backend failure there (for
example an inductor compile worker raising ``RuntimeError: Attempting to use a
closed pool``) escaped into the caller and aborted the inference request. These
tests pin the runtime contract: the compile failure is reported once and the
real eager implementation serves the call, while genuine user-code and
control-flow exceptions still propagate.
"""

import types

import pytest
import torch

import gptqmodel.models.base as model_base
import gptqmodel.utils.torch as torch_utils
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


class _EagerOnlyBackend:
    """Backend that fails on the first invocation, after construction succeeded."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise self.error


def _closed_pool_error():
    """Build the real failure dynamo raises for a dead inductor worker pool."""
    dynamo_exc = pytest.importorskip("torch._dynamo.exc")
    return dynamo_exc.BackendCompilerFailed(
        "inductor", RuntimeError("Attempting to use a closed pool"), None
    )


_REAL_COMPILE = torch.compile


def _dynamo_compile(monkeypatch, error):
    """Replace ``torch.compile`` with a constructor that returns a failing result.

    Module targets keep the real ``OptimizedModule`` shape so the fallback is
    exercised against what ``torch.compile`` actually returns; callable targets
    get a stand-in whose first invocation raises.
    """
    def fake_compile(target, **_kwargs):
        if isinstance(target, torch.nn.Module):
            compiled = _REAL_COMPILE(target, backend="eager")
            compiled.forward = _EagerOnlyBackend(error)
            return compiled
        return _EagerOnlyBackend(error)
    monkeypatch.setattr(torch_utils.torch, "compile", fake_compile)


def test_compile_failure_at_first_call_uses_eager_result(monkeypatch):
    calls = []

    def eager(value):
        calls.append(value)
        return value * 3

    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(eager)
    assert compiled is not eager

    # First call hits the compiler, which fails: eagerly compute the same value.
    assert compiled(2) == 6
    assert calls == [2]

    # The fallback is sticky: the compiler is not retried on every call.
    assert compiled(4) == 12
    assert calls == [2, 4]


def test_compile_failure_runs_eager_only_once_per_call(monkeypatch):
    executions = []

    def eager(value):
        executions.append(value)
        return value + 1

    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(eager)
    compiled(1)
    assert executions == [1], "a compiler failure must not also double-run eager code"


@pytest.mark.parametrize("error", [
    ValueError("bad input"),
    TypeError("bad dtype"),
    RuntimeError("user body failed"),
])
def test_user_code_errors_still_propagate(error):
    """A raising body must not be retried: dynamo's own frame sits in the traceback."""
    executions = []

    def eager(value):
        executions.append(value)
        raise error

    compiled = torch_utils.torch_compile(eager)
    with pytest.raises(type(error), match=str(error)):
        compiled(1)
    # If the origin check misread the user error as a compiler failure, eager
    # would run a second time and the caller would see the retry, not the first.
    assert executions == [1]


@pytest.mark.parametrize("control_error", [KeyboardInterrupt, SystemExit])
def test_control_flow_errors_are_never_swallowed(monkeypatch, control_error):
    def eager(_value):
        return 0

    _dynamo_compile(monkeypatch, control_error("stop"))
    compiled = torch_utils.torch_compile(eager)
    with pytest.raises(control_error):
        compiled(1)


def test_fullgraph_capture_failure_stays_fatal(monkeypatch):
    """``fullgraph=True`` is a single-graph contract; do not silently degrade it."""
    dynamo_exc = pytest.importorskip("torch._dynamo.exc")
    unsupported = dynamo_exc.Unsupported("graph break")

    def eager(_value):
        return 0

    _dynamo_compile(monkeypatch, unsupported)
    compiled = torch_utils.torch_compile(eager, fullgraph=True)
    with pytest.raises(dynamo_exc.Unsupported):
        compiled(1)


def test_compiled_module_never_returns_a_plain_callable(monkeypatch):
    """Callers assign the result to an nn.Module attribute; it must stay a module."""
    module = torch.nn.Linear(4, 4, bias=False)
    monkeypatch.setattr(torch_utils.torch, "compile", lambda *_a, **_k: lambda _x: None)
    compiled = torch_utils.torch_compile(module)
    assert compiled is module


def test_compiled_module_stays_eager_after_a_deferred_failure(monkeypatch):
    """A module forward is not retried; later calls run the eager module."""
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(4))
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    expected = module(x)

    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(module)
    assert isinstance(compiled, torch.nn.Module)
    with pytest.raises(Exception, match="closed pool"):
        compiled(x)
    # The failure armed the sticky eager path; the next call never recompiles.
    torch.testing.assert_close(compiled(x), expected)


def test_compiled_module_never_replays_side_effects(monkeypatch):
    """An earlier graph segment may already have run: never retry the forward."""
    executions = []

    class Counted(torch.nn.Module):
        def forward(self, x):
            executions.append(1)
            return x + 1

    module = Counted()
    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(module)
    with pytest.raises(Exception, match="closed pool"):
        compiled(torch.zeros(2))
    assert executions == [], "the failing call must not also run the eager module"


def test_fullgraph_backend_failure_stays_fatal(monkeypatch):
    """Nothing degrades under ``fullgraph=True``, including a backend failure."""

    def eager(value):
        return value

    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(eager, fullgraph=True)
    with pytest.raises(Exception, match="closed pool"):
        compiled(1)
    assert torch_utils._is_compiler_failure(_closed_pool_error(), fullgraph=True) is False


def _raise_in(filename, message="boom"):
    """Raise a RuntimeError whose innermost frame is ``filename``."""
    try:
        exec(compile(f"raise RuntimeError({message!r})", filename, "exec"), {})
    except RuntimeError as exc:
        return exc
    raise AssertionError("unreachable")


def test_runtime_error_from_a_compiler_frame_is_a_compiler_failure():
    exc = _raise_in("/site-packages/torch/_inductor/compile_worker/subproc_pool.py")
    assert torch_utils._is_compiler_failure(exc) is True


def test_runtime_error_with_only_an_outer_compiler_frame_is_user_code():
    """A compiler frame above the raise point must not make user code eligible."""
    user_ns, driver_ns = {}, {}
    exec(
        compile("def boom():\n    raise RuntimeError('user body')\n", "/tmp/user_model.py", "exec"),
        user_ns,
    )
    exec(
        compile(
            "def drive(fn):\n    return fn()\n",
            "/site-packages/torch/_inductor/driver.py",
            "exec",
        ),
        driver_ns,
    )
    with pytest.raises(RuntimeError, match="user body") as info:
        driver_ns["drive"](user_ns["boom"])
    assert torch_utils._is_compiler_failure(info.value) is False


def test_control_flow_is_never_classified_as_a_compiler_failure():
    assert torch_utils._is_compiler_failure(KeyboardInterrupt()) is False
    assert torch_utils._is_compiler_failure(SystemExit()) is False


def test_torchdynamo_disable_returns_the_eager_object(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")

    def eager(value):
        return value

    assert torch_utils.torch_compile(eager) is eager
    module = torch.nn.Linear(2, 2)
    assert torch_utils.torch_compile(module) is module


def _torch_linear(compile_flag=None):
    kwargs = {
        "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
        "in_features": 512, "out_features": 64, "bias": False,
        "pack_dtype": torch.int32, "register_buffers": True,
        "format": FORMAT.GPTQ_V2, "dtype": torch.float16,
    }
    if compile_flag is not None:
        kwargs["compile"] = compile_flag
    module = TorchLinear(**kwargs)
    module.eval()
    return module


def test_torch_linear_compile_false_skips_compile():
    module = _torch_linear(compile_flag=False)
    eager = module.dequantize_weight.__func__
    module.optimize()
    assert module.optimized
    assert module.dequantize_weight.__func__ is eager


def test_torch_linear_dequantize_retries_eagerly(monkeypatch):
    """The reported failure: a deferred compile failure inside dequantize_weight."""
    module = _torch_linear()
    _dynamo_compile(monkeypatch, _closed_pool_error())
    module.optimize()
    assert module.optimized

    eager = TorchLinear.dequantize_weight(module)
    compiled = module.dequantize_weight()
    torch.testing.assert_close(compiled, eager)
    # Sticky: the second call never touches the broken compiler again.
    torch.testing.assert_close(module.dequantize_weight(), eager)


def _fake_model(compile_flag):
    return types.SimpleNamespace(
        quantized=True,
        quantize_config=types.SimpleNamespace(compile=compile_flag),
        model=torch.nn.Linear(2, 2),
        compiled=False,
    )


def test_model_optimize_honours_compile_false(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_base, "torch_compile", lambda target, **_kwargs: calls.append(target) or target
    )
    fake = _fake_model(False)
    model_base.BaseQModel.optimize(fake)
    assert calls == [], "compile=False must also stop the whole-model compile"
    assert fake.compiled is False


def test_model_optimize_compiles_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_base, "torch_compile", lambda target, **_kwargs: calls.append(target) or target
    )
    fake = _fake_model(True)
    model_base.BaseQModel.optimize(fake)
    assert calls == [fake.model]


def test_torch_linear_rejects_non_bool_compile():
    with pytest.raises(TypeError, match="`compile` must be bool"):
        TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                    in_features=64, out_features=64, bias=False,
                    pack_dtype=torch.int32, register_buffers=True,
                    format=FORMAT.GPTQ_V2, compile="yes")


def test_quantize_config_compile_is_runtime_only():
    from gptqmodel import QuantizeConfig

    config = QuantizeConfig(bits=4, group_size=128, compile=False)
    assert config.quant_linear_init_kwargs()["compile"] is False
    assert "compile" not in config.to_dict()
    assert QuantizeConfig.from_quant_config(config.to_dict()).compile is True
