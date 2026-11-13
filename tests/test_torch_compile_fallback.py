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

import pytest
import torch

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


def test_compiled_module_falls_back_to_its_own_forward(monkeypatch):
    module = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.eye(4))
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    expected = module(x)

    _dynamo_compile(monkeypatch, _closed_pool_error())
    compiled = torch_utils.torch_compile(module)
    assert isinstance(compiled, torch.nn.Module)
    torch.testing.assert_close(compiled(x), expected)


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
