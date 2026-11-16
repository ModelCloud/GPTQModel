# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4465: a cached kernel-validation failure must not pin the caller's model.

`BaseQuantLinear.cached_validate_once` used to `lru_cache` the exception that
`validate_once` returned. `with_traceback(None)` cleared only that exception's
own traceback. Its chained `__cause__` (for example the `ModuleNotFoundError`
behind "GGUFCppKernel requires llama-cpp-python") kept the frames of the
failed probe and, through `f_back`, the loader frames that held the model
being loaded. The cache now keeps only `(ok, type, message)`.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from gptqmodel.nn_modules.qlinear.torch import TorchLinear


_MESSAGE = "TorchLinear requires a missing optional dependency for this test."


class _Model:
    """Stands in for the model a loader frame holds while it validates kernels."""


def _failing_validate_once(cls):
    try:
        import gptqmodel_test_missing_dependency_4465  # noqa: F401
    except ModuleNotFoundError as exc:
        try:
            raise ImportError(_MESSAGE) from exc
        except ImportError as chained:
            return False, chained
    return True, None


@pytest.fixture
def failing_kernel(monkeypatch):
    TorchLinear.cached_validate_once.cache_clear()
    monkeypatch.setattr(TorchLinear, "validate_once", classmethod(_failing_validate_once))
    yield TorchLinear
    TorchLinear.cached_validate_once.cache_clear()


def _load_and_validate(kernel):
    model = _Model()
    ok, error = kernel.cached_validate_once()
    return weakref.ref(model), ok, error


def test_failed_validation_does_not_pin_the_loading_frame(failing_kernel):
    model_ref, ok, error = _load_and_validate(failing_kernel)

    assert ok is False
    assert isinstance(error, ImportError)
    assert str(error) == _MESSAGE
    del error
    gc.collect()
    assert model_ref() is None, "the cached validation failure keeps the loader's model alive"

    # The failure is still cached: a later caller sees it without re-probing.
    assert failing_kernel.cached_validate_once.cache_info().hits == 0
    ok, error = failing_kernel.cached_validate_once()
    assert ok is False and str(error) == _MESSAGE
    assert failing_kernel.cached_validate_once.cache_info().hits == 1


def test_raising_a_cached_failure_does_not_leak_into_later_callers(failing_kernel):
    def raise_while_holding_a_model():
        model = _Model()
        _ok, error = failing_kernel.cached_validate_once()
        try:
            raise error
        except ImportError:
            pass
        return weakref.ref(model), error

    model_ref, raised = raise_while_holding_a_model()
    assert raised.__traceback__ is not None

    ok, fresh = failing_kernel.cached_validate_once()
    assert ok is False
    assert fresh is not raised
    assert fresh.__traceback__ is None and fresh.__cause__ is None and fresh.__context__ is None
    assert str(fresh) == _MESSAGE

    del raised
    gc.collect()
    assert model_ref() is None


def test_successful_validation_is_cached(monkeypatch):
    calls = []

    def ok_validate_once(cls):
        calls.append(cls)
        return True, None

    TorchLinear.cached_validate_once.cache_clear()
    monkeypatch.setattr(TorchLinear, "validate_once", classmethod(ok_validate_once))
    try:
        assert TorchLinear.cached_validate_once() == (True, None)
        assert TorchLinear.cached_validate_once() == (True, None)
        assert calls == [TorchLinear]
    finally:
        TorchLinear.cached_validate_once.cache_clear()


def test_missing_llama_cpp_does_not_pin_the_first_gguf_loader(monkeypatch):
    # The GGUF repro: without llama-cpp-python the module-level import failure
    # was kept as an exception whose traceback chained to the loader frames.
    import sys

    from gptqmodel.nn_modules.qlinear import gguf_cpp

    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr(gguf_cpp, "_LLAMA_CPP_LOADED", False)
    monkeypatch.setattr(gguf_cpp, "_LLAMA_CPP_IMPORT_ERROR", None)
    monkeypatch.setattr(gguf_cpp, "_GGML_BRIDGE", None)
    kernel = gguf_cpp.GGUFCppKernel
    kernel.cached_validate_once.cache_clear()
    try:
        model_ref, ok, error = _load_and_validate(kernel)

        assert ok is False
        assert isinstance(error, ModuleNotFoundError)
        message = str(error)
        del error
        gc.collect()
        assert model_ref() is None, "the llama-cpp import failure keeps the loader's model alive"
        # The import failure reason is still reported, as text.
        assert "llama-cpp-python" in message and "llama_cpp" in message
    finally:
        kernel.cached_validate_once.cache_clear()
