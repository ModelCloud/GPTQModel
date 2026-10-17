# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
"""Coverage for the ROCm quantize-phase BLAS backend override."""

import pytest

from gptqmodel.utils import rocm


_BACKEND_NAMES = {
    "rocblas": "Cublas",
    "hipblas": "Cublas",
    "cublas": "Cublas",
    "hipblaslt": "Cublaslt",
    "cublaslt": "Cublaslt",
    "ck": "Ck",
}


class _FakeBackend:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_BlasBackend.{self.name}"


class _FakeTorchBlas:
    """Minimal stand-in for ``torch.backends.cuda.preferred_blas_library``."""

    def __init__(self, current="Cublaslt", refuse=None):
        self.current = _FakeBackend(current)
        self.refuse = refuse
        self.calls = []

    def preferred_blas_library(self, value=None):
        if value is None:
            return self.current
        # The real setter accepts both backend names and the backend enum.
        name = getattr(value, "name", value).lower()
        self.calls.append(name)
        if self.refuse is not None and name == self.refuse:
            raise RuntimeError("refused")
        self.current = _FakeBackend(_BACKEND_NAMES[name])
        return None


def _install(monkeypatch, fake):
    monkeypatch.setattr(rocm, "_preferred_blas_library_getter", lambda: fake.preferred_blas_library)


def test_switch_is_scoped_to_the_context(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublas"

    assert fake.calls == ["hipblas", "cublaslt"]
    assert fake.current.name == "Cublaslt"


def test_restore_happens_on_error(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with pytest.raises(RuntimeError):
        with rocm.quant_blas_backend():
            assert fake.current.name == "Cublas"
            raise RuntimeError("quantize blew up")

    assert fake.calls == ["hipblas", "cublaslt"]
    assert fake.current.name == "Cublaslt"


def test_env_can_keep_hipblaslt(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setenv(rocm.QUANT_BLAS_BACKEND_ENV, "hipblaslt")

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == []


def test_user_selected_backend_is_respected(monkeypatch):
    fake = _FakeTorchBlas(current="Ck")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with rocm.quant_blas_backend():
        assert fake.current.name == "Ck"

    assert fake.calls == []


def test_non_rocm_hosts_are_untouched(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", False)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == []


def test_refused_switch_keeps_the_original_backend(monkeypatch):
    fake = _FakeTorchBlas(current="Cublaslt", refuse="hipblas")
    _install(monkeypatch, fake)
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with rocm.quant_blas_backend():
        assert fake.current.name == "Cublaslt"

    assert fake.calls == ["hipblas"]
    assert fake.current.name == "Cublaslt"


def test_missing_torch_api_is_tolerated(monkeypatch):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setattr(rocm, "_preferred_blas_library_getter", lambda: None)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    with rocm.quant_blas_backend():
        pass
