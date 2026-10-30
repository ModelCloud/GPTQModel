# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU coverage for the ROCm quantize-phase tuning table."""

import pytest
import torch

from gptqmodel.quantization import gsq_cuda_relaxation
from gptqmodel.utils import linalg_warmup, rocm


@pytest.mark.parametrize(
    ("arch", "blas_backend", "prewarm_ops"),
    [
        ("gfx950", "rocblas", None),
        ("gfx950:sramecc+:xnack-", "rocblas", None),
        ("gfx942", "rocblas", None),
        ("gfx1100", "rocblas", None),
        ("gfx1200", "rocblas", rocm._RDNA4_PREWARM_GROUPS),
        ("gfx1201", "rocblas", rocm._RDNA4_PREWARM_GROUPS),
        ("gfx1250", "rocblas", rocm._RDNA4_PREWARM_GROUPS),
        ("gfx9999", "rocblas", None),
    ],
)
def test_quantize_tuning_on_rocm_hosts(monkeypatch, arch, blas_backend, prewarm_ops):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.delenv(rocm.QUANT_BLAS_BACKEND_ENV, raising=False)

    tuning = rocm.quantize_tuning(arch)

    assert tuning == {"blas_backend": blas_backend, "amp": False, "prewarm_ops": prewarm_ops}
    # AMP stays off on every ROCm architecture, RDNA4 included.
    assert rocm.rocm_amp_disabled(arch, is_rocm=True) is True

@pytest.mark.parametrize("value", ["hipblaslt", "cublaslt", "HIPBLASLT"])
def test_quantize_tuning_keeps_the_hipblaslt_opt_in(monkeypatch, value):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setenv(rocm.QUANT_BLAS_BACKEND_ENV, value)

    assert rocm.quantize_tuning("gfx1201")["blas_backend"] == value.strip().lower()

@pytest.mark.parametrize("value", ["", "0", "off", "default", "bogus"])
def test_quantize_tuning_falls_back_to_rocblas_for_unknown_values(monkeypatch, value):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setenv(rocm.QUANT_BLAS_BACKEND_ENV, value)

    assert rocm.quantize_tuning("gfx1201")["blas_backend"] == "rocblas"
    # "Keep torch's default backend" requests make the switch a no-op.
    if value in ("", "0", "off", "default"):
        assert value in rocm.QUANT_BLAS_BACKEND_DISABLED

def test_quantize_tuning_on_non_rocm_hosts(monkeypatch):
    monkeypatch.setattr(rocm, "IS_ROCM", False)
    monkeypatch.setenv(rocm.QUANT_BLAS_BACKEND_ENV, "hipblaslt")

    assert rocm.quantize_tuning("gfx1201") == {
        "blas_backend": "default",
        "amp": True,
        "prewarm_ops": None,
    }
    assert rocm.rocm_amp_disabled("gfx1201", is_rocm=False) is False

def test_quantize_tuning_default_device_uses_arch_lookup(monkeypatch):
    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setattr(rocm, "arch_class", lambda device=None: "rdna4")

    tuning = rocm.quantize_tuning(torch.device("cuda:0"))

    assert tuning["blas_backend"] == "rocblas"
    assert tuning["amp"] is False
    assert tuning["prewarm_ops"] == rocm._RDNA4_PREWARM_GROUPS

def test_quant_blas_backend_uses_quantize_tuning(monkeypatch):
    class _FakeBackend:
        def __init__(self, name):
            self.name = name

    calls = []
    state = {"current": _FakeBackend("Cublaslt")}

    def getter():
        def preferred(value=None):
            if value is None:
                return state["current"]
            calls.append(value)
            state["current"] = _FakeBackend("Cublas")

        return preferred

    monkeypatch.setattr(rocm, "IS_ROCM", True)
    monkeypatch.setattr(rocm, "_preferred_blas_library_getter", getter)
    monkeypatch.setattr(rocm, "quantize_tuning", lambda device=None: {
        "blas_backend": "rocblas",
        "amp": False,
        "prewarm_ops": None,
    })

    with rocm.quant_blas_backend("gfx1201"):
        pass

    assert calls[0] == "hipblas"

def test_quantize_prewarm_groups_select_a_subset(monkeypatch):
    dispatched = []

    class _Recorder(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            dispatched.append(str(func))
            return func(*args, **(kwargs or {}))

    with _Recorder():
        linalg_warmup._run_quantize_prewarm(
            torch.device("cpu"), torch.float16, ("elementwise", "softmax")
        )
    subset = list(dispatched)

    dispatched.clear()
    monkeypatch.setattr(
        linalg_warmup,
        "quantize_tuning",
        lambda device=None: {"blas_backend": "rocblas", "amp": False, "prewarm_ops": ("softmax",)},
    )
    with _Recorder():
        linalg_warmup._run_quantize_prewarm(torch.device("cpu"), torch.float16)

    # The RDNA4 subset must skip the "shape" copy kernels.
    assert not any(name.startswith("aten.cat.") for name in dispatched)
    assert any(name.startswith("aten._softmax") for name in dispatched)
    assert len(dispatched) < len(subset)

def test_warp_multiple_block_keeps_cdna_and_rdna_blocks_aligned():
    assert gsq_cuda_relaxation._warp_multiple_block(128, 64) == 128
    assert gsq_cuda_relaxation._warp_multiple_block(128, 32) == 128
    # A hypothetical non-divisor wavefront is rounded up, never truncated.
    assert gsq_cuda_relaxation._warp_multiple_block(128, 48) == 144
    assert gsq_cuda_relaxation._warp_multiple_block(128, 0) == 128

