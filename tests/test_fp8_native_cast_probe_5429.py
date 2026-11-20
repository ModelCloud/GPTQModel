# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import torch

from gptqmodel.nn_modules.qlinear import fp8
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)
OUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
CPU = torch.device("cpu")


def test_probe_accepts_every_fp8_code_on_the_reference_device():
    for fp8_dtype in FP8_DTYPES:
        for out_dtype in OUT_DTYPES:
            verdict = fp8._probe_native_fp8_cast(CPU, fp8_dtype, out_dtype)
            assert type(verdict) is torch.Tensor
            assert verdict.dtype is torch.bool
            assert verdict.numel() == 1
            assert bool(verdict)


def test_probe_returns_a_device_side_verdict():
    verdict = fp8._probe_native_fp8_cast(CPU, torch.float8_e4m3fn, torch.float16)
    assert verdict.dim() == 0


def test_cuda_skips_the_measurement(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", False)

    def _boom(*args, **kwargs):
        raise AssertionError("the CUDA cast must not be probed")

    monkeypatch.setattr(fp8, "_probe_native_fp8_cast", _boom)
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e4m3fn, torch.float16) is True
    assert fp8._native_fp8_cast_supported(CPU, torch.float8_e4m3fn, torch.float16) is True


def test_rocm_measures_once_and_caches(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", True)
    monkeypatch.setattr(fp8, "_NATIVE_FP8_CAST_SUPPORT", {})
    calls = []

    def _probe(device, fp8_dtype, out_dtype):
        calls.append((device, fp8_dtype, out_dtype))
        return torch.tensor(True)

    monkeypatch.setattr(fp8, "_probe_native_fp8_cast", _probe)
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e4m3fn, torch.float16) is True
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e4m3fn, torch.float16) is True
    assert len(calls) == 1


def test_rocm_caches_a_failed_cast(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", True)
    monkeypatch.setattr(fp8, "_NATIVE_FP8_CAST_SUPPORT", {})
    monkeypatch.setattr(fp8, "_probe_native_fp8_cast", lambda *a: torch.tensor(False))
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e5m2, torch.bfloat16) is False
    assert list(fp8._NATIVE_FP8_CAST_SUPPORT.values()) == [False]


def test_rocm_does_not_cache_while_capturing(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", True)
    monkeypatch.setattr(fp8, "_NATIVE_FP8_CAST_SUPPORT", {})
    monkeypatch.setattr(fp8, "_probe_native_fp8_cast", lambda *a: torch.tensor(True))
    monkeypatch.setattr(fp8, "_probe_must_wait", lambda: True)
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e4m3fn, torch.float16) is None
    assert fp8._NATIVE_FP8_CAST_SUPPORT == {}
    monkeypatch.setattr(fp8, "_probe_must_wait", lambda: False)
    assert fp8._native_fp8_cast_verdict(CPU, torch.float8_e4m3fn, torch.float16) is True
    assert list(fp8._NATIVE_FP8_CAST_SUPPORT.values()) == [True]


class _Stub:
    _native_dequant_disabled = False
    fp8_format = "float8_e4m3fn"
    fp8_dtype = torch.float8_e4m3fn
    weight_scale_method = "row"
    weight_scale_semantics = "inverse"
    weight = torch.zeros(4, 8)
    weight_scale_inv = torch.ones(4, 1)

    def _native_gemm_eligible(self, x_flat):
        return True

    def _inverse_scales_are_nonzero(self):
        return True


def _gate(x_dtype=torch.float16):
    return TorchFP8Linear._lean_epilogue_eligible(_Stub(), torch.zeros(2, 8, dtype=x_dtype))


def test_lean_gate_follows_the_cast_verdict(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", True)
    monkeypatch.setattr(fp8, "_native_fp8_cast_supported", lambda *a: True)
    assert _gate() is True
    monkeypatch.setattr(fp8, "_native_fp8_cast_supported", lambda *a: False)
    assert _gate() is False


def test_lean_gate_skips_the_probe_on_cuda(monkeypatch):
    monkeypatch.setattr(fp8, "IS_ROCM", False)

    def _boom(*args, **kwargs):
        raise AssertionError("the CUDA gate must not probe the cast")

    monkeypatch.setattr(fp8, "_probe_native_fp8_cast", _boom)
    assert _gate() is True
