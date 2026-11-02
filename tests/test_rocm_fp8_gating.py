# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU coverage for the ROCm FP8 Torch kernel's dispatch and fallbacks."""

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import fp8 as fp8_module


def test_fp8_declares_rocm_support():
    assert DEVICE.ROCM in fp8_module.TorchFP8Linear.SUPPORTS_DEVICES

def test_fp8_scaled_mm_failure_is_logged_once_and_falls_back(monkeypatch):
    """On ROCm ``_scaled_mm`` may reject an encoding; the guarded call logs once."""

    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    monkeypatch.setattr(fp8_module, "IS_ROCM", True)

    def _broken_scaled_mm(_x_flat):
        # Measured on gfx1201: ``_scaled_mm`` rejects the AMD-native fnuz
        # encodings with HIPBLAS_STATUS_NOT_SUPPORTED while accepting e4m3fn.
        raise RuntimeError("float8_e4m3fnuz is not supported by _scaled_mm on this HIP build")

    fallbacks = []

    def _fallback(_x_flat):
        fallbacks.append(True)
        return torch.zeros((2, module.out_features), dtype=torch.float16)

    monkeypatch.setattr(module, "_can_use_scaled_mm", lambda _x: True)
    monkeypatch.setattr(module, "_forward_scaled_mm", _broken_scaled_mm)
    monkeypatch.setattr(module, "_forward_dequant_matmul", _fallback)

    x = torch.zeros((2, module.in_features), dtype=torch.float16)

    module.forward(x)
    module.forward(x)

    assert fallbacks == [True, True]
    assert module._scaled_mm_hard_disabled is True
    # After the first failure the shape gate refuses ``_scaled_mm`` outright, so
    # the one-time fallback notice cannot fire again on subsequent forwards.
    x_flat = torch.zeros((2, module.in_features), dtype=torch.float16)
    assert fp8_module.TorchFP8Linear._can_use_scaled_mm(module, x_flat) is False

def test_fp8_native_dequant_failure_falls_back_to_cpu(monkeypatch):
    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    calls = []

    def _broken_native(*, target_device, target_dtype):
        calls.append("native")
        raise RuntimeError("cast not implemented for float8_e4m3fn on this HIP build")

    def _cpu_fallback(*, target_device, target_dtype):
        calls.append("cpu")
        return torch.zeros((16, 16), dtype=target_dtype)

    monkeypatch.setattr(module, "_dequantize_native", _broken_native)
    monkeypatch.setattr(module, "_dequantize_cpu", _cpu_fallback)
    # This asserts the HIP "attempt native, then fall back" semantics.  On a
    # real CUDA host the native-capability probe short-circuits to the CPU path
    # before the attempt, so pin the ROCm flag to exercise the same branch on
    # both environments.
    monkeypatch.setattr(fp8_module, "IS_ROCM", True)

    out = module.dequantize_weight(device=torch.device("cuda:0"), dtype=torch.float16)

    assert calls == ["native", "cpu"]
    assert out.shape == (16, 16)


def test_fp8_native_dequant_failure_is_sticky(monkeypatch):
    """A failed HIP native cast must not be retried (and re-copied) every forward."""

    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    calls = []

    def _broken_native(*, target_device, target_dtype):
        calls.append("native")
        raise RuntimeError("cast not implemented for float8_e4m3fn on this HIP build")

    def _cpu_fallback(*, target_device, target_dtype):
        calls.append("cpu")
        return torch.zeros((16, 16), dtype=target_dtype)

    monkeypatch.setattr(module, "_dequantize_native", _broken_native)
    monkeypatch.setattr(module, "_dequantize_cpu", _cpu_fallback)
    monkeypatch.setattr(fp8_module, "IS_ROCM", True)

    for _ in range(3):
        module.dequantize_weight(device=torch.device("cuda:0"), dtype=torch.float16)

    assert module._native_dequant_disabled is True
    assert calls == ["native", "cpu", "cpu", "cpu"]


def test_fp8_native_dequant_failure_raises_on_nvidia(monkeypatch):
    """The CPU fallback is a ROCm probe; on NVIDIA a native failure must propagate."""

    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )

    def _broken_native(*, target_device, target_dtype):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(module, "_dequantize_native", _broken_native)
    monkeypatch.setattr(module, "_dequantize_cpu", lambda **_: pytest.fail("must not fall back"))
    monkeypatch.setattr(fp8_module, "IS_ROCM", False)
    monkeypatch.setattr(fp8_module, "device_supports_native_fp8", lambda *_a, **_k: True)

    with pytest.raises(RuntimeError, match="out of memory"):
        module.dequantize_weight(device=torch.device("cuda:0"), dtype=torch.float16)
    assert module._native_dequant_disabled is False


@pytest.mark.parametrize("scale_inv", [0.5, 1.0, 2.0, 0.25])
def test_fp8_cpu_fallback_preserves_inverse_scale_semantics(scale_inv):
    """The CPU fallback must divide by an inverse scale, exactly like the native path."""

    torch.manual_seed(0)
    in_features, out_features = 32, 16
    module = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        format="float8_e4m3fn",
        weight_scale_method="row",
    )
    codes = (torch.randn(out_features, in_features) * 3).to(torch.float8_e4m3fn)
    module.weight = codes
    module.weight_scale_inv = torch.full((out_features, 1), scale_inv, dtype=torch.float32)

    expected = (codes.to(torch.float32) / scale_inv).transpose(0, 1).contiguous()
    got = module._dequantize_cpu(target_device=torch.device("cpu"), target_dtype=torch.float32)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
