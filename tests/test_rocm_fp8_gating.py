# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""CPU coverage for the ROCm FP8 Torch kernel's dispatch and fallbacks."""

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

