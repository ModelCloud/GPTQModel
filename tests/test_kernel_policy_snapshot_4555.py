# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4555: forward dispatch reads the module's snapshotted kernel policy.

A quantized linear owns ``self.kernel`` (and, for Paro, the resolved
``self.fp32_accum``). Later ``configure_kernel`` calls must not change the
path or the accumulation mode of a module that was already built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear import bitsandbytes as bnb_module
from gptqmodel.nn_modules.qlinear import bitsandbytes_triton as bt
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.paro_triton import ParoTritonLinear
from gptqmodel.quantization.paro.modules.triton import gemm as paro_gemm
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, reset_kernel_config


@pytest.fixture(autouse=True)
def _reset_policy():
    reset_kernel_config()
    yield
    reset_kernel_config()


def test_bitsandbytes_triton_gate_uses_explicit_module_kernel(monkeypatch):
    # Simulate an NVIDIA host, where the Triton path is opt-in.
    monkeypatch.setattr(bt, "IS_ROCM", False)

    forced = KernelConfig(bitsandbytes_triton=True)
    configure_kernel(KernelConfig(bitsandbytes_triton=False))  # later load
    assert bt.available_for("sm_89", torch.float16, kernel=forced) is True
    assert bt.should_use("sm_89", torch.float16, 64, kernel=forced) is True

    # A module built after the policy change keeps the new policy.
    not_forced = KernelConfig(bitsandbytes_triton=False)
    assert bt.available_for("sm_89", torch.float16, kernel=not_forced) is False
    assert bt.should_use("sm_89", torch.float16, 64, kernel=not_forced) is False

    # The reverse direction: process policy forced, module snapshot not.
    configure_kernel(KernelConfig(bitsandbytes_triton=True))
    assert bt.should_use("sm_89", torch.float16, 64, kernel=not_forced) is False


class _StopForward(Exception):
    pass


class _TritonRecorder:
    def __init__(self):
        self.kernels = []

    def should_use(self, device, dtype, rows, *, kernel=None):
        self.kernels.append(kernel)
        raise _StopForward


def _build_bitsandbytes_linear(kernel: KernelConfig) -> BitsAndBytesLinear:
    module = BitsAndBytesLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=256,
        bias=False,
        register_buffers=False,
        dtype=torch.float16,
        kernel=kernel,
    )
    # The dispatch gate only needs the weight device; the recorder stops the
    # forward before any packed-weight access.
    module.weight = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    module.bias = None
    return module


def test_bitsandbytes_forward_passes_the_module_kernel(monkeypatch):
    monkeypatch.setattr(
        BitsAndBytesLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    monkeypatch.setattr(bnb_module, "import_bitsandbytes", lambda: SimpleNamespace())
    recorder = _TritonRecorder()
    monkeypatch.setattr(bnb_module, "_triton_backend", lambda: recorder)

    first = _build_bitsandbytes_linear(KernelConfig(bitsandbytes_triton=True))
    configure_kernel(KernelConfig(bitsandbytes_triton=False))  # later load

    with pytest.raises(_StopForward):
        with torch.inference_mode():
            first(torch.randn(64, 256, dtype=torch.float16))

    assert recorder.kernels[-1] is first.kernel
    assert first.kernel.bitsandbytes_triton is True

    second = _build_bitsandbytes_linear(KernelConfig(bitsandbytes_triton=False))
    with pytest.raises(_StopForward):
        with torch.inference_mode():
            second(torch.randn(64, 256, dtype=torch.float16))

    assert recorder.kernels[-1] is second.kernel
    assert second.kernel.bitsandbytes_triton is False


def _build_paro(kernel: KernelConfig | None) -> ParoTritonLinear:
    return ParoTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=256,
        bias=False,
        register_buffers=True,
        kernel=kernel,
    )


def test_paro_triton_forward_uses_the_module_fp32_accum(monkeypatch):
    monkeypatch.setattr(
        ParoTritonLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    seen = []

    def fake_gemm(*args, **kwargs):
        seen.append(kwargs.get("fp32_accum"))
        return torch.zeros(1, dtype=torch.float16)

    monkeypatch.setattr(paro_gemm, "_paro_gemm_triton", fake_gemm)

    configure_kernel(KernelConfig(fp32_accum=False))
    first = _build_paro(KernelConfig(fp32_accum=False))
    assert first.kernel.fp32_accum is False
    assert first.fp32_accum is False

    configure_kernel(KernelConfig(fp32_accum=True))  # later load
    x = torch.randn(64, 256, dtype=torch.float16)
    first._forward_triton_prefill(x)
    first._forward_triton_decode(x[:1])
    assert seen == [False, False]

    second = _build_paro(None)  # snapshots the current process policy
    assert second.kernel.fp32_accum is True
    assert second.fp32_accum is True
    seen.clear()
    second._forward_triton_prefill(x)
    second._forward_triton_decode(x[:1])
    assert seen == [True, True]


def test_paro_gemm_helpers_forward_explicit_fp32_accum(monkeypatch):
    seen = []

    def fake_gemm(*args, **kwargs):
        seen.append(kwargs.get("fp32_accum"))
        return torch.zeros(1, dtype=torch.float16)

    monkeypatch.setattr(paro_gemm, "_paro_gemm_triton", fake_gemm)
    configure_kernel(KernelConfig(fp32_accum=True))

    x = torch.randn(2, 256, dtype=torch.float16)
    qweight = torch.zeros(256, 64, dtype=torch.int32)
    scales = torch.zeros(256, 2, dtype=torch.float16)
    qzeros = torch.zeros(256, 2, dtype=torch.int32)

    paro_gemm.paro_gemm_triton_decode(
        x, qweight, scales, qzeros, fp32_accum=False
    )
    paro_gemm.paro_gemm_triton_prefill(
        x, qweight, scales, qzeros, fp32_accum=False
    )

    assert seen == [False, False]
