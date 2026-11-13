# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""ROCm gating for the native-only AWQ GEMV / GEMV_FAST / LLM_AWQ kernels.

These tests are pure CPU: the HIP runtime is simulated with a monkeypatched
``IS_ROCM`` flag, so they assert the dispatch decision rather than claiming a
kernel ran on ROCm.  Those three layouts are only readable by the
``gptqmodel_ext/awq`` CUDA JIT extension, which hipcc cannot build (the sources
include ``cuda_fp16.h`` and CUDA-only inline asm, and the hipified wave-safe warp
reduction rejects a 32-bit mask on wave32 RDNA4).
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemv_awq import (
    AWQ_GEMV_ROCM_UNSUPPORTED,
    AwqGEMVLinear,
)
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.quantization import FORMAT


NATIVE_AWQ_GEMV_KERNELS = (
    (AwqGEMVLinear, FORMAT.GEMV),
    (AwqGEMVFastLinear, FORMAT.GEMV_FAST),
    (LLMAwqLinear, FORMAT.LLM_AWQ),
)


def test_native_awq_gemv_kernels_declare_cuda_only():
    # Advertising DEVICE.ROCM made AUTO select these kernels on gfx1201 and then
    # raise at forward time inside the failed JIT build.
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        assert cls.SUPPORTS_DEVICES == [DEVICE.CUDA], cls.__name__
        assert DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__


def _legal_awq_gemv_shape(cls, in_features=128, out_features=128):
    """Return an input/output width pair the native kernel can actually launch.

    ``AwqGEMVLinear``/``LLMAwqLinear`` require ``out_features`` divisible by 64
    and ``AwqGEMVFastLinear`` requires 128, so a shared case must satisfy every
    kernel's ``SUPPORTS_OUT_FEATURES_DIVISIBLE_BY``. Device-family validation is
    what these tests assert; an illegal shape would reject for the wrong reason.
    """
    out_align = 1
    for value in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY:
        out_align = out_align * value // math.gcd(out_align, value)
    in_align = 1
    for value in cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY:
        in_align = in_align * value // math.gcd(in_align, value)
    return (
        in_features + (-in_features % in_align),
        out_features + (-out_features % out_align),
    )


def test_native_awq_gemv_validate_rejects_rocm_placement():
    # The device family check is what the AUTO selector and the loader consult.
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        in_features, out_features = _legal_awq_gemv_shape(cls)
        ok, error = cls.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=cls.SUPPORTS_PACK_DTYPES[0],
            device=DEVICE.ROCM,
        )
        assert ok is False, cls.__name__
        assert isinstance(error, NotImplementedError), cls.__name__
        # The rejection must be the device family, not a shape guard.
        assert "out_features" not in str(error), cls.__name__


def test_native_awq_gemv_validate_still_accepts_cuda(monkeypatch):
    # Test the NVIDIA contract even when this suite runs on a HIP host.
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", False)
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        in_features, out_features = _legal_awq_gemv_shape(cls)
        ok, error = cls.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=cls.SUPPORTS_PACK_DTYPES[0],
            device=DEVICE.CUDA,
        )
        assert ok and error is None, cls.__name__


def test_native_awq_gemv_validate_rejects_unaligned_out_features(monkeypatch):
    # A legal device with an illegal width must still be refused, and the error
    # must name the shape guard so callers can tell it apart from the device
    # family rejection above. ``64`` is legal for the GEMV kernel whose tile is
    # 64, so pick the largest value below each kernel's own alignment.
    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", False)
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        in_features, _ = _legal_awq_gemv_shape(cls)
        alignment = math.lcm(*cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
        ok, error = cls.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=alignment - 1,
            pack_dtype=cls.SUPPORTS_PACK_DTYPES[0],
            device=DEVICE.CUDA,
        )
        assert ok is False, cls.__name__
        assert isinstance(error, NotImplementedError), cls.__name__
        assert "out_features" in str(error), cls.__name__


def test_native_awq_gemv_refusal_message_is_actionable():
    # The loader surfaces this text, so it must name a kernel that does work on
    # ROCm instead of leaving the user with "No valid quant linear".
    assert "AWQ_GEMM_TRITON" in AWQ_GEMV_ROCM_UNSUPPORTED
    assert "ROCm" in AWQ_GEMV_ROCM_UNSUPPORTED
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        assert torch.float16 in cls.SUPPORTS_DTYPES, cls.__name__


def test_loader_rejects_native_awq_gemv_layouts_on_rocm(monkeypatch):
    # A ROCm *load* of these formats must fail with the actionable message at
    # load time, not with a bare "No valid quant linear" from AUTO selection.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    for fmt in (FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ):
        with pytest.raises(NotImplementedError) as excinfo:
            loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, fmt)
        assert "AWQ_GEMM_TRITON" in str(excinfo.value), fmt


def test_loader_rocm_rejection_is_scoped_to_native_awq_gemv(monkeypatch):
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    # The AWQ GEMM layout and every non-AWQ method are unaffected.
    loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, FORMAT.GEMM)
    loader_module._reject_native_awq_gemv_on_rocm(METHOD.GPTQ, FORMAT.GPTQ)
    loader_module._reject_native_awq_gemv_on_rocm(METHOD.QQQ, FORMAT.QQQ)


def test_loader_cuda_host_keeps_native_awq_gemv(monkeypatch):
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", False)

    for fmt in (FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ):
        loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, fmt)


def test_loader_allows_explicit_triton_gemv_on_rocm(monkeypatch):
    # The portable Triton GEMV reads the same GEMV packing without the CUDA
    # extension, so an explicit selection of it must not be refused.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.utils.backend import BACKEND

    monkeypatch.setattr(loader_module, "IS_ROCM", True)
    loader_module._reject_native_awq_gemv_on_rocm(
        __import__("gptqmodel.quantization", fromlist=["METHOD"]).METHOD.AWQ,
        FORMAT.GEMV,
        BACKEND.AWQ_GEMV_TRITON,
    )
    # Any other backend, including AUTO (None), still hits the actionable error.
    import pytest as _pytest
    with _pytest.raises(NotImplementedError):
        loader_module._reject_native_awq_gemv_on_rocm(
            __import__("gptqmodel.quantization", fromlist=["METHOD"]).METHOD.AWQ,
            FORMAT.GEMV,
            None,
        )


def test_triton_gemv_declares_rocm_and_native_stays_cuda_only():
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear

    assert DEVICE.ROCM in AwqGEMVTritonLinear.SUPPORTS_DEVICES
    # Priority 0 is the documented "supported but never AUTO-selected" marker.
    assert AwqGEMVTritonLinear.SUPPORTS_FORMATS[FORMAT.GEMV] == 0
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        assert DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__
