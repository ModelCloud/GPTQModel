# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""ROCm gating for the AWQ GEMV / GEMV_FAST / LLM_AWQ kernels.

These tests are pure CPU: the HIP runtime is simulated with a monkeypatched
``IS_ROCM`` flag, so they assert the dispatch decision rather than claiming a
kernel ran on ROCm.  The native classes are only readable by the
``gptqmodel_ext/awq`` CUDA JIT extension, which hipcc cannot build (the sources
include ``cuda_fp16.h`` and CUDA-only inline asm, and the hipified wave-safe warp
reduction rejects a 32-bit mask on wave32 RDNA4).  ``FORMAT.GEMV`` has a
portable Triton reader; ``FORMAT.GEMV_FAST``/``FORMAT.LLM_AWQ`` have portable
Torch readers (#5109).
"""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemv_awq import (
    AWQ_GEMV_ROCM_UNSUPPORTED,
    AwqGEMVLinear,
    AwqGEMVTorchLinear,
)
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import (
    AwqGEMVFastLinear,
    AwqGEMVFastTorchLinear,
    LLMAwqLinear,
    LLMAwqTorchLinear,
)
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


def test_loader_rejects_native_awq_gemv_gemv_on_rocm(monkeypatch):
    # A ROCm *load* of FORMAT.GEMV must fail with the actionable message at
    # load time, not with a bare "No valid quant linear" from AUTO selection.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    with pytest.raises(NotImplementedError) as excinfo:
        loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, FORMAT.GEMV)
    assert "AWQ_GEMM_TRITON" in str(excinfo.value)


def test_loader_allows_portable_gemv_fast_layouts_on_rocm(monkeypatch):
    # #5109: FORMAT.GEMV_FAST / FORMAT.LLM_AWQ have portable Torch readers that
    # declare ROCm, so an AUTO load must not be refused before selection.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    for fmt in (FORMAT.GEMV_FAST, FORMAT.LLM_AWQ):
        loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, fmt)


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


def test_loader_allows_gemv_cpu_placement_in_a_rocm_process(monkeypatch):
    # #5219: the gate used to key on ``IS_ROCM`` alone, so a HIP torch wheel
    # refused ``device="cpu"`` even though AwqGEMVTorchLinear serves exactly
    # that layout. The refusal must follow the target family, not the wheel.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.backend import BACKEND

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    for backend in (BACKEND.AUTO, BACKEND.AWQ_TORCH):
        loader_module._reject_native_awq_gemv_on_rocm(
            METHOD.AWQ, FORMAT.GEMV, backend, DEVICE.CPU
        )
    # An explicit accelerator family still gets the actionable refusal.
    for backend in (BACKEND.AUTO, BACKEND.AWQ_TORCH):
        with pytest.raises(NotImplementedError) as excinfo:
            loader_module._reject_native_awq_gemv_on_rocm(
                METHOD.AWQ, FORMAT.GEMV, backend, DEVICE.ROCM
            )
        assert "AWQ_GEMV_TRITON" in str(excinfo.value)


def test_loader_gemv_cpu_placement_keeps_refusal_when_target_is_unresolved(monkeypatch):
    # No placement argument means the caller has not resolved a target family;
    # keep the conservative ROCm refusal rather than guessing.
    import gptqmodel.models.loader as loader_module
    from gptqmodel.quantization import METHOD

    monkeypatch.setattr(loader_module, "IS_ROCM", True)

    with pytest.raises(NotImplementedError):
        loader_module._reject_native_awq_gemv_on_rocm(METHOD.AWQ, FORMAT.GEMV)


@pytest.mark.parametrize("backend_name", ["AUTO", "AWQ_TORCH"])
def test_awq_gemv_cpu_placement_selects_the_portable_torch_reader_on_rocm(
    monkeypatch, backend_name
):
    # The load now falls through to selection, which must land on the portable
    # CPU reader for the GEMV layout in a HIP process (#5219).
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)

    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=getattr(BACKEND, backend_name),
        format=FORMAT.GEMV,
        quant_method=METHOD.AWQ,
        pack=True,
        pack_dtype=torch.int32,
    )
    assert selected is AwqGEMVTorchLinear


def test_triton_gemv_declares_rocm_and_native_stays_cuda_only():
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear

    assert DEVICE.ROCM in AwqGEMVTritonLinear.SUPPORTS_DEVICES
    # Priority 0 is the documented "supported but never AUTO-selected" marker.
    assert AwqGEMVTritonLinear.SUPPORTS_FORMATS[FORMAT.GEMV] == 0
    for cls, _format in NATIVE_AWQ_GEMV_KERNELS:
        assert DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__


def test_awq_gemv_auto_backend_resolves_triton_on_rocm():
    # The quantize path used to hardcode BACKEND.AWQ_GEMV, a CUDA-only class,
    # so AUTO aborted before calibration on ROCm.
    from gptqmodel.utils.backend import BACKEND, awq_gemv_pack_backend

    assert awq_gemv_pack_backend(is_rocm=True) is BACKEND.AWQ_GEMV_TRITON
    assert awq_gemv_pack_backend(is_rocm=False) is BACKEND.AWQ_GEMV


def test_awq_gemv_fast_auto_backend_resolves_portable_torch_on_rocm():
    # #5109: GEMV_FAST/LLM_AWQ have no Triton kernel, so ROCm must pack with the
    # portable Torch reader, not the CUDA-only native class or a backend that
    # does not declare these formats.
    from gptqmodel.utils.backend import BACKEND, awq_gemv_fast_pack_backend

    for device_type in ("cuda", "rocm"):
        assert (
            awq_gemv_fast_pack_backend(device_type=device_type, is_rocm=True)
            is BACKEND.AWQ_TORCH
        )
    assert awq_gemv_fast_pack_backend(device_type="cpu", is_rocm=False) is BACKEND.AWQ_TORCH
    assert awq_gemv_fast_pack_backend(device_type="cuda", is_rocm=False) is BACKEND.AWQ_GEMV_FAST


@pytest.mark.parametrize(
    "fmt,torch_cls",
    [
        pytest.param(FORMAT.GEMV_FAST, AwqGEMVFastTorchLinear, id="gemv_fast"),
        pytest.param(FORMAT.LLM_AWQ, LLMAwqTorchLinear, id="llm_awq"),
    ],
)
def test_awq_gemv_fast_selects_the_portable_torch_reader_on_rocm(fmt, torch_cls):
    # AUTO and the resolved pack backend must both land on the portable reader.
    # This used to raise "No valid quant linear" (AUTO) or the CPU-only device
    # rejection (AWQ_TORCH) before any calibration ran.
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.backend import BACKEND, awq_gemv_fast_pack_backend
    from gptqmodel.utils.importer import select_quant_linear

    for backend in (BACKEND.AUTO, awq_gemv_fast_pack_backend(device_type="cuda", is_rocm=True)):
        selected = select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device=DEVICE.ROCM,
            backend=backend,
            format=fmt,
            quant_method=METHOD.AWQ,
            pack=True,
            pack_dtype=torch.int16,
        )
        assert selected is torch_cls, f"{fmt} via {backend}"


def test_awq_gemv_fast_portable_readers_declare_rocm_and_cpu_only_native_ones_do_not():
    for cls in (AwqGEMVFastTorchLinear, LLMAwqTorchLinear):
        assert DEVICE.ROCM in cls.SUPPORTS_DEVICES, cls.__name__
        assert DEVICE.CPU in cls.SUPPORTS_DEVICES, cls.__name__
        # Explicit-only: AUTO order on CUDA must not change.
        assert all(priority == 0 for priority in cls.SUPPORTS_FORMATS.values()), cls.__name__
    for cls, _fmt in NATIVE_AWQ_GEMV_KERNELS:
        assert DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__


def test_awq_gemv_auto_quantize_selects_triton_on_rocm(monkeypatch):
    # Quantize resolves AUTO through awq_gemv_pack_backend() and then validates
    # that backend. On ROCm the resolved kernel must be the portable Triton
    # GEMV; the native AwqGEMVLinear is CUDA-only.
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.backend import awq_gemv_pack_backend
    from gptqmodel.utils.importer import select_quant_linear

    monkeypatch.setattr("gptqmodel.models._const.IS_ROCM", True)
    backend = awq_gemv_pack_backend(is_rocm=True)
    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.ROCM,
        backend=backend,
        format=FORMAT.GEMV,
        quant_method=METHOD.AWQ,
        pack=True,
        pack_dtype=torch.int32,
    )
    assert selected is AwqGEMVTritonLinear


def test_torch_fused_rocm_rejection_names_its_real_families(monkeypatch):
    # TorchFusedLinear/TorchFusedAwqLinear are CPU/XPU kernels. Reporting them
    # as "CUDA-only" sent ROCm users looking for a CUDA build instead of the
    # actual family mismatch.
    from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
    from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear

    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear._is_rocm_target", lambda device: True
    )
    for cls in (TorchFusedLinear, TorchFusedAwqLinear):
        ok, error = cls.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=128,
            out_features=128,
            pack_dtype=torch.int32,
            device=DEVICE.ROCM,
        )
        assert ok is False, cls.__name__
        message = str(error)
        assert "CUDA-only" not in message, cls.__name__
        assert "cpu" in message and "xpu" in message, cls.__name__
