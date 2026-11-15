# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Audit the entire kernel registry's ROCm family contract without GPU execution."""

import sys
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.models._const as model_const
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear
from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.ttnn_blockfloat import TTNNBlockFloat8Linear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer, macaq_triton
from gptqmodel.utils.backend import BACKEND


# Explicit audit categories: adding a kernel requires reviewing its implementation,
# not inferring ROCm support from a name containing "Torch", "CUDA", or "Triton".
ROCM_KERNELS = set("""
    AllSparkLinear AllSparkAwqLinear AwqGEMMTritonLinear AwqGEMVTritonLinear
    GGUFTritonKernel TorchFP8Linear BitsAndBytesLinear
    TorchLinear TorchQuantEmbeddings AwqTorchLinear QQQTorchLinear
    GGUFTorchLinear ParoLinear ParoQuantTritonLinear VPTQLinear NF12Linear
    MacaqLinear MacaqTritonLinear
    TTBlockfloatLinear
""".split())
CUDA_ONLY_KERNELS = set("""
    TritonV2Linear W4AFP8Linear W4ANVFP4Linear ExllamaV2Linear AwqExllamaV2Linear
    MarlinLinear AwqMarlinLinear MacheteLinear AwqMacheteLinear
    SwordfishLinear AwqSwordfishLinear BitblasLinear AWQBitBlasKernel
    AwqGEMMLinear AwqGEMVLinear AwqGEMVFastLinear LLMAwqLinear QQQLinear
    VPTQCudaLinear GGUFCudaKernel
""".split())
OTHER_KERNELS = set("""
    TorchFusedLinear TorchFusedAwqLinear TorchAtenLinear TorchAtenAwqLinear
    TorchCpuAvx512Linear TorchCpuAvx512AwqLinear TorchInt8Linear TorchInt8AwqLinear
    GGUFCppKernel MlxQuantLinear GPTQMarlinMlxQuantLinear GPTQBitBLASMlxQuantLinear
    AwqMlxQuantLinear AWQMarlinMlxQuantLinear AWQBitBLASMlxQuantLinear
    AwqGemvMlxQuantLinear AwqGemvFastMlxQuantLinear LLMAwqMlxQuantLinear
    ParoMlxQuantLinear QQQMlxQuantLinear GGUFMlxQuantLinear FP8MlxQuantLinear
    BitsAndBytesMlxQuantLinear
    XLALinear
    TTNNLinear TTNNBlockFloat8Linear
""".split())
# Kernels registered for a non-Linux platform only. An MPS/Darwin kernel can
# never execute on a ROCm host, so it is classified by its declared platform
# instead of by device family. `_platform_restricted_kernels()` derives the
# bucket from `SUPPORTS_PLATFORM`, so the next MPS-only kernel is covered
# without editing this inventory; the explicit entries below are the ones
# reviewed for #4130 and are asserted to really be platform-restricted.
PLATFORM_RESTRICTED_KERNELS = set("""
    MacaqMlxQuantLinear NF12MlxQuantLinear
""".split())
KERNELS = sorted(
    [*importer.iter_quant_linear_kernels(), TorchQuantEmbeddings, TTNNBlockFloat8Linear],
    key=lambda c: c.__name__,
)


def _platform_restricted_kernels() -> set[str]:
    """Registered kernels that declare no Linux platform.

    These cannot run on the Linux ROCm hosts this audit targets. The set is
    derived from the shipped declaration rather than a hand-written list so a
    newly registered MPS-only kernel is classified automatically.
    """

    return {
        cls.__name__
        for cls in KERNELS
        if model_const.PLATFORM.LINUX not in cls.SUPPORTS_PLATFORM
        and model_const.PLATFORM.ALL not in cls.SUPPORTS_PLATFORM
    }


@pytest.fixture
def hip_runtime(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.16")
    monkeypatch.setattr(model_const, "IS_ROCM", True)
    monkeypatch.setattr(importer, "IS_ROCM", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(gcnArchName="gfx1201"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: pytest.fail("NVIDIA capability probe on HIP"))
    for cls in KERNELS:
        module = sys.modules[cls.__module__]
        if hasattr(module, "IS_ROCM"):
            monkeypatch.setattr(module, "IS_ROCM", True)
    monkeypatch.setattr(macaq_triton, "IS_ROCM", True)
    macaq_triton._rdna4_macaq_supported.cache_clear()
    yield
    macaq_triton._rdna4_macaq_supported.cache_clear()


def test_macaq_triton_rejects_mixed_rocm_architectures(hip_runtime, monkeypatch):
    def properties(target=None):
        index = target if isinstance(target, int) else (
            0 if target is None else torch.device(target).index
        )
        return SimpleNamespace(gcnArchName="gfx942" if index == 1 else "gfx1201")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    macaq_triton._rdna4_macaq_supported.cache_clear()
    MacaqTritonLinear.validate_device(torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="gfx1201"):
        MacaqTritonLinear.validate_device(torch.device("cuda:1"))
    with pytest.raises(NotImplementedError, match="gfx1201"):
        MacaqTritonLinear.validate_device(model_const.DEVICE.ROCM)


def test_audit_covers_every_registered_kernel():
    platform_restricted = _platform_restricted_kernels()
    # The explicitly reviewed entries must really be platform-restricted; a
    # kernel listed here without a non-Linux platform is a stale classification.
    assert PLATFORM_RESTRICTED_KERNELS <= platform_restricted
    assert {c.__name__ for c in KERNELS} == (
        ROCM_KERNELS | CUDA_ONLY_KERNELS | OTHER_KERNELS | platform_restricted
    )


def test_platform_restricted_kernels_declare_reachable_families():
    # A kernel that excludes Linux cannot advertise ROCm, and CUDA does not
    # exist on Darwin. Without these checks the derived bucket would let an
    # MPS-only kernel claim an accelerator family it can never reach and still
    # pass the audit.
    platform_restricted = _platform_restricted_kernels()
    assert platform_restricted
    for cls in KERNELS:
        if cls.__name__ not in platform_restricted:
            continue
        assert model_const.PLATFORM.LINUX not in cls.SUPPORTS_PLATFORM, cls.__name__
        assert model_const.DEVICE.ROCM not in cls.SUPPORTS_DEVICES, cls.__name__
        if model_const.PLATFORM.DARWIN in cls.SUPPORTS_PLATFORM:
            assert model_const.DEVICE.CUDA not in cls.SUPPORTS_DEVICES, cls.__name__


@pytest.mark.parametrize("cls,devices", [
    (ExllamaV3Linear, [model_const.DEVICE.CUDA]),
    (ExllamaV3TorchLinear, [model_const.DEVICE.ALL]),
    (TorchFP4Linear, [model_const.DEVICE.CPU, model_const.DEVICE.CUDA]),
])
def test_specialized_loader_kernel_declarations(cls, devices):
    # These nn.Module implementations use dedicated loader paths rather than
    # BaseQuantLinear's registry; their device contracts must be explicit too.
    assert cls.SUPPORTS_DEVICES == devices


@pytest.mark.parametrize("is_rocm", [False, True])
def test_explicit_rocm_family_never_probes_nvidia_capability(monkeypatch, is_rocm):
    monkeypatch.setattr(importer, "IS_ROCM", is_rocm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: pytest.fail("AMD family has no NVIDIA SM"))
    assert importer._device_capability(model_const.DEVICE.ROCM) is None


@pytest.mark.parametrize("cls", KERNELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("device", [model_const.DEVICE.ROCM, model_const.DEVICE.CUDA, "rocm", "cuda", "cuda:1", torch.device("cuda:1")])
def test_rocm_device_contract(cls, device, hip_runtime):
    expected = cls.__name__ in ROCM_KERNELS
    advertised = model_const.DEVICE.ROCM in cls.SUPPORTS_DEVICES or model_const.DEVICE.ALL in cls.SUPPORTS_DEVICES
    assert advertised is expected
    # Exercise shared validation, including class overrides, while avoiding
    # optional extension imports and unrelated bit/group/shape contracts.
    ok, error = cls._validate_shared(device=device, pack_dtype=cls.SUPPORTS_PACK_DTYPES[0])
    assert ok is expected, (cls.__name__, device, error)
    if not expected:
        assert isinstance(error, NotImplementedError)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("backend", [BACKEND.AUTO, BACKEND.BITSANDBYTES])
@pytest.mark.parametrize("device", [model_const.DEVICE.ROCM, model_const.DEVICE.CUDA, "cuda:1", torch.device("cuda:1")])
def test_bitsandbytes_rocm_selection(bits, backend, device, hip_runtime, monkeypatch):
    monkeypatch.setattr(BitsAndBytesLinear, "cached_validate_once", classmethod(lambda cls: (True, None)))
    importer.clear_validation_cache()
    try:
        assert importer.select_quant_linear(
            bits=bits, group_size=-1, desc_act=False, sym=True,
            backend=backend, format=FORMAT.BITSANDBYTES, quant_method=METHOD.BITSANDBYTES,
            device=device, pack_dtype=torch.int32,
        ) is BitsAndBytesLinear
    finally:
        importer.clear_validation_cache()


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
@pytest.mark.parametrize("device", ["cuda:1", torch.device("cuda:1"), model_const.DEVICE.ROCM, model_const.DEVICE.CUDA])
def test_allspark_rejects_non_rdna4_target(cls, device, hip_runtime, monkeypatch):
    def properties(target=None):
        index = 0 if target is None else torch.device(target).index
        return SimpleNamespace(gcnArchName="gfx942" if index == 1 else "gfx1201")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    # The current device is supported. A different target, or a family spanning
    # the unsupported target, must still be rejected before repacking/launch.
    assert cls.validate_once() == (True, None)
    cls.validate_device(torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="RDNA4"):
        cls.validate_device(device)


@pytest.mark.parametrize("cls", [AllSparkLinear, AllSparkAwqLinear])
def test_allspark_accepts_rdna4_target_when_current_device_is_unsupported(cls, hip_runtime, monkeypatch):
    def properties(target=None):
        index = 0 if target is None else torch.device(target).index
        return SimpleNamespace(gcnArchName="gfx1201" if index == 1 else "gfx942")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(cls, "cached_validate_once", classmethod(lambda kernel: kernel.validate_once()))
    ok, error = cls.validate(bits=8, group_size=-1, sym=True, desc_act=False,
                             pack_dtype=torch.int32, device=torch.device("cuda:1"))
    assert ok, error
    with pytest.raises(NotImplementedError, match="RDNA4"):
        cls.validate_device(torch.device("cuda:0"))
