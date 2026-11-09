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
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND


# Explicit audit categories: adding a kernel requires reviewing its implementation,
# not inferring ROCm support from a name containing "Torch", "CUDA", or "Triton".
ROCM_KERNELS = set("""
    AllSparkLinear AllSparkAwqLinear AwqGEMMTritonLinear AwqGEMVTritonLinear
    ParoQuantTritonLinear GGUFTritonKernel TorchFP8Linear BitsAndBytesLinear
    TorchLinear TorchQuantEmbeddings AwqTorchLinear QQQTorchLinear
    GGUFTorchLinear ParoLinear VPTQLinear NF12ADLinear
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
""".split())
KERNELS = sorted([*importer.iter_quant_linear_kernels(), TorchQuantEmbeddings], key=lambda c: c.__name__)


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


def test_audit_covers_every_registered_kernel():
    assert {c.__name__ for c in KERNELS} == ROCM_KERNELS | CUDA_ONLY_KERNELS | OTHER_KERNELS


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
