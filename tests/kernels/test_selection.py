# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections import OrderedDict

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import AUTO_BACKEND_KERNEL_MAPPING, auto_select_device, select_quant_linear
from gptqmodel.utils.rocm import IS_ROCM
from gptqmodel.utils.torch import HAS_CUDA, HAS_MPS, HAS_XPU


def _iter_kernel_classes():
    seen = set()
    stack = list(BaseQuantLinear.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if "SUPPORTS_FORMATS" in cls.__dict__:
            yield cls


def _infer_quant_methods(cls):
    supported = getattr(cls, "SUPPORTS_METHODS", None)
    if supported is None:
        raise ValueError(f"{cls.__name__} is missing SUPPORTS_METHODS.")
    return [
        METHOD(method) if isinstance(method, METHOD) else METHOD(str(method).lower())
        for method in supported
    ]


def _pick_device(cls):
    devices = getattr(cls, "SUPPORTS_DEVICES", [])
    if DEVICE.ALL in devices:
        return DEVICE.CPU
    if DEVICE.CPU in devices:
        return DEVICE.CPU
    if DEVICE.CUDA in devices and HAS_CUDA:
        return DEVICE.CUDA
    if DEVICE.ROCM in devices and IS_ROCM:
        return DEVICE.ROCM
    if DEVICE.XPU in devices and HAS_XPU:
        return DEVICE.XPU
    if DEVICE.MPS in devices and HAS_MPS:
        return DEVICE.MPS
    return None


def _pick_group_size(cls):
    group_sizes = list(getattr(cls, "SUPPORTS_GROUP_SIZE", []))
    for candidate in group_sizes:
        if candidate != -1:
            return candidate
    return group_sizes[0] if group_sizes else -1


def _pick_desc_act(cls):
    values = list(getattr(cls, "SUPPORTS_DESC_ACT", []))
    return values[0] if values else False


def _pick_sym(cls):
    values = list(getattr(cls, "SUPPORTS_SYM", []))
    return values[0] if values else True


def _pick_bits(cls):
    supported_bits = list(getattr(cls, "SUPPORTS_BITS", []))
    for candidate in supported_bits:
        if candidate in {2, 3, 4, 5, 6, 8}:
            return candidate
    return None


def _force_auto_candidates_valid(monkeypatch, method, fmt):
    for cls in set(AUTO_BACKEND_KERNEL_MAPPING[method][fmt].values()):
        monkeypatch.setattr(
            cls,
            "cached_validate_once",
            classmethod(lambda qlinear_cls: (True, None)),
        )


def test_auto_select_normalizes_torch_device_before_device_prefilter(monkeypatch):
    class CudaKernel:
        SUPPORTS_DEVICES = [DEVICE.CUDA]

        @classmethod
        def validate(cls, **_):
            return True, None

    class AnyDeviceKernel:
        SUPPORTS_DEVICES = [DEVICE.ALL]

        @classmethod
        def validate(cls, **_):
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict(
            [
                (BACKEND.QQQ, CudaKernel),
                (BACKEND.QQQ_TORCH, AnyDeviceKernel),
            ]
        ),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=torch.device("cuda:0"),
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is CudaKernel


CASES = []
for kernel_cls in sorted(_iter_kernel_classes(), key=lambda cls: cls.__name__):
    for method in _infer_quant_methods(kernel_cls):
        for fmt in kernel_cls.SUPPORTS_FORMATS:
            CASES.append((kernel_cls, method, fmt))


@pytest.mark.parametrize("kernel_cls,method,fmt", CASES)
def test_select_quant_linear_smoke(kernel_cls, method, fmt):
    device = _pick_device(kernel_cls)
    if device is None:
        pytest.skip(f"No supported device available for {kernel_cls.__name__}.")

    ok, err = kernel_cls.cached_validate_once()
    if not ok:
        pytest.skip(f"{kernel_cls.__name__} unavailable: {err}")

    pack_dtype = kernel_cls.SUPPORTS_PACK_DTYPES[0]
    bits = _pick_bits(kernel_cls)
    if bits is None:
        pytest.skip(f"No selector-compatible bit-width available for {kernel_cls.__name__}.")
    group_size = _pick_group_size(kernel_cls)
    desc_act = _pick_desc_act(kernel_cls)
    sym = _pick_sym(kernel_cls)

    qlinear_cls = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        device=device,
        backend=kernel_cls.SUPPORTS_BACKENDS[0],
        format=fmt,
        quant_method=method,
        pack_dtype=pack_dtype,
    )

    assert qlinear_cls is kernel_cls


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_cpu_auto_select_prioritizes_torch_aten_for_gptq(monkeypatch, fmt):
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, fmt)

    candidates = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is TorchAtenLinear


def test_cpu_auto_select_prioritizes_torch_aten_for_awq(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.AWQ, FORMAT.GEMM)

    candidates = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is TorchAtenAwqLinear


def test_cpu_auto_select_prioritizes_cpp_kernel_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFCppKernel
    assert GGUFTorchLinear in candidates


def test_cuda_auto_select_prioritizes_triton_then_cpp_then_torch_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert candidates[1] is GGUFCudaKernel
    assert candidates[2] is GGUFTorchLinear


def test_cuda_auto_select_prioritizes_triton_then_torch_for_sign_only_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=1,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert GGUFCudaKernel not in candidates
    assert candidates[1] is GGUFTorchLinear


def test_cpu_pack_auto_select_skips_cpp_kernel_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack=True,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert GGUFCppKernel not in candidates
    assert candidates[0] is GGUFTorchLinear


def test_cuda_pack_auto_select_prioritizes_triton_for_gguf(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GGUF, FORMAT.GGUF)

    candidates = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack=True,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is GGUFTritonKernel
    assert GGUFCudaKernel not in candidates
    assert GGUFTorchLinear in candidates


def test_explicit_gguf_cpu_backend_selects_cpp_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFCppKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.GGUF_CPP_CPU,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCppKernel


def test_explicit_gguf_cuda_backend_selects_cuda_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFCudaKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.GGUF_CPP_CUDA,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFCudaKernel


def test_explicit_gguf_torch_backend_selects_torch_kernel():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.GGUF_TORCH,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTorchLinear


def test_explicit_gguf_triton_backend_selects_triton_kernel(monkeypatch):
    monkeypatch.setattr(
        GGUFTritonKernel,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.GGUF_TRITON,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTritonKernel


def test_explicit_awq_marlin_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        AwqMarlinLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        AwqMarlinLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MARLIN,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqMarlinLinear


def test_explicit_awq_machete_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        AwqMacheteLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        AwqMacheteLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MACHETE,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqMacheteLinear


def test_explicit_gptq_machete_backend_selects_asymmetric_kernel(monkeypatch):
    monkeypatch.setattr(
        MacheteLinear,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    monkeypatch.setattr(
        MacheteLinear,
        "validate_device",
        classmethod(lambda qlinear_cls, _device: None),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        device=DEVICE.CUDA,
        backend=BACKEND.MACHETE,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is MacheteLinear


def test_torch_fused_auto_device_prefers_xpu_or_cpu(monkeypatch):
    monkeypatch.setattr(importer, "HAS_CUDA", True)
    monkeypatch.setattr(importer, "HAS_XPU", False)
    monkeypatch.setattr(importer, "HAS_MPS", False)

    assert auto_select_device(None, BACKEND.TORCH_FUSED) is DEVICE.CPU
    assert auto_select_device(None, BACKEND.TORCH_FUSED_AWQ) is DEVICE.CPU


def test_gguf_does_not_accept_generic_torch_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        select_quant_linear(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            device=DEVICE.CPU,
            backend=BACKEND.TORCH,
            format=FORMAT.GGUF,
            quant_method=METHOD.GGUF,
            pack_dtype=torch.int32,
        )
