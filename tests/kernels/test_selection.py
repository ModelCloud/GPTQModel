# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.models._const as model_const
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    AUTO_BACKEND_KERNEL_MAPPING,
    _iter_dynamic_contracts,
    auto_select_device,
    build_kernel_support_maps,
    clear_validation_cache,
    expand_selector_device_family,
    iter_quant_linear_kernels,
    normalize_device_device_map,
    select_quant_linear,
    validate_quant_linear,
)
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


def test_role_only_kernel_is_excluded_from_backend_discovery():
    assert TorchQuantEmbeddings.SUPPORTS_BACKEND_SELECTION is False
    assert TorchQuantEmbeddings not in iter_quant_linear_kernels()
    auto_mapping, _ = build_kernel_support_maps()
    assert all(
        TorchQuantEmbeddings not in backend_mapping.values()
        for format_mapping in auto_mapping.values()
        for backend_mapping in format_mapping.values()
    )


CASES = []
for kernel_cls in sorted(_iter_kernel_classes(), key=lambda cls: cls.__name__):
    if not getattr(kernel_cls, "SUPPORTS_BACKEND_SELECTION", True):
        continue
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


def test_iter_dynamic_contracts_yields_base_and_distinct_overrides():
    dynamic = {
        "+:^lm_head$": {"bits": 4, "group_size": 64},
        "+:^embed_tokens$": {"bits": 4, "group_size": 64},
        "model.layers.0.self_attn.q_proj": {"bits": 2, "sym": False},
    }
    contracts = list(
        _iter_dynamic_contracts(
            dynamic=dynamic,
            bits=3,
            group_size=32,
            desc_act=True,
            sym=True,
            pack_dtype=torch.int32,
            format_value=FORMAT.GPTQ,
        )
    )
    assert len(contracts) == 3
    assert {"bits": 3, "group_size": 32, "desc_act": True, "sym": True, "pack_dtype": torch.int32} in contracts
    assert {"bits": 4, "group_size": 64, "desc_act": True, "sym": True, "pack_dtype": torch.int32} in contracts
    assert {"bits": 2, "group_size": 32, "desc_act": True, "sym": False, "pack_dtype": torch.int32} in contracts


def test_select_quant_linear_multi_select_expands_dynamic_contracts(monkeypatch):
    """A 4-bit dynamic override must be visible to kernels that only support 4-bit weights,
    even when the base quantize config is 3-bit."""
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, FORMAT.GPTQ)
    # Marlin validates device compute capability before accepting the contract.
    monkeypatch.setattr(MarlinLinear, "validate_device", classmethod(lambda cls, _device: None))

    dynamic = {
        "model.layers.0.mlp.down_proj": {"bits": 4, "group_size": 128, "sym": True, "desc_act": False},
        "model.layers.0.mlp.up_proj": {"bits": 3, "group_size": 128, "sym": True, "desc_act": False},
    }
    candidates = select_quant_linear(
        bits=3,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=dynamic,
        multi_select=True,
    )
    # The 4-bit contract enables 4-bit-only kernels; the 3-bit contract and base
    # config keep 3-bit-capable kernels; Torch covers all widths.
    assert MarlinLinear in candidates
    assert ExllamaV2Linear in candidates
    assert TritonV2Linear in candidates
    assert TorchLinear in candidates


def test_select_quant_linear_single_select_keeps_model_wide_kernel(monkeypatch):
    """Single-select must return one kernel that is compatible with the whole model,
    including the base bits, so 4-bit-only kernels are not chosen here."""
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, FORMAT.GPTQ)

    dynamic = {
        "model.layers.0.mlp.down_proj": {"bits": 4, "group_size": 128, "sym": True, "desc_act": False},
    }
    selected = select_quant_linear(
        bits=3,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=dynamic,
    )
    # Marlin/Exllama/Triton do not support base bits=3, so the model-wide kernel
    # must be TorchLinear, which supports both the base 3-bit and dynamic 4-bit layers.
    assert selected is TorchLinear


def test_sharded_select_tolerates_kernel_without_shard_attrs(monkeypatch):
    class BareKernel:
        # Defines neither SUPPORTS_SHARDED_LOAD nor SUPPORTS_SHARDS; the
        # sharded-load guard must not evaluate a missing attribute eagerly
        # and treats absent declarations as shard-capable.
        SUPPORTS_DEVICES = [DEVICE.ALL]

        @classmethod
        def validate(cls, **_):
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict([(BACKEND.QQQ_TORCH, BareKernel)]),
    )

    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=torch.device("cpu"),
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
        is_sharded=True,
    )

    assert qlinear_cls is BareKernel


def test_select_quant_linear_validates_each_exact_multi_gpu_target(monkeypatch):
    calls = []

    class IndexedKernel:
        SUPPORTS_DEVICES = [DEVICE.CUDA]

        @classmethod
        def validate(cls, **kwargs):
            calls.append(kwargs["device"])
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict([(BACKEND.QQQ, IndexedKernel)]),
    )

    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=[torch.device("cuda:0"), torch.device("cuda:1")],
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    )

    assert selected is IndexedKernel
    assert calls == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_device_map_preserves_ordinals_and_rejects_mixed_capabilities(monkeypatch):
    monkeypatch.setattr(
        importer.torch.cuda,
        "get_device_capability",
        lambda device=None: (8, 0) if torch.device(device).index == 0 else (9, 0),
    )

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        normalize_device_device_map(None, {"first": "cuda:0", "second": "cuda:1"})

    monkeypatch.setattr(importer.torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    assert normalize_device_device_map(None, {"first": "cuda:0", "second": "cuda:1"}) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


def test_auto_device_map_rejects_mixed_visible_capabilities(monkeypatch):
    monkeypatch.setattr(importer.torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    monkeypatch.setattr(importer.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        importer.torch.cuda,
        "get_device_capability",
        lambda device=None: (8, 0) if torch.device(device).index == 0 else (9, 0),
    )

    with pytest.raises(ValueError, match=r"cuda:0=8\.0, cuda:1=9\.0"):
        normalize_device_device_map(None, "auto")


def test_expand_selector_device_family_preserves_all_visible_ordinals(monkeypatch):
    monkeypatch.setattr(importer.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(importer.torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    assert expand_selector_device_family(DEVICE.CUDA) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


def test_expand_selector_device_family_expands_unindexed_torch_device(monkeypatch):
    monkeypatch.setattr(importer.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(importer.torch.cuda, "get_device_capability", lambda device=None: (8, 0))

    assert expand_selector_device_family(torch.device("cuda")) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


def test_selector_device_descriptor_has_consistent_shape():
    assert importer._selector_device_descriptor(DEVICE.ALL) == (
        "family",
        "all",
        None,
        None,
        None,
    )
    assert importer._selector_device_descriptor(torch.device("cpu")) == (
        "torch",
        "cpu",
        "cpu",
        None,
        None,
    )


def test_selector_device_descriptor_uses_rocm_family_for_cuda_device(monkeypatch):
    monkeypatch.setattr(importer, "IS_ROCM", True)
    monkeypatch.setattr(importer, "_device_capability", lambda _device: (9, 0))

    assert importer._selector_device_descriptor(torch.device("cuda:1")) == (
        "torch",
        "rocm",
        "cuda",
        1,
        (9, 0),
    )


@pytest.mark.parametrize("accelerator_type", ["cuda", "xpu"])
def test_integer_device_map_uses_active_accelerator(monkeypatch, accelerator_type):
    monkeypatch.setattr(
        importer.torch.accelerator,
        "current_accelerator",
        lambda: torch.device(accelerator_type),
    )

    assert importer._as_selector_device(2) == torch.device(f"{accelerator_type}:2")
    assert normalize_device_device_map(None, {"layer": 2}) == torch.device(f"{accelerator_type}:2")


def test_integer_device_map_uses_npu_without_torch_npu(monkeypatch):
    class FakeDevice:
        def __init__(self, spec):
            self.spec = spec

    monkeypatch.setattr(
        importer.torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="npu"),
    )
    monkeypatch.setattr(importer.torch, "device", FakeDevice)

    assert importer._as_selector_device(2).spec == "npu:2"


def test_rocm_string_preserves_device_family_on_non_rocm_host(monkeypatch):
    monkeypatch.setattr(importer, "IS_ROCM", False)

    assert importer._as_selector_device("rocm") is DEVICE.ROCM
    assert normalize_device_device_map(None, {"": "rocm"}) is DEVICE.ROCM


def test_qqq_accepts_exact_rocm_cuda_device(monkeypatch):
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.qqq.IS_ROCM", True)

    QQQLinear.validate_device(torch.device("cuda:1"))
    assert DEVICE.ROCM in QQQLinear.SUPPORTS_DEVICES


@pytest.mark.parametrize(
    ("has_cuda", "has_xpu", "has_npu", "expected"),
    [
        (False, True, False, DEVICE.XPU),
        (False, False, True, DEVICE.NPU),
    ],
)
def test_public_integer_device_uses_active_accelerator(monkeypatch, has_cuda, has_xpu, has_npu, expected):
    monkeypatch.setattr(model_const, "HAS_CUDA", has_cuda)
    monkeypatch.setattr(model_const, "HAS_XPU", has_xpu)
    monkeypatch.setattr(model_const, "HAS_NPU", has_npu)
    monkeypatch.setattr(model_const, "HAS_MPS", False)

    assert normalize_device_device_map(0, None) is expected

    class ActiveAcceleratorKernel:
        SUPPORTS_DEVICES = [expected]

        @classmethod
        def validate(cls, **_kwargs):
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict([(BACKEND.QQQ, ActiveAcceleratorKernel)]),
    )
    assert select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=0,
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    ) is ActiveAcceleratorKernel


def test_validation_cache_partitions_capability_and_index(monkeypatch):
    clear_validation_cache()
    capability = {0: (8, 0), 1: (8, 0)}
    monkeypatch.setattr(
        importer.torch.cuda,
        "get_device_capability",
        lambda device=None: capability[torch.device(device).index],
    )
    calls = []

    class DeviceKernel:
        @classmethod
        def validate(cls, **kwargs):
            calls.append(kwargs["device"])
            return True, None

    contract = dict(bits=4, group_size=128, trainable=False)
    validate_quant_linear(DeviceKernel, **contract, device=torch.device("cuda:0"))
    validate_quant_linear(DeviceKernel, **contract, device=torch.device("cuda:0"))
    validate_quant_linear(DeviceKernel, **contract, device=torch.device("cuda:1"))
    capability[0] = (9, 0)
    validate_quant_linear(DeviceKernel, **contract, device=torch.device("cuda:0"))

    assert calls == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:0"),
    ]


def test_validation_cache_is_contract_and_device_sensitive():
    clear_validation_cache()
    calls = []

    class ContractKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]

        @classmethod
        def validate(cls, **kwargs):
            calls.append(kwargs)
            return True, None

    common = dict(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=None,
        trainable=False,
        adapter=None,
    )
    validate_quant_linear(ContractKernel, **common, device=torch.device("cpu"), in_features=128, out_features=256)
    validate_quant_linear(ContractKernel, **common, device=torch.device("cpu"), in_features=128, out_features=256)

    variants = [
        {"device": torch.device("meta")},
        {"in_features": 256},
        {"out_features": 512},
        {"bits": 3},
        {"group_size": 64},
        {"dtype": torch.bfloat16},
        {"pack_dtype": torch.int16},
        {"desc_act": True},
        {"sym": False},
        {"trainable": True},
    ]
    for variant in variants:
        contract = dict(common, device=torch.device("cpu"), in_features=128, out_features=256)
        contract.update(variant)
        validate_quant_linear(ContractKernel, **contract)

    assert len(calls) == 1 + len(variants)


def test_allow_marlin_false_filters_auto_and_rejects_explicit(monkeypatch):
    class MarlinKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]
        SUPPORTS_BACKENDS = [BACKEND.GPTQ_MARLIN]

        @classmethod
        def validate(cls, **_kwargs):
            return True, None

    class FallbackKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]
        SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]

        @classmethod
        def validate(cls, **_kwargs):
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.GPTQ],
        FORMAT.GPTQ,
        OrderedDict(
            [
                (BACKEND.GPTQ_MARLIN, MarlinKernel),
                (BACKEND.GPTQ_TORCH, FallbackKernel),
            ]
        ),
    )

    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=torch.device("cpu"),
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
        allow_marlin=False,
    )
    assert selected is FallbackKernel

    with pytest.raises(ValueError, match="allow_marlin=False"):
        select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device=torch.device("cpu"),
            backend=BACKEND.GPTQ_MARLIN,
            format=FORMAT.GPTQ,
            quant_method=METHOD.GPTQ,
            pack_dtype=torch.int32,
            allow_marlin=False,
        )
