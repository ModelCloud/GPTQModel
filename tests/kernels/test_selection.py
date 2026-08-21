# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections import OrderedDict

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.humming import HummingAwqLinear, HummingGptqLinear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.pangolin import PangolinQuantLinear
from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear, SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, TorchQuantEmbeddings
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.trilin import AwqTrilinLinear, TrilinLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    AUTO_BACKEND_KERNEL_MAPPING,
    _iter_dynamic_contracts,
    auto_select_device,
    iter_quant_linear_kernels,
    select_quant_linear,
)
from gptqmodel.utils.model import create_quant_layer
from gptqmodel.utils.rocm import IS_ROCM
from gptqmodel.utils.torch import HAS_CUDA, HAS_MPS, HAS_NPU, HAS_XPU


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
    if DEVICE.NPU in devices and HAS_NPU:
        return DEVICE.NPU
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
    # Prefer False; it is the most compatible default and avoids kernel-specific
    # restrictions such as AWQ Triton 3-bit fused inference requiring desc_act=False.
    if False in values:
        return False
    return values[0] if values else False


def _pick_sym(cls):
    values = list(getattr(cls, "SUPPORTS_SYM", []))
    return values[0] if values else True


def _pick_bits(cls, fmt, *, device, group_size, desc_act, sym, pack_dtype, dtype):
    supported_bits = list(cls.supported_bits(fmt))
    for candidate in supported_bits:
        if candidate not in {2, 3, 4, 5, 6, 8}:
            continue
        valid, _ = cls.validate(
            bits=candidate,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=None,
            device=device,
            trainable=False,
            format=fmt,
        )
        if valid:
            return candidate
    return None


def _force_auto_candidates_valid(monkeypatch, method, fmt):
    # These are selector contract tests, not hardware probes. Simulate a
    # capable CUDA runtime so they do not depend on the host running pytest.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (9, 0))
    for cls in set(AUTO_BACKEND_KERNEL_MAPPING[method][fmt].values()):
        monkeypatch.setattr(
            cls,
            "cached_validate_once",
            classmethod(lambda qlinear_cls: (True, None)),
        )
        monkeypatch.setattr(
            cls,
            "validate_device",
            classmethod(lambda _cls, _device: None),
        )


def _disable_humming(monkeypatch):
    """Force Humming kernels to fail validation so older fallback tests remain deterministic."""
    for cls in (HummingGptqLinear, HummingAwqLinear):
        monkeypatch.setattr(
            cls,
            "cached_validate_once",
            classmethod(lambda _cls: (False, None)),
        )


def test_pangolin_validation_checks_its_runtime(monkeypatch):
    runtime_error = ImportError("Pangolin runtime unavailable")
    monkeypatch.setattr(
        PangolinQuantLinear,
        "cached_validate_once",
        classmethod(lambda _cls: (False, runtime_error)),
    )

    valid, error = PangolinQuantLinear.validate(
        bits=3,
        group_size=32,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=None,
        device=DEVICE.CUDA,
        trainable=False,
        format=FORMAT.GPTQ_P,
    )

    assert valid is False
    assert error is runtime_error


def test_pangolin_mps_validation_rejects_missing_metal_runtime(monkeypatch):
    monkeypatch.setattr(
        PangolinQuantLinear,
        "cached_validate_once",
        classmethod(lambda _cls: (True, None)),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.pangolin_mps.pangolin_mps_supported",
        lambda: False,
    )

    valid, error = PangolinQuantLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=None,
        device=DEVICE.MPS,
        trainable=False,
        format=FORMAT.GPTQ_P,
    )

    assert valid is False
    assert isinstance(error, NotImplementedError)
    assert "compile_shader" in str(error)


def test_pangolin_mps_selection_requires_fp16_dtype(monkeypatch):
    monkeypatch.setattr(
        PangolinQuantLinear,
        "cached_validate_once",
        classmethod(lambda _cls: (True, None)),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.pangolin_mps.pangolin_mps_supported",
        lambda: True,
    )
    kwargs = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "device": DEVICE.MPS,
        "backend": BACKEND.GPTQ_PANGOLIN,
        "format": FORMAT.GPTQ_P,
        "quant_method": METHOD.GPTQ,
        "pack_dtype": torch.int32,
    }

    assert select_quant_linear(**kwargs, dtype=torch.float16) is PangolinQuantLinear
    with pytest.raises(ValueError, match="float16 inference only"):
        select_quant_linear(**kwargs, dtype=None)


@pytest.mark.parametrize(
    ("kernel_cls", "fmt", "bits"),
    [
        (SwordfishLinear, FORMAT.GPTQ, (4, 8)),
        (AwqSwordfishLinear, FORMAT.GEMM, (4,)),
    ],
)
def test_swordfish_declares_complete_format_bit_contract(kernel_cls, fmt, bits):
    kernel_cls.verify_supports_params()

    assert kernel_cls.supported_bits(fmt) == bits


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


def test_explicit_select_tolerates_kernel_without_legacy_shard_capability(monkeypatch):
    class MinimalKernel:
        @classmethod
        def validate(cls, **_kwargs):
            return True, None

    monkeypatch.setattr(importer, "get_kernel_for_backend", lambda *_args: MinimalKernel)

    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.QQQ_TORCH,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    )

    assert selected is MinimalKernel


def test_auto_and_explicit_selection_both_forward_adapter_to_validation(monkeypatch):
    adapter = object()
    seen_adapters = []
    unsupported = NotImplementedError("test adapter is unsupported")

    class AdapterAwareKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]

        @classmethod
        def validate(cls, **kwargs):
            seen_adapters.append(kwargs.get("adapter"))
            if kwargs.get("adapter") is adapter:
                return False, unsupported
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict(((BACKEND.QQQ_TORCH, AdapterAwareKernel),)),
    )
    selection_kwargs = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "device": DEVICE.CPU,
        "format": FORMAT.QQQ,
        "quant_method": METHOD.QQQ,
        "pack_dtype": torch.int32,
        "adapter": adapter,
    }

    with pytest.raises(NotImplementedError, match="test adapter is unsupported"):
        select_quant_linear(**selection_kwargs, backend=BACKEND.AUTO)

    monkeypatch.setattr(importer, "get_kernel_for_backend", lambda *_args: AdapterAwareKernel)
    with pytest.raises(ValueError, match="test adapter is unsupported"):
        select_quant_linear(**selection_kwargs, backend=BACKEND.QQQ_TORCH)

    assert seen_adapters == [adapter, adapter]


def test_auto_falls_back_on_dependency_probe_exception_but_explicit_preserves_it(monkeypatch):
    missing_dependency = ModuleNotFoundError("optional kernel runtime is missing")

    class MissingDependencyKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]

        @classmethod
        def validate(cls, **_kwargs):
            raise missing_dependency

    class FallbackKernel:
        SUPPORTS_DEVICES = [DEVICE.CPU]

        @classmethod
        def validate(cls, **_kwargs):
            return True, None

    monkeypatch.setitem(
        AUTO_BACKEND_KERNEL_MAPPING[METHOD.QQQ],
        FORMAT.QQQ,
        OrderedDict(
            (
                (BACKEND.QQQ, MissingDependencyKernel),
                (BACKEND.QQQ_TORCH, FallbackKernel),
            )
        ),
    )
    selection_kwargs = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "device": DEVICE.CPU,
        "format": FORMAT.QQQ,
        "quant_method": METHOD.QQQ,
        "pack_dtype": torch.int32,
    }

    assert select_quant_linear(**selection_kwargs, backend=BACKEND.AUTO) is FallbackKernel

    monkeypatch.setattr(importer, "get_kernel_for_backend", lambda *_args: MissingDependencyKernel)
    with pytest.raises(ModuleNotFoundError, match="optional kernel runtime is missing"):
        select_quant_linear(**selection_kwargs, backend=BACKEND.QQQ_TORCH)


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_auto_select_excludes_embedding_only_kernel(fmt):
    candidates = AUTO_BACKEND_KERNEL_MAPPING[METHOD.GPTQ][fmt].values()

    assert TorchQuantEmbeddings not in candidates
    assert TorchLinear in candidates
    assert TorchQuantEmbeddings not in iter_quant_linear_kernels()


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
    group_size = _pick_group_size(kernel_cls)
    desc_act = _pick_desc_act(kernel_cls)
    sym = _pick_sym(kernel_cls)
    dtype = kernel_cls.SUPPORTS_DTYPES[0] if kernel_cls.SUPPORTS_DTYPES else None
    bits = _pick_bits(
        kernel_cls,
        fmt,
        device=device,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        pack_dtype=pack_dtype,
        dtype=dtype,
    )
    if bits is None:
        pytest.skip(f"No selector-compatible bit-width available for {kernel_cls.__name__}.")

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
        dtype=dtype,
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


@pytest.mark.parametrize(
    "method,fmt,kernel_cls,group_size",
    [
        *((METHOD.GPTQ, FORMAT.GPTQ, TrilinLinear, group_size)
          for group_size in TrilinLinear.SUPPORTS_GROUP_SIZE),
        *((METHOD.AWQ, FORMAT.GEMM, AwqTrilinLinear, group_size)
          for group_size in AwqTrilinLinear.SUPPORTS_GROUP_SIZE),
    ],
)
def test_cuda_auto_selects_grouped_3bit_trilin_backend(monkeypatch, method, fmt, kernel_cls, group_size):
    _disable_humming(monkeypatch)
    monkeypatch.setattr(
        kernel_cls,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )

    selected = select_quant_linear(
        bits=3,
        group_size=group_size,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=method,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is kernel_cls


@pytest.mark.parametrize(
    ("method", "fmt", "expected"),
    [
        (METHOD.GPTQ, FORMAT.GPTQ, TritonV2Linear),
        (METHOD.AWQ, FORMAT.GEMM, AwqGEMMTritonLinear),
    ],
)
def test_cuda_auto_uses_triton_for_unsupported_trilin_group_size(monkeypatch, method, fmt, expected):
    _force_auto_candidates_valid(monkeypatch, method, fmt)
    _disable_humming(monkeypatch)

    selected = select_quant_linear(
        bits=3,
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=method,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is expected


@pytest.mark.parametrize(
    ("method", "fmt", "expected"),
    [
        (METHOD.GPTQ, FORMAT.GPTQ, TrilinLinear),
        (METHOD.AWQ, FORMAT.GEMM, AwqTrilinLinear),
    ],
)
def test_explicit_trilin_backend_selects_layout_specific_kernel(monkeypatch, method, fmt, expected):
    monkeypatch.setattr(
        expected,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )

    selected = select_quant_linear(
        bits=3,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.TRILIN,
        format=fmt,
        quant_method=method,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is expected


@pytest.mark.parametrize(
    ("method", "fmt", "kernel_cls"),
    [
        (METHOD.GPTQ, FORMAT.GPTQ, TritonV2Linear),
        (METHOD.AWQ, FORMAT.GEMM, AwqGEMMTritonLinear),
    ],
)
def test_cuda_auto_selects_3bit_triton_with_compatible_4bit_overrides(
    monkeypatch,
    method,
    fmt,
    kernel_cls,
):
    _disable_humming(monkeypatch)
    monkeypatch.setattr(
        kernel_cls,
        "cached_validate_once",
        classmethod(lambda qlinear_cls: (True, None)),
    )
    dynamic = {
        "+:^model\\.embed_tokens$": {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
        },
        "+:^lm_head$": {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
        },
    }

    selected = select_quant_linear(
        bits=3,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=method,
        dynamic=dynamic,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is kernel_cls


@pytest.mark.parametrize("kernel_cls", [TritonV2Linear, AwqGEMMTritonLinear])
def test_3bit_triton_rejects_incompatible_dynamic_3bit_contract(kernel_cls):
    valid, error = kernel_cls.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        dynamic={
            "+:^model\\.layers\\.0\\.self_attn\\.q_proj$": {
                "bits": 3,
                "group_size": 128,
                "desc_act": True,
                "sym": True,
            },
        },
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert not valid
    assert "3-bit fused inference requires `desc_act=False`" in str(error)


def test_create_quant_layer_selects_trilin_decoder_and_triton_4bit_override(monkeypatch):
    for kernel_cls in (TrilinLinear, TritonV2Linear, TorchLinear):
        monkeypatch.setattr(
            kernel_cls,
            "cached_validate_once",
            classmethod(lambda qlinear_cls: (True, None)),
        )

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(128, 32, bias=False)
    model.lm_head = torch.nn.Linear(128, 32, bias=False)
    dynamic = {
        "+:^lm_head$": {
            "bits": 4,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
        },
    }

    primary = create_quant_layer(
        linear_candidates=[TrilinLinear, TritonV2Linear, TorchLinear],
        bits=3,
        desc_act=False,
        dynamic=dynamic,
        group_size=128,
        quant_result={"proj": {}, "lm_head": {}},
        module=model,
        sym=True,
        device=DEVICE.CUDA,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        dtype=torch.float16,
    )

    assert primary is TrilinLinear
    assert isinstance(model.proj, TrilinLinear)
    assert isinstance(model.lm_head, TritonV2Linear)
    assert not isinstance(model.lm_head, TrilinLinear)


@pytest.mark.parametrize("group_size", [96, 192, 256, 384, 512])
@pytest.mark.parametrize(
    ("method", "fmt", "expected_primary", "expected_fallback"),
    [
        (METHOD.GPTQ, FORMAT.GPTQ, TritonV2Linear, TorchLinear),
        (METHOD.AWQ, FORMAT.GEMM, AwqGEMMTritonLinear, AwqTorchLinear),
    ],
)
def test_cuda_auto_selects_extended_group_size_backends(
    monkeypatch,
    group_size,
    method,
    fmt,
    expected_primary,
    expected_fallback,
):
    _force_auto_candidates_valid(monkeypatch, method, fmt)

    candidates = select_quant_linear(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        device=DEVICE.CUDA,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=method,
        pack_dtype=torch.int32,
        multi_select=True,
    )

    assert candidates[0] is expected_primary
    assert expected_fallback in candidates


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
            desc_act=False,
            sym=True,
            pack_dtype=torch.int32,
            format_value=FORMAT.GPTQ,
        )
    )
    assert len(contracts) == 3
    assert {"bits": 3, "group_size": 32, "desc_act": False, "sym": True, "pack_dtype": torch.int32} in contracts
    assert {"bits": 4, "group_size": 64, "desc_act": False, "sym": True, "pack_dtype": torch.int32} in contracts
    assert {"bits": 2, "group_size": 32, "desc_act": False, "sym": False, "pack_dtype": torch.int32} in contracts


def test_select_quant_linear_multi_select_expands_dynamic_contracts(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, FORMAT.GPTQ)
    # Marlin's validate_device inspects actual GPU compute capability; neutralize it
    # so the test depends only on declared capability contracts.
    monkeypatch.setattr(
        MarlinLinear,
        "validate_device",
        classmethod(lambda _cls, _device: None),
    )

    dynamic = {
        "+:^lm_head$": {"bits": 4, "group_size": 32, "desc_act": False, "sym": True},
        "+:^model\\.layers\\.0\\.self_attn\\.q_proj$": {
            "bits": 2,
            "group_size": 32,
            "desc_act": False,
            "sym": True,
        },
    }
    candidates = select_quant_linear(
        bits=3,
        group_size=32,
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
    # 4-bit contract enables Swordfish/Marlin/Exllama; base 3-bit enables
    # Trilin; Triton/Torch cover all contracts.
    assert SwordfishLinear in candidates
    assert MarlinLinear in candidates
    assert ExllamaV2Linear in candidates
    assert TrilinLinear in candidates
    assert TritonV2Linear in candidates
    assert TorchLinear in candidates
    assert candidates[0] is SwordfishLinear


def test_select_quant_linear_single_select_stays_model_wide_compatible(monkeypatch):
    _force_auto_candidates_valid(monkeypatch, METHOD.GPTQ, FORMAT.GPTQ)
    _disable_humming(monkeypatch)
    monkeypatch.setattr(
        MarlinLinear,
        "validate_device",
        classmethod(lambda _cls, _device: None),
    )

    dynamic = {
        "+:^lm_head$": {"bits": 4, "group_size": 32, "desc_act": False, "sym": True},
    }
    selected = select_quant_linear(
        bits=3,
        group_size=32,
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
    # Marlin fails on base bits=3; Trilin rejects dynamic 4-bit; Triton handles the whole map.
    assert selected is TritonV2Linear


def test_create_quant_layer_selects_marlin_for_4bit_and_trilin_for_3bit(monkeypatch):
    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.marlin.marlin_import_exception",
        None,
    )
    for cls in (MarlinLinear, TrilinLinear, TritonV2Linear, TorchLinear):
        monkeypatch.setattr(
            cls,
            "cached_validate_once",
            classmethod(lambda _qlinear_cls: (True, None)),
        )
    monkeypatch.setattr(
        MarlinLinear,
        "validate_device",
        classmethod(lambda _cls, _device: None),
    )

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(128, 32, bias=False)
    model.lm_head = torch.nn.Linear(128, 32, bias=False)
    dynamic = {
        "+:^lm_head$": {"bits": 4, "group_size": 32, "desc_act": False, "sym": True},
    }
    _ = create_quant_layer(
        linear_candidates=[MarlinLinear, TrilinLinear, TritonV2Linear, TorchLinear],
        bits=3,
        desc_act=False,
        dynamic=dynamic,
        group_size=128,
        quant_result={"proj": {}, "lm_head": {}},
        module=model,
        sym=True,
        device=DEVICE.CUDA,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ,
        dtype=torch.float16,
    )
    assert isinstance(model.proj, TrilinLinear)
    assert isinstance(model.lm_head, MarlinLinear)
    assert not isinstance(model.lm_head, TrilinLinear)
