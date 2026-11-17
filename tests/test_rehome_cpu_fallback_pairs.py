# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every CPU-fallback pair ``rehome_quantized_kernel`` can reach (#4451).

``rehome_quantized_kernel`` moves the packed layers of an AWQ/QQQ/GGUF run from
the calibration-time kernel (often CUDA-only) to a CPU-capable kernel that
reads the same packed buffers. It swaps ``__class__`` without running the
target's ``__init__``, so a target field created only there is missing; QQQ
crashed at loop end with ``'QQQTorchLinear' object has no attribute
'_torch_int_mm_failed_shapes'`` (fixed for QQQ by #4490).

This guard enumerates every pair from the kernel registry, so a kernel added
later with constructor-only state fails here instead of at the end of a
quantization run: the switched module must hold every attribute a directly
built target holds and compute exactly what that target computes.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import rehome_quantized_kernel, select_in_session_quant_linear
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, GGUFConfig, QQQConfig, resolve_quant_format
from gptqmodel.utils.importer import _registry_kernel_classes, _supports_pack_api


IN_FEATURES = 256
OUT_FEATURES = 256

_CONFIGS = {
    "awq_gemm": lambda: AWQConfig(bits=4, format="gemm", group_size=128, offload_to_disk=False),
    "qqq": lambda: QQQConfig(bits=4, group_size=128, offload_to_disk=False),
    "gguf_q4_k_m": lambda: GGUFConfig(bits="q4_k_m", offload_to_disk=False),
}

# The calibration-time kernels a CUDA host selects for these formats. The
# registry enumeration below must cover them.
_CUDA_SOURCES = {
    "awq_gemm": AwqGEMMLinear,
    "qqq": QQQLinear,
    "gguf_q4_k_m": GGUFTritonKernel,
}

_EXPECTED_TARGETS = {
    "awq_gemm": AwqTorchLinear,
    "qqq": QQQTorchLinear,
    "gguf_q4_k_m": GGUFTorchLinear,
}


def _constructor_kwargs(qcfg, *, bias: bool) -> dict:
    method = qcfg.export_quant_method()
    return {
        "bits": qcfg.bits if str(method) == "gguf" else qcfg.runtime_bits,
        "group_size": qcfg.group_size,
        "sym": qcfg.sym,
        "desc_act": qcfg.desc_act,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": bias,
        "pack_dtype": qcfg.pack_dtype,
        "dtype": torch.float16,
        "name": "model.layers.0.mlp.down_proj",
    }


def _cpu_fallback_sources(qcfg) -> list[type[BaseQuantLinear]]:
    """Packing kernels for ``qcfg`` that cannot run on CPU."""

    method = qcfg.export_quant_method()
    fmt = resolve_quant_format(qcfg.format, method)
    return [
        cls
        for cls in _registry_kernel_classes()
        if method in (cls.SUPPORTS_METHODS or ())
        and fmt in (cls.SUPPORTS_FORMATS or {})
        and _supports_pack_api(cls)
        and DEVICE.CPU not in (cls.SUPPORTS_DEVICES or ())
        and DEVICE.ALL not in (cls.SUPPORTS_DEVICES or ())
    ]


def _populate(module: BaseQuantLinear, *, bias: bool) -> None:
    """Fill the packed payload of a CPU-capable reference kernel."""

    generator = torch.Generator().manual_seed(4451)
    if isinstance(module, GGUFTorchLinear):
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator) * 0.05)
            if bias:
                linear.bias.copy_(torch.randn(OUT_FEATURES, generator=generator) * 0.1)
        module.pack_original(linear=linear, scales=None, zeros=None, g_idx=None)
        return

    for name, tensor in list(module._buffers.items()):
        if tensor is None or name in module._non_persistent_buffers_set or tensor.ndim == 0:
            continue
        if tensor.dtype == torch.int32:
            value = torch.randint(-(2**31), 2**31 - 1, tensor.shape, dtype=torch.int32, generator=generator)
        elif name == "bias":
            value = (torch.randn(tensor.shape, generator=generator) * 0.1).to(tensor.dtype)
        else:
            value = (torch.rand(tensor.shape, generator=generator) * 0.01 + 0.001).to(tensor.dtype)
        module._buffers[name] = value


class _Tree(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.proj = module


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("config_name", sorted(_CONFIGS))
def test_rehome_switches_every_cpu_fallback_pair_with_complete_state(monkeypatch, config_name, bias):
    qcfg = _CONFIGS[config_name]()
    target = select_in_session_quant_linear(qcfg, torch.device("cpu"))
    assert target is _EXPECTED_TARGETS[config_name]

    sources = _cpu_fallback_sources(qcfg)
    assert _CUDA_SOURCES[config_name] in sources

    kwargs = _constructor_kwargs(qcfg, bias=bias)
    reference = target(**kwargs, register_buffers=True)
    _populate(reference, bias=bias)
    reference.post_init()
    activation = torch.randn(3, IN_FEATURES, generator=torch.Generator().manual_seed(7)).to(torch.float16)
    with torch.inference_mode():
        expected = reference(activation)

    switched = []
    for source_cls in sources:
        # Build the accelerator kernel on any host: only its constructor state
        # matters here, never its native forward.
        monkeypatch.setattr(source_cls, "validate", classmethod(lambda cls, **_kw: (True, None)))
        try:
            module = source_cls(**kwargs, register_buffers=True)
        except (ImportError, ValueError, NotImplementedError):
            # The quantizer builds its packing kernel the same way; a kernel
            # that rejects this layout (AllSpark is 8-bit only) or lacks its
            # package (BitBLAS) can never be the module the switch receives.
            continue
        for name, tensor in reference._buffers.items():
            if tensor is not None:
                module._buffers[name] = tensor.clone()

        fake_model = SimpleNamespace(
            model=_Tree(module),
            quantize_config=qcfg,
            qlinear_kernel=source_cls,
        )
        rehome_quantized_kernel(fake_model)

        assert type(module) is target, source_cls.__name__
        assert fake_model.qlinear_kernel is target
        assert module.backend == reference.backend
        missing = sorted(set(vars(reference)) - set(vars(module)))
        assert not missing, f"{source_cls.__name__} -> {target.__name__} missing {missing}"

        # Every cache and failure set the target owns must be usable.
        module.clear_weight_cache()
        module.post_init()
        with torch.inference_mode():
            actual = module(activation)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=source_cls.__name__)
        switched.append(source_cls)

    assert _CUDA_SOURCES[config_name] in switched
