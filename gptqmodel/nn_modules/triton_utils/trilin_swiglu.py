# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MethodType

import torch


_TRILIN_SWIGLU_IN_FEATURES = 4096
_TRILIN_SWIGLU_OUT_FEATURES = {11008, 14336}
_TRILIN_SWIGLU_MODEL_MODULES = {
    "llama": "LlamaMLP",
    "mistral": "MistralMLP",
}


def _runtime_qweight(projection: torch.nn.Module) -> torch.Tensor | None:
    runtime_qweight = getattr(projection, "_triton_3bit_qweight", None)
    if runtime_qweight is not None:
        return runtime_qweight
    return getattr(projection, "qweight", None)


def _trilin_swiglu_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_proj = module.gate_proj
    up_proj = module.up_proj
    installed_device = module._gptqmodel_trilin_swiglu_device
    if hidden_states.ndim == 0 or hidden_states.shape[-1] != _TRILIN_SWIGLU_IN_FEATURES:
        return module._gptqmodel_trilin_swiglu_original_forward(hidden_states)
    rows = hidden_states.numel() // hidden_states.shape[-1]
    gate_qweight = _runtime_qweight(gate_proj)
    up_qweight = _runtime_qweight(up_proj)
    if (
        module.training
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or hidden_states.device != installed_device
        or rows != 1
        or gate_qweight is None
        or up_qweight is None
        or gate_qweight.device != installed_device
        or up_qweight.device != installed_device
    ):
        return module._gptqmodel_trilin_swiglu_original_forward(hidden_states)

    from ...utils.trilin import trilin_silu_mul

    values = hidden_states.reshape(1, _TRILIN_SWIGLU_IN_FEATURES)
    if not values.is_contiguous():
        values = values.contiguous()
    intermediate = trilin_silu_mul(
        values,
        gate_qweight,
        gate_proj.scales,
        up_qweight,
        up_proj.scales,
    ).reshape(*hidden_states.shape[:-1], gate_proj.out_features)
    return module.down_proj(intermediate)


def _is_exact_projection(projection: torch.nn.Module, projection_types: tuple[type, ...]) -> bool:
    return (
        isinstance(projection, projection_types)
        and projection.bits == 3
        and projection.group_size == 128
        and not projection.desc_act
        and projection.sym
        and projection.in_features == _TRILIN_SWIGLU_IN_FEATURES
        and projection.out_features in _TRILIN_SWIGLU_OUT_FEATURES
        and projection.bias is None
        and not projection.adapter
        and getattr(projection, "_trilin_native_3bit", False)
    )


def install_trilin_3bit_swiglu(model: torch.nn.Module) -> int:
    """Fuse exact-shape Llama/Mistral 3-bit gate/up GEMVs and SwiGLU for one-row sm80 decode."""

    config = getattr(model, "config", None)
    mlp_class_name = _TRILIN_SWIGLU_MODEL_MODULES.get(getattr(config, "model_type", None))
    if mlp_class_name is None or getattr(config, "hidden_act", None) != "silu":
        return 0

    from ..qlinear.trilin import AwqTrilinLinear, TrilinLinear

    projection_types = (TrilinLinear, AwqTrilinLinear)
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != mlp_class_name or getattr(module, "_gptqmodel_trilin_swiglu", False):
            continue

        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if not _is_exact_projection(gate_proj, projection_types) or not _is_exact_projection(
            up_proj, projection_types
        ):
            continue
        if down_proj is None or gate_proj.out_features != up_proj.out_features:
            continue

        gate_qweight = _runtime_qweight(gate_proj)
        up_qweight = _runtime_qweight(up_proj)
        if gate_qweight is None or up_qweight is None:
            continue
        device = gate_qweight.device
        if (
            device.type != "cuda"
            or up_qweight.device != device
            or gate_proj.scales.device != device
            or up_proj.scales.device != device
            or torch.cuda.get_device_capability(device) != (8, 0)
        ):
            continue

        module._gptqmodel_trilin_swiglu = True
        module._gptqmodel_trilin_swiglu_device = device
        module._gptqmodel_trilin_swiglu_original_forward = module.forward
        module.forward = MethodType(_trilin_swiglu_forward, module)
        installed += 1

    return installed


__all__ = ["install_trilin_3bit_swiglu"]
