# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MethodType
from weakref import ref

import torch


_TRILIN_QKV_INPUT_FEATURES = 4096
_TRILIN_QKV_Q_FEATURES = 4096
_TRILIN_QKV_KV_FEATURES = {1024, 4096}
_TRILIN_QKV_CACHE_ATTRIBUTE = "_gptqmodel_trilin_qkv_cache"
_TRILIN_QKV_MODEL_MODULES = {
    "llama": "LlamaAttention",
    "mistral": "MistralAttention",
}


def _runtime_qweight(projection: torch.nn.Module) -> torch.Tensor | None:
    runtime_qweight = getattr(projection, "_triton_3bit_qweight", None)
    if runtime_qweight is not None:
        return runtime_qweight
    return getattr(projection, "qweight", None)


def _projection_group(module: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module] | None:
    projections = tuple(projection_ref() for projection_ref in module._gptqmodel_trilin_qkv_projection_refs)
    if any(projection is None for projection in projections):
        return None
    return projections


def _runtime_qweights(
    projections: tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    qweights = tuple(_runtime_qweight(projection) for projection in projections)
    if any(qweight is None for qweight in qweights):
        return None
    if any(
        qweight.device != device or not qweight.is_contiguous()
        for qweight in qweights
    ):
        return None
    if any(projection.scales.device != device or not projection.scales.is_contiguous() for projection in projections):
        return None
    return qweights


def _trilin_q_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    projections = _projection_group(module)
    installed_device = module._gptqmodel_trilin_qkv_device
    if (
        projections is None
        or module.training
        or hidden_states.ndim == 0
        or hidden_states.shape[-1] != _TRILIN_QKV_INPUT_FEATURES
        or hidden_states.numel() != _TRILIN_QKV_INPUT_FEATURES
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or hidden_states.device != installed_device
        or not hidden_states.is_contiguous()
    ):
        return module._gptqmodel_trilin_qkv_original_forward(hidden_states)

    qweights = _runtime_qweights(projections, installed_device)
    if qweights is None:
        return module._gptqmodel_trilin_qkv_original_forward(hidden_states)

    from ...utils.trilin import trilin_qkv

    q_proj, k_proj, v_proj = projections
    values = hidden_states.reshape(1, _TRILIN_QKV_INPUT_FEATURES)
    q_output, k_output, v_output = trilin_qkv(
        values,
        qweights[0],
        q_proj.scales,
        qweights[1],
        k_proj.scales,
        qweights[2],
        v_proj.scales,
    )
    original_shape = hidden_states.shape[:-1]
    q_output = q_output.reshape(*original_shape, q_proj.out_features)
    k_output = k_output.reshape(*original_shape, k_proj.out_features)
    v_output = v_output.reshape(*original_shape, v_proj.out_features)
    try:
        setattr(
            hidden_states,
            _TRILIN_QKV_CACHE_ATTRIBUTE,
            (module._gptqmodel_trilin_qkv_cache_key, k_output, v_output),
        )
    except (AttributeError, RuntimeError):
        pass
    return q_output


def _cached_projection(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    output_index: int,
    clear: bool,
) -> torch.Tensor:
    cache = getattr(hidden_states, _TRILIN_QKV_CACHE_ATTRIBUTE, None)
    if cache is None or cache[0] != module._gptqmodel_trilin_qkv_cache_key:
        return module._gptqmodel_trilin_qkv_original_forward(hidden_states)
    if clear:
        try:
            setattr(hidden_states, _TRILIN_QKV_CACHE_ATTRIBUTE, None)
        except (AttributeError, RuntimeError):
            pass
    return cache[output_index]


def _trilin_k_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return _cached_projection(module, hidden_states, output_index=1, clear=False)


def _trilin_v_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return _cached_projection(module, hidden_states, output_index=2, clear=True)


def _is_exact_projection(
    projection: torch.nn.Module,
    projection_types: tuple[type, ...],
    *,
    out_features: int | set[int],
) -> bool:
    expected_out_features = {out_features} if isinstance(out_features, int) else out_features
    return (
        isinstance(projection, projection_types)
        and projection.bits == 3
        and projection.group_size == 128
        and not projection.desc_act
        and projection.sym
        and projection.in_features == _TRILIN_QKV_INPUT_FEATURES
        and projection.out_features in expected_out_features
        and projection.bias is None
        and not projection.adapter
        and getattr(projection, "_trilin_native_3bit", False)
    )


def install_trilin_3bit_qkv(model: torch.nn.Module) -> int:
    """Fuse exact-shape Llama/Mistral 3-bit Q/K/V GEMVs for one-row sm80 decode."""

    config = getattr(model, "config", None)
    attention_class_name = _TRILIN_QKV_MODEL_MODULES.get(getattr(config, "model_type", None))
    if attention_class_name is None:
        return 0

    from ..qlinear.trilin import AwqTrilinLinear, TrilinLinear

    projection_types = (TrilinLinear, AwqTrilinLinear)
    installed = 0
    for attention in model.modules():
        if attention.__class__.__name__ != attention_class_name or getattr(
            attention, "_gptqmodel_trilin_qkv", False
        ):
            continue

        projections = tuple(getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj"))
        q_proj, k_proj, v_proj = projections
        if not _is_exact_projection(q_proj, projection_types, out_features=_TRILIN_QKV_Q_FEATURES):
            continue
        if not _is_exact_projection(k_proj, projection_types, out_features=_TRILIN_QKV_KV_FEATURES):
            continue
        if not _is_exact_projection(v_proj, projection_types, out_features=_TRILIN_QKV_KV_FEATURES):
            continue
        if k_proj.out_features != v_proj.out_features or not all(
            projection.__class__ is q_proj.__class__ for projection in projections[1:]
        ):
            continue

        qweights = tuple(_runtime_qweight(projection) for projection in projections)
        if any(qweight is None for qweight in qweights):
            continue
        device = qweights[0].device
        if (
            device.type != "cuda"
            or any(qweight.device != device or not qweight.is_contiguous() for qweight in qweights)
            or any(
                projection.scales.device != device or not projection.scales.is_contiguous()
                for projection in projections
            )
            or torch.cuda.get_device_capability(device) != (8, 0)
        ):
            continue

        projection_refs = tuple(ref(projection) for projection in projections)
        cache_key = id(q_proj)
        for projection in projections:
            projection._gptqmodel_trilin_qkv_projection_refs = projection_refs
            projection._gptqmodel_trilin_qkv_cache_key = cache_key
            projection._gptqmodel_trilin_qkv_device = device
            projection._gptqmodel_trilin_qkv_original_forward = projection.forward
        q_proj.forward = MethodType(_trilin_q_forward, q_proj)
        k_proj.forward = MethodType(_trilin_k_forward, k_proj)
        v_proj.forward = MethodType(_trilin_v_forward, v_proj)
        attention._gptqmodel_trilin_qkv = True
        installed += 1

    return installed


__all__ = ["install_trilin_3bit_qkv"]
