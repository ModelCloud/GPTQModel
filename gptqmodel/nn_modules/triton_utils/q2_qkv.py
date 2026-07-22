# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import MethodType
from weakref import ref

import torch


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-minimal environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_PRISM_Q2_Q_FEATURES = 2048
_PRISM_Q2_KV_FEATURES = 1024
_PRISM_Q2_INPUT_FEATURES = 2048
_PRISM_Q2_QKV_CACHE_ATTRIBUTE = "_gptqmodel_prism_q2_qkv_cache"


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_qkv_gemv_kernel(
        input_ptr,
        q_qweight_ptr,
        q_scale_ptr,
        k_qweight_ptr,
        k_scale_ptr,
        v_qweight_ptr,
        v_scale_ptr,
        output_ptr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
        Q_SIZE: tl.constexpr,
        KV_SIZE: tl.constexpr,
        q_stride_qn,
        q_stride_sb,
        q_stride_sn,
        k_stride_qn,
        k_stride_sb,
        k_stride_sn,
        v_stride_qn,
        v_stride_sb,
        v_stride_sn,
    ):
        output_index = tl.program_id(0)
        block_offsets = tl.arange(0, 128)
        byte_offsets = tl.arange(0, 32)
        code_shifts = tl.arange(0, 4) * 2
        accumulator = tl.zeros((128,), dtype=tl.float32)

        for block_index in range(0, NUM_BLOCKS):
            activation = tl.load(input_ptr + block_index * 128 + block_offsets)
            if output_index < Q_SIZE:
                projection_index = output_index
                packed = tl.load(
                    q_qweight_ptr
                    + projection_index * q_stride_qn
                    + block_index * BLOCK_BYTES
                    + 2
                    + byte_offsets
                )
                scale = tl.load(q_scale_ptr + block_index * q_stride_sb + projection_index * q_stride_sn)
            elif output_index < Q_SIZE + KV_SIZE:
                projection_index = output_index - Q_SIZE
                packed = tl.load(
                    k_qweight_ptr
                    + projection_index * k_stride_qn
                    + block_index * BLOCK_BYTES
                    + 2
                    + byte_offsets
                )
                scale = tl.load(k_scale_ptr + block_index * k_stride_sb + projection_index * k_stride_sn)
            else:
                projection_index = output_index - Q_SIZE - KV_SIZE
                packed = tl.load(
                    v_qweight_ptr
                    + projection_index * v_stride_qn
                    + block_index * BLOCK_BYTES
                    + 2
                    + byte_offsets
                )
                scale = tl.load(v_scale_ptr + block_index * v_stride_sb + projection_index * v_stride_sn)

            codes = (packed[:, None] >> code_shifts[None, :]) & 0x03
            values = tl.reshape(codes, (128,))
            weight = (tl.cast(values, tl.float16) - 1.0) * scale
            accumulator += tl.cast(activation * weight, tl.float32)

        tl.store(output_ptr + output_index, tl.sum(accumulator, axis=0))


def _launch_prism_q2_qkv(
    hidden_states: torch.Tensor,
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    v_proj: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    q_scale = q_proj._get_q2_native_scale(values.device)
    k_scale = k_proj._get_q2_native_scale(values.device)
    v_scale = v_proj._get_q2_native_scale(values.device)
    total_features = q_proj.out_features + k_proj.out_features + v_proj.out_features
    output = torch.empty((1, total_features), device=values.device, dtype=values.dtype)
    _prism_q2_qkv_gemv_kernel[(total_features,)](
        values,
        q_proj.qweight,
        q_scale,
        k_proj.qweight,
        k_scale,
        v_proj.qweight,
        v_scale,
        output,
        NUM_BLOCKS=q_scale.shape[0],
        BLOCK_BYTES=q_proj.gguf_type_size,
        Q_SIZE=q_proj.out_features,
        KV_SIZE=k_proj.out_features,
        q_stride_qn=q_proj.qweight.stride(0),
        q_stride_sb=q_scale.stride(0),
        q_stride_sn=q_scale.stride(1),
        k_stride_qn=k_proj.qweight.stride(0),
        k_stride_sb=k_scale.stride(0),
        k_stride_sn=k_scale.stride(1),
        v_stride_qn=v_proj.qweight.stride(0),
        v_stride_sb=v_scale.stride(0),
        v_stride_sn=v_scale.stride(1),
        num_warps=2,
        num_stages=1,
    )
    q_end = q_proj.out_features
    k_end = q_end + k_proj.out_features
    original_shape = hidden_states.shape[:-1]
    return (
        output[:, :q_end].reshape(*original_shape, q_proj.out_features),
        output[:, q_end:k_end].reshape(*original_shape, k_proj.out_features),
        output[:, k_end:].reshape(*original_shape, v_proj.out_features),
    )


def _projection_group(module: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module] | None:
    projections = tuple(projection_ref() for projection_ref in module._gptqmodel_prism_q2_qkv_projection_refs)
    if any(projection is None for projection in projections):
        return None
    return projections


def _prism_q2_q_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    projections = _projection_group(module)
    if (
        projections is None
        or module.training
        or hidden_states.dtype != torch.float16
        or hidden_states.device != module._gptqmodel_prism_q2_qkv_device
        or hidden_states.shape[-1] != _PRISM_Q2_INPUT_FEATURES
        or hidden_states.numel() != _PRISM_Q2_INPUT_FEATURES
    ):
        return module._gptqmodel_prism_q2_qkv_original_forward(hidden_states)

    q_output, k_output, v_output = _launch_prism_q2_qkv(hidden_states, *projections)
    try:
        setattr(
            hidden_states,
            _PRISM_Q2_QKV_CACHE_ATTRIBUTE,
            (module._gptqmodel_prism_q2_qkv_cache_key, k_output, v_output),
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
    cache = getattr(hidden_states, _PRISM_Q2_QKV_CACHE_ATTRIBUTE, None)
    if cache is None or cache[0] != module._gptqmodel_prism_q2_qkv_cache_key:
        return module._gptqmodel_prism_q2_qkv_original_forward(hidden_states)
    if clear:
        try:
            setattr(hidden_states, _PRISM_Q2_QKV_CACHE_ATTRIBUTE, None)
        except (AttributeError, RuntimeError):
            pass
    return cache[output_index]


def _prism_q2_k_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return _cached_projection(module, hidden_states, output_index=1, clear=False)


def _prism_q2_v_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return _cached_projection(module, hidden_states, output_index=2, clear=True)


def install_prism_q2_qkv(model: torch.nn.Module) -> int:
    """Fuse exact-shape Qwen3 Q/K/V Q2 GEMVs for batch-one sm80 decode."""

    if not _TRITON_AVAILABLE or getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
        return 0

    from ..qlinear.gguf_triton import GGUFTritonKernel

    installed = 0
    for attention in model.modules():
        if attention.__class__.__name__ != "Qwen3Attention" or getattr(
            attention, "_gptqmodel_prism_q2_qkv", False
        ):
            continue

        projections = tuple(getattr(attention, name, None) for name in ("q_proj", "k_proj", "v_proj"))
        if not all(isinstance(projection, GGUFTritonKernel) for projection in projections):
            continue
        q_proj, k_proj, v_proj = projections
        if any(projection.gguf_tensor_qtype != "Q2_0" for projection in projections):
            continue
        if any(projection.bias is not None or projection.adapter for projection in projections):
            continue
        if (q_proj.in_features, q_proj.out_features) != (_PRISM_Q2_INPUT_FEATURES, _PRISM_Q2_Q_FEATURES):
            continue
        if any(
            (projection.in_features, projection.out_features)
            != (_PRISM_Q2_INPUT_FEATURES, _PRISM_Q2_KV_FEATURES)
            for projection in (k_proj, v_proj)
        ):
            continue

        device = q_proj.qweight.device
        if device.type != "cuda" or any(projection.qweight.device != device for projection in projections[1:]):
            continue
        if torch.cuda.get_device_capability(device) != (8, 0):
            continue

        projection_refs = tuple(ref(projection) for projection in projections)
        cache_key = id(q_proj)
        for projection in projections:
            projection._gptqmodel_prism_q2_qkv_projection_refs = projection_refs
            projection._gptqmodel_prism_q2_qkv_cache_key = cache_key
            projection._gptqmodel_prism_q2_qkv_device = device
            projection._gptqmodel_prism_q2_qkv_original_forward = projection.forward
        q_proj.forward = MethodType(_prism_q2_q_forward, q_proj)
        k_proj.forward = MethodType(_prism_q2_k_forward, k_proj)
        v_proj.forward = MethodType(_prism_q2_v_forward, v_proj)
        attention._gptqmodel_prism_q2_qkv = True
        installed += 1

    return installed


__all__ = ["install_prism_q2_qkv"]
