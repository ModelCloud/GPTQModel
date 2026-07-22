# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import functools
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


_PRISM_Q2_HEADS = 8
_PRISM_Q2_HEAD_DIM = 128
_PRISM_Q2_CACHE_LENGTH = 84
_PRISM_Q2_STATE_SHAPE = (1, _PRISM_Q2_HEADS, 1, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_CACHE_SHAPE = (1, _PRISM_Q2_HEADS, _PRISM_Q2_CACHE_LENGTH, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_STATE_ELEMENTS = _PRISM_Q2_HEADS * _PRISM_Q2_HEAD_DIM


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_static_cache_update_kernel(
        key_state_ptr,
        value_state_ptr,
        key_cache_ptr,
        value_cache_ptr,
        cumulative_length_ptr,
        STATE_ELEMENTS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CACHE_LENGTH: tl.constexpr,
        BLOCK_ELEMENTS: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_ELEMENTS)
        is_value = offsets >= STATE_ELEMENTS
        state_offsets = offsets % STATE_ELEMENTS
        heads = state_offsets // HEAD_DIM
        dimensions = state_offsets % HEAD_DIM
        cache_position = tl.load(cumulative_length_ptr)
        cache_offsets = heads * CACHE_LENGTH * HEAD_DIM + cache_position * HEAD_DIM + dimensions

        keys = tl.load(key_state_ptr + state_offsets, mask=~is_value, other=0.0)
        values = tl.load(value_state_ptr + state_offsets, mask=is_value, other=0.0)
        tl.store(key_cache_ptr + cache_offsets, keys, mask=~is_value)
        tl.store(value_cache_ptr + cache_offsets, values, mask=is_value)
        tl.debug_barrier()
        tl.store(cumulative_length_ptr, cache_position + 1)


def _supports_prism_q2_static_cache_update(
    model_ref,
    installed_device: torch.device,
    layer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> bool:
    model = model_ref()
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    cumulative_length = getattr(layer, "cumulative_length", None)
    return (
        _TRITON_AVAILABLE
        and model is not None
        and not model.training
        and getattr(layer, "is_initialized", False)
        and key_states.shape == _PRISM_Q2_STATE_SHAPE
        and value_states.shape == _PRISM_Q2_STATE_SHAPE
        and torch.is_tensor(keys)
        and torch.is_tensor(values)
        and keys.shape == _PRISM_Q2_CACHE_SHAPE
        and values.shape == _PRISM_Q2_CACHE_SHAPE
        and torch.is_tensor(cumulative_length)
        and cumulative_length.numel() == 1
        and cumulative_length.dtype == torch.int64
        and key_states.dtype == torch.float16
        and value_states.dtype == torch.float16
        and keys.dtype == torch.float16
        and values.dtype == torch.float16
        and key_states.device == installed_device
        and value_states.device == installed_device
        and keys.device == installed_device
        and values.device == installed_device
        and cumulative_length.device == installed_device
        and key_states.is_contiguous()
        and value_states.is_contiguous()
        and keys.is_contiguous()
        and values.is_contiguous()
    )


def _prism_q2_static_cache_update(
    layer_ref,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = layer_ref()
    if layer is None:  # pragma: no cover - the callable is owned by the layer
        raise RuntimeError("Prism Q2 static-cache layer was released before its update callable.")
    original = layer._gptqmodel_prism_q2_cache_original_update
    if not _supports_prism_q2_static_cache_update(
        layer._gptqmodel_prism_q2_cache_model_ref,
        layer._gptqmodel_prism_q2_cache_device,
        layer,
        key_states,
        value_states,
    ):
        return original(layer, key_states, value_states, *args, **kwargs)

    _prism_q2_static_cache_update_kernel[(1,)](
        key_states,
        value_states,
        layer.keys,
        layer.values,
        layer.cumulative_length,
        STATE_ELEMENTS=_PRISM_Q2_STATE_ELEMENTS,
        HEAD_DIM=_PRISM_Q2_HEAD_DIM,
        CACHE_LENGTH=_PRISM_Q2_CACHE_LENGTH,
        BLOCK_ELEMENTS=2 * _PRISM_Q2_STATE_ELEMENTS,
        num_warps=8,
        num_stages=1,
    )
    return layer.keys, layer.values


def install_prism_q2_static_cache(model: torch.nn.Module, cache, *, device: torch.device) -> int:
    """Fuse exact-shape StaticCache decode updates for native Prism Q2 graph inference on sm80."""

    config = getattr(model, "config", None)
    layers = getattr(cache, "layers", None)
    prism_attention_count = sum(
        1
        for module in model.modules()
        if module.__class__.__name__ == "Qwen3Attention"
        and getattr(module, "_gptqmodel_prism_q2_qkv", False)
    )
    if (
        not _TRITON_AVAILABLE
        or getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_size", None) != 2048
        or getattr(config, "num_attention_heads", None) != 16
        or getattr(config, "num_key_value_heads", None) != _PRISM_Q2_HEADS
        or getattr(config, "head_dim", _PRISM_Q2_HEAD_DIM) != _PRISM_Q2_HEAD_DIM
        or not isinstance(device, torch.device)
        or device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (8, 0)
        or not isinstance(layers, list)
        or len(layers) != prism_attention_count
        or not layers
    ):
        return 0

    if any(
        layer.__class__.__name__ != "StaticLayer"
        or getattr(layer, "max_cache_len", None) != _PRISM_Q2_CACHE_LENGTH
        or getattr(layer, "is_sliding", False)
        or getattr(layer, "_gptqmodel_prism_q2_cache", False)
        or not callable(getattr(layer.update, "__func__", None))
        for layer in layers
    ):
        return 0

    model_reference = ref(model)
    for layer in layers:
        layer_reference = ref(layer)
        layer._gptqmodel_prism_q2_cache_original_update = layer.update.__func__
        layer._gptqmodel_prism_q2_cache_model_ref = model_reference
        layer._gptqmodel_prism_q2_cache_device = device
        layer.update = functools.partial(_prism_q2_static_cache_update, layer_reference)
        layer._gptqmodel_prism_q2_cache = True
    return len(layers)


__all__ = ["install_prism_q2_static_cache"]
