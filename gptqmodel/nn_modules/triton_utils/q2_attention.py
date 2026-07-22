# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import functools
import math
from types import FunctionType, MethodType
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


_PRISM_Q2_QUERY_HEADS = 16
_PRISM_Q2_KEY_VALUE_HEADS = 8
_PRISM_Q2_QUERY_GROUPS = 2
_PRISM_Q2_HEAD_DIM = 128
_PRISM_Q2_CACHE_LENGTH = 84
_PRISM_Q2_QUERY_SHAPE = (1, _PRISM_Q2_QUERY_HEADS, 1, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_CACHE_SHAPE = (1, _PRISM_Q2_KEY_VALUE_HEADS, _PRISM_Q2_CACHE_LENGTH, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_MASK_SHAPE = (1, 1, 1, _PRISM_Q2_CACHE_LENGTH)
_PRISM_Q2_OUTPUT_SHAPE = (1, 1, _PRISM_Q2_QUERY_HEADS, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_SCALE = _PRISM_Q2_HEAD_DIM**-0.5


if _TRITON_AVAILABLE:

    @triton.jit
    def _prism_q2_gqa_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        mask_ptr,
        output_ptr,
        CACHE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        QUERY_GROUPS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_CACHE: tl.constexpr,
    ):
        key_value_head = tl.program_id(0)
        query_rows = tl.arange(0, 16)
        dimensions = tl.arange(0, HEAD_DIM)
        cache_positions = tl.arange(0, BLOCK_CACHE)
        query_heads = key_value_head * QUERY_GROUPS + query_rows

        query_offsets = query_heads[:, None] * HEAD_DIM + dimensions[None, :]
        query = tl.load(query_ptr + query_offsets, mask=query_rows[:, None] < QUERY_GROUPS, other=0.0)
        cache_offsets = (
            key_value_head * CACHE_LENGTH * HEAD_DIM
            + cache_positions[:, None] * HEAD_DIM
            + dimensions[None, :]
        )
        key = tl.load(key_ptr + cache_offsets, mask=cache_positions[:, None] < CACHE_LENGTH, other=0.0)
        scores = tl.dot(query, tl.trans(key), out_dtype=tl.float32) * SCALE
        attention_mask = tl.load(
            mask_ptr + cache_positions,
            mask=cache_positions < CACHE_LENGTH,
            other=False,
        )
        attention_bias = tl.where(attention_mask, 0.0, -float("inf"))
        scores += attention_bias[None, :]
        maximum = tl.max(scores, 1, keep_dims=True)
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        numerator = tl.where(attention_mask[None, :], tl.exp(scores - maximum), 0.0)
        denominator = tl.sum(numerator, 1, keep_dims=True)
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        probabilities = numerator / denominator

        value = tl.load(value_ptr + cache_offsets, mask=cache_positions[:, None] < CACHE_LENGTH, other=0.0)
        output = tl.dot(probabilities.to(tl.float16), value, out_dtype=tl.float32)
        tl.store(output_ptr + query_offsets, output, mask=query_rows[:, None] < QUERY_GROUPS)


def _supports_prism_q2_gqa_attention(
    attention_ref,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float,
    scaling: float | None,
    installed_device: torch.device,
    kwargs: dict,
) -> bool:
    attention = attention_ref()
    return (
        _TRITON_AVAILABLE
        and attention is not None
        and module is attention
        and not attention.training
        and query.shape == _PRISM_Q2_QUERY_SHAPE
        and key.shape == _PRISM_Q2_CACHE_SHAPE
        and value.shape == _PRISM_Q2_CACHE_SHAPE
        and attention_mask is not None
        and attention_mask.shape == _PRISM_Q2_MASK_SHAPE
        and getattr(attention, "sliding_window", None) is None
        and dropout == 0.0
        and scaling is not None
        and math.isclose(float(scaling), _PRISM_Q2_SCALE, rel_tol=0.0, abs_tol=0.0)
        and not kwargs.get("output_attentions", False)
        and kwargs.get("position_bias") is None
        and query.dtype == torch.float16
        and key.dtype == torch.float16
        and value.dtype == torch.float16
        and attention_mask.dtype == torch.bool
        and query.device == installed_device
        and key.device == installed_device
        and value.device == installed_device
        and attention_mask.device == installed_device
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and attention_mask.is_contiguous()
    )


def _prism_q2_attention_interface(
    original,
    attention_ref,
    installed_device: torch.device,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if not _supports_prism_q2_gqa_attention(
        attention_ref,
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        installed_device=installed_device,
        kwargs=kwargs,
    ):
        return original(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    output = torch.empty(_PRISM_Q2_OUTPUT_SHAPE, device=query.device, dtype=query.dtype)
    _prism_q2_gqa_attention_kernel[(_PRISM_Q2_KEY_VALUE_HEADS,)](
        query,
        key,
        value,
        attention_mask,
        output,
        CACHE_LENGTH=_PRISM_Q2_CACHE_LENGTH,
        HEAD_DIM=_PRISM_Q2_HEAD_DIM,
        QUERY_GROUPS=_PRISM_Q2_QUERY_GROUPS,
        SCALE=_PRISM_Q2_SCALE,
        BLOCK_CACHE=128,
        num_warps=8,
        num_stages=1,
    )
    return output, None


class _PrismQ2AttentionInterfaces:
    def __init__(self, original, attention_ref, installed_device: torch.device):
        self._original = original
        self._attention_ref = attention_ref
        self._installed_device = installed_device
        self._cached_key = None
        self._cached_fallback = None
        self._cached_interface = None

    def get_interface(self, key, fallback):
        if key == self._cached_key and fallback is self._cached_fallback:
            return self._cached_interface
        original_interface = self._original.get_interface(key, fallback)
        interface = functools.partial(
            _prism_q2_attention_interface,
            original_interface,
            self._attention_ref,
            self._installed_device,
        )
        self._cached_key = key
        self._cached_fallback = fallback
        self._cached_interface = interface
        return interface


def _clone_attention_forward(attention: torch.nn.Module, *, installed_device: torch.device) -> bool:
    original_bound = attention.forward
    original_function = getattr(original_bound, "__func__", None)
    if original_function is None:
        return False
    original_interfaces = original_function.__globals__.get("ALL_ATTENTION_FUNCTIONS")
    if not callable(getattr(original_interfaces, "get_interface", None)):
        return False

    cloned_globals = dict(original_function.__globals__)
    cloned_globals["ALL_ATTENTION_FUNCTIONS"] = _PrismQ2AttentionInterfaces(
        original_interfaces,
        ref(attention),
        installed_device,
    )
    cloned = FunctionType(
        original_function.__code__,
        cloned_globals,
        name=original_function.__name__,
        argdefs=original_function.__defaults__,
        closure=original_function.__closure__,
    )
    cloned.__kwdefaults__ = original_function.__kwdefaults__
    cloned.__annotations__ = original_function.__annotations__
    functools.update_wrapper(cloned, original_function)
    attention._gptqmodel_prism_q2_attention_original_forward = original_bound
    attention._gptqmodel_prism_q2_attention_device = installed_device
    attention.forward = MethodType(cloned, attention)
    return True


def install_prism_q2_gqa_attention(model: torch.nn.Module) -> int:
    """Fuse exact-shape Qwen3 grouped-query decode attention for native Prism Q2 on sm80."""

    config = getattr(model, "config", None)
    if (
        not _TRITON_AVAILABLE
        or getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_size", None) != 2048
        or getattr(config, "num_attention_heads", None) != _PRISM_Q2_QUERY_HEADS
        or getattr(config, "num_key_value_heads", None) != _PRISM_Q2_KEY_VALUE_HEADS
        or getattr(config, "head_dim", _PRISM_Q2_HEAD_DIM) != _PRISM_Q2_HEAD_DIM
    ):
        return 0

    installed = 0
    for attention in model.modules():
        if (
            attention.__class__.__name__ != "Qwen3Attention"
            or not getattr(attention, "_gptqmodel_prism_q2_qkv", False)
            or getattr(attention, "_gptqmodel_prism_q2_attention", False)
        ):
            continue
        q_proj = getattr(attention, "q_proj", None)
        device = getattr(q_proj, "_gptqmodel_prism_q2_qkv_device", None)
        if not isinstance(device, torch.device) or device.type != "cuda":
            continue
        if torch.cuda.get_device_capability(device) != (8, 0):
            continue
        if not _clone_attention_forward(attention, installed_device=device):
            continue
        attention._gptqmodel_prism_q2_attention = True
        installed += 1
    return installed


__all__ = ["install_prism_q2_gqa_attention"]
