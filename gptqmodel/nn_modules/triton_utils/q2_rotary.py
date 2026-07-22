# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import functools
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
_PRISM_Q2_KEY_HEADS = 8
_PRISM_Q2_HEAD_DIM = 128
_PRISM_Q2_QUERY_SHAPE = (1, _PRISM_Q2_QUERY_HEADS, 1, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_KEY_SHAPE = (1, _PRISM_Q2_KEY_HEADS, 1, _PRISM_Q2_HEAD_DIM)
_PRISM_Q2_EMBEDDING_SHAPE = (1, 1, _PRISM_Q2_HEAD_DIM)


if _TRITON_AVAILABLE:

    @triton.jit
    def _round_fp32_to_fp16_as_fp32(value):
        """Match PyTorch FP16 elementwise kernels while keeping the fused expression in one launch."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f16 rounded;
                cvt.rn.f16.f32 rounded, $1;
                cvt.f32.f16 $0, rounded;
            }
            """,
            constraints="=f,f",
            args=[value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _prism_q2_fused_rotary_kernel(
        query_ptr,
        key_ptr,
        cos_ptr,
        sin_ptr,
        query_output_ptr,
        key_output_ptr,
        QUERY_HEADS: tl.constexpr,
        KEY_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        head_index = tl.program_id(0)
        offsets = tl.arange(0, HEAD_DIM)
        is_query = head_index < QUERY_HEADS
        source_head = tl.where(is_query, head_index, head_index - QUERY_HEADS)
        source_offsets = source_head * HEAD_DIM + offsets
        partner_offsets = source_head * HEAD_DIM + (offsets + HEAD_DIM // 2) % HEAD_DIM

        query = tl.load(query_ptr + source_offsets, mask=is_query, other=0.0)
        query_partner = tl.load(query_ptr + partner_offsets, mask=is_query, other=0.0)
        key = tl.load(key_ptr + source_offsets, mask=~is_query, other=0.0)
        key_partner = tl.load(key_ptr + partner_offsets, mask=~is_query, other=0.0)
        value = tl.where(is_query, query, key)
        partner = tl.where(is_query, query_partner, key_partner)
        rotated = tl.where(offsets < HEAD_DIM // 2, -partner, partner)
        cosine = tl.load(cos_ptr + offsets)
        sine = tl.load(sin_ptr + offsets)

        product = _round_fp32_to_fp16_as_fp32(tl.cast(value, tl.float32) * tl.cast(cosine, tl.float32))
        rotated_product = _round_fp32_to_fp16_as_fp32(
            tl.cast(rotated, tl.float32) * tl.cast(sine, tl.float32)
        )
        output = _round_fp32_to_fp16_as_fp32(product + rotated_product)
        tl.store(query_output_ptr + source_offsets, output, mask=is_query)
        tl.store(key_output_ptr + source_offsets, output, mask=~is_query)


def _supports_prism_q2_rotary(
    attention_ref,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int,
    installed_device: torch.device,
) -> bool:
    attention = attention_ref()
    return (
        _TRITON_AVAILABLE
        and attention is not None
        and not attention.training
        and unsqueeze_dim == 1
        and query.dtype == torch.float16
        and key.dtype == torch.float16
        and cos.dtype == torch.float16
        and sin.dtype == torch.float16
        and query.device == installed_device
        and key.device == installed_device
        and cos.device == installed_device
        and sin.device == installed_device
        and query.shape == _PRISM_Q2_QUERY_SHAPE
        and key.shape == _PRISM_Q2_KEY_SHAPE
        and cos.shape == _PRISM_Q2_EMBEDDING_SHAPE
        and sin.shape == _PRISM_Q2_EMBEDDING_SHAPE
        and query.stride(1) == _PRISM_Q2_HEAD_DIM
        and key.stride(1) == _PRISM_Q2_HEAD_DIM
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )


def _prism_q2_apply_rotary_pos_emb(
    original,
    attention_ref,
    installed_device: torch.device,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _supports_prism_q2_rotary(
        attention_ref,
        query,
        key,
        cos,
        sin,
        unsqueeze_dim=unsqueeze_dim,
        installed_device=installed_device,
    ):
        return original(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)

    query_output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    key_output = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    _prism_q2_fused_rotary_kernel[(_PRISM_Q2_QUERY_HEADS + _PRISM_Q2_KEY_HEADS,)](
        query,
        key,
        cos,
        sin,
        query_output,
        key_output,
        QUERY_HEADS=_PRISM_Q2_QUERY_HEADS,
        KEY_HEADS=_PRISM_Q2_KEY_HEADS,
        HEAD_DIM=_PRISM_Q2_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return query_output, key_output


def _clone_attention_forward(attention: torch.nn.Module, *, installed_device: torch.device) -> bool:
    original_bound = attention.forward
    original_function = getattr(original_bound, "__func__", None)
    if original_function is None:
        return False
    original_apply = original_function.__globals__.get("apply_rotary_pos_emb")
    if not callable(original_apply):
        return False

    cloned_globals = dict(original_function.__globals__)
    cloned_globals["apply_rotary_pos_emb"] = functools.partial(
        _prism_q2_apply_rotary_pos_emb,
        original_apply,
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
    attention._gptqmodel_prism_q2_rotary_original_forward = original_bound
    attention._gptqmodel_prism_q2_rotary_device = installed_device
    attention.forward = MethodType(cloned, attention)
    return True


def install_prism_q2_rotary(model: torch.nn.Module) -> int:
    """Fuse exact-shape Qwen3 Q/K rotary embedding for native Prism Q2 batch-one sm80 decode."""

    config = getattr(model, "config", None)
    if (
        not _TRITON_AVAILABLE
        or getattr(config, "model_type", None) != "qwen3"
        or getattr(config, "hidden_size", None) != 2048
        or getattr(config, "num_attention_heads", None) != _PRISM_Q2_QUERY_HEADS
        or getattr(config, "num_key_value_heads", None) != _PRISM_Q2_KEY_HEADS
        or getattr(config, "head_dim", _PRISM_Q2_HEAD_DIM) != _PRISM_Q2_HEAD_DIM
    ):
        return 0

    installed = 0
    for attention in model.modules():
        if (
            attention.__class__.__name__ != "Qwen3Attention"
            or not getattr(attention, "_gptqmodel_prism_q2_qkv", False)
            or getattr(attention, "_gptqmodel_prism_q2_rotary", False)
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
        attention._gptqmodel_prism_q2_rotary = True
        installed += 1
    return installed


__all__ = ["install_prism_q2_rotary"]
