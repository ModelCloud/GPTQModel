# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""Visibility builders for the block-diffusion Qwen2 decoder (SPEC sections 6.3 and 6.4).

Every builder works on boolean visibility tables ``[B, L, K]`` (True = the query may attend
to the key). ``to_additive_mask`` turns a table into the ``[B, 1, L, K]`` 0/-inf mask the
native attention functions consume.
"""

from __future__ import annotations

import torch


def causal_visibility(
    batch_size: int,
    query_len: int,
    past_len: int,
    device,
    sliding_window: int | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    key_len = past_len + query_len
    query_pos = torch.arange(query_len, device=device)[:, None] + past_len
    key_pos = torch.arange(key_len, device=device)[None, :]
    visible = key_pos <= query_pos
    if sliding_window is not None:
        visible = visible & (key_pos > query_pos - sliding_window)
    visible = visible.unsqueeze(0).expand(batch_size, query_len, key_len)
    if attention_mask is not None:
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != (batch_size, key_len):
            raise ValueError("attention_mask must include the cached prefix")
        visible = visible & attention_mask.to(device=device).bool()[:, None, :]
    return visible.clone()


def apply_cached_window(visible: torch.Tensor, block_size: int, causal_attn: bool) -> torch.Tensor:
    """Trailing MTP window on a cached decode: bidirectional inside, blind to its predecessor."""
    visible = visible.clone()
    query_len, key_len = visible.shape[-2:]
    rows = slice(max(query_len - block_size, 0), query_len)
    if not causal_attn:
        visible[:, rows, max(key_len - block_size, 0) :] = True
    predecessor = key_len - block_size - 1
    if predecessor >= 0:
        visible[:, rows, predecessor] = False
    return visible


def window_positions(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Positions that hold MASK or are directly followed by MASK."""
    is_mask = input_ids == mask_token_id
    followed = torch.zeros_like(is_mask)
    followed[..., :-1] = is_mask[..., 1:]
    return is_mask | followed


def apply_uncached_windows(
    visible: torch.Tensor,
    input_ids: torch.Tensor,
    mask_token_id: int,
    causal_attn: bool,
) -> torch.Tensor:
    """Speculative windows inside a full (past = 0) sequence."""
    visible = visible.clone()
    length = input_ids.shape[-1]
    device = input_ids.device
    index = torch.arange(length, device=device)
    rows = index[:, None]
    for b in range(input_ids.shape[0]):
        in_window = window_positions(input_ids[b], mask_token_id)
        if not bool(in_window.any()):
            continue
        prev_real = torch.where(~in_window, index, torch.full_like(index, -1)).cummax(dim=0).values
        prev_real = prev_real.clamp_min(0)
        next_candidates = torch.where(~in_window, index, torch.full_like(index, length))
        next_real = next_candidates.flip(0).cummin(dim=0).values.flip(0)
        hidden = in_window[None, :] & (index > prev_real)[None, :] & (rows >= next_real[None, :])
        visible[b] &= ~hidden
        if not causal_attn:
            opened = in_window[None, :] & (rows > prev_real[None, :]) & (rows < index[None, :])
            visible[b] |= opened
    return visible


def training_visibility(position_ids: torch.Tensor, block_size: int, causal_attn: bool) -> torch.Tensor:
    """Clean prefix plus block-diffusion blocks, split where positions drop."""
    batch_size, length = position_ids.shape
    device = position_ids.device
    index = torch.arange(length, device=device)
    q = index[:, None]
    k = index[None, :]
    tables = []
    for b in range(batch_size):
        pos = position_ids[b]
        drops = torch.nonzero(pos[1:] < pos[:-1]).flatten()
        split = int(drops[0]) + 1 if drops.numel() else -1
        clean = index < split
        block = torch.div(index - split, block_size, rounding_mode="floor")
        block_start = (split + block * block_size).clamp(0, length - 1)
        prefix_len = pos[block_start]
        in_blocks = ~clean
        both_clean = clean[:, None] & clean[None, :] & (k <= q)
        same_block = in_blocks[:, None] & in_blocks[None, :] & (block[:, None] == block[None, :])
        if causal_attn:
            same_block = same_block & (k <= q)
        block_to_clean = in_blocks[:, None] & clean[None, :] & (k < prefix_len[:, None])
        tables.append(both_clean | same_block | block_to_clean)
    return torch.stack(tables, dim=0)


def to_additive_mask(visible: torch.Tensor, dtype: torch.dtype, build_dtype: torch.dtype | None = None) -> torch.Tensor:
    build_dtype = build_dtype or dtype
    mask = torch.zeros(visible.shape, dtype=build_dtype, device=visible.device)
    mask = mask.masked_fill(~visible, float("-inf"))
    return mask.to(dtype).unsqueeze(1)


__all__ = [
    "apply_cached_window",
    "apply_uncached_windows",
    "causal_visibility",
    "to_additive_mask",
    "training_visibility",
    "window_positions",
]
