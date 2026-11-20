# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch


def normalize_seq_mask(mask: torch.Tensor | None, seq_len: int | None = None) -> torch.Tensor | None:
    """
    Normalize a variety of HF attention mask formats to a boolean keep-mask [B, S].
    True = keep (attended), False = drop (padding/fully-masked).

    Accepts typical HF forms:
      - [B, S] with 1/0
      - [B, 1, 1, S] 'extended' masks with {0 or positive} keep and {negative large} masked
      - [B, 1, S] (rare)
      - [B, T, T] and [B, 1, T, T] causal masks; the query axis is collapsed
        with `any` so a token is kept when at least one query may attend to it.
    """
    if mask is None:
        return None

    m = mask
    # Convert numeric to bool 'keep' (HF tends to use >0 for keep; extended masks use big negatives for masked)
    if m.dtype != torch.bool:
        # A floating-point mask (or any rank >= 3 mask) that is entirely zero
        # is the unpadded extended/additive form: additive masks encode
        # "attend" as 0 and "masked" as a large negative bias, so an all-zero
        # mask means "no padding at all". Without this rule the mask has no
        # negative sentinel, falls through to the 0/1 inference below, and
        # drops every token -- PanGu-alpha's `(1.0 - mask) * -10000.0` layer
        # input mask hits exactly this case (#5134). A 2-D integer 0/1 mask
        # keeps the opposite meaning: an all-zero row is all padding.
        zero_is_keep = m.is_floating_point() or m.dim() >= 3
        if m.device.type in {"cuda", "mps"}:
            has_negative = (m < 0).any()
            all_zero = ~m.any()
            additive = has_negative | (all_zero & bool(zero_is_keep))
            m = torch.where(additive, m >= 0, m > 0)
        elif torch.any(m < 0) or (zero_is_keep and not bool(m.any())):
            m = m >= 0
        else:
            m = m > 0

    # Squeeze broadcast dims to reach [B, S]
    if m.dim() == 4 and m.size(1) == 1:
        if m.size(2) == 1:
            # Common HF format: [B, 1, 1, S]
            m = m[:, 0, 0, :]
        elif m.size(2) == m.size(3):
            # Causal mask expanded to [B, 1, S, S]; collapse query axis so any
            # attended position is marked for retention.
            m = m[:, 0].any(dim=1)
        else:
            # Fallback: collapse singleton dim, leave seq axis for later handling
            m = m[:, 0, 0, :]
    elif m.dim() == 3:
        if m.size(1) == 1:
            m = m[:, 0, :]  # [B, S]
        elif m.size(1) == m.size(2):
            # [B, T, T] causal mask without a singleton query dim. Collapse the
            # query axis the same way the [B, 1, T, T] branch does; flattening
            # would otherwise keep only the first query row.
            m = m.any(dim=1)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(mask.shape)}")
    elif m.dim() == 2:
        pass  # already [B, S]
    else:
        # Fallback: try to flatten to [B, S] if seq_len is known
        if seq_len is not None and m.dim() > 2 and m.size(-1) == seq_len:
            m = m.reshape(m.size(0), -1)[..., :seq_len]
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(mask.shape)}")

    return m.to(dtype=torch.bool)


def align_single_head_capture(
    x: torch.Tensor | None, keep_mask: torch.Tensor | None
) -> torch.Tensor | None:
    """Return an MLA latent capture as ``[B, S, ...]`` so a keep mask can match it.

    transformers' MLA attention expands the compressed KV latent as a
    single-head ``[B, 1, S, kv_lora_rank]`` tensor before calling ``kv_b_proj``
    (``modeling_deepseek_v2/v3``, ``glm4_moe_lite``, ``kimi_linear`` and
    siblings). Every capture hook matches a mask by its leading dims, so
    ``shape[:2]`` is ``(B, 1)`` and the padded positions of the whole layer went
    into that one projection's sample (#5389). Dropping the unit head axis
    makes the existing ``[B, S, ...]`` handling apply unchanged.

    Returns ``x`` untouched unless it is exactly a single-head 4-D capture whose
    batch and sequence dims describe ``keep_mask``; no copy is made.
    """

    if not torch.is_tensor(x) or not torch.is_tensor(keep_mask):
        return x
    if x.dim() != 4 or x.shape[1] != 1 or keep_mask.ndim != 2:
        return x
    if (int(x.shape[0]), int(x.shape[2])) != (int(keep_mask.shape[0]), int(keep_mask.shape[1])):
        return x
    return x.squeeze(1)


def apply_keep_mask_bt(x: torch.Tensor, keep_mask_bs: torch.Tensor | None) -> torch.Tensor:
    """
    Apply [B, S] keep-mask to a tensor x of shape [B, S, ...].
    Returns a flattened tensor of shape [N_kept, ...] (collapses batch/time on the kept rows).
    If keep_mask is None or x doesn't have [B, S, ...] leading dims, returns x unchanged.
    """
    if keep_mask_bs is None or x.dim() < 2:
        return x

    x = align_single_head_capture(x, keep_mask_bs)
    B, S = x.size(0), x.size(1)
    if keep_mask_bs.shape != (B, S):
        raise AssertionError(f"Mask shape {keep_mask_bs.shape} does not match leading dims {(B, S)} of x={tuple(x.shape)}")

    # Concatenate variable-length selections per batch along the sequence axis:
    kept_rows = [x[b, keep_mask_bs[b]] for b in range(B)]
    if len(kept_rows) == 0:
        return x.new_zeros((0,) + x.shape[2:], dtype=x.dtype, device=x.device)
    return torch.cat(kept_rows, dim=0).contiguous()


def calibration_row_keep(input_ids, attention_mask=None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(ids, keep)`` as ``[B, S]`` tensors for one prepared calibration row.

    ``keep`` marks attended positions (``True`` when no mask is given). ``None``
    means the row carries no usable ``input_ids`` (for example embeddings-only
    rows).
    """

    if input_ids is None:
        return None
    try:
        ids = torch.as_tensor(input_ids)
    except (TypeError, ValueError, RuntimeError):
        return None
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2 or ids.shape[-1] == 0:
        return None
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        return None

    keep = None
    if attention_mask is not None:
        try:
            mask = torch.as_tensor(attention_mask)
        except (TypeError, ValueError, RuntimeError):
            mask = None
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() > 2:
                mask = normalize_seq_mask(mask, seq_len=ids.shape[-1])
            elif mask.dtype != torch.bool:
                mask = mask > 0
            if mask is not None and tuple(mask.shape) == tuple(ids.shape):
                keep = mask.to(device=ids.device, dtype=torch.bool)
    if keep is None:
        keep = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
    return ids, keep


def tokenizer_bos_token_id(tokenizer) -> int | None:
    for candidate in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        bos = getattr(candidate, "bos_token_id", None)
        if isinstance(bos, int) and not isinstance(bos, bool):
            return bos
    return None


def bos_token_positions(
    input_ids,
    attention_mask=None,
    bos_token_id: int | None = None,
) -> torch.Tensor | None:
    """Return the ``[B, S]`` attended positions that hold the BOS token.

    This identifies the existing BOS-based exclusion policy, not all possible
    attention/activation-derived sinks. It covers leading and interior document
    boundaries. ``None`` means the tokenizer has no BOS or the row contains none.
    """

    if bos_token_id is None:
        return None
    row = calibration_row_keep(input_ids, attention_mask)
    if row is None:
        return None
    ids, keep = row
    positions = (ids == int(bos_token_id)) & keep
    if not bool(positions.any()):
        return None
    return positions.cpu()


def exclude_attention_sinks(
    keep_mask: torch.Tensor | None,
    bos_positions: torch.Tensor | None = None,
    *,
    batch: int | None = None,
    seq_len: int | None = None,
    device=None,
) -> torch.Tensor | None:
    """Remove the attended BOS positions from a ``[B, S]`` calibration keep-mask.

    All attended BOS tokens are excluded, including interior document boundaries
    (issue #4393). Nothing else is removed: for BOS-less tokenizers (Qwen2/2.5/3),
    this policy deliberately does not infer a sink from the first text position
    (issue #4459). Such models can still have non-BOS positional sinks.

    ``bos_positions`` has the same ``[B, S]`` shape as the calibration batch.
    ``keep_mask`` is the padding keep-mask derived from ``attention_mask``;
    ``None`` means every position is attended, and ``batch``/``seq_len`` then
    give the replayed grid. ``keep_mask`` is returned unchanged (same object)
    when there is no BOS position to remove, or when ``bos_positions`` does not
    match the replayed grid (for example a VLM whose forward expands image
    placeholders), so such batches keep their exact statistics. Otherwise the
    result is a new tensor; the input is never mutated.
    """

    if not torch.is_tensor(bos_positions) or bos_positions.ndim != 2:
        return keep_mask
    if not bool(bos_positions.any()):
        return keep_mask
    if keep_mask is not None:
        if not torch.is_tensor(keep_mask) or keep_mask.ndim != 2:
            return keep_mask
        if tuple(bos_positions.shape) != tuple(keep_mask.shape):
            return keep_mask
        result = keep_mask.to(dtype=torch.bool)
    else:
        if batch is None or seq_len is None:
            return keep_mask
        if (int(batch), int(seq_len)) != tuple(bos_positions.shape):
            return keep_mask
        result = torch.ones((int(batch), int(seq_len)), dtype=torch.bool, device=device)
    return result & ~bos_positions.to(device=result.device, dtype=torch.bool)
