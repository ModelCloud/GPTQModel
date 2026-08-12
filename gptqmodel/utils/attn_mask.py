# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import cast

import torch


def _as_tensor(value, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular tensor or sequence.") from exc


def _ragged_rows(value, *, name: str) -> list[torch.Tensor]:
    """Normalize a Python ragged batch into validated one-dimensional rows."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a rectangular tensor or sequence.")

    rows = []
    for row in value:
        if isinstance(row, torch.Tensor):
            row_tensor = row
        elif isinstance(row, (list, tuple)):
            row_tensor = _as_tensor(row, name=name)
        else:
            raise ValueError(f"{name} must be a rectangular tensor or sequence.")
        if row_tensor.ndim != 1:
            raise ValueError(f"{name} ragged rows must be one-dimensional sequences.")
        rows.append(row_tensor)
    return rows


def normalize_seq_mask(mask: torch.Tensor | None, seq_len: int | None = None) -> torch.Tensor | None:
    """
    Normalize a variety of HF attention mask formats to a boolean keep-mask [B, S].
    True = keep (attended), False = drop (padding/fully-masked).

    Accepts typical HF forms:
      - [B, S] with 1/0
      - [B, 1, 1, S] 'extended' masks with {0 or positive} keep and {negative large} masked
      - [B, 1, S] (rare)
    """
    if mask is None:
        return None

    m = mask
    if m.ndim == 0:
        raise ValueError("Unsupported scalar attention_mask.")
    if m.ndim == 1:
        m = m.unsqueeze(0)

    if m.is_floating_point() and torch.isnan(m).any():
        raise ValueError("attention_mask must not contain NaN values.")

    # Binary masks use positive=keep. HF additive/extended masks use zero for
    # allowed positions and a negative bias (usually -inf) for masked positions.
    # An all-zero floating mask with broadcast/query axes is therefore an
    # unmasked additive mask, not an all-padding binary mask.
    if m.dtype == torch.bool:
        keep = m
    else:
        has_negative = bool(torch.any(m < 0))
        extended_all_nonpositive = m.is_floating_point() and m.ndim > 2 and not bool(torch.any(m > 0))
        keep = m >= 0 if has_negative or extended_all_nonpositive else m > 0

    if keep.ndim > 2:
        # Treat the last axis as key/token position and union every broadcast,
        # head, and causal-query axis. Taking the first causal query would keep
        # only token zero and silently discard the rest of the sequence.
        normalized = keep.reshape(keep.shape[0], -1, keep.shape[-1]).any(dim=1)
    else:
        normalized = keep

    if seq_len is not None and normalized.shape[-1] != seq_len:
        raise ValueError(
            f"attention_mask sequence width {normalized.shape[-1]} does not match expected seq_len {seq_len}."
        )
    return normalized.to(dtype=torch.bool)


def attention_mask_sequence_lengths(
    mask,
    seq_len: int | None = None,
    batch_size: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Return one valid-token count per sequence using normalized mask semantics."""

    if mask is None:
        return []
    if seq_len is not None and sequence_lengths is not None:
        raise ValueError("Specify either seq_len or sequence_lengths, not both.")
    try:
        mask_tensor = _as_tensor(mask, name="attention_mask")
    except ValueError:
        rows = _ragged_rows(mask, name="attention_mask")
        if batch_size is not None and len(rows) != batch_size:
            raise ValueError(
                f"attention_mask batch size {len(rows)} does not match input_ids batch size {batch_size}."
            )
        if sequence_lengths is not None and len(rows) != len(sequence_lengths):
            raise ValueError(
                f"attention_mask batch size {len(rows)} does not match input_ids batch size {len(sequence_lengths)}."
            )
        lengths = []
        for index, row in enumerate(rows):
            expected_length = sequence_lengths[index] if sequence_lengths is not None else seq_len
            keep = cast(torch.Tensor, normalize_seq_mask(row, seq_len=expected_length))
            lengths.extend(int(length) for length in keep.sum(dim=1, dtype=torch.int64).tolist())
        return lengths

    keep = cast(torch.Tensor, normalize_seq_mask(mask_tensor, seq_len=seq_len))
    if batch_size is not None and keep.shape[0] != batch_size:
        raise ValueError(
            f"attention_mask batch size {keep.shape[0]} does not match input_ids batch size {batch_size}."
        )
    if sequence_lengths is not None:
        if keep.shape[0] != len(sequence_lengths):
            raise ValueError(
                f"attention_mask batch size {keep.shape[0]} does not match input_ids batch size "
                f"{len(sequence_lengths)}."
            )
        mismatched_row = next(
            (index for index, expected in enumerate(sequence_lengths) if keep.shape[1] != expected),
            None,
        )
        if mismatched_row is not None:
            raise ValueError(
                f"attention_mask sequence width {keep.shape[1]} does not match input_ids row "
                f"{mismatched_row} length {sequence_lengths[mismatched_row]}."
            )
    return [int(length) for length in keep.sum(dim=1, dtype=torch.int64).tolist()]


def input_id_sequence_lengths(input_ids) -> list[int]:
    """Return one physical token-position count per unbatched or batched input-id row."""

    if input_ids is None:
        return []
    try:
        ids = _as_tensor(input_ids, name="input_ids")
    except ValueError:
        return [int(row.numel()) for row in _ragged_rows(input_ids, name="input_ids")]
    if ids.ndim == 0:
        raise ValueError("input_ids must have at least one dimension.")
    if ids.ndim == 1:
        return [int(ids.numel())]
    if ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [S] or [B, S], got {tuple(ids.shape)}.")
    return [int(ids.shape[1])] * int(ids.shape[0])


def apply_keep_mask_bt(x: torch.Tensor, keep_mask_bs: torch.Tensor | None) -> torch.Tensor:
    """
    Apply [B, S] keep-mask to a tensor x of shape [B, S, ...].
    Returns a flattened tensor of shape [N_kept, ...] (collapses batch/time on the kept rows).
    If keep_mask is None or x doesn't have [B, S, ...] leading dims, returns x unchanged.
    """
    if keep_mask_bs is None or x.dim() < 2:
        return x

    B, S = x.size(0), x.size(1)
    if keep_mask_bs.shape != (B, S):
        raise AssertionError(f"Mask shape {keep_mask_bs.shape} does not match leading dims {(B, S)} of x={tuple(x.shape)}")

    # Concatenate variable-length selections per batch along the sequence axis:
    kept_rows = [x[b, keep_mask_bs[b]] for b in range(B)]
    if len(kept_rows) == 0:
        return x.new_zeros((0,) + x.shape[2:], dtype=x.dtype, device=x.device)
    return torch.cat(kept_rows, dim=0).contiguous()
