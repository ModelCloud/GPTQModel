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
        if m.device.type in {"cuda", "mps"}:
            has_negative = (m < 0).any()
            m = torch.where(has_negative, m >= 0, m > 0)
        elif torch.any(m < 0):
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
