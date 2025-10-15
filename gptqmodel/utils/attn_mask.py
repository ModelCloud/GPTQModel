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
    """
    if mask is None:
        return None

    m = mask
    if m.dtype == torch.bool:
        keep = m
    else:
        has_negative = torch.any(m < 0)
        if has_negative:
            keep = m >= 0
        else:
            maxv = torch.amax(m)
            if m.is_floating_point():
                minv = torch.amin(m)
                approx_zero = torch.isclose(minv, torch.zeros((), device=m.device, dtype=m.dtype))
                approx_one = torch.isclose(maxv, torch.ones((), device=m.device, dtype=m.dtype))
                same_extreme = torch.isclose(maxv, minv)
                has_mid = torch.any((m > minv) & (m < maxv))
                is_binary = approx_zero and (approx_one or same_extreme) and not has_mid
            else:
                minv = torch.amin(m)
                outside = torch.any((m != 0) & (m != 1))
                is_binary = minv >= 0 and maxv <= 1 and not outside

            has_positive = torch.any(m > 0)
            if has_positive and not is_binary:
                scaled_mismatch = torch.any((m > 0) & (m != maxv))
                is_scaled_binary = not scaled_mismatch
            else:
                is_scaled_binary = False

            if not has_positive:
                keep = torch.zeros_like(m, dtype=torch.bool)
            else:
                keep = (m > 0) if (is_binary or is_scaled_binary) else (m >= 0)
    m = keep

    # Squeeze broadcast dims to reach [B, S]
    if m.dim() == 4 and m.size(1) == 1 and m.size(2) == 1:
        m = m[:, 0, 0, :]  # [B, S]
    elif m.dim() == 3 and m.size(1) == 1:
        m = m[:, 0, :]  # [B, S]
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
