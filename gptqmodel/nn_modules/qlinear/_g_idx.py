# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Validation for GPTQ activation-order group indices."""

from __future__ import annotations

from typing import Optional

import torch


def validate_g_idx_bounds(
    g_idx: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    layer_name: str = "GPTQ",
) -> None:
    """Reject indices that would read scales or zero points out of bounds."""

    if g_idx is None or scales is None or g_idx.numel() == 0:
        return

    num_groups = scales.shape[0]
    g_min = int(g_idx.min())
    g_max = int(g_idx.max())
    if g_min < -num_groups or g_max >= num_groups:
        raise ValueError(
            f"{layer_name}: checkpoint g_idx is out of range for {num_groups} "
            f"scale group(s) (got min={g_min}, max={g_max}, valid "
            f"[{-num_groups}, {num_groups - 1}]); the group index would read "
            "scales/qzeros out of bounds."
        )


__all__ = ["validate_g_idx_bounds"]
