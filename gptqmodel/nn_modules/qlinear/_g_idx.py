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
    *,
    allow_negative_wrap: bool = True,
) -> None:
    """Reject indices that would read scales or zero points out of bounds.

    ``allow_negative_wrap`` preserves the Triton kernel's contract: it adds
    ``num_groups`` to a negative index, so ``[-num_groups, -1]`` is in range.
    Kernels that index the table directly (ExLlamaV2) pass ``False`` and accept
    only ``[0, num_groups)``.
    """

    if g_idx is None or scales is None or g_idx.numel() == 0:
        return

    num_groups = scales.shape[0]
    g_min = int(g_idx.min())
    g_max = int(g_idx.max())
    lower = -num_groups if allow_negative_wrap else 0
    if g_min < lower or g_max >= num_groups:
        valid = f"[{-num_groups}, {num_groups - 1}]" if allow_negative_wrap else f"[0, {num_groups - 1}]"
        raise ValueError(
            f"{layer_name}: checkpoint g_idx is out of range for {num_groups} "
            f"scale group(s) (got min={g_min}, max={g_max}, valid "
            f"{valid}); the group index would read "
            "scales/qzeros out of bounds."
        )


__all__ = ["validate_g_idx_bounds"]
