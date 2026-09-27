# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent Torch replay of token-scaled E4M3 activations."""

import torch

FP8_E4M3_MAX = 448.0


def fp8_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """Round each Linear input row to E4M3 with its own dynamic scale."""
    if x.shape[-1] == 0 or x.numel() == 0:
        return x
    x32 = x.float()
    if not bool(torch.isfinite(x32).all()):
        raise ValueError("W4AFP8 Linear input contains NaN or infinity.")
    amax = x32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
    rounded = (x32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return (rounded.float() * scale).to(x.dtype)
