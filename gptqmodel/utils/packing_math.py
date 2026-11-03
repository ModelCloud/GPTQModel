# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch


def scaled_zero_offset(scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """Build an affine offset for backends that store it in the scale dtype."""
    offset = (-scales.float() * zeros.float()).to(scales.dtype)
    if not torch.isfinite(offset).all():
        raise ValueError(
            f"Scaled zero-point offset is not finite in {scales.dtype}; "
            "use a wider compute dtype or a backend that stores integer zero points."
        )
    return offset.contiguous()
