# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Scale-aware FP8 activation values passed between W4A operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class W4AActivation:
    mode: str
    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    model_dtype: torch.dtype
    global_scale: torch.Tensor | None = None
    recipe: str | None = None
    rotation_applied: bool = False

    def decode(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Decode only at an operator that requires ordinary arithmetic."""
        out_dtype = dtype or self.model_dtype
        rows, width = self.codes.shape[0], self.shape[-1]
        if self.mode == "w4afp8":
            if self.codes.dtype != torch.float8_e4m3fn or self.scales.shape != (rows,):
                raise ValueError("Malformed FP8 activation operand.")
            result = self.codes.float() * self.scales[:, None]
            return result.reshape(self.shape).to(out_dtype)
        raise ValueError(f"Unknown W4A activation mode: {self.mode}")


def pack_activation(x: torch.Tensor, mode: str, *, global_scale: torch.Tensor | None = None,
                    model_dtype: torch.dtype | None = None,
                    recipe: str | None = None,
                    rotation_applied: bool = False) -> W4AActivation:
    """Encode a model or FP32 tensor, retaining all scales needed by consumers."""
    if x.device.type != "cuda" or x.ndim < 2 or x.shape[-1] % 128:
        raise ValueError("W4A activation transport requires CUDA and width divisible by 128.")
    shape = tuple(x.shape)
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    x2 = x.contiguous().reshape(rows, width)
    model_dtype = model_dtype or (x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16)
    if model_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W4A transport needs an FP16 or BF16 model dtype.")
    if mode == "w4afp8":
        if recipe is not None:
            raise ValueError("FP8 activation transport does not use an activation recipe.")
        from .w4a_triton import fp8_pack

        codes, scales = fp8_pack(x2)
        return W4AActivation(mode, codes, scales, shape, model_dtype,
                             rotation_applied=rotation_applied)
    raise ValueError(f"Unknown W4A activation mode: {mode}")
