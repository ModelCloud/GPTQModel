# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Scale-aware FP8/NVFP4 activation values passed between W4A operators."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    # Optional outer token multiplier. FP4 codes and hardware block scales
    # stay unchanged through scalar operations such as a fused RMSNorm.
    token_scale: torch.Tensor | None = None
    # Exact compute-dtype value this operand was packed from. Hardware GEMMs
    # always consume `codes`/`scales`; non-GEMM math (residual adds, RMSNorm,
    # the unquantized head) uses `exact()` so the residual stream is never
    # carried at 4-bit precision.
    reference: torch.Tensor | None = None

    def rescale_tokens(self, factor: torch.Tensor) -> "W4AActivation":
        rows = self.codes.shape[0]
        if self.mode != "w4a_nvfp4" or factor.shape != (rows,):
            raise ValueError("Token rescaling requires an NVFP4 carrier and one scalar per token.")
        if factor.device != self.codes.device or factor.dtype != torch.float32:
            raise ValueError("Token multipliers must be FP32 on the carrier device.")
        if not bool(torch.isfinite(factor).all()):
            raise ValueError("Token multipliers must be finite.")
        combined = factor if self.token_scale is None else self.token_scale * factor
        reference = self.reference
        if reference is not None:
            width = self.shape[-1]
            reference = (reference.reshape(rows, width).float() * factor[:, None])
            reference = reference.to(self.model_dtype).reshape(self.shape)
        return replace(self, token_scale=combined.contiguous(), reference=reference)

    def to(self, device: torch.device | str | int, non_blocking: bool = False) -> "W4AActivation":
        """Move every device-resident field so placement hooks can re-home the carrier.

        ``accelerate.utils.send_to_device`` only relocates arguments that expose
        ``to``. A checkpoint sharded across several devices hands this operand
        to the next device's layer, so the carrier has to follow the module
        placement exactly like a plain tensor. Codes, scales, and the optional
        global/token multipliers always move together to keep the operand
        internally consistent on the destination device.
        """
        target = torch.device(device)
        tensors = (self.codes, self.scales, self.global_scale, self.token_scale, self.reference)
        if all(value is None or value.device == target for value in tensors):
            return self
        moved = {
            "codes": self.codes.to(target, non_blocking=non_blocking),
            "scales": self.scales.to(target, non_blocking=non_blocking),
        }
        for name in ("global_scale", "token_scale", "reference"):
            value = getattr(self, name)
            if value is not None:
                moved[name] = value.to(target, non_blocking=non_blocking)
        return replace(self, **moved)

    def exact(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return the compute-dtype value the hardware operand was packed from.

        The residual stream, RMSNorm inputs, and the unquantized head must see
        the same values a BF16 deployment would. Callers that genuinely need
        the hardware rounding (audits, replay) keep using ``decode``.
        """
        if self.reference is None:
            return self.decode(dtype)
        out_dtype = dtype or self.model_dtype
        return self.reference.to(out_dtype)

    def decode(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Decode only at an operator that requires ordinary arithmetic."""
        out_dtype = dtype or self.model_dtype
        rows, width = self.codes.shape[0], self.shape[-1]
        if self.mode == "w4afp8":
            if self.codes.dtype != torch.float8_e4m3fn or self.scales.shape != (rows,):
                raise ValueError("Malformed FP8 activation operand.")
            result = self.codes.float() * self.scales[:, None]
            return result.reshape(self.shape).to(out_dtype)
        if self.mode == "w4a_nvfp4":
            if self.global_scale is None or self.codes.dtype != torch.float4_e2m1fn_x2:
                raise ValueError("Malformed NVFP4 activation operand.")
            from .w4a_nvfp4_triton import nvfp4_decode
            result = nvfp4_decode(
                self.codes, self.scales, self.global_scale, width,
                torch.float32 if self.token_scale is not None else out_dtype,
            )
            if self.token_scale is not None:
                result = result * self.token_scale[:, None]
            return result.reshape(self.shape).to(out_dtype)
        raise ValueError(f"Unknown W4A activation mode: {self.mode}")


def pack_activation(x: torch.Tensor, mode: str, *, global_scale: torch.Tensor | None = None,
                    model_dtype: torch.dtype | None = None,
                    recipe: str | None = None,
                    rotation_applied: bool = False,
                    reference: torch.Tensor | None = None) -> W4AActivation:
    """Encode a model or FP32 tensor, retaining all scales needed by consumers."""
    if x.device.type != "cuda" or x.ndim < 2 or x.shape[-1] % 128:
        raise ValueError("W4A activation transport requires CUDA and width divisible by 128.")
    shape = tuple(x.shape)
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    x2 = x.contiguous().reshape(rows, width)
    model_dtype = model_dtype or (x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16)
    if model_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W4A transport needs an FP16 or BF16 model dtype.")
    if reference is not None:
        if tuple(reference.shape) != shape or reference.device != x.device:
            raise ValueError("A W4A reference must match the packed shape and device.")
    if mode == "w4afp8":
        if recipe is not None:
            raise ValueError("FP8 activation transport does not use an NVFP4 recipe.")
        from .w4a_triton import fp8_pack

        codes, scales = fp8_pack(x2)
        return W4AActivation(mode, codes, scales, shape, model_dtype,
                             rotation_applied=rotation_applied, reference=reference)
    if mode == "w4a_nvfp4":
        from ...quantization.activation_floatx import normalize_nvfp4_recipe, nvfp4_global_scale
        from .w4a_nvfp4_triton import nvfp4_pack_and_swizzle
        recipe = normalize_nvfp4_recipe(recipe or "least_squares")
        if global_scale is None:
            if rows:
                global_scale = nvfp4_global_scale(
                    x2.abs().amax(), grid_dtype=x2.dtype, recipe=recipe or "least_squares"
                )
            else:
                global_scale = torch.ones((), device=x.device, dtype=torch.float32)
        else:
            global_scale = global_scale.to(device=x.device, dtype=torch.float32)
        if global_scale.numel() != 1 or not bool(torch.isfinite(global_scale)) or not bool(global_scale > 0):
            raise ValueError("NVFP4 activation global scale must be one positive finite FP32 scalar.")
        recipe = recipe or "least_squares"
        if recipe not in {
            "nvidia", "nvidia_headroom", "four_six", "least_squares", "least_squares_headroom", "least_squares_grid"
        }:
            raise ValueError(f"Unsupported NVFP4 scale recipe: {recipe}.")
        codes, scales = nvfp4_pack_and_swizzle(x2, global_scale, recipe=recipe)
        return W4AActivation(
            mode, codes, scales, shape, model_dtype, global_scale, recipe,
            rotation_applied, reference=reference,
        )
    raise ValueError(f"Unknown W4A activation mode: {mode}")
