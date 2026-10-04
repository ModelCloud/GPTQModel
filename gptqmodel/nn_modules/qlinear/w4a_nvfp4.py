# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 activation operands for native GPTQ INT4 weight checkpoints.

This backend uses two exact E2M1 planes for centered INT4 codes. The planes
are runtime caches; the saved checkpoint retains native GPTQ qweight.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.activation_floatx import _nvfp4_mse_quantize_blocks, nvfp4_global_scale
from ...utils.backend import BACKEND
from . import PackableQuantLinear
from .w4a_floatx import W4AFP8Linear


def _fp4_codes(value: torch.Tensor) -> torch.Tensor:
    """Round to E2M1, nearest even; return the four bit code per scalar."""
    magnitude = value.float().abs().clamp(max=6.0)
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary, index, ties_up in (
        (0.25, 1, False), (0.75, 2, True), (1.25, 3, False),
        (1.75, 4, True), (2.5, 5, False), (3.5, 6, True), (5.0, 7, False),
    ):
        code = torch.where(magnitude >= boundary if ties_up else magnitude > boundary, index, code)
    return code | ((value < 0).to(torch.uint8) << 3)


def _fp4_integer_codes(value: torch.Tensor) -> torch.Tensor:
    """Encode exactly representable integer planes without floating rounding."""
    mapping = torch.tensor((6, 5, 4, 2, 0, 2, 4, 5, 6), dtype=torch.uint8, device=value.device)
    code = mapping[(value + 4).long()]
    return code | ((value < 0).to(torch.uint8) << 3)


def _pack_k(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-2] % 2:
        raise ValueError("FP4 K dimension must be even.")
    packed = (codes[..., 0::2, :] | (codes[..., 1::2, :] << 4)).contiguous()
    return packed.view(torch.float4_e2m1fn_x2)


def _pack_last(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("FP4 K dimension must be even.")
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed.view(torch.float4_e2m1fn_x2)


def _swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Map a [rows, K/16] E4M3 scale matrix to NVIDIA's 32x4x4 layout."""
    rows, cols = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    padded = torch.zeros((padded_rows, padded_cols), device=scales.device, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    return (
        padded.view(padded_rows // 128, 128, padded_cols // 4, 4)
        .permute(0, 2, 1, 3).reshape(-1, 4, 32, 4)
        .transpose(1, 2).reshape(-1, 32, 16).flatten().contiguous()
    )


def nvfp4_input(x: torch.Tensor, global_scale: torch.Tensor,
                recipe: str = "least_squares") -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-16 E4M3 scale optimization and packed E2M1 input."""
    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("NVFP4 input expects [tokens, K] with K divisible by 128.")
    blocks = x.float().reshape(x.shape[0], x.shape[1] // 16, 16)
    local, _values = _nvfp4_mse_quantize_blocks(blocks, global_scale.float(), recipe)
    local = local.squeeze(-1)
    decoded = local.float() * global_scale.float()
    codes = _fp4_codes(blocks / decoded[..., None]).reshape_as(x)
    return _pack_last(codes), local


class W4ANVFP4Linear(W4AFP8Linear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_W4A_NVFP4]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    QUANT_TYPE = "w4a_nvfp4"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            from . import w4a_nvfp4_triton  # noqa: F401
        except (ImportError, OSError) as exc:
            return False, RuntimeError(f"W4ANVFP4 needs Triton for activation packing: {exc}")
        if not hasattr(torch, "float4_e2m1fn_x2"):
            return False, RuntimeError("W4ANVFP4 requires native PyTorch FP4 support.")
        if not all(hasattr(torch.nn.functional, name) for name in ("scaled_mm", "ScalingType", "SwizzleType")):
            return False, RuntimeError("W4ANVFP4 requires PyTorch block-scaled FP4 GEMM support.")
        return True, None

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = kwargs.pop("backend", BACKEND.GPTQ_W4A_NVFP4)
        super().__init__(*args, **kwargs)
        # The generic loader casts floating checkpoint buffers to its chosen
        # model dtype. Store the FP32 calibration scalar as exact INT32 bits.
        self.register_buffer("activation_global_scale_bits", torch.ones((), dtype=torch.float32).view(torch.int32))
        self.register_buffer("_weight_both", torch.empty(0, dtype=torch.float4_e2m1fn_x2), persistent=False)
        self.register_buffer("_unit_weight_scales", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)
        # Mixed-precision stream: an attention projection consumes an FP8
        # carrier while the MLP keeps NVFP4. Both operands describe the same
        # native GPTQ INT4 weights, so only the staging differs.
        self.register_buffer("_weight_e4m3", torch.empty(0, dtype=torch.float8_e4m3fn), persistent=False)

    @property
    def activation_global_scale(self) -> torch.Tensor:
        return self.activation_global_scale_bits.view(torch.float32)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._weight_both = self._weight_both.new_empty(0)
        self._unit_weight_scales = self._unit_weight_scales.new_empty(0)
        self._weight_e4m3 = self._weight_e4m3.new_empty(0)

    @torch.no_grad()
    def post_init(self):
        if not bool(torch.isfinite(self.activation_global_scale).all()) or not bool((self.activation_global_scale > 0).all()):
            raise ValueError("NVFP4 calibrated global scale must be positive and finite.")
        PackableQuantLinear.post_init(self)
        centered = self._centered_int4_codes()
        self._weight_e4m3 = centered.to(torch.float8_e4m3fn).contiguous()
        high = torch.floor((centered + 2.0) / 4.0)
        low = centered - 4.0 * high
        low_bytes = _pack_k(_fp4_integer_codes(low)).view(torch.uint8)
        high_bytes = _pack_k(_fp4_integer_codes(high)).view(torch.uint8)
        both_bytes = torch.cat((low_bytes, high_bytes), dim=1).T.contiguous().T
        self._weight_both = both_bytes.view(torch.float4_e2m1fn_x2)
        unit = torch.ones((self.out_features * 2, 8), device=centered.device, dtype=torch.float8_e4m3fn)
        self._unit_weight_scales = _swizzle_scales(unit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_both.numel() == 0:
            raise RuntimeError("NVFP4 weight cache is absent; call post_init() after loading or packing.")
        from .w4a_activation import W4AActivation, pack_activation
        from .w4a_nvfp4_triton import nvfp4_accumulate_group, nvfp4_pack_and_swizzle

        encoded = isinstance(x, W4AActivation)
        if not encoded and getattr(self, "_require_activation_stream", False):
            raise TypeError("This W4ANVFP4 Linear requires encoded FP4 input with block scales.")
        if encoded and x.mode == "w4afp8":
            return self._forward_fp8_operand(x)
        if encoded:
            if x.mode != "w4a_nvfp4" or x.shape[-1] != self.in_features or x.global_scale is None:
                raise ValueError("NVFP4 requires a matching scale-aware activation operand.")
            needs_rotation = bool(self.online_full_had or self.online_partial_had)
            if x.rotation_applied and not needs_rotation:
                raise ValueError("This NVFP4 operand was rotated for a different Linear contract.")
            if needs_rotation and not x.rotation_applied:
                rotated = self._apply_rotation_to_input(x.decode(torch.float32))
                x = pack_activation(
                    rotated, "w4a_nvfp4",
                    global_scale=(self.activation_global_scale
                                  if x.recipe in {"nvidia_headroom", "least_squares_headroom"}
                                  else None),
                    model_dtype=x.model_dtype, recipe=x.recipe
                )
            original_shape = x.shape[:-1] + (self.out_features,)
            rows = x.codes.shape[0]
            packed, activation_scales, global_scale = x.codes, x.scales, x.global_scale
            if (packed.device.type != "cuda" or packed.device != self._weight_both.device or
                    activation_scales.device != packed.device or global_scale.device != packed.device):
                raise ValueError("NVFP4 codes, scales, and prepared weights must share one CUDA device.")
            output_dtype = x.model_dtype
            device = packed.device
        else:
            if x.dtype not in (torch.float16, torch.bfloat16) or x.device.type != "cuda":
                raise ValueError("NVFP4 requires CUDA FP16 or BF16 input.")
            original_shape = x.shape[:-1] + (self.out_features,)
            rows = x.numel() // self.in_features
            output_dtype = x.dtype
            device = x.device
        if rows == 0:
            if encoded:
                return torch.empty(original_shape, device=device, dtype=x.model_dtype)
            return x.new_empty(original_shape)
        if not encoded:
            x = self._apply_rotation_to_input(x)
            x2 = x.contiguous().reshape(rows, self.in_features)
            global_scale = self.activation_global_scale
            recipe = getattr(self, "_w4a_activation_recipe", "least_squares")
            packed, activation_scales = nvfp4_pack_and_swizzle(x2, global_scale, recipe=recipe)
        accumulator = torch.empty((rows, self.out_features), device=device, dtype=torch.float32)
        output = torch.empty((rows, self.out_features), device=device, dtype=output_dtype)
        groups = self.in_features // 128
        for group in range(groups):
            start = group * 64
            a = packed[:, start:start + 64]
            a_scales = activation_scales[group]
            both = scaled_mm(
                a, self._weight_both[start:start + 64], a_scales, ScalingType.BlockWise1x16,
                self._unit_weight_scales, ScalingType.BlockWise1x16,
                SwizzleType.SWIZZLE_32_4_4, SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.float32,
            )
            nvfp4_accumulate_group(
                both, self.scales[group], global_scale,
                self.bias, accumulator, output,
                first=group == 0, last=group == groups - 1,
                token_scale=x.token_scale if encoded else None,
            )
        result = output.reshape(original_shape)
        return result

    def _forward_fp8_operand(self, x) -> torch.Tensor:
        """Consume an FP8 carrier over the same native GPTQ INT4 weights.

        Mixed-precision recipes keep the MLP on NVFP4 and step attention down
        to FP8. Both operands describe one INT4 weight tensor, so the FP8 lane
        reuses the E4M3 staging that ``W4AFP8Linear.post_init`` builds.
        """
        from .w4a_activation import pack_activation
        from .w4a_triton import fp8_linear_prepacked

        if self._weight_e4m3.numel() != self.in_features * self.out_features:
            raise RuntimeError("FP8 weight cache is absent; call post_init() after loading or packing.")
        if x.shape[-1] != self.in_features:
            raise ValueError("W4AFP8 requires a matching scale-aware FP8 activation.")
        needs_rotation = bool(self.online_full_had or self.online_partial_had)
        if x.rotation_applied and not needs_rotation:
            raise ValueError("This FP8 operand was rotated for a different Linear contract.")
        if needs_rotation and not x.rotation_applied:
            rotated = self._apply_rotation_to_input(x.decode(torch.float32))
            x = pack_activation(rotated, "w4afp8", model_dtype=x.model_dtype)
        result = fp8_linear_prepacked(
            x.codes, x.scales, self._weight_e4m3, self.scales, self.bias, torch.float32,
        )
        result = result.reshape(x.shape[:-1] + (self.out_features,))
        return result.to(x.model_dtype)


__all__ = ["W4ANVFP4Linear", "nvfp4_global_scale", "nvfp4_input"]
