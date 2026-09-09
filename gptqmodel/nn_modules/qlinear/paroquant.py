# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant runtime implementation adapted from the ParoQuant paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant quantized linear layer with CUDA and NPU fast paths."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import FormatSupport
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.paroquant import (
    apply_paroquant_rotation,
    apply_paroquant_rotation_awq,
    build_identity_rotation_buffers,
    is_identity_rotation,
)
from .gemm_awq import FP32_ACCUM, _awq_cuda_gemm_forward
from .komodo import AwqKomodoLinear, _assert_fp16_inference_input, _weight_quant_matmul


# Rotated activations benchmark faster with a shallower K split than generic AWQ.
_PAROQUANT_AWQ_SPLIT_K = 4
_PAROQUANT_CACHE_RUNTIME_DTYPE = env_flag("GPTQMODEL_PAROQUANT_CACHE_RUNTIME_DTYPE", default=False)
_PAROQUANT_AUTO_CACHE_BF16_RUNTIME_DTYPE = env_flag(
    "GPTQMODEL_PAROQUANT_AUTO_CACHE_BF16_RUNTIME_DTYPE", default=True
)
# Cache typed rotation metadata so BF16 runs do not re-cast theta/scales every call.
_PAROQUANT_CACHE_ROTATION_DTYPE = env_flag("GPTQMODEL_PAROQUANT_CACHE_ROTATION_DTYPE", default=False)
_PAROQUANT_AUTO_CACHE_BF16_ROTATION_DTYPE = env_flag(
    "GPTQMODEL_PAROQUANT_AUTO_CACHE_BF16_ROTATION_DTYPE", default=True
)


class ParoLinear(AwqKomodoLinear):
    """Run ParoQuant inference by rotating inputs and reusing AWQ packed GEMM."""

    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_CUDA]
    SUPPORTS_METHODS = [METHOD.PARO]
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.PAROQUANT: FormatSupport(priority=55, bits=(4,)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "awq_paroquant"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = False,
        krot: int = 8,
        fp32_accum: bool = FP32_ACCUM,
        cache_runtime_dtype: bool = _PAROQUANT_CACHE_RUNTIME_DTYPE,
        auto_cache_bf16_runtime_dtype: bool = _PAROQUANT_AUTO_CACHE_BF16_RUNTIME_DTYPE,
        cache_rotation_dtype: bool = _PAROQUANT_CACHE_ROTATION_DTYPE,
        auto_cache_bf16_rotation_dtype: bool = _PAROQUANT_AUTO_CACHE_BF16_ROTATION_DTYPE,
        **kwargs,
    ):
        """Initialize AWQ buffers plus the extra ParoQuant rotation state."""
        self.krot = int(krot)
        if self.krot <= 0:
            raise ValueError(f"ParoLinear: `krot` must be positive, got {krot}.")
        self.fp32_accum = bool(fp32_accum)
        self.cache_runtime_dtype = bool(cache_runtime_dtype)
        self.auto_cache_bf16_runtime_dtype = bool(auto_cache_bf16_runtime_dtype)
        self.cache_rotation_dtype = bool(cache_rotation_dtype)
        self.auto_cache_bf16_rotation_dtype = bool(auto_cache_bf16_rotation_dtype)
        self._rotation_runtime_dtype: Optional[torch.dtype] = None
        self._rotation_runtime_device: Optional[torch.device] = None
        self._runtime_theta: Optional[torch.Tensor] = None
        self._runtime_channel_scales: Optional[torch.Tensor] = None
        self.paroquant_cuda_awq_fused_dispatch_enabled = True

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            backend=kwargs.pop("backend", BACKEND.PAROQUANT_CUDA),
            **kwargs,
        )
        self._register_rotation_buffers()
        self._rotation_identity = True

    def _register_rotation_buffers(self) -> None:
        """Allocate the per-layer buffers that encode runtime rotations."""
        # Fresh runtime modules must start from a valid identity matching so the
        # fused kernel never sees duplicate pair indices before optimized
        # buffers are loaded from quantization or checkpoints.
        pairs, theta, channel_scales = build_identity_rotation_buffers(
            in_features=self.in_features,
            group_size=self.group_size,
            krot=self.krot,
            dtype=torch.float16,
        )

        if "theta" not in self._buffers:
            self.register_buffer("theta", theta)
        else:
            self.theta = theta

        if "pairs" not in self._buffers:
            self.register_buffer("pairs", pairs)
        else:
            self.pairs = pairs

        if "channel_scales" not in self._buffers:
            self.register_buffer("channel_scales", channel_scales)
        else:
            self.channel_scales = channel_scales

    def post_init(self):
        """Refresh cached runtime state after weights or rotation buffers change."""
        super().post_init()
        self._clear_rotation_runtime_cache()
        self._rotation_identity = is_identity_rotation(self.theta, self.channel_scales)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        """ParoQuant relies on AWQ validation and needs no extra one-time checks here."""
        return True, None

    def extra_repr(self) -> str:
        """Expose ParoQuant-specific fields in `repr(module)` for debugging."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}, "
            f"krot={self.krot}, awq_split_k={_PAROQUANT_AWQ_SPLIT_K}, "
            f"cache_runtime_dtype={self.cache_runtime_dtype}, "
            f"auto_cache_bf16={self.auto_cache_bf16_runtime_dtype}, "
            f"cache_rotation_dtype={self.cache_rotation_dtype}, "
            f"auto_cache_bf16_rotation={self.auto_cache_bf16_rotation_dtype}, "
            f"fp32_accum={self.fp32_accum}"
        )

    def _clear_rotation_runtime_cache(self) -> None:
        self._rotation_runtime_dtype = None
        self._rotation_runtime_device = None
        self._runtime_theta = None
        self._runtime_channel_scales = None

    def _apply(self, fn):
        result = super()._apply(fn)
        self._clear_rotation_runtime_cache()
        self._rotation_identity = is_identity_rotation(self.theta, self.channel_scales)
        return result

    def _maybe_eager_native_prepack(self) -> bool:
        if self.scales is not None and self.scales.dtype != torch.float16:
            return False
        return super()._maybe_eager_native_prepack()

    def _can_prefetch_native_plan(
        self, *, device: torch.device, dtype: torch.dtype, allow_training: bool = False
    ) -> bool:
        if dtype != torch.float16 or (self.scales is not None and self.scales.dtype != torch.float16):
            return False
        return super()._can_prefetch_native_plan(device=device, dtype=dtype, allow_training=allow_training)

    def _can_use_native_int4(self, x: torch.Tensor, compute_dtype: torch.dtype) -> bool:
        if x.dtype != torch.float16 or (self.scales is not None and self.scales.dtype != torch.float16):
            return False
        return super()._can_use_native_int4(x, compute_dtype)

    def _ensure_runtime_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.scales is not None and (self.scales.device != device or self.scales.dtype != dtype or not self.scales.is_contiguous()):
            self.scales = self.scales.to(device=device, dtype=dtype).contiguous()
        if self.bias is not None and (self.bias.device != device or self.bias.dtype != dtype or not self.bias.is_contiguous()):
            self.bias = self.bias.to(device=device, dtype=dtype).contiguous()

    def _ensure_rotation_runtime_dtype(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self._rotation_runtime_device != device
            or self._rotation_runtime_dtype != dtype
            or self._runtime_theta is None
            or self._runtime_channel_scales is None
            or not self._runtime_theta.is_contiguous()
            or not self._runtime_channel_scales.is_contiguous()
        ):
            self._runtime_theta = self.theta.to(device=device, dtype=dtype).contiguous()
            self._runtime_channel_scales = self.channel_scales.to(device=device, dtype=dtype).contiguous()
            self._rotation_runtime_device = device
            self._rotation_runtime_dtype = dtype
        return self._runtime_theta, self._runtime_channel_scales

    def _rotate_inputs(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Apply the learned input transform before quantized matmul."""
        if self._rotation_identity:
            return x_flat
        use_cached_rotation_dtype = self.cache_rotation_dtype or (
            self.auto_cache_bf16_rotation_dtype and x_flat.dtype == torch.bfloat16
        )
        theta = self.theta
        channel_scales = self.channel_scales
        if use_cached_rotation_dtype:
            theta, channel_scales = self._ensure_rotation_runtime_dtype(x_flat.device, x_flat.dtype)
        return apply_paroquant_rotation(
            x_flat,
            self.pairs,
            theta,
            scales=channel_scales,
            group_size=self.group_size,
        )

    def _forward_dense(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Fallback reference path: dequantize AWQ weights and run dense matmul."""
        input_dtype = x_flat.dtype
        compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
        if x_flat.dtype != compute_dtype or not x_flat.is_contiguous():
            x_flat = x_flat.to(dtype=compute_dtype).contiguous()

        self._ensure_runtime_dtype(device=x_flat.device, dtype=compute_dtype)
        weight = self._dequantized_weight(device=x_flat.device, dtype=compute_dtype)

        out = torch.matmul(x_flat, weight)
        if self.bias is not None:
            out = out + self.bias.to(device=x_flat.device, dtype=compute_dtype)
        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)
        return out

    def _forward_npu_native(self, rotated_flat: torch.Tensor, original_shape: torch.Size, adapter_input: torch.Tensor):
        """Run rotated activations through Komodo's native NPU AWQ int4 path."""
        _assert_fp16_inference_input(rotated_flat, self.__class__.__name__)
        compute_dtype = torch.float16
        if rotated_flat.dtype != compute_dtype or not rotated_flat.is_contiguous():
            rotated_flat = rotated_flat.to(dtype=compute_dtype).contiguous()

        packed_weight, scales, offsets, native_group_size, _ = self._native_plan(
            device=rotated_flat.device,
            dtype=compute_dtype,
        )
        out = _weight_quant_matmul(rotated_flat, packed_weight, scales, offsets, native_group_size)

        if self.bias is not None:
            bias = self.bias
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out = out + bias

        if self.adapter:
            out = self.adapter.apply(x=adapter_input, out=out)

        self._maybe_schedule_lookahead(compute_dtype)
        return out.reshape(original_shape)

    def _forward_cuda_awq_kernel(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Fast path that feeds rotated activations into the AWQ CUDA GEMM kernel."""
        if x_flat.device.type != "cuda":
            return None

        compute_dtype = x_flat.dtype if x_flat.dtype in (torch.float16, torch.bfloat16) else torch.float16
        kernel_input = (
            x_flat
            if x_flat.dtype == compute_dtype and x_flat.is_contiguous()
            else x_flat.to(device=x_flat.device, dtype=compute_dtype).contiguous()
        )
        use_cached_runtime_dtype = self.cache_runtime_dtype or (
            self.auto_cache_bf16_runtime_dtype and compute_dtype == torch.bfloat16
        )
        if use_cached_runtime_dtype:
            self._ensure_runtime_dtype(kernel_input.device, compute_dtype)
            kernel_scales = self.scales
            kernel_bias = self.bias
        else:
            kernel_scales = self.scales
            if (
                kernel_scales.device != kernel_input.device
                or kernel_scales.dtype != compute_dtype
                or not kernel_scales.is_contiguous()
            ):
                kernel_scales = kernel_scales.to(device=kernel_input.device, dtype=compute_dtype).contiguous()
            kernel_bias = self.bias
            if (
                kernel_bias is not None
                and (kernel_bias.device != kernel_input.device or kernel_bias.dtype != compute_dtype or not kernel_bias.is_contiguous())
            ):
                kernel_bias = kernel_bias.to(device=kernel_input.device, dtype=compute_dtype).contiguous()
        out = _awq_cuda_gemm_forward(
            kernel_input.reshape(-1, kernel_input.shape[-1]),
            self.qweight,
            kernel_scales,
            self.qzeros,
            _PAROQUANT_AWQ_SPLIT_K,
            fp32_accum=self.fp32_accum,
        )
        if kernel_bias is not None:
            out = out + kernel_bias
        if out.dtype != x_flat.dtype:
            out = out.to(dtype=x_flat.dtype)
        return out

    def _forward_cuda_awq_fused(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """Submit rotation and AWQ GEMM from one native op on the inference fast path."""
        if (
            not self.paroquant_cuda_awq_fused_dispatch_enabled
            or self.training
            or torch.is_grad_enabled()
            or self._rotation_identity
            or x_flat.device.type != "cuda"
        ):
            return None

        compute_dtype = x_flat.dtype if x_flat.dtype in (torch.float16, torch.bfloat16) else torch.float16
        kernel_input = (
            x_flat
            if x_flat.dtype == compute_dtype and x_flat.is_contiguous()
            else x_flat.to(device=x_flat.device, dtype=compute_dtype).contiguous()
        )
        use_cached_runtime_dtype = self.cache_runtime_dtype or (
            self.auto_cache_bf16_runtime_dtype and compute_dtype == torch.bfloat16
        )
        if use_cached_runtime_dtype:
            self._ensure_runtime_dtype(kernel_input.device, compute_dtype)
            kernel_scales = self.scales
            kernel_bias = self.bias
        else:
            kernel_scales = self.scales
            if (
                kernel_scales.device != kernel_input.device
                or kernel_scales.dtype != compute_dtype
                or not kernel_scales.is_contiguous()
            ):
                kernel_scales = kernel_scales.to(device=kernel_input.device, dtype=compute_dtype).contiguous()
            kernel_bias = self.bias
            if (
                kernel_bias is not None
                and (
                    kernel_bias.device != kernel_input.device
                    or kernel_bias.dtype != compute_dtype
                    or not kernel_bias.is_contiguous()
                )
            ):
                kernel_bias = kernel_bias.to(device=kernel_input.device, dtype=compute_dtype).contiguous()

        use_cached_rotation_dtype = self.cache_rotation_dtype or (
            self.auto_cache_bf16_rotation_dtype and compute_dtype == torch.bfloat16
        )
        theta = self.theta
        channel_scales = self.channel_scales
        if use_cached_rotation_dtype:
            theta, channel_scales = self._ensure_rotation_runtime_dtype(kernel_input.device, compute_dtype)

        try:
            out = apply_paroquant_rotation_awq(
                kernel_input.reshape(-1, kernel_input.shape[-1]),
                self.pairs,
                theta,
                channel_scales,
                self.qweight,
                kernel_scales,
                self.qzeros,
                kernel_bias,
                group_size=self.group_size,
                split_k_iters=_PAROQUANT_AWQ_SPLIT_K,
                fp32_accum=self.fp32_accum,
            )
        except (NotImplementedError, RuntimeError):
            return None
        if out is not None and out.dtype != x_flat.dtype:
            out = out.to(dtype=x_flat.dtype)
        return out

    def forward(self, x: torch.Tensor):
        """Rotate inputs, run quantized matmul, then apply adapters in input space."""
        original_shape = x.shape[:-1] + (self.out_features,)
        if self.input_rows(x) == 0:
            # Rotation and AWQ GEMM both require a positive row count; keep the
            # empty batch on the shared shape-preserving path instead.
            return self.empty_linear_output(x)
        x_flat = x.reshape(-1, x.shape[-1])
        out = self._forward_cuda_awq_fused(x_flat)
        if out is None:
            rotated = self._rotate_inputs(x_flat)

            compute_dtype = torch.float16
            if self._can_use_native_int4(rotated, compute_dtype):
                return self._forward_npu_native(rotated, original_shape, x_flat)

            out = self._forward_cuda_awq_kernel(rotated)
            if out is None:
                out = self._forward_dense(rotated)

        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out.reshape(original_shape)


__all__ = ["ParoLinear"]
