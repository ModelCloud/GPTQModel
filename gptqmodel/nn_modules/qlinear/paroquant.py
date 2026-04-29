# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant runtime implementation adapted from the ParoQuant paper and public
# project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant CUDA-backed quantized linear layer."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.paroquant import apply_paroquant_rotation, build_identity_rotation_buffers, is_identity_rotation
from .gemm_awq import FP32_ACCUM, _awq_cuda_gemm_forward
from .torch_awq import AwqTorchLinear


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


class ParoLinear(AwqTorchLinear):
    """Run ParoQuant inference by rotating inputs and reusing AWQ packed GEMM."""

    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_CUDA]
    SUPPORTS_METHODS = [METHOD.PARO]
    SUPPORTS_FORMATS = {FORMAT.PAROQUANT: 55}
    SUPPORTS_BITS = [4]
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
        weight = dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        )
        if weight.dtype != x_flat.dtype or weight.device != x_flat.device:
            weight = weight.to(device=x_flat.device, dtype=x_flat.dtype)

        out = torch.matmul(x_flat, weight)
        if self.bias is not None:
            out = out + self.bias.to(device=x_flat.device, dtype=x_flat.dtype)
        return out

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

    def forward(self, x: torch.Tensor):
        """Rotate inputs, run quantized matmul, then apply adapters in input space."""
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        rotated = self._rotate_inputs(x_flat)

        out = self._forward_cuda_awq_kernel(rotated)
        if out is None:
            out = self._forward_dense(rotated)

        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out.reshape(original_shape)


__all__ = ["ParoLinear"]
