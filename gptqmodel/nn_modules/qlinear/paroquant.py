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
from ...utils.awq import awq_runtime_available
from ...utils.backend import BACKEND
from ...utils.paroquant import apply_paroquant_rotation, is_identity_rotation
from .gemm_awq import FP32_ACCUM, _awq_cuda_gemm_forward
from .torch_awq import AwqTorchQuantLinear


# Rotated activations benchmark faster with a shallower K split than generic AWQ.
_PAROQUANT_AWQ_SPLIT_K = 4


class ParoQuantQuantLinear(AwqTorchQuantLinear):
    """Run ParoQuant inference by rotating inputs and reusing AWQ packed GEMM."""

    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_CUDA]
    SUPPORTS_METHODS = [METHOD.PAROQUANT]
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
        **kwargs,
    ):
        """Initialize AWQ buffers plus the extra ParoQuant rotation state."""
        self.krot = int(krot)
        if self.krot <= 0:
            raise ValueError(f"ParoQuantQuantLinear: `krot` must be positive, got {krot}.")
        self.fp32_accum = bool(fp32_accum)

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
        theta = torch.zeros((self.krot, self.in_features // 2), dtype=torch.float16)
        pairs = torch.zeros((self.krot, self.in_features), dtype=torch.int16)
        channel_scales = torch.ones((1, self.in_features), dtype=torch.float16)

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
            f"krot={self.krot}, awq_split_k={_PAROQUANT_AWQ_SPLIT_K}, fp32_accum={self.fp32_accum}"
        )

    def _rotate_inputs(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Apply the learned input transform before quantized matmul."""
        if self._rotation_identity:
            return x_flat
        return apply_paroquant_rotation(
            x_flat,
            self.pairs,
            self.theta,
            scales=self.channel_scales,
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
        if x_flat.device.type != "cuda" or not awq_runtime_available():
            return None

        compute_dtype = x_flat.dtype if x_flat.dtype in (torch.float16, torch.bfloat16) else torch.float16
        kernel_input = (
            x_flat
            if x_flat.dtype == compute_dtype and x_flat.is_contiguous()
            else x_flat.to(device=x_flat.device, dtype=compute_dtype).contiguous()
        )
        kernel_scales = self.scales
        if (
            kernel_scales.device != kernel_input.device
            or kernel_scales.dtype != compute_dtype
            or not kernel_scales.is_contiguous()
        ):
            kernel_scales = kernel_scales.to(device=kernel_input.device, dtype=compute_dtype).contiguous()
        out = _awq_cuda_gemm_forward(
            kernel_input.reshape(-1, kernel_input.shape[-1]),
            self.qweight,
            kernel_scales,
            self.qzeros,
            _PAROQUANT_AWQ_SPLIT_K,
            fp32_accum=self.fp32_accum,
        )
        if self.bias is not None:
            out = out + self.bias.to(device=kernel_input.device, dtype=out.dtype)
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


__all__ = ["ParoQuantQuantLinear"]
