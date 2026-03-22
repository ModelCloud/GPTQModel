# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.module import try_import
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.paroquant import apply_paroquant_rotation, is_identity_rotation
from .gemm_awq import FP32_ACCUM, _awq_cuda_gemm_forward
from .torch_awq import AwqTorchQuantLinear


awq_ext, awq_import_error = try_import("gptqmodel_awq_kernels")


class ParoQuantQuantLinear(AwqTorchQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_CUDA]
    SUPPORTS_METHODS = [METHOD.PAROQUANT]
    SUPPORTS_FORMATS = {FORMAT.PAROQUANT: 55}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16]

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
        super().post_init()
        self._rotation_identity = is_identity_rotation(self.theta, self.channel_scales)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        return True, None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}, "
            f"krot={self.krot}, fp32_accum={self.fp32_accum}"
        )

    def _rotate_inputs(self, x_flat: torch.Tensor) -> torch.Tensor:
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
        if awq_ext is None or x_flat.device.type != "cuda":
            return None

        kernel_input = x_flat if x_flat.dtype == torch.float16 else x_flat.to(torch.float16)
        out = _awq_cuda_gemm_forward(
            kernel_input.reshape(-1, kernel_input.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            8,
            fp32_accum=self.fp32_accum,
        )
        if self.bias is not None:
            out = out + self.bias
        if x_flat.dtype != torch.float16:
            out = out.to(dtype=x_flat.dtype)
        return out

    def forward(self, x: torch.Tensor):
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
