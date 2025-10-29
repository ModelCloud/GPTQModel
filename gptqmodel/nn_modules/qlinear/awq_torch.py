# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.awq.utils.packing_utils import dequantize_gemm
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear


log = setup_logger()


class AwqTorchQuantLinear(AWQuantLinear):
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

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_torch"

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
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH_AWQ),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )

        self._cached_weights: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _invalidate_cache(self) -> None:
        self._cached_weights.clear()

    def post_init(self):
        self._invalidate_cache()
        super().post_init()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )

    def _materialize_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        qweight = self.qweight.to(device=device, non_blocking=True)
        qzeros = self.qzeros.to(device=device, non_blocking=True) if self.qzeros is not None else None
        scales = self.scales.to(device=device, non_blocking=True)

        weight = dequantize_gemm(qweight, qzeros, scales, self.bits, self.group_size)
        return weight.to(dtype=dtype)

    def _get_dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._cached_weights.get(key)
        if cached is None:
            cached = self._materialize_weight(device=device, dtype=dtype)
            self._cached_weights[key] = cached
        return cached

    def _get_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(device=device, dtype=dtype, non_blocking=True)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        original_dtype = x.dtype
        device = x.device

        target_dtype = x.dtype
        x_flat = x.reshape(-1, x.shape[-1]).to(dtype=target_dtype)

        weight = self._get_dequantized_weight(device=device, dtype=target_dtype)
        output = torch.matmul(x_flat, weight)

        bias = self._get_bias(device=device, dtype=output.dtype)
        if bias is not None:
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        output = output.reshape(original_shape)

        return output

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._invalidate_cache()
        return result

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._invalidate_cache()
        return module


__all__ = ["AwqTorchQuantLinear"]
