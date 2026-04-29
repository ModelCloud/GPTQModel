# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import AWQuantLinear
from .torch_aten_kernel import TorchAtenLinear, _cpu_int4pack_zero_offsets, _has_local_int4pack_cpu_ops
from .torch_fused import pack_scales_and_zeros


log = setup_logger()


class TorchAtenAwqLinear(AWQuantLinear):
    """AWQ CPU int4pack backend implemented with local ATen ops."""

    QUANT_TYPE = "awq_torch_aten_kernel"

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_ATEN]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 110}

    SUPPORTS_BITS = TorchAtenLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = TorchAtenLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = TorchAtenLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchAtenLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchAtenLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchAtenLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchAtenLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchAtenLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchAtenLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = TorchAtenLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = TorchAtenLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchAtenLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchAtenLinear.SUPPORTS_ADAPTERS
    REQUIRES_FORMAT_V2 = False

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    gemm_int4_forward_kernel = None

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
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.AWQ_TORCH_ATEN)
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
            register_buffers=False,
            **kwargs,
        )

        self.linear_mode = None

        if register_buffers:
            pack_cols = max(1, self.out_features // self.pack_factor)
            qweight_shape = (self.in_features, pack_cols)
            group_size = max(int(self.group_size), 1)
            group_rows = max(1, math.ceil(self.in_features / group_size))

            self.register_buffer(
                "qweight",
                torch.zeros(qweight_shape, dtype=self.pack_dtype),
            )

            self.register_buffer(
                "qzeros",
                torch.zeros((group_rows, pack_cols), dtype=self.pack_dtype),
            )

            self.register_buffer(
                "scales",
                torch.zeros((group_rows, self.out_features), dtype=torch.float16),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16))
            else:
                self.bias = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        ok, err = TorchAtenLinear.validate_once()
        if ok:
            cls.gemm_int4_forward_kernel = TorchAtenLinear.gemm_int4_forward_kernel
        else:
            cls.gemm_int4_forward_kernel = None
        return ok, err

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self):
        if self.optimized:
            return

        super().optimize()

    def transform_cpu(self):
        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val).to(torch.uint8)
        izeros = torch.bitwise_and(izeros, max_val).reshape(self.scales.shape).to(torch.uint8)

        self.scales = self.scales.to(torch.bfloat16).contiguous()
        self.qweight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(iweight.t().int(), 1).contiguous()
        self.qzeros = _cpu_int4pack_zero_offsets(izeros, self.scales, self.bits).contiguous()
        self.scales_and_zeros = pack_scales_and_zeros(self.scales, self.qzeros)

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
        else:
            raise NotImplementedError(
                "TorchAtenAwqLinear only supports fused transforms on CPU devices."
            )

    def awq_weight_dequantize(self, device, dtype):
        return dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        ).to(device=device, dtype=dtype)

    @torch.no_grad()
    def _fused_op_forward(self, x):
        if x.device.type != "cpu":
            raise NotImplementedError

        original_dtype = x.dtype
        if original_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        out = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x,
            self.qweight,
            self.group_size,
            self.scales_and_zeros,
        )
        if original_dtype != torch.bfloat16:
            out = out.to(original_dtype)
        return out

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if (
            not self.training
            and not x.requires_grad
            and self.linear_mode is None
            and _has_local_int4pack_cpu_ops()
            and x.device.type == "cpu"
        ):
            self.transform(x.device.type)
            self.linear_mode = "inference"
        elif self.linear_mode is None:
            self.linear_mode = "train"

        if self.linear_mode == "inference":
            out = self._fused_op_forward(x).reshape(out_shape)
        else:
            weight = self.awq_weight_dequantize(device=x.device, dtype=x.dtype)
            out = torch.matmul(x, weight).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


__all__ = ["TorchAtenAwqLinear"]
