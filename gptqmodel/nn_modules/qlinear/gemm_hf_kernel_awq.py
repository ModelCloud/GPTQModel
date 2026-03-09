# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import torch

from ...adapter.adapter import Adapter
from ...nn_modules.qlinear import BaseQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    reverse_awq_order,
    unpack_awq,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from .gemm_hf_kernel import HFKernelLinear


log = setup_logger()


class HFKernelAwqLinear(HFKernelLinear):
    """AWQ variant of HFKernelLinear — uses kernels-community gemm_int4 with AWQ weights."""

    QUANT_TYPE = "hf_kernel_awq"

    SUPPORTS_BACKENDS = [BACKEND.HF_KERNEL_AWQ]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 110}

    # inherit from HFKernelLinear
    SUPPORTS_BITS = HFKernelLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = HFKernelLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = HFKernelLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = HFKernelLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = HFKernelLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = HFKernelLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = HFKernelLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = HFKernelLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = HFKernelLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = HFKernelLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = HFKernelLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = HFKernelLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = HFKernelLinear.SUPPORTS_ADAPTERS
    REQUIRES_FORMAT_V2 = HFKernelLinear.REQUIRES_FORMAT_V2

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

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
        kwargs.setdefault("backend", BACKEND.HF_KERNEL_AWQ)
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
            # Skip base buffer init, we need to manually init buffers for awq
            register_buffers=False,
            **kwargs,
        )

        # Create awq buffers
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

    def post_init(self):
        # AWQ has no g_idx — create wf buffers using qweight.device instead
        device = self.qweight.device
        if self.bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=torch.int32).unsqueeze(0).to(device=device)
        elif self.bits == 3:
            wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12).to(device=device)

        self.register_buffer("wf_unsqueeze_zero", wf.unsqueeze(0).to(device=device), persistent=False)
        self.register_buffer("wf_unsqueeze_neg_one", wf.unsqueeze(-1).to(device=device), persistent=False)

        # Lora: Keep adapter post-init behavior aligned with BaseQuantLinear.
        BaseQuantLinear.post_init(self)
        self.linear_mode = None
        self.dequant_dtype = torch.int8
        self.optimize()

    def transform_cpu(self):
        # Unpack AWQ weights directly to integer form
        iweight, izeros = unpack_awq(self.qweight, self.qzeros, self.bits)
        iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
        max_val = (1 << self.bits) - 1
        iweight = torch.bitwise_and(iweight, max_val).to(torch.uint8)
        izeros = torch.bitwise_and(izeros, max_val).to(torch.uint8)

        self.scales = self.scales.to(torch.bfloat16).contiguous()

        # AWQ has no g_idx — weights are already in natural order, just transpose
        # iweight: (in_features, out_features) -> (out_features, in_features)
        self.qweight = iweight.t().contiguous()
        self.qzeros = izeros.contiguous()

    def transform(self, device):
        if device == "cpu":
            self.transform_cpu()
            self.convert_weight_packed_zp()
        else:
            raise NotImplementedError(
                "HFKernelAwqLinear only supports fused transforms on CPU devices."
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
        # AWQ has no g_idx reordering — skip ret_idx
        if x.device.type == "cpu":
            out = self.gemm_int4_forward_kernel(x, self.qweight, self.qzeros, self.scales, self.group_size)
        else:
            raise NotImplementedError
        return out

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        if not self.training and not x.requires_grad and self.linear_mode is None and self.gemm_int4_forward_kernel is not None:
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


__all__ = ["HFKernelAwqLinear"]
