# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.utils.module import try_import
from ...utils.backend import BACKEND
from ...utils.gemv import calculate_zeros_width
from ...utils.logger import setup_logger


log = setup_logger()

awq_ext, msg = try_import("gptqmodel_awq_kernels")

class AwqGEMVQuantLinear(AWQuantLinear):
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

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemv"

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
        backend = kwargs.pop("backend", BACKEND.GEMV)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=backend,
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        self.split_k_iters = 8

        self.bias = None

        if register_buffers:
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, in_features // self.pack_factor), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    out_features,
                    calculate_zeros_width(in_features, self.group_size),
                    dtype=self.pack_dtype,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    out_features,
                    calculate_zeros_width(in_features, self.group_size) * self.pack_factor,
                    dtype=torch.float16,
                ),
            )

            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        # awq only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)

        super().post_init()

    def forward(self, x: torch.Tensor):
        if awq_ext is None:
            raise ModuleNotFoundError("External AWQ kernels are not properly installed." + msg)

        out_shape = x.shape[:-1] + (self.out_features,)
        inputs = x.reshape(-1, x.shape[-1])

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()

        if inputs.shape[0] > 8:
            out = awq_ext.gemmv2_forward_cuda(
                inputs,
                self.qweight,
                self.scales,
                self.qzeros,
                self.group_size,
                self.split_k_iters,
            )
        else:
            out = awq_ext.gemv_forward_cuda(
                inputs, self.qweight, self.scales, self.qzeros, self.group_size
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, bits={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.bits,
                self.group_size,
            )
        )

__all__ = ["AwqGEMVQuantLinear"]
