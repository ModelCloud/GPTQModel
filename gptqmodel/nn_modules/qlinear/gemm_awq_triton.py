# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from contextlib import nullcontext

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization.awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton
from ...utils.backend import BACKEND
from . import tritonv2


class AwqGemmTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
        prefer_backend=None,
    ):
        if not tritonv2.TRITON_AVAILABLE:
            raise ValueError(tritonv2.TRITON_INSTALL_HINT)

        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # Above compute density threshold it is faster to just dequantize the whole thing and do simple matmul
        FULL_DEQUANT_MATMUL_THRESHOLD = x.shape[0] * x.shape[1] > 128
        if FULL_DEQUANT_MATMUL_THRESHOLD:
            out = awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x, out.to(x.dtype))
        else:
            out = awq_gemm_triton(
                x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8,
            )

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, qweight, qzeros, scales, bias = ctx.saved_tensors
        if not tritonv2.TRITON_AVAILABLE:
            raise ValueError(tritonv2.TRITON_INSTALL_HINT)

        weights = awq_dequantize_triton(qweight, scales, qzeros).to(grad_output.dtype)

        grad_input = None
        if ctx.needs_input_grad[0]:
            batch_size = grad_output.shape[0]
            grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))

        return grad_input, None, None, None, None, None, None, None, None


class AwqGEMMTritonQuantLinear(AWQuantLinear):
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

    QUANT_TYPE = "awq_gemm_triton"

    @classmethod
    def validate(cls, **args):
        if not tritonv2.TRITON_AVAILABLE:
            return False, ValueError(tritonv2.TRITON_INSTALL_HINT)

        return cls._validate(**args)

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
            backend=kwargs.pop("backend", BACKEND.TRITON),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

    def post_init(self):
        if self.scales is not None:
            self.scales = self.scales.to(dtype=torch.float16)
        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        ctx = nullcontext() if self.training else torch.inference_mode()
        with ctx:
            out = AwqGemmTritonFn.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.bits,
                self.group_size,
                self.bias,
                self.out_features,
                "triton",
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)


__all__ = [
    "AwqGemmTritonFn",
    "AwqGEMMTritonQuantLinear",
]
