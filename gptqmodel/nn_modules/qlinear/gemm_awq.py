# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
from torch import nn

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, FormatSupport
from ...quantization import FORMAT, METHOD
from ...utils.awq import awq_dequantize_weights, awq_gemm_forward, awq_runtime_available, awq_runtime_error
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from .pack_block_ext import pack_awq_cpu


# Shared runtime default: prefer accuracy first unless the user explicitly opts out.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)

def _awq_cuda_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum: bool = FP32_ACCUM):
    return awq_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum)


class AwqGemmFn(torch.autograd.Function):
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
        fp32_accum=FP32_ACCUM,
    ):
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        if x.shape[0] == 0:
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # Above compute density threshold it is faster to just dequantize the whole thing and do simple matmul
        FULL_DEQUANT_MATMUL_THRESHOLD = x.shape[0] * x.shape[1] > 1024
        if FULL_DEQUANT_MATMUL_THRESHOLD:
            out = awq_dequantize_weights(qweight, scales, qzeros, 0, 0, 0, False)
            out = torch.matmul(x, out.to(dtype=x.dtype))
        else:
            out = _awq_cuda_gemm_forward(
                x.reshape(-1, x.shape[-1]),
                qweight,
                scales,
                qzeros,
                8,
                fp32_accum=fp32_accum,
            )

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, qweight, qzeros, scales, bias = ctx.saved_tensors

        weights = awq_dequantize_weights(
            qweight, scales, qzeros, 1, 0, 0, False
        ).to(grad_output.dtype)

        grad_input = None
        if ctx.needs_input_grad[0]:
            batch_size = grad_output.shape[0]
            grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))

        return grad_input, None, None, None, None, None, None, None, None, None


class AwqGEMMLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMM]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GEMM: FormatSupport(priority=60, bits=(4,)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemm"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not awq_runtime_available():
            return False, ValueError(awq_runtime_error() or "CUDA AWQ extension not available; cannot select AwqGEMMLinear")
        else:
            return True, None

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
        fp32_accum: bool = FP32_ACCUM,
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
            backend=kwargs.pop("backend", BACKEND.AWQ_GEMM),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)
        self.fp32_accum = bool(fp32_accum)

    def _ensure_runtime_dtype(self, dtype: torch.dtype):
        if self.scales is not None and (self.scales.dtype != dtype or not self.scales.is_contiguous()):
            self.scales = self.scales.to(dtype=dtype).contiguous()
        if self.bias is not None and (self.bias.dtype != dtype or not self.bias.is_contiguous()):
            self.bias = self.bias.to(dtype=dtype).contiguous()

    def post_init(self):
        if self.scales is not None and self.scales.dtype not in (torch.float16, torch.bfloat16):
            self.scales = self.scales.to(dtype=torch.float16)
        if self.bias is not None and self.bias.dtype not in (torch.float16, torch.bfloat16):
            self.bias = self.bias.to(dtype=torch.float16)

        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
        if input_dtype != compute_dtype:
            x = x.to(compute_dtype)
        elif not x.is_contiguous():
            x = x.contiguous()

        self._ensure_runtime_dtype(compute_dtype)

        ctx = nullcontext() if self.training else torch.inference_mode()
        with ctx:
            out = AwqGemmFn.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.bits,
                self.group_size,
                self.bias,
                self.out_features,
                "cuda",
                self.fp32_accum,
            )

        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def pack(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor=None,
        workers: Optional[int]=None,
    ):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()

        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        self.register_buffer("scales", scales.clone().to(scale_dtype))
        if linear.bias is not None:
            bias_dtype = linear.bias.dtype if linear.bias.dtype in (torch.float16, torch.bfloat16) else scale_dtype
            self.register_buffer("bias", linear.bias.clone().to(bias_dtype))
        else:
            self.bias = None

        pack_num = 32 // self.bits

        # Vectorized AWQ quantization on CPU, then bit-pack with the CPU extension.
        weight = linear.weight.detach().to("cpu", copy=False)
        scales_cpu = self.scales.to("cpu", copy=False)
        zeros_cpu = zeros.to("cpu", copy=False)

        if self.group_size > 0:
            g_idx = torch.arange(self.in_features, device="cpu", dtype=torch.int64) // self.group_size
        else:
            g_idx = torch.zeros(self.in_features, device="cpu", dtype=torch.int64)

        scale_zeros_exp = (zeros_cpu * scales_cpu).to(weight.dtype)[g_idx]
        scales_exp = scales_cpu[g_idx]
        intweight = torch.round((weight.t() + scale_zeros_exp) / scales_exp)
        intweight = torch.clamp(intweight, 0, self.maxq).to(torch.int32)
        zeros_int = zeros_cpu.to(torch.int32)

        try:
            awq_threads = workers if workers and workers > 0 else -1
            qweight, qzeros = pack_awq_cpu(intweight, zeros_int, self.bits, awq_threads)
        except Exception:
            # Fallback to the original Python packing loops if the extension is unavailable.
            qweight = torch.zeros(
                (intweight.shape[0], intweight.shape[1] // pack_num),
                dtype=torch.int32,
                device=intweight.device,
            )
            qzeros = torch.zeros(
                (zeros_int.shape[0], zeros_int.shape[1] // pack_num),
                dtype=torch.int32,
                device=zeros_int.device,
            )

            if self.bits != 4:
                raise NotImplementedError("Only 4-bit are supported for now.")
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]

            for col in range(intweight.shape[1] // pack_num):
                for i in range(pack_num):
                    qweight_col = intweight[:, col * pack_num + order_map[i]]
                    qweight[:, col] |= qweight_col << (i * self.bits)

            for col in range(zeros_int.shape[1] // pack_num):
                for i in range(pack_num):
                    qzero_col = zeros_int[:, col * pack_num + order_map[i]]
                    qzeros[:, col] |= qzero_col << (i * self.bits)

        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)



__all__ = [
    "AwqGemmFn",
    "AwqGEMMLinear",
]
