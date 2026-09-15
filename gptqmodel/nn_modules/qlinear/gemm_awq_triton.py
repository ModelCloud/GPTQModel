# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from contextlib import nullcontext
from typing import Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear, FormatSupport
from ...quantization import FORMAT, METHOD
from ...utils import has_gil_disabled
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.torch import HAS_XPU
from .utils import validate_mixed_precision_3bit_contract


# Shared runtime default: prefer accuracy first unless the user explicitly opts out.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)


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
        # Only quantized weights and input metadata are needed for the
        # input-gradient backward; input and bias values are not retained.
        ctx.save_for_backward(qweight, qzeros, scales)
        ctx.input_shape = tuple(x.shape)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)
        rows = input_rows(x)
        ctx.input_rows = rows
        if rows == 0:
            # Triton also requires a positive M dimension, so use the shared
            # empty path and avoid compiling/launching a zero-row kernel.
            return empty_linear_output(x, out_features)

        from ...quantization.awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton

        # Dense matmul is faster once the flattened row count is large enough.
        FULL_DEQUANT_MATMUL_THRESHOLD = rows > 128
        # Triton consumes [rows, features]; restore leading dimensions later.
        x_2d = x.reshape(rows, x.shape[-1])
        if FULL_DEQUANT_MATMUL_THRESHOLD:
            out = awq_dequantize_triton(qweight, scales, qzeros)
            out = torch.matmul(x_2d, out.to(x.dtype))
        else:
            out = awq_gemm_triton(
                x_2d,
                qweight,
                scales,
                qzeros,
                split_k_iters=8,
                fp32_accum=FP32_ACCUM,
                output_dtype=x.dtype,
            )

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        qweight, qzeros, scales = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            if ctx.input_rows == 0:
                grad_input = grad_output.new_empty(ctx.input_shape)
                return grad_input, None, None, None, None, None, None, None, None
            from ...quantization.awq.modules.triton.gemm import awq_dequantize_triton

            weights = awq_dequantize_triton(qweight, scales, qzeros).to(grad_output.dtype)
            # Mirror forward's flatten/restore so backward supports every rank.
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.matmul(grad_output_2d, weights.transpose(-1, -2))
            grad_input = grad_input.reshape(ctx.input_shape)

        return grad_input, None, None, None, None, None, None, None, None


class AwqGEMMTritonLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMM_TRITON]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GEMM: FormatSupport(priority=50, bits=(3, 4)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 96, 128, 192, 256, 384, 512]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    # TODO: ROCM also has Triton support. Need to validate ROCM for triton
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "awq_gemm_triton"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        from packaging import version
        from triton import __version__ as triton_version
        triton_v = version.parse(triton_version)

        if triton_v < version.parse("2.0.0"):
            raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

        # GIL=0 is tested with Triton 3.4.0 and it works
        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            raise Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        valid, error = super().validate(**args)
        if not valid:
            return valid, error

        valid, error = validate_mixed_precision_3bit_contract(
            kernel_name=cls.__name__,
            bits=args.get("bits"),
            desc_act=args.get("desc_act"),
            sym=args.get("sym"),
            pack_dtype=args.get("pack_dtype"),
            dynamic=args.get("dynamic"),
        )
        if not valid:
            return valid, error

        if args.get("bits") != 3 and args.get("dtype") == torch.bfloat16:
            return False, NotImplementedError(
                f"{cls.__name__} BF16 support is currently limited to the 3-bit fused path."
            )

        if args.get("bits") == 3:
            in_features = args.get("in_features")
            out_features = args.get("out_features")
            group_size = args.get("group_size", 128)
            effective_group_size = in_features if group_size == -1 else group_size
            if in_features is not None and (
                in_features % 32 != 0
                or effective_group_size is None
                or effective_group_size <= 0
                or in_features % effective_group_size != 0
            ):
                return False, NotImplementedError(
                    f"{cls.__name__} 3-bit fused inference requires in_features divisible by 32 and by "
                    f"group_size (or group_size=-1), got in_features={in_features}, group_size={group_size}."
                )
            if out_features is not None and out_features % 32 != 0:
                return False, NotImplementedError(
                    f"{cls.__name__} 3-bit fused inference requires out_features divisible by 32, "
                    f"got {out_features}."
                )

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
        **kwargs,
    ):
        register_3bit_buffers = register_buffers and bits == 3
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_GEMM_TRITON),
            adapter=adapter,
            register_buffers=register_buffers and bits != 3,
            **kwargs)

        if register_3bit_buffers:
            self.register_buffer(
                "qweight",
                torch.zeros((in_features, out_features // 32 * 3), dtype=pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                torch.zeros((in_features // self.group_size, out_features // 32 * 3), dtype=pack_dtype),
            )
            self.register_buffer(
                "scales",
                torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16),
            )
            if bias:
                self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
            else:
                self.bias = None

    def post_init(self):
        if self.scales is not None:
            self.scales = self.scales.to(dtype=torch.float16)
        super().post_init()
        if self.bits == 3:
            from ..triton_utils.three_bit import (
                repack_awq_to_gptq_3bit,
                unpack_3bit,
            )

            zeros = unpack_3bit(self.qzeros, axis=1, count=self.out_features)
            if not torch.all(zeros == 4):
                raise ValueError(
                    f"{type(self).__name__}: 3-bit symmetric AWQ inference requires every zero point to equal 4."
                )
            self.register_buffer(
                "_triton_3bit_qweight",
                repack_awq_to_gptq_3bit(self.qweight),
                persistent=False,
            )

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)

        if self.bits == 3:
            from ..triton_utils.three_bit import (
                LAYOUT_GPTQ,
                matmul_3bit,
            )

            input_dtype = x.dtype
            compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
            x_compute = x if x.dtype == compute_dtype else x.to(compute_dtype)
            x_flat = x_compute.reshape(-1, x_compute.shape[-1])
            if x_flat.stride(-1) != 1:
                x_flat = x_flat.contiguous()

            out = matmul_3bit(
                x_flat,
                self._triton_3bit_qweight,
                self.scales,
                layout=LAYOUT_GPTQ,
                group_size=self.requested_group_size,
            )

            if self.bias is not None:
                out = out + self.bias
            out = out.reshape(out_shape)
            if self.adapter:
                out = self.adapter.apply(x=x_compute, out=out)
            return out.to(dtype=input_dtype)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        if not x.is_contiguous():
            x = x.contiguous()

        # Select from x.device instead of whichever accelerator is globally available.
        device_context = (
            torch.xpu.device(x.device)
            if x.device.type == "xpu" and HAS_XPU
            else torch.cuda.device(x.device)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            with nullcontext() if self.training else torch.inference_mode():
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

    def pack(
        self,
        linear: torch.nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor = None,
        workers: Optional[int] = None,
    ):
        if self.bits != 3:
            from .gemm_awq import AwqGEMMLinear

            return AwqGEMMLinear.pack(
                self, linear=linear, scales=scales, zeros=zeros, g_idx=g_idx, workers=workers
            )

        from ..triton_utils.three_bit import pack_3bit

        if g_idx is not None:
            expected_g_idx = torch.arange(self.in_features, dtype=g_idx.dtype, device=g_idx.device) // self.group_size
            if not torch.equal(g_idx, expected_g_idx):
                raise ValueError("3-bit AWQ packing requires natural group indices for desc_act=False.")

        scales_group_n = scales.t().contiguous()
        zeros_group_n = zeros.t().contiguous()
        expected_metadata_shape = (self.in_features // self.group_size, self.out_features)
        if tuple(scales_group_n.shape) != expected_metadata_shape:
            raise ValueError(
                f"3-bit AWQ scales expected shape {expected_metadata_shape}, got {tuple(scales_group_n.shape)}"
            )
        if tuple(zeros_group_n.shape) != expected_metadata_shape:
            raise ValueError(
                f"3-bit AWQ zeros expected shape {expected_metadata_shape}, got {tuple(zeros_group_n.shape)}"
            )
        if not torch.all(zeros_group_n == 4):
            raise ValueError("3-bit symmetric AWQ packing requires every zero point to equal 4.")

        weight = linear.weight.detach()
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                "3-bit AWQ packing expects a Linear weight with shape "
                f"({self.out_features}, {self.in_features}), got {tuple(weight.shape)}"
            )
        groups = torch.arange(self.in_features, device=weight.device) // self.group_size
        scales_device = scales_group_n.to(device=weight.device)
        zeros_device = zeros_group_n.to(device=weight.device)
        integer_weight = torch.round(
            (weight.t().contiguous() + zeros_device[groups] * scales_device[groups]) / scales_device[groups]
        ).to(torch.int32)

        self.register_buffer("qweight", pack_3bit(integer_weight, axis=1).to(device=weight.device))
        self.register_buffer("qzeros", pack_3bit(zeros_device.to(torch.int32), axis=1))
        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        self.register_buffer("scales", scales_group_n.to(device=weight.device, dtype=scale_dtype))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(device=weight.device, dtype=scale_dtype))
        else:
            self.bias = None


__all__ = [
    "AwqGemmTritonFn",
    "AwqGEMMTritonLinear",
]
