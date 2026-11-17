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
from ...nn_modules.qlinear import AWQuantLinear, empty_linear_output, input_rows
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns
from ...utils.awq import (
    awq_dequantize_weights,
    awq_gemm_forward,
    awq_gemm_forward_fp32_reduce,
    awq_runtime_available,
    awq_runtime_error,
)
from ...utils.backend import BACKEND
from ...utils.kernel import resolve_kernel_config
from ...utils.packing_math import scale_to_storage_dtype
from ...utils.rocm import IS_ROCM


def _awq_cuda_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum: bool = None, kernel=None):
    kernel = resolve_kernel_config(kernel)
    if fp32_accum is None:
        fp32_accum = kernel.fp32_accum
    return awq_gemm_forward(
        input,
        qweight,
        scales,
        qzeros,
        split_k_iters,
        fp32_accum,
        kernel.awq_fused_splitk_reduce,
    )


def _awq_cuda_gemm_forward_fp32_reduce(input, qweight, scales, qzeros, split_k_iters, kernel=None):
    return awq_gemm_forward_fp32_reduce(
        input,
        qweight,
        scales,
        qzeros,
        split_k_iters,
        True,
        resolve_kernel_config(kernel).awq_fused_splitk_reduce,
    )


def _awq_fp32_output_mm(x_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply into fp32 so the bias add and final cast round only once.

    ``out_dtype`` is a CUDA-only ``torch.mm`` overload; the portable fp32-input
    form produces the same fp32 output on CPU and keeps the unit tests runnable
    without a GPU. Production ``AwqGEMMLinear`` is CUDA-only.
    """

    if x_2d.device.type == "cuda":
        return torch.mm(x_2d, weight, out_dtype=torch.float32)
    return torch.mm(x_2d.to(torch.float32), weight.to(torch.float32))


def _pack_awq_columns(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack AWQ code columns with the classic shift/OR order.

    Thin wrapper over the shared :func:`pack_awq_columns` helper that applies
    the AWQ ``[0, 2, 4, 6, 1, 3, 5, 7]`` interleave. Each code is masked
    before shifting to keep sign or excess bits out of adjacent fields.
    """

    return pack_awq_columns(codes, bits, order=AWQ_ORDER)


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
        fp32_accum=None,
        kernel=None,
    ):
        if fp32_accum is None:
            fp32_accum = resolve_kernel_config(kernel).fp32_accum
        # Input and bias values are not needed for the input-gradient backward;
        # retain only input metadata and the quantized weight tensors.
        qweight, qzeros, scales = qweight.contiguous(), qzeros.contiguous(), scales.contiguous()
        ctx.save_for_backward(qweight, qzeros, scales)
        ctx.input_shape = tuple(x.shape)
        ctx.out_features = out_features
        ctx.bias_shape = None if bias is None else bias.shape
        ctx.bias_dtype = None if bias is None else bias.dtype

        out_shape = x.shape[:-1] + (out_features,)
        rows = input_rows(x)
        ctx.input_rows = rows
        if rows == 0:
            # CUDA AWQ kernels do not accept M=0; preserve Linear's shape and
            # autograd behavior without launching a native kernel.
            return empty_linear_output(x, out_features)

        # Dense matmul is faster once the flattened row count is large enough.
        # Below 640 rows the native split-K kernel still wins. From 640 rows
        # the dequantise-then-fp32-GEMM path is faster and at least as accurate,
        # but only when fp32 accumulation was requested; the existing large-M
        # path keeps its historical dispatch and now also produces fp32.
        FULL_DEQUANT_MATMUL_THRESHOLD = 640
        # Native GEMM consumes [rows, features]; restore leading dimensions later.
        x_2d = x.reshape(rows, x.shape[-1])
        if rows > 1024 or (rows >= FULL_DEQUANT_MATMUL_THRESHOLD and fp32_accum):
            weight = awq_dequantize_weights(qweight, scales, qzeros, 0, 0, 0, False)
            # Accumulate and add bias in fp32, then cast once. An fp16-output
            # matmul rounds before the bias add and is measurably less accurate.
            out = _awq_fp32_output_mm(x_2d, weight.to(dtype=x.dtype))
            if bias is not None:
                out = out + bias.to(torch.float32)
            out = out.to(dtype=x.dtype)
        elif bias is not None:
            # The split-K partials are already FP32. Ask for an FP32 result and
            # add the stored bias there, so a finite sum that overflows the
            # activation dtype is not rounded before the bias cancels it.
            out = _awq_cuda_gemm_forward_fp32_reduce(
                x_2d,
                qweight,
                scales,
                qzeros,
                8,
                **({"kernel": kernel} if kernel is not None else {}),
            )
            out = (out + bias.float()).to(dtype=x.dtype)
        else:
            out = _awq_cuda_gemm_forward(
                x_2d,
                qweight,
                scales,
                qzeros,
                8,
                fp32_accum=fp32_accum,
                **({"kernel": kernel} if kernel is not None else {}),
            )

        out = out.reshape(out_shape)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        qweight, qzeros, scales = ctx.saved_tensors

        grad_input = None
        grad_bias = None
        if len(ctx.needs_input_grad) > 6 and ctx.needs_input_grad[6]:
            grad_bias = grad_output.float().sum_to_size(ctx.bias_shape).to(ctx.bias_dtype)
        if ctx.needs_input_grad[0]:
            if ctx.input_rows == 0:
                grad_input = grad_output.new_empty(ctx.input_shape)
                gradients = (grad_input, None, None, None, None, None, grad_bias, None, None, None, None)
                return gradients[:len(ctx.needs_input_grad)]
            weights = awq_dequantize_weights(
                qweight, scales, qzeros, 1, 0, 0, False
            ).to(grad_output.dtype)
            # Mirror forward's flatten/restore so backward supports every rank.
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.matmul(grad_output_2d, weights.transpose(-1, -2))
            grad_input = grad_input.reshape(ctx.input_shape)

        gradients = (grad_input, None, None, None, None, None, grad_bias, None, None, None, None)
        return gradients[:len(ctx.needs_input_grad)]


class AwqGEMMLinear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_GEMM]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 60}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_gemm"

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if IS_ROCM:
            # The native AWQ GEMM extension is CUDA-only (cusparse.h + CUDA
            # inline asm), so it never builds on ROCm. Fail fast with an
            # actionable message instead of attempting a doomed JIT build.
            return False, NotImplementedError(
                "AwqGEMMLinear requires the native CUDA AWQ extension, which is not "
                "available on ROCm. Use the portable AWQ GEMM kernel instead "
                "(`backend=AUTO` or `BACKEND.AWQ_TORCH`)."
            )
        if not awq_runtime_available():
            return False, ValueError(awq_runtime_error() or "CUDA AWQ extension not available; cannot select AwqGEMMLinear")
        else:
            return True, None

    @classmethod
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        group_size = args.get("group_size")
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        effective_group_size = (
            in_features if group_size == -1 and in_features is not None else group_size
        )
        if (
            effective_group_size is not None
            and out_features is not None
            and (
                effective_group_size % 32
                or out_features % effective_group_size
            )
        ):
            return False, NotImplementedError(
                "AWQ GEMM requires a group size divisible by 32 and "
                "out_features divisible by that group size: "
                f"got out_features={out_features}, group_size={effective_group_size}"
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
        fp32_accum: bool = None,
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
        kernel = resolve_kernel_config(self)
        self.fp32_accum = bool(kernel.fp32_accum if fp32_accum is None else fp32_accum)

    def _scales_for_compute(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the stored scales converted for this computation only.

        ``self.scales`` keeps the checkpoint dtype. The native kernel needs a
        half-precision operand matching the activation, but rewriting the
        registered buffer in place (the previous ``_ensure_runtime_dtype``
        behavior) permanently narrowed the stored precision: one BF16 forward
        on an FP16 checkpoint changed 87% of the scales, so every later FP16
        forward and any re-saved checkpoint used the rounded values.
        """

        scales = self.scales
        if scales is not None and (scales.dtype != dtype or not scales.is_contiguous()):
            scales = scales.to(dtype=dtype).contiguous()
        return scales

    def _bias_for_compute(self, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return the stored bias converted for this computation only.

        Like :meth:`_scales_for_compute`, the conversion is a temporary
        operand; the checkpoint bias keeps its original dtype.
        """

        bias = self.bias
        if bias is not None and (bias.dtype != dtype or not bias.is_contiguous()):
            bias = bias.to(dtype=dtype).contiguous()
        return bias

    def post_init(self):
        # Scales and bias keep the dtype they were loaded with; forward casts
        # temporary operands to the activation dtype. The native kernel only
        # sees those temporaries, so post_init must not rewrite the checkpoint.
        super().post_init()

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[:-1] + (self.out_features,)

        # The extension dereferences these pointers directly; fail early with
        # a useful message instead of reporting an opaque CUDA device error.
        for name in ("qweight", "qzeros", "scales"):
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.device != x.device:
                raise RuntimeError(
                    f"AWQ GEMM input and {name} must be on the same device: "
                    f"input={x.device}, {name}={tensor.device}"
                )

        input_dtype = x.dtype
        compute_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16) else torch.float16
        if input_dtype != compute_dtype:
            x = x.to(compute_dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        compute_scales = self._scales_for_compute(compute_dtype)
        compute_bias = self._bias_for_compute(compute_dtype)

        # Keep eval-mode autograd and in-place adapter application viable;
        # only ordinary no-grad inference needs an inference tensor here.
        needs_grad = torch.is_grad_enabled() and (
            x.requires_grad or (self.bias is not None and self.bias.requires_grad)
        )
        ctx = nullcontext() if self.training or needs_grad or self.adapter else torch.inference_mode()
        with ctx:
            out = AwqGemmFn.apply(
                x,
                self.qweight,
                self.qzeros,
                compute_scales,
                self.bits,
                self.group_size,
                compute_bias,
                self.out_features,
                "cuda",
                self.fp32_accum,
                self.kernel,
            )

        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros.float() * scales.float()

        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
        # Validate the storage cast before replacing any module state: a scale
        # that overflows or underflows here would silently corrupt every code.
        stored_scales = scale_to_storage_dtype(scales, scale_dtype).clone()
        self.register_buffer("scales", stored_scales)
        if linear.bias is not None:
            bias_dtype = linear.bias.dtype if linear.bias.dtype in (torch.float16, torch.bfloat16) else scale_dtype
            self.register_buffer("bias", linear.bias.clone().to(bias_dtype))
        else:
            self.bias = None

        if self.bits != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        # Gather each input column's group scale/zero in a single pass instead
        # of launching one tiny kernel per input column.
        weight = linear.weight.data
        group_index = torch.arange(self.in_features, device=weight.device) // self.group_size
        intweight = torch.round(
            (weight.t().float() + scale_zeros.index_select(0, group_index))
            / self.scales.index_select(0, group_index).float()
        ).clamp_(0, (1 << self.bits) - 1).to(torch.int32).contiguous()

        best_device = str(linear.weight.device)

        # Avoid: The operator 'aten::__lshift__.Scalar' is not currently implemented for the MPS device
        if "mps" in best_device:
            intweight = intweight.to("cpu")

        self.register_buffer("qweight", _pack_awq_columns(intweight, self.bits))

        zeros = zeros.to(dtype=torch.int32, device=best_device)

        if "mps" in best_device:
            zeros = zeros.to("cpu")

        self.register_buffer("qzeros", _pack_awq_columns(zeros, self.bits))



__all__ = [
    "AwqGemmFn",
    "AwqGEMMLinear",
]
