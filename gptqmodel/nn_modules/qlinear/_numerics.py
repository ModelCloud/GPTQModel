# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from ...utils.rocm import IS_ROCM


# cuBLAS accumulates float16/bfloat16 GEMMs in the operand dtype once the
# problem is large enough to split the reduction, so the Torch epilogue has to
# ask for an FP32 output to keep partial sums wide. HIP already accumulates
# these operands in FP32 and a `float32` output request only adds a cast:
# measured on gfx1201 (R9700) every activation/weight pair was bitwise
# identical to the low-precision matmul across 96 shape/dtype/bit cases and 150
# random matmuls, while the wide path cost +4% to +40% latency and up to
# +144 MiB of transient allocation. Keep the wide request on CUDA only, but
# preserve the fused bias epilogue for HIP linear calls below.
_WIDE_REDUCE_DEVICES = ("cuda",)


def _wants_fp32_reduce(x: torch.Tensor) -> bool:
    return (
        x.device.type in _WIDE_REDUCE_DEVICES
        and not IS_ROCM
        and x.dtype in (torch.float16, torch.bfloat16)
    )


def _needs_wide_autograd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )


class _CudaWideMatmul(torch.autograd.Function):
    """Differentiate a CUDA FP32-output GEMM, whose aten backward is unavailable."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        ctx.input_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_shape = None if bias is None else bias.shape
        ctx.bias_dtype = None if bias is None else bias.dtype
        # Keep only the low-precision operands needed by backward. In particular,
        # a bias-only gradient must not hold a dequantized weight until backward.
        ctx.save_for_backward(x if weight.requires_grad else None, weight if x.requires_grad else None)
        flat_x = x.reshape(-1, x.shape[-1])
        if bias is None:
            output = torch.mm(flat_x, weight, out_dtype=torch.float32)
        else:
            output = torch.addmm(bias.float(), flat_x, weight, out_dtype=torch.float32)
        return output.to(x.dtype).reshape(x.shape[:-1] + (weight.shape[-1],))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved_x, saved_weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mm(grad, saved_weight.t(), out_dtype=torch.float32)
            grad_x = grad_x.to(ctx.x_dtype).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            flat_x = saved_x.reshape(-1, saved_x.shape[-1])
            grad_weight = torch.mm(flat_x.t(), grad, out_dtype=torch.float32).to(ctx.weight_dtype)
        if ctx.needs_input_grad[2]:
            grad_bias = grad.float().sum_to_size(ctx.bias_shape).to(ctx.bias_dtype)
        return grad_x, grad_weight, grad_bias


def matmul_fp32_reduce(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Keep the CUDA reduction and optional bias in FP32 until output rounding."""
    if _wants_fp32_reduce(x):
        if _needs_wide_autograd(x, weight, bias):
            return _CudaWideMatmul.apply(x, weight, bias)
        shape = x.shape[:-1] + (weight.shape[-1],)
        # FP32 output also keeps split reductions wide, while retaining low-precision
        # tensor-core operands. No expanded FP32 weight copy is needed.
        flat_x = x.reshape(-1, x.shape[-1])
        if bias is not None:
            output = torch.addmm(bias.float(), flat_x, weight, out_dtype=torch.float32)
        else:
            output = torch.mm(flat_x, weight, out_dtype=torch.float32)
        return output.to(x.dtype).reshape(shape)
    output = torch.matmul(x, weight)
    return output + bias if bias is not None else output


def linear_fp32_reduce(x, weight, bias=None):
    if _wants_fp32_reduce(x):
        if _needs_wide_autograd(x, weight, bias):
            return _CudaWideMatmul.apply(x, weight.t(), bias)
        shape = x.shape[:-1] + (weight.shape[0],)
        output = torch.mm(x.reshape(-1, x.shape[-1]), weight.t(), out_dtype=torch.float32)
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype).reshape(shape)
    if (
        IS_ROCM
        and x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and bias is not None
        and x.ndim > 2
    ):
        # F.linear can use matmul + add for non-contiguous batched inputs,
        # rounding (or overflowing) before bias cancels the partial result.
        # A 2D input selects the fused addmm epilogue without an FP32 output
        # allocation or an expanded weight copy.
        shape = x.shape[:-1] + (weight.shape[0],)
        return torch.nn.functional.linear(x.flatten(0, -2), weight, bias).reshape(shape)
    return torch.nn.functional.linear(x, weight, bias)
