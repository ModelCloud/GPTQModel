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
# preserve the fused bias epilogues for HIP linear/matmul calls below.
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


def autocast_active(device_type: str) -> bool:
    """Return whether an autocast context is active for ``device_type``."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4 takes no device argument (CUDA only)
        return torch.is_autocast_enabled()


def _can_fuse_hip_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Whether ``addmm`` must replace the separate bias add on this platform.

    HIP keeps the low-precision output because the wide FP32 request is
    CUDA-only, so a separate add rounds (and can overflow) the partial sum
    before the bias cancels it; ``addmm`` matches ``(x @ w + bias)`` in FP32 on
    HIP bitwise. The branch is deliberately scoped to ROCm low-precision
    activations: NVIDIA FP32 inputs never round before the add, so their
    expression stays unchanged.

    Two shapes of input keep the unfused expression:

    * an active autocast context narrows ``addmm`` operands to the autocast
      dtype (a BF16 bias becomes FP16) while the standalone addition kept its
      range through type promotion;
    * a bias that needs broadcasting beyond the flattened output shape cannot
      be expressed by the 2D ``addmm`` input.
    """
    if bias is None or not IS_ROCM or x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or bias.dtype != x.dtype:
        return False
    if bias.device != x.device or autocast_active(x.device.type):
        return False
    return bias.dim() == 1 or bias.shape == x.shape[:-1] + (weight.shape[-1],)


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
        if autocast_active(x.device.type):
            # Autocast narrows the GEMM operands, so a bias fused into the
            # epilogue is narrowed too (a BF16 bias can overflow as FP16).
            # Keep the original expression, where the standalone addition
            # retains the promoted range.
            output = torch.matmul(x, weight)
            return output + bias if bias is not None else output
        # FP32 output also keeps split reductions wide, while retaining low-precision
        # tensor-core operands. No expanded FP32 weight copy is needed.
        flat_x = x.reshape(-1, x.shape[-1])
        if bias is not None and (bias.dim() == 1 or bias.shape == shape):
            flat_bias = bias if bias.dim() == 1 else bias.reshape(-1, weight.shape[-1])
            output = torch.addmm(flat_bias.float(), flat_x, weight, out_dtype=torch.float32)
        elif bias is not None:
            # A bias that broadcasts beyond the flattened output shape cannot
            # be expressed by the 2D `addmm` input; add it after reshaping.
            output = torch.mm(flat_x, weight, out_dtype=torch.float32).reshape(shape)
            return (output + bias).to(x.dtype)
        else:
            output = torch.mm(flat_x, weight, out_dtype=torch.float32)
        return output.to(x.dtype).reshape(shape)
    if _can_fuse_hip_bias(x, weight, bias):
        # Fuse the bias into the GEMM epilogue instead of rounding first.
        flat_x = x.reshape(-1, x.shape[-1])
        flat_bias = bias if bias.dim() == 1 else bias.reshape(-1, weight.shape[-1])
        return torch.addmm(flat_bias, flat_x, weight).reshape(x.shape[:-1] + (weight.shape[-1],))
    output = torch.matmul(x, weight)
    return output + bias if bias is not None else output


def linear_wide_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Multiply, add the stored bias in FP32, and round once to the output dtype.

    Callers select this path when the bias must not be narrowed to the
    activation dtype first. Two cases need it:

    * the stored bias is wider than the activation dtype, so casting it can
      overflow (an FP32 ``-65536`` bias is ``-inf`` as FP16) or lose the exact
      value that cancels the reduction;
    * the partial sum is outside the activation dtype's finite range and only
      the bias brings it back, which a reduced-precision product cannot
      represent.

    The weight is already at its reconstructed precision when this runs, and
    the wider operands exist only for this call; nothing is cached.
    """

    shape = x.shape[:-1] + (weight.shape[0],)
    flat_x = x.reshape(-1, x.shape[-1])
    if bias is not None and bias.device != x.device:
        bias = bias.to(device=x.device)
    weight_t = weight.t()
    wide_autograd = _needs_wide_autograd(x, weight, bias)
    # A bias that does not broadcast against the flattened 2-D product (for
    # example a per-token bias over a 3-D input) must be added after reshaping,
    # where the original broadcast shape applies.
    flat_shape = (flat_x.shape[0], weight.shape[0])
    reshape_bias = bias is not None and bias.dim() != 1 and bias.shape != flat_shape

    def reduce() -> torch.Tensor:
        if x.device.type == "cuda" and not wide_autograd:
            # FP32 output keeps the low-precision operands (no expanded FP32
            # weight copy) and keeps split reductions wide.
            output = torch.mm(flat_x, weight_t, out_dtype=torch.float32)
        else:
            output = torch.mm(flat_x.float(), weight_t.float())
        if bias is not None:
            if reshape_bias:
                output = output.reshape(shape) + bias.float()
            else:
                output = output + bias.float()
        return output.to(x.dtype).reshape(shape)

    if not autocast_active(x.device.type):
        return reduce()
    # Autocast narrows the GEMM operands, which would round the product before
    # the bias cancels it.
    with torch.autocast(device_type=x.device.type, enabled=False):
        return reduce()


# Relative width of the floating dtypes a checkpoint may store a bias in. A
# promotion is only allowed when it is lossless for every value of the source
# dtype; FP16 and BF16 are deliberately equal because neither contains the
# other (BF16 has the range, FP16 the mantissa).
_FLOAT_WIDTH = {torch.float16: 0, torch.bfloat16: 0, torch.float32: 1, torch.float64: 2}


def bias_narrows_losslessly(bias: torch.Tensor, dtype: torch.dtype) -> bool:
    """Whether ``bias`` has the same values after a cast to ``dtype``.

    Used for kernels that add the bias after their own wide accumulation (VPTQ
    CUDA sums in FP32, or FP64 for BF16) and therefore only restrict the bias
    storage dtype, not its arithmetic. A value FP16 cannot hold, such as
    ``-65536``, is rejected so the caller keeps the FP32 epilogue.

    The comparison is computed in FP64, so it also rejects NaN (``torch.equal``
    never reports NaN as equal) and preserves the sign of zero.
    """

    if bias.dtype == dtype:
        return True
    # A device equality check synchronizes and cannot run in a CUDA graph;
    # Dynamo also cannot trace its data-dependent Python boolean. Choose the
    # safe wide epilogue in either mode. The graph still reads the live bias
    # tensor on every replay, including writes made through `.data`.
    if torch.compiler.is_compiling():
        return False
    if bias.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return False
    return torch.equal(bias.to(dtype=dtype).to(torch.float64), bias.to(torch.float64))


def linear_needs_wide_bias(x: torch.Tensor, bias: torch.Tensor | None) -> bool:
    """Whether the stored bias must stay wide through the reduction epilogue."""

    return (
        bias is not None
        and x.dtype in (torch.float16, torch.bfloat16)
        and bias.dtype != x.dtype
        # A narrower bias can be widened losslessly inside the GEMM; only a
        # wider (or incomparable) one must stay out of the low-precision
        # operands.
        and _FLOAT_WIDTH.get(bias.dtype, 3) >= _FLOAT_WIDTH.get(x.dtype, 3)
    )


def promote_bias(x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
    """Promote a narrower stored bias to the activation dtype, never narrow one.

    ``F.linear`` requires the bias and the operands to share a dtype. A bias
    stored in a dtype narrower than the activation (for example an FP16 buffer
    with an FP32 activation) is widened losslessly, which is safe. A bias in a
    wider or incomparable dtype is returned unchanged so the caller can add it
    after the reduction instead of rounding its value first.
    """

    if bias is None or bias.dtype == x.dtype:
        return bias
    if _FLOAT_WIDTH.get(bias.dtype, 3) < _FLOAT_WIDTH.get(x.dtype, 3):
        return bias.to(dtype=x.dtype)
    return bias


def _fuse_bias_epilogue(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> bool:
    """Whether a low-precision separate bias add must move into the GEMM.

    On CPU and HIP, ``F.linear`` may use matmul plus add for non-contiguous
    batched inputs, so a product outside the activation dtype's finite range
    becomes ``inf`` before a cancelling bias is added. A fused ``addmm`` keeps
    the reduction and the bias in one epilogue without an FP32 output
    allocation or an expanded weight copy. Only a 1-D bias, or one already
    matching the output shape, can be expressed by the flattened 2-D input; an
    active autocast context narrows the operands and keeps the original
    expression, where type promotion applies.
    """

    if bias is None or bias.dtype != x.dtype or weight.dtype != x.dtype:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if x.device.type != "cpu" and not (IS_ROCM and x.device.type == "cuda"):
        return False
    if bias.device != x.device or autocast_active(x.device.type):
        return False
    return bias.dim() == 1 or bias.shape == x.shape[:-1] + (weight.shape[0],)


def _fused_addmm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    shape = x.shape[:-1] + (weight.shape[0],)
    flat_bias = bias if bias.dim() == 1 else bias.reshape(-1, weight.shape[0])
    return torch.addmm(flat_bias, x.reshape(-1, x.shape[-1]), weight.t()).reshape(shape)


def linear_fp32_reduce(x, weight, bias=None):
    if linear_needs_wide_bias(x, bias):
        # A bias that cannot be represented in the activation dtype must be
        # added after the product in FP32, never narrowed first.
        return linear_wide_bias(x, weight, bias)
    bias = promote_bias(x, bias)
    if _wants_fp32_reduce(x):
        if _needs_wide_autograd(x, weight, bias):
            return _CudaWideMatmul.apply(x, weight.t(), bias)
        shape = x.shape[:-1] + (weight.shape[0],)
        output = torch.mm(x.reshape(-1, x.shape[-1]), weight.t(), out_dtype=torch.float32)
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype).reshape(shape)
    if _fuse_bias_epilogue(x, weight, bias):
        # CPU/HIP may round or overflow the product before a separate bias
        # addition; keep both in a single epilogue.
        return _fused_addmm(x, weight, bias)
    return torch.nn.functional.linear(x, weight, bias)
