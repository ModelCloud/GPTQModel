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


def matmul_fp32_reduce(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep CUDA low-precision partial sums in FP32 without changing global cuBLAS flags."""
    if _wants_fp32_reduce(x):
        shape = x.shape[:-1] + (weight.shape[-1],)
        # FP32 output also keeps split reductions wide, while retaining low-precision
        # tensor-core operands. No expanded FP32 weight copy is needed.
        return torch.mm(x.reshape(-1, x.shape[-1]), weight, out_dtype=torch.float32).to(x.dtype).reshape(shape)
    return torch.matmul(x, weight)


def linear_fp32_reduce(x, weight, bias=None):
    if _wants_fp32_reduce(x):
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
