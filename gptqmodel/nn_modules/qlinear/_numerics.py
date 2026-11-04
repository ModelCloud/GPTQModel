# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch


def matmul_fp32_reduce(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep CUDA low-precision partial sums in FP32 without changing global cuBLAS flags."""
    if x.device.type == "cuda" and x.dtype in (torch.float16, torch.bfloat16):
        shape = x.shape[:-1] + (weight.shape[-1],)
        # FP32 output also keeps split reductions wide, while retaining low-precision
        # tensor-core operands. No expanded FP32 weight copy is needed.
        return torch.mm(x.reshape(-1, x.shape[-1]), weight, out_dtype=torch.float32).to(x.dtype).reshape(shape)
    return torch.matmul(x, weight)


def linear_fp32_reduce(x, weight, bias=None):
    if x.device.type == "cuda" and x.dtype in (torch.float16, torch.bfloat16):
        shape = x.shape[:-1] + (weight.shape[0],)
        output = torch.mm(x.reshape(-1, x.shape[-1]), weight.t(), out_dtype=torch.float32)
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype).reshape(shape)
    return torch.nn.functional.linear(x, weight, bias)
