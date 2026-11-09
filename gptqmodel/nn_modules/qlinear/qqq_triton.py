# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Operation-local QQQ activation quantization with FP32 scales."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * K + columns, columns < K, other=0).to(tl.float32)
    # Match Torch's scalar division: multiply by the FP32 reciprocal of 127.
    # Replacing this with div.rn changes codes at exact half-step boundaries.
    scale = tl.max(tl.abs(values), 0) * (1.0 / 127.0)
    scale = tl.where(scale == 0, 1.0, scale)
    codes = libdevice.nearbyint(tl.div_rn(values, scale))
    codes = tl.minimum(tl.maximum(codes, -128.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + columns, codes, columns < K)
    tl.store(S + row, scale)


def dynamic_quant(x):
    # The Triton kernel indexes each activation row with a dense K stride.
    x = x.contiguous()
    codes = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    rows = x.numel() // x.shape[-1]
    if rows:
        with torch.cuda.device(x.device):
            _quantize_rows[(rows,)](
                x, codes, scales, x.shape[-1], triton.next_power_of_2(x.shape[-1]),
                num_warps=4 if x.shape[-1] <= 4096 else 8,
                enable_fp_fusion=False,
            )
    return codes, scales
