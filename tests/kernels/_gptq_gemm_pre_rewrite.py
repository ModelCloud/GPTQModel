# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Frozen pre-#3982 ``gptq_gemm`` kernel used as the bitwise reference.

The split-off path of the packed-load rewrite must reproduce this kernel's FP32
and visible outputs exactly.  Keep this file frozen: it is the old behaviour,
not an implementation under test.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pre_rewrite_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    m,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    bits: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1

    rows_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    m_mask = rows_m < m
    n_mask = cols < out_features

    accumulator = tl.zeros((block_m, block_n), tl.float32)
    k_offsets = tl.arange(0, block_k)

    for chunk in range(tl.cdiv(in_features, block_k)):
        rows = chunk * block_k + k_offsets
        row_mask = rows < in_features
        groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
        groups = tl.where(groups < 0, groups + num_groups, groups)
        group_mask = row_mask & (groups >= 0) & (groups < num_groups)
        weight_mask = group_mask[:, None] & n_mask[None, :]

        weight_words = tl.load(
            qweight_ptr + (rows[:, None] // pack_factor) * out_features + cols[None, :],
            weight_mask, other=0,
        )
        zero_words = tl.load(
            qzeros_ptr + groups[:, None] * (out_features // pack_factor) + cols[None, :] // pack_factor,
            weight_mask, other=0,
        )
        scales = tl.load(
            scales_ptr + groups[:, None] * out_features + cols[None, :],
            weight_mask, other=0,
        ).to(tl.float32)

        codes = (weight_words >> ((rows[:, None] % pack_factor) * bits)) & maxq
        zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
        weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty)

        activations = tl.load(
            x_ptr + rows_m[:, None] * in_features + rows[None, :],
            m_mask[:, None] & row_mask[None, :], other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator)

    tl.store(
        out_ptr + rows_m[:, None] * out_features + cols[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        m_mask[:, None] & n_mask[None, :],
    )


def pre_rewrite_gptq_gemm(
    x, qweight, scales, qzeros, g_idx, out_dtype=None, bits=4,
    block_m=16, block_n=16, block_k=256, num_warps=4, num_stages=2,
):
    """Launch the frozen kernel with the arch's pre-rewrite launch tile."""
    m, in_features = x.shape
    out_features = qweight.shape[1]
    output = torch.empty(
        (m, out_features), device=x.device,
        dtype=x.dtype if out_dtype is None else out_dtype,
    )
    grid = (triton.cdiv(m, block_m), triton.cdiv(out_features, block_n))
    with torch.cuda.device(x.device):
        _pre_rewrite_kernel[grid](
            x, qweight, scales, qzeros, g_idx, output,
            m, in_features, out_features, scales.shape[0],
            block_m, block_n, block_k, bits,
            num_warps=num_warps, num_stages=num_stages,
        )
    return output
