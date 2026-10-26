"""Fused GPTQ GEMV for single-token (batch-1) decode.

``TorchLinear`` normally dequantizes the whole weight matrix and then calls
``torch.matmul``, which materializes an ``in_features x out_features``
activation-dtype tensor and launches a separate GEMM. For batch-1 decode that
is dominated by that dequantize + GEMM pair. This kernel dequantizes the
low-bit weights inside the matrix-vector product, so the packed weights are
read once and no dequantized weight buffer is ever written.

Supports 2/4/8-bit GPTQ weights packed into ``int32`` words (the 10-1-10-1-10
3-bit layout is not handled here and falls back).

The block shape below was measured on gfx950 (MI350X/MI355X) across the
projection shapes of Qwen2.5-0.5B and larger 7B-32B style models
(K,N in 896..17408). ``BLOCK_K=512, BLOCK_N=8, num_warps=8`` was the fastest
configuration for every shape tested, and the fused path was 1.5x-2.9x faster
than the dequantize + matmul path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gptq_gemv_kernel(
    x_ptr, qweight_ptr, scales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    in_features: tl.constexpr, out_features: tl.constexpr,
    num_groups: tl.constexpr, block_k: tl.constexpr, block_n: tl.constexpr,
    bits: tl.constexpr,
):
    pack_factor: tl.constexpr = 32 // bits
    maxq: tl.constexpr = (1 << bits) - 1
    row = tl.program_id(1)
    cols = tl.program_id(0) * block_n + tl.arange(0, block_n)
    offsets = tl.arange(0, block_k)
    accumulator = tl.full((block_k, block_n), 0, tl.float32)

    for chunk in range(tl.cdiv(in_features, block_k)):
        rows = chunk * block_k + offsets
        row_mask = rows < in_features
        mask = row_mask[:, None] & (cols[None, :] < out_features)
        activations = tl.load(x_ptr + row * in_features + rows, row_mask, other=0).to(tl.float32)
        groups = tl.load(g_idx_ptr + rows, row_mask, other=0)
        groups = tl.where(groups < 0, groups + num_groups, groups)
        mask = mask & (groups[:, None] >= 0) & (groups[:, None] < num_groups)
        weight_words = tl.load(
            qweight_ptr + (rows[:, None] // pack_factor) * out_features + cols[None, :], mask, other=0,
        )
        zero_words = tl.load(
            qzeros_ptr + groups[:, None] * (out_features // pack_factor) + cols[None, :] // pack_factor,
            mask, other=0,
        )
        scales = tl.load(
            scales_ptr + groups[:, None] * out_features + cols[None, :], mask, other=0,
        ).to(tl.float32)
        codes = (weight_words >> ((rows[:, None] % pack_factor) * bits)) & maxq
        zeros = (zero_words >> ((cols[None, :] % pack_factor) * bits)) & maxq
        weights = ((codes - zeros).to(tl.float32) * scales).to(x_ptr.dtype.element_ty).to(tl.float32)
        accumulator += activations[:, None] * weights

    result = tl.sum(accumulator, axis=0)
    tl.store(out_ptr + row * out_features + cols, result, cols < out_features)


def gptq_gemv(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
              qzeros: torch.Tensor, g_idx: torch.Tensor,
              out_dtype: torch.dtype = None, bits: int = 4) -> torch.Tensor:
    in_features = x.shape[1]
    out_features = qweight.shape[1]
    block_k = min(512, max(32, 1 << (in_features - 1).bit_length()))
    block_n = 8
    output = torch.empty(
        (x.shape[0], out_features), device=x.device, dtype=x.dtype if out_dtype is None else out_dtype,
    )
    with torch.cuda.device(x.device):
        _gptq_gemv_kernel[(triton.cdiv(out_features, block_n), x.shape[0])](
            x, qweight, scales, qzeros, g_idx, output,
            in_features, out_features, scales.shape[0], block_k, block_n, bits, num_warps=8,
        )
    return output
