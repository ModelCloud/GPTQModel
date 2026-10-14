"""Fused GPTQ GEMM for small-batch (M in 2..16) forward passes.

``TorchLinear`` normally dequantizes the whole weight matrix and then calls
``torch.matmul``. That materializes an ``in_features x out_features``
activation-dtype tensor and then streams it straight back in: for small M the
extra write + read of that buffer costs more than the matmul itself. This
kernel dequantizes the low-bit weights *inside* the GEMM, so the packed weights
are read once and no dequantized weight buffer is ever written.

The single-token (M == 1) case is handled by :mod:`gptq_gemm`'s sibling
``gptq_gemv``; this module covers the small-M band used by speculative decoding
and small-batch serving, where the whole batch fits in one ``BLOCK_M`` tile and
the weights are therefore dequantized exactly once (same dequant work as the
split path, minus the buffer traffic).

Supports 2/4/8-bit GPTQ weights packed into ``int32`` words (the 10-1-10-1-10
3-bit layout is not handled here and falls back).

Measured on gfx950 (MI350X/MI355X) against the dequantize + matmul path with
``BLOCK_M=16, BLOCK_N=16, BLOCK_K=256, num_warps=4, num_stages=2`` (module
forward, M=2..16):

    K=896  N=128   : 1.44x-1.49x      K=4096 N=4096  : 1.46x
    K=896  N=896   : 1.43x-1.47x      K=4096 N=11008 : 1.24x-1.28x
    K=896  N=4864  : 1.45x-1.46x      K=5120 N=5120  : 1.05x-1.07x
    K=4864 N=896   : 1.42x-1.45x      K=5120 N=12288 : 1.17x-1.20x
    K=3584 N=3584  : 1.61x-1.65x      K=5120 N=13824 : 1.08x
    K=3584 N=18944 : 1.26x-1.28x

End to end on Qwen2.5-0.5B-Instruct (4-bit, M=2..16) the forward is 1.51x-1.54x
faster; M=1 and M>``SMALL_M_MAX`` are unchanged.

``BLOCK_N=16`` is deliberately small: it maximizes the grid size so the
memory-bound dequant side saturates the device (wider ``BLOCK_N`` starves
occupancy and was slower on every shape tested).

``MAX_IN_FEATURES`` bounds the shape gate. Past roughly K=6144 the kernel's
serial K-loop becomes latency-bound (there is not enough N-parallelism left to
hide it) and the split path wins again -- e.g. 0.68x at K=11008/N=4096 and
0.81x at K=8192/N=4096. The crossover is not monotonic in K or N, so the gate
uses the conservative bound that was measured to win on every shape at or below
it (1.05x-1.65x across K in 896..5120).
"""

import torch
import triton
import triton.language as tl


# Largest M handled by the fused small-batch GEMM. Above this the weight would
# be dequantized once per BLOCK_M tile, which loses to dequantize-once + GEMM.
SMALL_M_MAX = 16

# Largest in_features for which the fused kernel was measured to beat the split
# path. See the module docstring.
MAX_IN_FEATURES = 5120

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 256
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _gptq_gemm_small_m_kernel(
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


def gptq_gemm(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
              qzeros: torch.Tensor, g_idx: torch.Tensor,
              out_dtype: torch.dtype = None, bits: int = 4) -> torch.Tensor:
    m, in_features = x.shape
    out_features = qweight.shape[1]
    output = torch.empty(
        (m, out_features), device=x.device, dtype=x.dtype if out_dtype is None else out_dtype,
    )
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(out_features, BLOCK_N))
    with torch.cuda.device(x.device):
        _gptq_gemm_small_m_kernel[grid](
            x, qweight, scales, qzeros, g_idx, output,
            m, in_features, out_features, scales.shape[0],
            BLOCK_M, BLOCK_N, BLOCK_K, bits,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
    return output
