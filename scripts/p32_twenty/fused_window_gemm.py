"""Experimental P32 decode/MMA tile with configurable activation-row reuse.

Window bits and bank/codebook mapping are unchanged. Decoded B fragments feed
FP16-input/FP32-output MMA within each K16 tile. No dense weight cache is used.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(
    X,
    W,
    BANK,
    LEVELS,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    ALT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    DECODE_ONLY: tl.constexpr = False,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    npair = cols // 2
    kk = tl.arange(0, 16)
    acc = tl.full((BM, BN), 0, tl.float32)
    start = tl.program_id(2) * (K // SPLIT)
    for offset in range(0, K // SPLIT, 16):
        krow = start + offset + kk
        tile = (krow[:, None] // 16) * (N // 16) + npair[None, :] // 8
        pair = (kk[:, None] % 16) * 8 + npair[None, :] % 8
        bit = (127 - pair) * T
        word, shift = bit // 32, bit % 32
        mask = npair[None, :] < N // 2
        lo = tl.load(W + tile * (4 * T) + word, mask, 0).to(tl.uint32)
        hi = tl.load(W + tile * (4 * T) + (word + 1) % (4 * T), mask, 0).to(tl.uint32)
        state = ((lo >> shift) | tl.where(shift > 0, hi << (32 - shift), 0)) & 65535
        bank = tl.load(BANK + tile, mask, 0).to(tl.uint32)
        state = state ^ (((bank >> (pair // 16)) & 1) * ALT)
        mixed = state ^ (state >> 8)
        mixed = (mixed * 40503 + 17011) & 65535
        mixed = mixed ^ (mixed >> 7)
        index = tl.where((cols[None, :] % 2) == 0, mixed >> 8, mixed & 255)
        b = tl.load(LEVELS + index)
        if DECODE_ONLY:
            tl.store(Y + krow[:, None] * N + cols[None, :], b, cols[None, :] < N)
        a = tl.load(X + rows[:, None] * K + krow[None, :], rows[:, None] < M, 0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(
        Y + tl.program_id(2) * M * N + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N) & (not DECODE_ONLY),
    )


def fused_window_mm(
    x,
    window,
    levels,
    bank,
    bits,
    *,
    out_features,
    bank_alt_id,
    block_m=16,
    block_n=32,
    split=1,
):
    if x.dtype != torch.float16 or levels.dtype != torch.float16:
        raise ValueError("FP16 activation and canonical level buffers required")
    if (
        x.ndim != 2
        or not x.is_cuda
        or any(
            t.device != x.device or not t.is_contiguous()
            for t in [x, window, levels, bank]
        )
    ):
        raise ValueError("Expected contiguous tensors on one CUDA device")
    if torch.cuda.get_device_capability(x.device) != (8, 0):
        raise ValueError("This experimental kernel is validated only on sm80")
    if levels.numel() != 256 or window.dtype != torch.int32 or bank.dtype != torch.uint8:
        raise ValueError("Expected canonical 256-level, int32-word, uint8-bank buffers")
    m, k = x.shape
    n = out_features
    t = int(2 * bits)
    if (
        bits not in (2, 2.5, 3, 3.5)
        or block_m not in (16, 32, 64)
        or block_n not in (32, 64)
    ):
        raise ValueError("Unsupported study configuration")
    if split < 1 or k % (16 * split) or n % 16:
        raise ValueError("Split must partition full K16 tiles")
    if window.numel() != (k // 16) * (n // 16) * 4 * t or bank.numel() != (k // 16) * (
        n // 16
    ):
        raise ValueError("Payload geometry mismatch")
    masks = {
        4: [0, 0x5A5A, 0x3C3C, 0xC3C3],
        5: [0, 0x9696, 0x3C3C, 0xC3C3],
        6: [0, 0x6969, 0x5A5A, 0x3C3C],
        7: [0, 0xC3C3, 0x9696, 0x5A5A],
    }
    if bank_alt_id not in range(4):
        raise ValueError("Invalid bank alternate")
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _gemm[(triton.cdiv(m, block_m), triton.cdiv(n, block_n), split)](
        x,
        window,
        bank,
        levels,
        partial,
        m,
        k,
        n,
        t,
        masks[t][bank_alt_id],
        block_m,
        block_n,
        split,
        num_warps=4,
        num_stages=2,
    )
    return partial[0] if split == 1 else partial.sum(0)
