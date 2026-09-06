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
    PROMOTION_K: tl.constexpr,
    FOLDED_ADDRESSES: tl.constexpr = False,
    PREDICATED_BANK_XOR: tl.constexpr = False,
    PAIR_LEVEL_LUT: tl.constexpr = False,
    LUT_CACHE_CA: tl.constexpr = False,
    LUT_CACHE_CG: tl.constexpr = False,
    RESIDENT_WINDOW_WORDS: tl.constexpr = False,
    DECODE_ONLY: tl.constexpr = False,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    npair = cols // 2
    kk = tl.arange(0, 16)
    acc = tl.full((BM, BN), 0, tl.float32)
    partial = tl.full((BM, BN), 0, tl.float16)
    start = tl.program_id(2) * (K // SPLIT)
    for offset in range(0, K // SPLIT, 16):
        krow = start + offset + kk
        if RESIDENT_WINDOW_WORDS:
            kblock = (start + offset) // 16
            pair_col = npair % 8
            pair = kk[:, None] * 8 + pair_col[None, :]
            tile_slots = tl.arange(0, BN // 16)
            tile_ids = (
                kblock * (N // 16)
                + tl.program_id(1) * (BN // 16)
                + tile_slots
            )
            # Triton arange bounds must be powers of two.  The largest P32
            # window tile is 28 words, so load a masked 32-word register tile.
            word_ids = tl.arange(0, 32)
            tile_words = tl.load(
                W + tile_ids[:, None] * (4 * T) + word_ids[None, :],
                mask=(tile_ids[:, None] < (K // 16) * (N // 16))
                & (word_ids[None, :] < 4 * T),
                other=0,
            )
            flat_words = tl.reshape(tile_words, (BN // 16) * 32)
            tile_slot = npair // 8 - tl.program_id(1) * (BN // 16)
            bit = (127 - pair) * T
            word, shift = bit // 32, bit % 32
            word_index = tile_slot[None, :] * 32 + word
            next_word_index = tile_slot[None, :] * 32 + (word + 1) % (4 * T)
            lo = tl.reshape(
                tl.gather(flat_words, tl.reshape(word_index, (16 * BN,)), axis=0),
                (16, BN),
            ).to(tl.uint32)
            hi = tl.reshape(
                tl.gather(flat_words, tl.reshape(next_word_index, (16 * BN,)), axis=0),
                (16, BN),
            ).to(tl.uint32)
            state = ((lo >> shift) | tl.where(shift > 0, hi << (32 - shift), 0)) & 65535
            tile_banks = tl.load(
                BANK + tile_ids,
                mask=tile_ids < (K // 16) * (N // 16),
                other=0,
            ).to(tl.uint32)
            bank = tl.gather(tile_banks, tile_slot, axis=0)[None, :]
        elif FOLDED_ADDRESSES:
            # For every K16 step, all 16 lanes in the K dimension address the
            # same window tile column. Keep that address vector one-dimensional
            # so the compiler does not regenerate it for each decoded value.
            kblock = (start + offset) // 16
            tile = kblock * (N // 16) + npair // 8
            pair = kk[:, None] * 8 + npair[None, :] % 8
            tile_for_load = tile[None, :]
        else:
            tile = (krow[:, None] // 16) * (N // 16) + npair[None, :] // 8
            pair = (kk[:, None] % 16) * 8 + npair[None, :] % 8
            tile_for_load = tile
            bit = (127 - pair) * T
            word, shift = bit // 32, bit % 32
            mask = npair[None, :] < N // 2
            lo = tl.load(W + tile_for_load * (4 * T) + word, mask, 0).to(tl.uint32)
            hi = tl.load(W + tile_for_load * (4 * T) + (word + 1) % (4 * T), mask, 0).to(tl.uint32)
            state = ((lo >> shift) | tl.where(shift > 0, hi << (32 - shift), 0)) & 65535
            bank = tl.load(BANK + tile_for_load, mask, 0).to(tl.uint32)
        bank_bit = (bank >> (pair // 16)) & 1
        if PREDICATED_BANK_XOR:
            state = state ^ tl.where(bank_bit != 0, ALT, 0)
        else:
            state = state ^ (bank_bit * ALT)
        if PAIR_LEVEL_LUT:
            parity = cols[None, :] & 1
            if LUT_CACHE_CA:
                b = tl.load(LEVELS + state * 2 + parity, cache_modifier=".ca")
            elif LUT_CACHE_CG:
                b = tl.load(LEVELS + state * 2 + parity, cache_modifier=".cg")
            else:
                b = tl.load(LEVELS + state * 2 + parity)
        else:
            mixed = state ^ (state >> 8)
            mixed = (mixed * 40503 + 17011) & 65535
            mixed = mixed ^ (mixed >> 7)
            index = tl.where((cols[None, :] % 2) == 0, mixed >> 8, mixed & 255)
            if LUT_CACHE_CA:
                b = tl.load(LEVELS + index, cache_modifier=".ca")
            elif LUT_CACHE_CG:
                b = tl.load(LEVELS + index, cache_modifier=".cg")
            else:
                b = tl.load(LEVELS + index)
        if DECODE_ONLY:
            tl.store(Y + krow[:, None] * N + cols[None, :], b, cols[None, :] < N)
        a = tl.load(X + rows[:, None] * K + krow[None, :], rows[:, None] < M, 0)
        if PROMOTION_K == 0:
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        else:
            partial = tl.dot(a, b, partial, out_dtype=tl.float16)
            if (offset + 16) % PROMOTION_K == 0:
                acc += partial.to(tl.float32)
                partial = tl.full((BM, BN), 0, tl.float16)
    if PROMOTION_K != 0:
        acc += partial.to(tl.float32)
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
    promotion_k=0,
    address_mode="baseline",
    bank_mode="multiply",
    decode_mode="scalar",
    lut_cache="default",
    window_mode="standard",
    num_warps=4,
    num_stages=2,
    maxnreg=0,
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
    if bank.dtype != torch.uint8:
        raise ValueError("Expected uint8-bank buffers")
    m, k = x.shape
    n = out_features
    t = int(2 * bits)
    if (
        bits not in (2, 2.5, 3, 3.5)
        or block_m not in (16, 32, 64, 128)
        or block_n not in (32, 64)
    ):
        raise ValueError("Unsupported study configuration")
    if split < 1 or k % (16 * split) or n % 16:
        raise ValueError("Split must partition full K16 tiles")
    if promotion_k not in (0, 16, 32, 64, 128, 256):
        raise ValueError("Unsupported FP32 promotion interval")
    if promotion_k and (k // split) % promotion_k:
        raise ValueError("Promotion interval must divide each split K range")
    if address_mode not in ("baseline", "factored"):
        raise ValueError("Unsupported address algebra mode")
    if bank_mode not in ("multiply", "predicated"):
        raise ValueError("Unsupported bank algebra mode")
    if decode_mode not in ("scalar", "pair-lut"):
        raise ValueError("Unsupported decode mode")
    if lut_cache not in ("default", "ca", "cg"):
        raise ValueError("Unsupported LUT cache mode")
    if window_mode not in ("standard", "resident-words"):
        raise ValueError("Unsupported window mode")
    expected_levels = 256 if decode_mode == "scalar" else 65536 * 2
    if levels.numel() != expected_levels:
        raise ValueError(f"Expected {expected_levels} decode levels for {decode_mode}")
    expected_payload_bytes = (k // 16) * (n // 16) * 16 * t
    if window.numel() * window.element_size() != expected_payload_bytes or bank.numel() != (k // 16) * (
        n // 16
    ):
        raise ValueError("Payload geometry mismatch")
    if window.dtype != torch.int32:
        raise ValueError("Standard window payload must use torch.int32")
    if num_warps not in (2, 4, 8) or num_stages not in (1, 2, 3, 4):
        raise ValueError("Unsupported launch configuration")
    if maxnreg not in (0, 48, 56, 64, 72, 80, 96):
        raise ValueError("Unsupported register cap")
    masks = {
        4: [0, 0x5A5A, 0x3C3C, 0xC3C3],
        5: [0, 0x9696, 0x3C3C, 0xC3C3],
        6: [0, 0x6969, 0x5A5A, 0x3C3C],
        7: [0, 0xC3C3, 0x9696, 0x5A5A],
    }
    if bank_alt_id not in range(4):
        raise ValueError("Invalid bank alternate")
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    launch = dict(num_warps=num_warps, num_stages=num_stages)
    if maxnreg:
        launch["maxnreg"] = maxnreg
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
        promotion_k,
        address_mode == "factored",
        bank_mode == "predicated",
        decode_mode == "pair-lut",
        lut_cache == "ca",
        lut_cache == "cg",
        window_mode == "resident-words",
        **launch,
    )
    return partial[0] if split == 1 else partial.sum(0)
