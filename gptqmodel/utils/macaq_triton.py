# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-FileCopyrightText: 2026 the MACAQ authors
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Triton kernels for MACaQ W4A4 / W4A16 MXFP4 Linear layers.
#
# These kernels are a port of Huawei CSL's MACaQ reference kernels:
#   https://github.com/huawei-csl/MACaQ        (Apache License 2.0)
#   `quantize/macaq_vllm_mxfp4_dense.py` (dense W4A4 + activation quantizer)
#   `vllm/macaq_weight_only.py`          (W4A16 dequant path)
# Algorithm: "MACaQ: Fast, Model Agnostic Calibrated Quantization for
# Low-Precision Weights and Activations", Muller et al., arXiv:2611.10649.
#
# The forward contract is unchanged: the saved weight is already in the
# (D1-balance, block-Hadamard) frame, so the runtime only applies the same
# transform to the activation, MXFP4-quantizes it, and runs a block-scaled
# E2M1 GEMM.  Weights and activations both follow the OCP convention
# `value = code * 2 ** (e8m0 - 127)`, so the GEMM needs no correction.
#
# Port scope: the dense activation quantizer, fused rotate+quantize kernel,
# dense W4A4 `tl.dot_scaled` GEMM, and W4A16 dequant GEMM.  The upstream MoE
# kernels, W4A8 FP8 path, and cuDNN/CUTLASS backend are not included.

from __future__ import annotations

import math
from typing import Optional

import torch

from .logger import setup_logger
from .rocm import IS_ROCM


log = setup_logger()

try:  # pragma: no cover - import availability is environment dependent
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

MXFP4_BLOCK = 32
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def triton_mxfp4_available() -> bool:
    """Whether the Triton MXFP4 kernels can run on the current host."""

    return TRITON_AVAILABLE and torch.cuda.is_available() and not IS_ROCM


def make_hadamard(block: int, device="cpu") -> torch.Tensor:
    """Normalized ``block x block`` Sylvester Hadamard matrix."""

    if block <= 0 or block & (block - 1):
        raise ValueError(f"MACaQ: Hadamard size must be a positive power of two, got {block}.")
    rows = [[(-1.0) ** (i & j).bit_count() for j in range(block)] for i in range(block)]
    return torch.tensor(rows, dtype=torch.float32, device=device) / math.sqrt(block)


def rotate_blocks(x: torch.Tensor, bh: torch.Tensor) -> torch.Tensor:
    """Right-multiply ``x`` by the block-diagonal Hadamard matrix on the last axis."""

    block = bh.shape[0]
    lead = x.shape[:-1]
    return (x.reshape(*lead, x.shape[-1] // block, block) @ bh.to(x.dtype)).reshape(*lead, x.shape[-1])


if TRITON_AVAILABLE:

    @triton.jit
    def _quant_mxfp4_kernel(x_ptr, pk_ptr, sc_ptr, K,
                            stride_xr, stride_pr, stride_sr,
                            BLOCK_G: tl.constexpr):
        """One program = ``BLOCK_G`` scale groups (of 32) of one row."""

        row = tl.program_id(0)
        g0 = tl.program_id(1) * BLOCK_G
        gid = g0 + tl.arange(0, BLOCK_G)
        half = tl.arange(0, 16)
        col_lo = gid[:, None] * 32 + half[None, :] * 2
        col_hi = col_lo + 1
        xl = tl.load(x_ptr + row * stride_xr + col_lo, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
        xh = tl.load(x_ptr + row * stride_xr + col_hi, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xl), axis=1), tl.max(tl.abs(xh), axis=1))
        amax = tl.maximum(amax, 1e-20)
        e = tl.floor(tl.log2(amax)) - 2.0
        e = tl.minimum(tl.maximum(e, -127.0), 127.0)
        scale = tl.exp2(e)[:, None]

        al = tl.abs(xl) / scale
        clo = ((al >= 0.25).to(tl.int32) + (al >= 0.75) + (al >= 1.25) + (al >= 1.75)
               + (al >= 2.5) + (al >= 3.5) + (al >= 5.0))
        nlo = ((xl < 0).to(tl.int32) << 3) | clo
        ah = tl.abs(xh) / scale
        chi = ((ah >= 0.25).to(tl.int32) + (ah >= 0.75) + (ah >= 1.25) + (ah >= 1.75)
               + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0))
        nhi = ((xh < 0).to(tl.int32) << 3) | chi
        packed = (nlo | (nhi << 4)).to(tl.uint8)

        pk_off = row * stride_pr + gid[:, None] * 16 + half[None, :]
        tl.store(pk_ptr + pk_off, packed, mask=gid[:, None] < K // 32)
        sc_byte = (e + 127.0).to(tl.uint8)
        tl.store(sc_ptr + row * stride_sr + gid, sc_byte, mask=gid < K // 32)

    @triton.jit
    def _rot_scale_quant_kernel(x_ptr, ps_ptr, bh_ptr, pk_ptr, sc_ptr, K,
                                stride_xr, stride_pr, stride_sr,
                                BLOCK_G: tl.constexpr):
        """Fused ``(x * pre_scale) @ blkdiag(BH)`` then MXFP4 RTN."""

        row = tl.program_id(0)
        g0 = tl.program_id(1) * BLOCK_G
        gid = g0 + tl.arange(0, BLOCK_G)
        cols = tl.arange(0, 32)
        off = gid[:, None] * 32 + cols[None, :]
        x = tl.load(x_ptr + row * stride_xr + off, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
        ps = tl.load(ps_ptr + off, mask=gid[:, None] < K // 32, other=1).to(tl.float32)
        xs = (x * ps).to(tl.bfloat16)
        bh = tl.load(bh_ptr + cols[:, None] * 32 + cols[None, :]).to(tl.bfloat16)
        r = tl.dot(xs, bh, out_dtype=tl.float32)
        r = r.to(tl.bfloat16).to(tl.float32)
        rl, rh = tl.split(tl.reshape(r, (BLOCK_G, 16, 2)))
        amax = tl.maximum(tl.max(tl.abs(rl), axis=1), tl.max(tl.abs(rh), axis=1))
        amax = tl.maximum(amax, 1e-20)
        e = tl.floor(tl.log2(amax)) - 2.0
        e = tl.minimum(tl.maximum(e, -127.0), 127.0)
        scale = tl.exp2(e)[:, None]
        al = tl.abs(rl) / scale
        clo = ((al >= 0.25).to(tl.int32) + (al >= 0.75) + (al >= 1.25) + (al >= 1.75)
               + (al >= 2.5) + (al >= 3.5) + (al >= 5.0))
        nlo = ((rl < 0).to(tl.int32) << 3) | clo
        ah = tl.abs(rh) / scale
        chi = ((ah >= 0.25).to(tl.int32) + (ah >= 0.75) + (ah >= 1.25) + (ah >= 1.75)
               + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0))
        nhi = ((rh < 0).to(tl.int32) << 3) | chi
        packed = (nlo | (nhi << 4)).to(tl.uint8)
        half = tl.arange(0, 16)
        pk_off = row * stride_pr + gid[:, None] * 16 + half[None, :]
        tl.store(pk_ptr + pk_off, packed, mask=gid[:, None] < K // 32)
        tl.store(sc_ptr + row * stride_sr + gid, (e + 127.0).to(tl.uint8), mask=gid < K // 32)

    @triton.jit
    def _dense_w4a4_kernel(
        a_ptr, b_ptr, asc_ptr, bsc_ptr, c_ptr, M, N, K,
        stride_am, stride_bn, stride_asm, stride_bsn, stride_cm,
        OUT_SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_kb = tl.arange(0, BLOCK_K // 2)
        offs_ks = tl.arange(0, BLOCK_K // 32)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_kb[None, :]
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_kb[None, :]
        asc_ptrs = asc_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :]
        bsc_ptrs = bsc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block in range(0, tl.cdiv(K, BLOCK_K)):
            # BLOCK_K need not divide K (896, 1152, 2176).  Mask the tail tile
            # with the loop offset, as the upstream MoE kernel does; a mask
            # built once from offs_kb alone only clips the first tile and lets
            # later tiles read the next row's packed bytes and E8M0 exponents.
            kb_mask = block * (BLOCK_K // 2) + offs_kb < K // 2
            ks_mask = block * (BLOCK_K // 32) + offs_ks < K // 32
            a = tl.load(a_ptrs, mask=m_mask[:, None] & kb_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=n_mask[:, None] & kb_mask[None, :], other=0)
            asc = tl.load(asc_ptrs, mask=m_mask[:, None] & ks_mask[None, :], other=127)
            bsc = tl.load(bsc_ptrs, mask=n_mask[:, None] & ks_mask[None, :], other=127)
            acc = tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1", acc)
            a_ptrs += BLOCK_K // 2
            b_ptrs += BLOCK_K // 2
            asc_ptrs += BLOCK_K // 32
            bsc_ptrs += BLOCK_K // 32

        acc = acc * OUT_SCALE
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _dequant_w4_to_bf16_kernel(pk_ptr, sc_ptr, out_ptr, K, BLOCK_G: tl.constexpr):
        """Packed MXFP4 row -> bf16.  ``value = magnitude * 2 ** (e8m0 - 127)``."""

        row = tl.program_id(0)
        gid = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
        half = tl.arange(0, 16)
        Kg = K // 32
        mask = (gid < Kg)[:, None]
        pk = tl.load(pk_ptr + row * (K // 2) + gid[:, None] * 16 + half[None, :], mask=mask, other=0)
        e = tl.load(sc_ptr + row * Kg + gid, mask=gid < Kg, other=127).to(tl.float32)
        lo = (pk & 0xF).to(tl.int32)
        hi = ((pk >> 4) & 0xF).to(tl.int32)
        mlo = lo & 7
        mhi = hi & 7
        maglo = tl.where(mlo < 4, mlo.to(tl.float32) * 0.5,
                         tl.where(mlo == 4, 2.0, tl.where(mlo == 5, 3.0, tl.where(mlo == 6, 4.0, 6.0))))
        maghi = tl.where(mhi < 4, mhi.to(tl.float32) * 0.5,
                         tl.where(mhi == 4, 2.0, tl.where(mhi == 5, 3.0, tl.where(mhi == 6, 4.0, 6.0))))
        factor = tl.exp2(e - 127.0)[:, None]
        vlo = tl.where((lo >> 3) == 1, -1.0, 1.0) * maglo * factor
        vhi = tl.where((hi >> 3) == 1, -1.0, 1.0) * maghi * factor
        base = out_ptr + row * K + gid[:, None] * 32 + half[None, :] * 2
        tl.store(base, vlo.to(tl.bfloat16), mask=mask)
        tl.store(base + 1, vhi.to(tl.bfloat16), mask=mask)


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4-quantize a 2-D activation matrix with the ported Triton kernel."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MACaQ: Triton is not available.")
    if x.ndim != 2 or x.shape[1] % MXFP4_BLOCK:
        raise ValueError("MACaQ: activation quantization expects [rows, K] with K divisible by 32.")
    rows, k = x.shape
    groups = k // MXFP4_BLOCK
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(rows, groups, dtype=torch.uint8, device=x.device)
    block_groups = min(triton.next_power_of_2(groups), 16)
    grid = (rows, triton.cdiv(groups, block_groups))
    _quant_mxfp4_kernel[grid](
        x, packed, e8m0, k,
        x.stride(0), packed.stride(0), e8m0.stride(0),
        BLOCK_G=block_groups,
    )
    return packed, e8m0


def rot_scale_quantize_mxfp4(
    x: torch.Tensor,
    pre_scale: Optional[torch.Tensor] = None,
    bh: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``(x * pre_scale) @ blkdiag(BH)`` then MXFP4 quantization."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MACaQ: Triton is not available.")
    if bh is None or bh.shape[0] != MXFP4_BLOCK:
        raise ValueError("MACaQ: the fused kernel requires a 32x32 block-Hadamard.")
    if x.ndim != 2 or x.shape[1] % MXFP4_BLOCK:
        raise ValueError("MACaQ: activation quantization expects [rows, K] with K divisible by 32.")
    rows, k = x.shape
    groups = k // MXFP4_BLOCK
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(rows, groups, dtype=torch.uint8, device=x.device)
    scale = (
        pre_scale
        if pre_scale is not None
        else torch.ones(k, device=x.device, dtype=x.dtype)
    ).contiguous()
    block_groups = min(triton.next_power_of_2(groups), 16)
    grid = (rows, triton.cdiv(groups, block_groups))
    _rot_scale_quant_kernel[grid](
        x, scale, bh.contiguous(), packed, e8m0, k,
        x.stride(0), packed.stride(0), e8m0.stride(0),
        BLOCK_G=block_groups,
    )
    return packed, e8m0


def dequantize_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor, k: int) -> torch.Tensor:
    """Torch inverse of the activation quantizer, for reference and tests."""

    rows = packed.shape[0]
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    code = torch.stack([lo, hi], dim=-1).reshape(rows, k).long()
    sign = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
    value = sign * _E2M1_MAG.to(packed.device)[code & 7]
    scale_bytes = e8m0.view(torch.uint8)[:rows, :k // MXFP4_BLOCK]
    scale = torch.exp2((scale_bytes.float() - 127.0)).repeat_interleave(32, dim=1)
    return value * scale


def dequant_w4_to_bf16(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode packed MXFP4 weights to the exact bf16 rotated-frame matrix."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MACaQ: Triton is not available.")
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("MACaQ: MXFP4 weights must be a 2-D uint8 tensor.")
    rows, half_k = packed.shape
    k = half_k * 2
    if k % MXFP4_BLOCK:
        raise ValueError("MACaQ: MXFP4 requires 32-wide groups.")
    scale_bytes = scales.view(torch.uint8)[:rows, : k // MXFP4_BLOCK].contiguous()
    out = torch.empty(rows, k, dtype=torch.bfloat16, device=packed.device)
    _dequant_w4_to_bf16_kernel[(rows, triton.cdiv(k // MXFP4_BLOCK, 16))](
        packed.contiguous(), scale_bytes, out, k, BLOCK_G=16
    )
    return out


def mxfp4_w4a4_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    *,
    pre_scale: Optional[torch.Tensor] = None,
    bh: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 256,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Packed MXFP4 weights x MXFP4 activations (native block-scaled GEMM)."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MACaQ: Triton is not available.")
    lead = x.shape[:-1]
    k = x.shape[-1]
    x2 = x.reshape(-1, k).contiguous()
    if bh is not None and bh.shape[0] == MXFP4_BLOCK:
        a_packed, a_scales = rot_scale_quantize_mxfp4(x2, pre_scale, bh)
    else:
        work = x2.float()
        if pre_scale is not None:
            work = work * pre_scale.float()
        work = work.to(torch.bfloat16).float()
        if bh is not None:
            work = rotate_blocks(work, bh.float()).to(torch.bfloat16).float()
        a_packed, a_scales = quantize_mxfp4(work.contiguous())

    rows = x2.shape[0]
    n = w_packed.shape[0]
    # `tl.dot_scaled` tiles must not straddle the K boundary; pick the largest
    # power-of-two tile (up to the caller's request) that divides K.
    while block_k > 32 and (block_k > k or k % block_k):
        block_k //= 2
    out = torch.empty(rows, n, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    # Triton 3.9's pointer canonicalizer does not accept the E8M0 dtype
    # directly; the bytes are the same and `dot_scaled` consumes them as E8M0.
    w_scales_bytes = w_scales.view(torch.uint8)
    _dense_w4a4_kernel[grid](
        a_packed, w_packed, a_scales, w_scales_bytes, out, rows, n, k,
        a_packed.stride(0), w_packed.stride(0), a_scales.stride(0), w_scales_bytes.stride(0),
        out.stride(0),
        OUT_SCALE=out_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype).reshape(*lead, n)


def mxfp4_w4a16_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    *,
    pre_scale: Optional[torch.Tensor] = None,
    bh: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Packed MXFP4 weights with full-precision activations in ``compute_dtype``."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MACaQ: Triton is not available.")
    compute_dtype = compute_dtype or x.dtype
    xb = x.to(compute_dtype)
    if pre_scale is not None:
        xb = xb * pre_scale.to(compute_dtype)
    if bh is not None:
        xb = rotate_blocks(xb, bh)
    weight = dequant_w4_to_bf16(w_packed, w_scales).to(compute_dtype)
    return torch.nn.functional.linear(
        xb, weight, None if bias is None else bias.to(xb.dtype)
    )
