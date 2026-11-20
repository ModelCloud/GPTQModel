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
# dense W4A4 `tl.dot_scaled` GEMM, decode-aware tiles and fused output epilogue
# from `vllm/triton_mxfp4_dense.py`, and W4A16 dequant GEMM. GPT-QModel adds
# measured RDNA4 gfx1201 launch tiles. The upstream MoE kernels, W4A8 FP8
# path, and cuDNN/CUTLASS backend are not included.

from __future__ import annotations

import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch

from ..nn_modules.qlinear._numerics import linear_fp32_reduce
from .logger import setup_logger
from .rocm import IS_ROCM


log = setup_logger()

try:  # pragma: no cover - import availability is environment dependent
    import triton
    import triton.language as tl

    from .triton_jit import jit as triton_jit
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False

MXFP4_BLOCK = 32
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _launch_device(device: torch.device):
    """Bind a Triton launch to the device that owns the operands.

    Triton launches into the *current* CUDA/HIP context, but a checkpoint
    sharded over several devices hands each layer operands on that layer's
    device. Without this guard a layer on `cuda:1` raises
    ``ValueError: Pointer argument cannot be accessed from Triton`` while
    `cuda:0` is current. GGUF Triton and the W4A kernels use the same guard.
    """

    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(device)
    return nullcontext()


def _cuda_ordinals(device) -> tuple[int, ...]:
    """Resolve an indexed runtime device or all devices named by a family."""

    if device is None:
        return (torch.cuda.current_device(),)
    value = getattr(device, "value", device)
    if isinstance(value, int):
        return (value,)
    try:
        target = torch.device("cuda" if value == "rocm" else value)
    except (TypeError, ValueError, RuntimeError):
        return ()
    if target.type != "cuda":
        return ()
    return tuple(range(torch.cuda.device_count())) if target.index is None else (target.index,)


@lru_cache(maxsize=None)
def _rdna4_macaq_supported(ordinal: int) -> bool:
    """The R9700 gfx1201 is the validated ROCm MXFP4 Triton target."""

    arch = getattr(torch.cuda.get_device_properties(ordinal), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1201"


def triton_mxfp4_available(device=None) -> bool:
    """Whether every selected accelerator supports the MACaQ Triton path."""

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    if not IS_ROCM and getattr(device, "value", device) == "rocm":
        return False
    ordinals = _cuda_ordinals(device)
    if not ordinals:
        return False
    try:
        return all(
            0 <= index < torch.cuda.device_count()
            and (not IS_ROCM or _rdna4_macaq_supported(index))
            for index in ordinals
        )
    except (RuntimeError, AssertionError):
        return False


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

    @triton_jit
    def _quant_mxfp4_kernel(x_ptr, pk_ptr, sc_ptr, K,
                            stride_xr, stride_pr, stride_sr,
                            BLOCK_G: tl.constexpr):
        """One program = ``BLOCK_G`` scale groups (of 32) of one row."""

        # rows * K can exceed 2^31 for one large prefill batch; a 32-bit row
        # offset wraps and reads the wrong row (CUDA) or faults (HIP).
        row = tl.program_id(0).to(tl.int64)
        g0 = tl.program_id(1) * BLOCK_G
        gid = g0 + tl.arange(0, BLOCK_G)
        half = tl.arange(0, 16)
        col_lo = gid[:, None] * 32 + half[None, :] * 2
        col_hi = col_lo + 1
        xl = tl.load(x_ptr + row * stride_xr + col_lo, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
        xh = tl.load(x_ptr + row * stride_xr + col_hi, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xl), axis=1), tl.max(tl.abs(xh), axis=1))
        # `tl.max` drops NaN, so a NaN block would keep a finite exponent and
        # the NaN would fold into a plausible finite code. E8M0 0xFF is the
        # format's NaN and the block-scaled GEMM consumes it as such. Inf has no
        # E8M0 encoding either, so it takes the same path and the two paths stay
        # consistent.
        nonfinite = (tl.max(((xl != xl) | (xh != xh)).to(tl.int32), axis=1) > 0) | (
            amax == float("inf")
        )
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
        sc_byte = tl.where(nonfinite, 255.0, e + 127.0).to(tl.uint8)
        tl.store(sc_ptr + row * stride_sr + gid, sc_byte, mask=gid < K // 32)

    @triton_jit
    def _rot_scale_quant_kernel(x_ptr, ps_ptr, bh_ptr, pk_ptr, sc_ptr, K,
                                stride_xr, stride_pr, stride_sr,
                                BLOCK_G: tl.constexpr):
        """Fused ``(x * pre_scale) @ blkdiag(BH)`` then MXFP4 RTN."""

        row = tl.program_id(0).to(tl.int64)
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
        # Same poisoning as `_quant_mxfp4_kernel`: the rotation already spread a
        # NaN/Inf input over its 32-value block, so test the rotated values.
        nonfinite = (tl.max(((rl != rl) | (rh != rh)).to(tl.int32), axis=1) > 0) | (
            amax == float("inf")
        )
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
        tl.store(
            sc_ptr + row * stride_sr + gid,
            tl.where(nonfinite, 255.0, e + 127.0).to(tl.uint8),
            mask=gid < K // 32,
        )

    @triton_jit
    def _dense_mx_gemm_kernel(
        a_ptr, b_ptr, asc_ptr, bsc_ptr, c_ptr, bias_ptr, M, N, K,
        stride_am, stride_bn, stride_asm, stride_bsn, stride_cm,
        OUT_SCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        A_FORMAT: tl.constexpr = "e2m1", B_FORMAT: tl.constexpr = "e2m1",
        SCALE_ROWS: tl.constexpr = 1, FP4_AS_FP8: tl.constexpr = False,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        group = tl.program_id(2).to(tl.int64)
        A_PACK: tl.constexpr = 2 if A_FORMAT == "e2m1" else 1
        B_PACK: tl.constexpr = 2 if B_FORMAT == "e2m1" else 1
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_n_grouped = group * N + offs_n
        offs_ka = tl.arange(0, BLOCK_K // A_PACK)
        offs_kb = tl.arange(0, BLOCK_K // B_PACK)
        offs_ks = tl.arange(0, BLOCK_K // 32)
        a_ptrs = a_ptr + group * (K // A_PACK) + offs_m[:, None] * stride_am + offs_ka[None, :]
        b_ptrs = b_ptr + offs_n_grouped[:, None] * stride_bn + offs_kb[None, :]
        asc_ptrs = asc_ptr + group * (K // 32) + offs_m[:, None] * stride_asm + offs_ks[None, :]
        bsc_ptrs = bsc_ptr + (offs_n_grouped[:, None] // SCALE_ROWS) * stride_bsn + offs_ks[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block in range(0, tl.cdiv(K, BLOCK_K)):
            # BLOCK_K need not divide K (896, 1152, 2176).  Mask the tail tile
            # with the loop offset, as the upstream MoE kernel does; a mask
            # built once from offs_kb alone only clips the first tile and lets
            # later tiles read the next row's packed bytes and E8M0 exponents.
            ka_mask = block * (BLOCK_K // A_PACK) + offs_ka < K // A_PACK
            kb_mask = block * (BLOCK_K // B_PACK) + offs_kb < K // B_PACK
            ks_mask = block * (BLOCK_K // 32) + offs_ks < K // 32
            a = tl.load(a_ptrs, mask=m_mask[:, None] & ka_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=n_mask[:, None] & kb_mask[None, :], other=0)
            asc = tl.load(asc_ptrs, mask=m_mask[:, None] & ks_mask[None, :], other=127)
            bsc = tl.load(bsc_ptrs, mask=n_mask[:, None] & ks_mask[None, :], other=127)
            if FP4_AS_FP8:
                # The SM12x Triton lowering takes its hardware path for equal
                # formats. All E2M1 values have an exact E4M3 encoding.
                b = _fp4_to_fp8_codes(b)
                acc = tl.dot_scaled(a, asc, A_FORMAT, b.T, bsc, "e4m3", acc)
            else:
                acc = tl.dot_scaled(a, asc, A_FORMAT, b.T, bsc, B_FORMAT, acc)
            a_ptrs += BLOCK_K // A_PACK
            b_ptrs += BLOCK_K // B_PACK
            asc_ptrs += BLOCK_K // 32
            bsc_ptrs += BLOCK_K // 32

        acc = acc * OUT_SCALE
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n_grouped, mask=n_mask, other=0).to(tl.float32)
            acc = acc + bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n_grouped[None, :]
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

    @triton_jit
    def _fp4_to_fp8_codes(packed):
        codes = tl.interleave(packed & 15, packed >> 4)
        magnitude = codes & 7
        # 0x30 encodes 0.5 in E4M3. E2M1 magnitudes 2–7 step by four.
        fp8_magnitude = tl.where(magnitude > 1, 0x30 + 4 * magnitude, tl.where(magnitude == 1, 0x30, 0))
        sign = (codes & 8) << 4
        return (sign | fp8_magnitude).to(tl.uint8)

    @triton_jit
    def _dequant_w4_to_bf16_kernel(pk_ptr, sc_ptr, out_ptr, K, BLOCK_G: tl.constexpr):
        """Packed MXFP4 row -> bf16.  ``value = magnitude * 2 ** (e8m0 - 127)``."""

        row = tl.program_id(0).to(tl.int64)
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
        # E8M0 0xFF is NaN, not 2**128.
        factor = tl.where((e == 255.0)[:, None], float("nan"), tl.exp2(e - 127.0)[:, None])
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
    with _launch_device(x.device):
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
    with _launch_device(x.device):
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
    scale_bytes = e8m0.view(torch.uint8)[:rows, :k // MXFP4_BLOCK].to(torch.float32)
    # E8M0 0xFF is the format's NaN, written for a block that held a NaN/Inf
    # value; `exp2(255 - 127)` would be a finite-range overflow instead.
    scale = torch.where(
        scale_bytes == 255.0,
        torch.full_like(scale_bytes, float("nan")),
        torch.exp2(scale_bytes - 127.0),
    ).repeat_interleave(32, dim=1)
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
    with _launch_device(packed.device):
        _dequant_w4_to_bf16_kernel[(rows, triton.cdiv(k // MXFP4_BLOCK, 16))](
            packed.contiguous(), scale_bytes, out, k, BLOCK_G=16
        )
    return out


def _pick_tiles(rows: int, k: int) -> tuple[int, int, int, int, int]:
    """Measured NVIDIA/RDNA4 launch tiles.

    ``BLOCK_K`` never drops below 64. On sm_120 ``tl.dot_scaled`` lowers to
    the native ``mma.sync`` MXFP4 instruction, which requires a K tile of at
    least 64; a 32-wide tile fails to compile and the module silently falls
    back to Torch. The GEMM masks its K tail, so the tile does not need to
    divide K.
    """

    if IS_ROCM:
        # gfx1201 lowers scaled dot through BF16 WMMA. Its 128-wide K tile
        # avoids the register pressure of Huawei's CUDA 512-wide decode tile.
        block_m, block_n, k_candidates, warps = (
            16 if rows <= 16 else 32, 64, (128, 64), 4
        )
    elif rows <= 32:
        # The upstream fast path uses <=64; the 4090 benchmark favors the
        # larger tile at M=64, so keep its skinny tile through M=32 only.
        block_m, block_n, k_candidates, warps = 16, 64, (512, 256, 128, 64), 4
    else:
        block_m, block_n, k_candidates, warps = 64, 128, (256, 128, 64), 8
    block_k = k_candidates[0]
    return block_m, block_n, block_k, warps, 3


def mxfp4_w4a4_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    *,
    pre_scale: Optional[torch.Tensor] = None,
    bh: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """Packed MXFP4 weights x MXFP4 activations (native block-scaled GEMM)."""

    if not triton_mxfp4_available(x.device):
        raise RuntimeError("MACaQ: Triton MXFP4 requires NVIDIA CUDA or ROCm gfx1201.")
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
    default_m, default_n, default_k, default_warps, default_stages = _pick_tiles(rows, k)
    block_m = block_m if block_m is not None else default_m
    block_n = block_n if block_n is not None else default_n
    block_k = block_k if block_k is not None else default_k
    num_warps = num_warps if num_warps is not None else default_warps
    num_stages = num_stages if num_stages is not None else default_stages
    # sm_120's native MXFP4 `tl.dot_scaled` needs a K tile of at least 64.
    # Prefer a tile that divides K, but the kernel's `kb_mask`/`ks_mask` tails
    # make a non-divisor correct, so never shrink below 64.
    block_k = max(int(block_k), 64)
    while block_k > 64 and (block_k > k or k % block_k):
        block_k //= 2
    block_k = max(block_k, 64)
    out = torch.empty(rows, n, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    # Triton 3.9's pointer canonicalizer does not accept the E8M0 dtype
    # directly; the bytes are the same and `dot_scaled` consumes them as E8M0.
    w_scales_bytes = w_scales.view(torch.uint8)
    bias_tensor = bias.contiguous() if bias is not None else w_scales_bytes
    with _launch_device(x.device):
        _dense_mx_gemm_kernel[grid](
            a_packed, w_packed, a_scales, w_scales_bytes, out, bias_tensor, rows, n, k,
            a_packed.stride(0), w_packed.stride(0), a_scales.stride(0), w_scales_bytes.stride(0),
            out.stride(0),
            OUT_SCALE=out_scale, HAS_BIAS=bias is not None,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=num_stages,
        )
    return out.reshape(*lead, n)


def native_mx_launch_config(capability: tuple[int, int], rows: int, fp4: bool) -> dict:
    """GEMM tiles that lower to hardware block-scaled MMA on Blackwell."""
    warp_mma = capability[0] == 12
    return {
        "BLOCK_M": (16 if rows <= 16 else 64) if warp_mma else 128,
        "BLOCK_N": 64 if warp_mma else 128,
        "BLOCK_K": 128,
        "FP4_AS_FP8": warp_mma and fp4,
        "num_warps": 4,
        "num_stages": 2,
    }


def mxfp8_scaled_linear(activations, activation_scales, weight, weight_scales, *, groups, dtype):
    """Native MXFP8 activations against E2M1 or E4M3 checkpoint weights."""
    k = activations.size(-1)
    n = weight.size(0) // groups
    rows = activations.numel() // (k * groups)
    out = activations.new_empty((*activations.shape[:-1], n), dtype=dtype)
    if not rows:
        return out
    fp4 = weight.dtype == torch.int8
    config = native_mx_launch_config(torch.cuda.get_device_capability(activations.device), rows, fp4)
    grid = (triton.cdiv(rows, config["BLOCK_M"]), triton.cdiv(n, config["BLOCK_N"]), groups)
    scales = weight_scales.view(torch.uint8)
    with _launch_device(activations.device):
        _dense_mx_gemm_kernel[grid](
            activations.view(torch.uint8), weight.view(torch.uint8), activation_scales, scales, out, scales,
            rows, n, k, groups * k, weight.stride(0), groups * (k // 32), scales.stride(0), groups * n,
            OUT_SCALE=1.0, HAS_BIAS=False, A_FORMAT="e4m3",
            B_FORMAT="e2m1" if fp4 else "e4m3", SCALE_ROWS=1 if fp4 else 32, **config,
        )
    return out


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
    # Keep a wider stored bias out of the low-precision operands; the helper
    # adds it in FP32 and rounds the output once (#5632).
    return linear_fp32_reduce(xb, weight, bias)
