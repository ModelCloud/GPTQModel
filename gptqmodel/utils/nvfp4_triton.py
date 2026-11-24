# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Fused W*A16 Triton GEMMs for fine-grained checkpoint formats.
#
# Both kernels decode the weight tile inside the K loop and never materialize a
# dense weight: a 4-bit NVFP4 expert stays 4 bits wide in VRAM, and an MXFP8
# expert stays 8 bits wide.  Activations stay in the model dtype and the dot
# accumulates in FP32, so the kernels run on every Ampere-and-newer part with a
# plain `tl.dot` -- no `tl.dot_scaled`, no SM100/SM120 requirement, and no
# CUTLASS or Marlin layout permutation.
#
# NVFP4 (OCP / NVIDIA ModelOpt, `quant_algo: NVFP4`):
#   value = e2m1(code) * e4m3(block_scale) * global_scale
#   two codes per byte, the even column in the low nibble, one E4M3 scale per
#   16 contiguous columns, one FP32 scale per tensor.
# MXFP8 (OCP MX):
#   value = e4m3(value_byte) * 2 ** (e8m0_scale - 127)
#   one byte per value, one E8M0 scale per 32 contiguous columns.
#
# The E4M3 block scale is applied to the decoded tile and the FP32 global scale
# to the FP32 accumulator, so a group's scale is rounded once per weight element
# and the global factor is rounded once per output element.

from __future__ import annotations

from typing import Optional

import torch

from .logger import setup_logger
from .macaq_triton import _launch_device


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

NVFP4_GROUP = 16
MXFP8_GROUP = 32


def scaled_wxa16_available(device=None) -> bool:
    """Whether the fused decode GEMM can run on ``device``.

    The kernels only need a BF16/FP16 ``tl.dot`` with FP32 accumulation, so any
    CUDA device Triton can target qualifies. ROCm is excluded until the tile
    choices are measured there; the Torch dequant path stays the reference.
    """

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        target = torch.device("cuda" if device is None else device)
    except (TypeError, ValueError, RuntimeError):
        return False
    if target.type != "cuda":
        return False
    if target.index is not None and not 0 <= target.index < torch.cuda.device_count():
        return False
    return not _is_rocm()


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


if TRITON_AVAILABLE:

    @triton_jit
    def _decode_e2m1(codes):
        """E2M1 nibble codes (0-15) -> float32 magnitudes with sign."""

        magnitude = codes & 7
        value = tl.where(
            magnitude < 4,
            magnitude.to(tl.float32) * 0.5,
            tl.where(
                magnitude == 4,
                2.0,
                tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
            ),
        )
        return tl.where((codes & 8) != 0, -value, value)

    @triton_jit
    def _decode_e4m3(byte):
        """E4M3 (OCP `fn`, no infinities) byte -> float32."""

        raw = byte.to(tl.int32)
        exponent = (raw >> 3) & 15
        mantissa = (raw & 7).to(tl.float32)
        # 0x7F/0xFF are the format's NaNs; every other byte is finite.
        value = tl.where(
            exponent == 0,
            mantissa * 0.001953125,  # 2 ** -9
            (1.0 + mantissa * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0),
        )
        value = tl.where((raw & 127) == 127, float("nan"), value)
        return tl.where((raw & 128) != 0, -value, value)

    @triton_jit
    def _nvfp4_w4a16_kernel(
        a_ptr, b_ptr, bsc_ptr, c_ptr, bias_ptr, M, N, K,
        stride_am, stride_bn, stride_bsn, stride_cm,
        GLOBAL_SCALE,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N

        offs_ka = tl.arange(0, BLOCK_K)
        offs_kb = tl.arange(0, BLOCK_K // 2)
        offs_ks = tl.arange(0, BLOCK_K // 16)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_ka[None, :]
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_kb[None, :]
        bsc_ptrs = bsc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = m_mask[:, None] & (block * BLOCK_K + offs_ka < K)[None, :]
            b_mask = n_mask[:, None] & (block * (BLOCK_K // 2) + offs_kb < K // 2)[None, :]
            s_mask = n_mask[:, None] & (block * (BLOCK_K // 16) + offs_ks < K // 16)[None, :]
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            packed = tl.load(b_ptrs, mask=b_mask, other=0)
            scales = tl.load(bsc_ptrs, mask=s_mask, other=0)
            # Even column in the low nibble, following column in the high one.
            codes = tl.interleave(packed & 15, packed >> 4)
            weight = _decode_e2m1(codes)
            weight = tl.reshape(weight, (BLOCK_N, BLOCK_K // 16, 16))
            weight = weight * tl.expand_dims(_decode_e4m3(scales), 2)
            weight = tl.reshape(weight, (BLOCK_N, BLOCK_K)).to(a.dtype)
            acc = tl.dot(a, tl.trans(weight), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K // 2
            bsc_ptrs += BLOCK_K // 16

        acc = acc * GLOBAL_SCALE
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)
            acc = acc + bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

    @triton_jit
    def _mxfp8_w8a16_kernel(
        a_ptr, b_ptr, bsc_ptr, c_ptr, bias_ptr, M, N, K,
        stride_am, stride_bn, stride_bsn, stride_cm,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        m_mask = offs_m < M
        n_mask = offs_n < N

        offs_ka = tl.arange(0, BLOCK_K)
        offs_ks = tl.arange(0, BLOCK_K // 32)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_ka[None, :]
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_ka[None, :]
        bsc_ptrs = bsc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block in range(0, tl.cdiv(K, BLOCK_K)):
            a_mask = m_mask[:, None] & (block * BLOCK_K + offs_ka < K)[None, :]
            s_mask = n_mask[:, None] & (block * (BLOCK_K // 32) + offs_ks < K // 32)[None, :]
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            raw = tl.load(b_ptrs, mask=n_mask[:, None] & (block * BLOCK_K + offs_ka < K)[None, :], other=0)
            exponents = tl.load(bsc_ptrs, mask=s_mask, other=127)
            weight = _decode_e4m3(raw)
            # E8M0 0xFF is the format's NaN, not 2 ** 128.
            factor = tl.exp2(exponents.to(tl.float32) - 127.0)
            factor = tl.where(exponents == 255, float("nan"), factor)
            weight = tl.reshape(weight, (BLOCK_N, BLOCK_K // 32, 32))
            weight = weight * tl.expand_dims(factor, 2)
            weight = tl.reshape(weight, (BLOCK_N, BLOCK_K)).to(a.dtype)
            acc = tl.dot(a, tl.trans(weight), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
            bsc_ptrs += BLOCK_K // 32

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)
            acc = acc + bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])




def _pick_tiles(rows: int, k: int) -> tuple[int, int, int]:
    """Decode-shaped tiles; the weight tile is re-expanded in registers."""

    block_m = 16 if rows <= 16 else (32 if rows <= 32 else 64)
    block_n = 64 if rows <= 32 else 128
    block_k = 128
    while block_k > 32 and (block_k > k or k % block_k):
        block_k //= 2
    return block_m, block_n, block_k


def _as_global_scale(value) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("NVFP4: the global scale must be a scalar.")
        return float(value.detach().float().cpu())
    return float(value)


def dequant_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale=1.0,
) -> torch.Tensor:
    """Torch reference decode of a ModelOpt NVFP4 matrix, in float32."""

    rows, half_k = packed.shape
    k = half_k * 2
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(rows, k).long()
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device, dtype=torch.float32
    )[codes & 7]
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    block = scales.to(torch.float32)[:rows, : k // NVFP4_GROUP]
    value = magnitude * sign * block.repeat_interleave(NVFP4_GROUP, dim=1)
    return value * _as_global_scale(global_scale)


def dequant_mxfp8(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Torch reference decode of an OCP MXFP8 matrix, in float32."""

    rows, k = weight.shape
    byte = scales.view(torch.uint8)[:rows, : k // MXFP8_GROUP].to(torch.float32)
    factor = torch.where(
        byte == 255.0, torch.full_like(byte, float("nan")), torch.exp2(byte - 127.0)
    ).repeat_interleave(MXFP8_GROUP, dim=1)
    return weight.to(torch.float32) * factor


def nvfp4_w4a16_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    global_scale=1.0,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """``x @ W^T`` for packed NVFP4 ``W`` and full-precision ``x``.

    ``w_packed`` is ``[N, K // 2]`` uint8 and ``w_scales`` ``[N, K // 16]``
    E4M3 bytes (float8 or uint8 view).  ``global_scale`` is the per-tensor FP32
    decode factor, applied to the FP32 accumulator.  The weight is decoded one
    K tile at a time and never materialized, so a 4-bit expert keeps its 4-bit
    footprint for the whole forward.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("NVFP4: Triton is not available.")
    if w_packed.dtype != torch.uint8 or w_packed.ndim != 2:
        raise ValueError("NVFP4: packed weights must be a 2-D uint8 tensor.")
    rows = x.numel() // x.shape[-1]
    k = w_packed.shape[1] * 2
    if x.shape[-1] != k:
        raise ValueError(f"NVFP4: activation width {x.shape[-1]} does not match K={k}.")
    if k % NVFP4_GROUP:
        raise ValueError("NVFP4: K must be a multiple of the 16-wide group.")
    if w_scales.shape[0] != w_packed.shape[0] or w_scales.shape[1] < k // NVFP4_GROUP:
        raise ValueError("NVFP4: one E4M3 block scale per 16 columns is required.")

    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(rows, k).to(out_dtype)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    weight = w_packed.contiguous()
    scales = w_scales.view(torch.uint8).contiguous()
    n = weight.shape[0]

    default_m, default_n, default_k = _pick_tiles(rows, k)
    block_m = block_m or default_m
    block_n = block_n or default_n
    block_k = block_k or default_k
    num_warps = num_warps or (4 if block_n <= 64 else 8)
    num_stages = num_stages or 3

    out = torch.empty(rows, n, device=x.device, dtype=out_dtype)
    bias_tensor = bias.contiguous() if bias is not None else scales
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    with _launch_device(x.device):
        _nvfp4_w4a16_kernel[grid](
            x2, weight, scales, out, bias_tensor, rows, n, k,
            x2.stride(0), weight.stride(0), scales.stride(0), out.stride(0),
            _as_global_scale(global_scale),
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=num_stages,
        )
    return out.reshape(*lead, n)


def mxfp8_w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    w_scales: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """``x @ W^T`` for an MXFP8 ``W`` with full-precision ``x``."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("MXFP8: Triton is not available.")
    if weight.ndim != 2:
        raise ValueError("MXFP8: weights must be a 2-D tensor.")
    rows = x.numel() // x.shape[-1]
    k = weight.shape[1]
    if x.shape[-1] != k:
        raise ValueError(f"MXFP8: activation width {x.shape[-1]} does not match K={k}.")
    if k % MXFP8_GROUP:
        raise ValueError("MXFP8: K must be a multiple of the 32-wide group.")
    if w_scales.shape[0] != weight.shape[0] or w_scales.shape[1] < k // MXFP8_GROUP:
        raise ValueError("MXFP8: one E8M0 scale per 32 columns is required.")

    out_dtype = out_dtype or x.dtype
    lead = x.shape[:-1]
    x2 = x.reshape(rows, k).to(out_dtype)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    raw = weight.view(torch.uint8).contiguous()
    scales = w_scales.view(torch.uint8).contiguous()
    n = raw.shape[0]

    default_m, default_n, default_k = _pick_tiles(rows, k)
    block_m = block_m or default_m
    block_n = block_n or default_n
    block_k = block_k or default_k
    num_warps = num_warps or (4 if block_n <= 64 else 8)
    num_stages = num_stages or 3

    out = torch.empty(rows, n, device=x.device, dtype=out_dtype)
    bias_tensor = bias.contiguous() if bias is not None else scales
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    with _launch_device(x.device):
        _mxfp8_w8a16_kernel[grid](
            x2, raw, scales, out, bias_tensor, rows, n, k,
            x2.stride(0), raw.stride(0), scales.stride(0), out.stride(0),
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=num_stages,
        )
    return out.reshape(*lead, n)


__all__ = [
    "MXFP8_GROUP",
    "NVFP4_GROUP",
    "TRITON_AVAILABLE",
    "dequant_mxfp8",
    "dequant_nvfp4",
    "mxfp8_w8a16_linear",
    "nvfp4_w4a16_linear",
    "scaled_wxa16_available",
]
