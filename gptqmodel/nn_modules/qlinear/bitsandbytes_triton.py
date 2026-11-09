# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for bitsandbytes 4-bit (NF4/FP4) inference on ROCm/RDNA.

bitsandbytes ships CUDA kernels and, since 0.49, a HIP build. The HIP 4-bit
GEMM is tuned for CDNA-style wave64 execution: measured on gfx1201 it is
2.4x-6.3x slower than the fused kernel below for ``M >= 16`` (e.g. 324 us vs
61 us at K=N=4096, M=16; 999 us vs 159 us at K=4096, N=11008). Below that
threshold bitsandbytes' own GEMV is the faster path and is kept.

These kernels dequantize the packed NF4/FP4 codes inside the matmul, so no
dense weight buffer is ever written and the packed weights are read once.

Packed layout (verified against ``bitsandbytes.functional.quantize_4bit`` and
``dequantize_4bit`` for NF4/FP4, block sizes 32/64/128, with and without
``compress_statistics``): the row-major flattening of ``(N, K)`` stores element
``i`` in byte ``i // 2``, with the **even** element in the high nibble and the
odd element in the low nibble. ``absmax`` is indexed by
``(n * K + k) // block_size``. When ``compress_statistics`` is set the stored
``absmax`` is itself blockwise quantized; it is expanded to FP32 once (a scale
tensor, never a dequantized weight).

Measured on an AMD Radeon AI PRO R9700 (gfx1201, wave32, HIP 7.16.26385,
torch 2.16.0.dev20261025+rocm10.1, bitsandbytes 0.50.2, FP16, median of 50
iterations), fused kernel vs ``bnb.matmul_4bit``::

    K=N=2048   M=16 40.3us vs  95.4us (2.4x)   M=64 39.9us vs 130.9us (3.3x)
    K=N=4096   M=16 61.2us vs 324.6us (5.3x)   M=64 102.6us vs 326.1us (3.2x)
    K=4096 N=11008 M=16 158.8us vs 999.2us (6.3x) M=64 285.8us vs 1004.8us (3.5x)

Ada (RTX 4090, SM 8.9) keeps bitsandbytes' native CUDA kernels by default:
they are cublas-backed and measured faster than this tile on every shape
tested, so the ``ada`` entry exists for the hardware-parity contract and for
forced A/B runs (``GPTQMODEL_BNB_TRITON=1``), not as the default.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from ...utils.arch import tuning_family
from ...utils.rocm import IS_ROCM


class BnbGemmLaunch(NamedTuple):
    """Launch constants for the fused 4-bit GEMM on one architecture class."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    #: Smallest row count at which the fused kernel beats the vendor path.
    min_m: int


#: ``rdna4`` was swept over ``(block_m, block_n, block_k)`` in
#: {16, 32, 64} x {16, 32} x {64, 128, 256} with warps {4, 8} and stages
#: {1, 2}; ``(32, 32, 128, 4, 1)`` had the lowest worst-case regret. ``ada``
#: mirrors the measured GPTQ small-M tile.
_BNB_GEMM_LAUNCH: dict[str, BnbGemmLaunch] = {
    "ada": BnbGemmLaunch(16, 16, 256, 4, 2, min_m=16),
    "rdna4": BnbGemmLaunch(32, 32, 128, 4, 1, min_m=16),
}

_DEFAULT_GEMM = BnbGemmLaunch(32, 32, 128, 4, 1, min_m=16)


def _family(device) -> str:
    if device is None:
        return "other"
    try:
        return tuning_family(device)
    except Exception:
        return "other"


def gemm_launch(device=None) -> BnbGemmLaunch:
    """Return the fused 4-bit GEMM tile for ``device``."""

    if device is None:
        return _DEFAULT_GEMM
    return _BNB_GEMM_LAUNCH.get(_family(device), _DEFAULT_GEMM)


def tuned_families() -> tuple[str, ...]:
    """Families with a measured tile (the hardware-parity contract)."""

    return tuple(sorted(_BNB_GEMM_LAUNCH))


def available_for(device=None, dtype: Optional[torch.dtype] = None) -> bool:
    """Return whether the Triton path is enabled for this device and dtype.

    Enabled on ROCm for a family with a measured tile. Elsewhere it is off
    unless ``GPTQMODEL_BNB_TRITON`` is set, because the vendor kernels are the
    measured winner on CUDA.
    """

    from ...utils.env import env_flag

    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        return False
    forced = env_flag("GPTQMODEL_BNB_TRITON", default=False)
    if not forced and not IS_ROCM:
        return False
    return _family(device) in _BNB_GEMM_LAUNCH


def should_use(device=None, dtype: Optional[torch.dtype] = None, rows: int = 0) -> bool:
    """Return whether a forward with ``rows`` rows should use the fused kernel."""

    if rows < gemm_launch(device).min_m:
        return False
    return available_for(device, dtype)


@triton.jit
def _bnb_4bit_gemm_kernel(
    x_ptr, q_ptr, am_ptr, code_ptr, bias_ptr, out_ptr, m,
    K: tl.constexpr, N: tl.constexpr, BS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mmask = rows < m
    nmask = cols < N
    kk = tl.arange(0, BK)
    half = tl.arange(0, BK // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    acc_dtype = x_ptr.dtype.element_ty

    for k0 in range(0, K, BK):
        ks = k0 + kk
        kvalid = ks < K
        # Even element -> high nibble, odd element -> low nibble.
        packed = tl.load(
            q_ptr + (cols[:, None] * K + k0) // 2 + half[None, :],
            mask=nmask[:, None] & ((k0 + 2 * half)[None, :] < K),
            other=0,
        )
        hi = ((packed >> 4) & 0xF).to(tl.int32)
        lo = (packed & 0xF).to(tl.int32)
        if K % 2:
            # For odd K, alternate rows begin in the low nibble of a byte.
            # Their next code is the high nibble of the following byte.
            odd_row = ((cols * K) & 1) != 0
            next_packed = tl.load(
                q_ptr + (cols[:, None] * K + k0) // 2 + half[None, :] + 1,
                mask=nmask[:, None] & odd_row[:, None] & ((k0 + 2 * half + 1)[None, :] < K),
                other=0,
            )
            first = tl.where(odd_row[:, None], lo, hi)
            second = tl.where(odd_row[:, None], (next_packed >> 4).to(tl.int32), lo)
            codes = tl.interleave(first, second)
        else:
            codes = tl.interleave(hi, lo)
        valid = nmask[:, None] & kvalid[None, :]
        am = tl.load(
            am_ptr + (cols[:, None] * K + ks[None, :]) // BS, mask=valid, other=0.0,
        ).to(tl.float32)
        cw = tl.load(code_ptr + codes, mask=valid, other=0.0).to(tl.float32)
        w = (cw * am).to(acc_dtype)
        xv = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=mmask[:, None] & kvalid[None, :], other=0.0,
        )
        acc = tl.dot(xv, tl.trans(w), acc)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols, mask=nmask, other=0.0)[None, :].to(tl.float32)
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mmask[:, None] & nmask[None, :],
    )


@triton.jit
def _bnb_4bit_dequant_kernel(
    q_ptr, am_ptr, code_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr, BS: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = pid_k * BK + tl.arange(0, BK)
    half = tl.arange(0, BK // 2)
    valid = (cols < N)[:, None] & (ks < K)[None, :]
    packed = tl.load(
        q_ptr + (cols[:, None] * K + pid_k * BK) // 2 + half[None, :],
        mask=(cols < N)[:, None] & ((pid_k * BK + 2 * half)[None, :] < K),
        other=0,
    )
    hi = ((packed >> 4) & 0xF).to(tl.int32)
    lo = (packed & 0xF).to(tl.int32)
    if K % 2:
        odd_row = ((cols * K) & 1) != 0
        next_packed = tl.load(
            q_ptr + (cols[:, None] * K + pid_k * BK) // 2 + half[None, :] + 1,
            mask=(cols < N)[:, None] & odd_row[:, None] & ((pid_k * BK + 2 * half + 1)[None, :] < K),
            other=0,
        )
        first = tl.where(odd_row[:, None], lo, hi)
        second = tl.where(odd_row[:, None], (next_packed >> 4).to(tl.int32), lo)
        codes = tl.interleave(first, second)
    else:
        codes = tl.interleave(hi, lo)
    am = tl.load(
        am_ptr + (cols[:, None] * K + ks[None, :]) // BS, mask=valid, other=0.0,
    ).to(tl.float32)
    cw = tl.load(code_ptr + codes, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + cols[:, None] * K + ks[None, :], (cw * am).to(out_ptr.dtype.element_ty), valid)


def _absmax_fp32(quant_state) -> torch.Tensor:
    """Return a flat FP32 absmax vector for ``quant_state``.

    ``compress_statistics`` stores absmax as blockwise-quantized ``uint8``; the
    expansion is a scale tensor (``numel / block_size`` entries), never a
    dequantized weight, so materializing it does not violate the no-dense-cache
    rule.
    """

    absmax = quant_state.absmax
    if absmax.dtype != torch.uint8:
        return absmax.reshape(-1).float().contiguous()

    state2 = getattr(quant_state, "state2", None)
    if state2 is None:
        raise RuntimeError("bitsandbytes compressed absmax is missing its nested state")
    code = state2.code.to(torch.float32)
    nested = state2.absmax.to(torch.float32).reshape(-1)
    blocksize = int(state2.blocksize)
    idx = torch.arange(absmax.numel(), device=absmax.device)
    return code[absmax.reshape(-1).long()] * nested[idx // blocksize] + quant_state.offset


def _codebook(quant_state, device) -> torch.Tensor:
    code = getattr(quant_state, "code", None)
    if code is None:
        import bitsandbytes as bnb

        code = bnb.functional.get_4bit_type(quant_state.quant_type, device="cpu")
    return code.to(device=device, dtype=torch.float32).reshape(-1).contiguous()


def dequantize_4bit(qweight: torch.Tensor, quant_state, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Dequantize a bitsandbytes 4-bit weight with a Triton kernel."""

    out_features, in_features = quant_state.shape
    dtype = out_dtype or quant_state.dtype
    out = torch.empty((out_features, in_features), device=qweight.device, dtype=dtype)
    bk = min(128, max(32, 1 << (in_features - 1).bit_length()))
    grid = (triton.cdiv(out_features, 16), triton.cdiv(in_features, bk))
    _bnb_4bit_dequant_kernel[grid](
        qweight.reshape(-1).contiguous(), _absmax_fp32(quant_state), _codebook(quant_state, qweight.device), out,
        in_features, out_features, int(quant_state.blocksize), 16, bk,
        num_warps=4,
    )
    return out


def matmul_4bit(
    x: torch.Tensor,
    qweight: torch.Tensor,
    quant_state,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fused ``x @ dequantize_4bit(qweight)`` reading the packed weights once."""

    out_features, in_features = quant_state.shape
    out_dtype = out_dtype or x.dtype
    # The GEMM kernel addresses x2 with a dense row-major stride.
    x2 = x.reshape(-1, in_features).contiguous()
    out = torch.empty((x2.shape[0], out_features), device=x.device, dtype=out_dtype)
    cfg = gemm_launch(x.device)
    grid = (triton.cdiv(x2.shape[0], cfg.block_m), triton.cdiv(out_features, cfg.block_n))
    _bnb_4bit_gemm_kernel[grid](
        x2, qweight.reshape(-1).contiguous(), _absmax_fp32(quant_state), _codebook(quant_state, x.device),
        bias.contiguous() if bias is not None else x, out, x2.shape[0],
        in_features, out_features, int(quant_state.blocksize),
        cfg.block_m, cfg.block_n, cfg.block_k, bias is not None,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return out.reshape(x.shape[:-1] + (out_features,))


__all__ = [
    "BnbGemmLaunch",
    "available_for",
    "dequantize_4bit",
    "gemm_launch",
    "matmul_4bit",
    "should_use",
    "tuned_families",
]
