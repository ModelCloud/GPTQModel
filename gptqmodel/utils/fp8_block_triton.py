# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton decode kernels for block-scaled FP8 (``FineGrainedFP8``) weights.

``TorchFP8Linear`` stores E4M3/E5M2 codes plus a per-block inverse scale -- the
``weight_block_size=[128, 128]`` layout that transformers' ``FineGrainedFP8``
and the DeepSeek/Qwen checkpoints write. The row/tensor-scale decode of #5173
moves such a scale into the FP32 output epilogue, but a block scale varies
along both axes of the reduction, so that reformulation does not apply and the
decoded weight has to exist for the GEMM.

The portable path built it with two full-weight passes (a cast, then an
in-place block scale). Each pass reads and writes the whole layer, so a
1 B/element format paid several times the weight bytes in traffic before the
GEMM ran. This module keeps the same arithmetic in one pass, and for the
batch-1 decode shapes where weight traffic dominates it folds the decode into
the GEMM so the packed payload is read once and never written.

Both kernels keep FP32 arithmetic and round to the activation dtype once, so
they match the reference ``weight.float() / expanded_scale`` decode. Neither
stores a decoded weight on the module: a temporary decodes into memory owned by
the calling forward and released before it returns, and the fused product has
no decoded temporary at all.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch


try:  # pragma: no cover - import availability is environment dependent
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


#: FP8 formats Triton can read as a load operand on NVIDIA. ``float8_e4m3fnuz``
#: and ``float8_e5m2fnuz`` are AMD-scaled layouts Triton does not lower there
#: (``ValueError: type fp8e4b8 not supported in this architecture``), so those
#: payloads keep the portable Torch decode.
TRITON_FP8_FORMATS = frozenset({"float8_e4m3fn", "float8_e5m2"})

#: Minimum NVIDIA compute capability per Triton FP8 operand type. Triton's
#: CUDA backend only adds ``fp8e4nv`` -- the type behind ``torch.float8_e4m3fn``
#: -- to a target's supported operands at SM 8.9 and later, so a kernel that
#: loads those codes on Ampere (SM 8.0/8.6) fails to compile with
#: ``ValueError: type fp8e4nv not supported in this architecture``. E5M2 lowers
#: to ``fp8e5``, which every target supports. A format outside this table has
#: no Triton operand at all and keeps the portable Torch decode.
TRITON_FP8_MIN_CAPABILITY = {
    "float8_e4m3fn": (8, 9),
    "float8_e5m2": (0, 0),
}


def fp8_operand_supported(fp8_format: str, capability: tuple[int, int]) -> bool:
    """Whether Triton can name *fp8_format* as an operand at *capability*."""

    minimum = TRITON_FP8_MIN_CAPABILITY.get(fp8_format)
    if minimum is None:
        return False
    return tuple(capability) >= minimum


def triton_fp8_supports_format(device: torch.device, fp8_format: str) -> bool:
    """Whether the Triton block kernels can load *fp8_format* on *device*.

    The operand type a kernel declares is fixed by the payload's format, and
    Triton's CUDA backend rejects ``fp8e4nv`` (``float8_e4m3fn``) on every
    target below SM 8.9 (#5400). ROCm keeps the portable Torch decode, as
    ``triton_fp8_available`` already decides for the whole path.
    """

    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    from .rocm import IS_ROCM

    if IS_ROCM:
        return False
    return fp8_operand_supported(fp8_format, torch.cuda.get_device_capability(device))


def _launch_device(device: torch.device):
    """Bind a Triton launch to the device that owns the operands.

    Triton launches into the current CUDA/HIP context, but a sharded checkpoint
    hands each layer operands on that layer's device. MACaQ and GGUF use the
    same guard.
    """

    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(device)
    return nullcontext()


if TRITON_AVAILABLE:

    @triton.jit
    def _fp8_block_decode_kernel(
        w_ptr,
        s_ptr,
        out_ptr,
        rows,
        cols,
        swr,
        swc,
        ssr,
        ssc,
        sor,
        soc,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
        INVERSE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Decode one tile of codes with its block scale in a single pass."""

        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        tr = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        tc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (tr[:, None] < rows) & (tc[None, :] < cols)
        codes = tl.load(
            w_ptr + tr[:, None] * swr + tc[None, :] * swc, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(
            s_ptr + (tr // BLOCK_H)[:, None] * ssr + (tc // BLOCK_W)[None, :] * ssc,
            mask=mask,
            other=1.0,
        )
        if INVERSE:
            value = codes / scale
        else:
            value = codes * scale
        tl.store(
            out_ptr + tr[:, None] * sor + tc[None, :] * soc,
            value.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _fp8_block_gemv_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        out_ptr,
        bias_ptr,
        rows,
        out_features,
        in_features,
        sxr,
        sxc,
        swr,
        swc,
        ssr,
        ssc,
        sor,
        soc,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
        INVERSE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Block-scaled W8A16 product without a decoded-weight temporary."""

        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n < out_features
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m < rows
        k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_block in range(0, in_features // BLOCK_K):
            # BLOCK_K equals the column block, so one scale covers this step.
            scale = tl.load(
                s_ptr + (n // BLOCK_H) * ssr + k_block * ssc, mask=n_mask, other=1.0
            )
            codes = tl.load(
                w_ptr + n[:, None] * swr + (k_block * BLOCK_K + k)[None, :] * swc,
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if INVERSE:
                weight = codes / scale[:, None]
            else:
                weight = codes * scale[:, None]
            x = tl.load(
                x_ptr + m[:, None] * sxr + (k_block * BLOCK_K + k)[None, :] * sxc,
                mask=m_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(weight).to(x.dtype))
        if HAS_BIAS:
            acc += tl.load(bias_ptr + n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(
            out_ptr + m[:, None] * sor + n[None, :] * soc,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def triton_fp8_available(device: Optional[torch.device] = None) -> bool:
    """Whether the block-scale Triton decode may run on *device*.

    It is qualified on NVIDIA CUDA only. HIP's activation-dtype narrowing and
    its FP8 GEMM differ enough that this path needs its own numerical
    qualification before it is enabled there, so ROCm keeps the portable Torch
    decode. This is the environment half of the gate: a payload also has to
    pass :func:`triton_fp8_supports_format` for its own FP8 operand, which is
    what the two ``*_supported`` payload predicates below check.
    """

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    target = device if device is not None else torch.device("cuda")
    if target.type != "cuda":
        return False
    from .rocm import IS_ROCM

    return not IS_ROCM


@lru_cache(maxsize=None)
def _decode_tile(*_key) -> tuple[int, int]:
    """Measured decode tile: one block-row band by one block-column band."""

    return (32, 128)


def fp8_block_decode_supported(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    fp8_format: str,
    block_size: tuple[int, int],
    dtype: torch.dtype,
) -> bool:
    """Whether :func:`fp8_block_decode` accepts this payload.

    The kernel follows arbitrary strides, so only the geometry and the dtype
    are constrained, together with the device and format the Triton operand
    needs. Non-divisible or padded/cropped checkpoint scales, and a format the
    target cannot lower, keep the portable decode.
    """

    if not TRITON_AVAILABLE or codes.device.type != "cuda":
        return False
    if not triton_fp8_supports_format(codes.device, fp8_format):
        return False
    if codes.ndim != 2 or scale.device != codes.device or scale.dtype != torch.float32:
        return False
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    rows, cols = codes.shape
    block_rows, block_cols = int(block_size[0]), int(block_size[1])
    if block_rows <= 0 or block_cols <= 0:
        return False
    if rows % block_rows or cols % block_cols:
        return False
    return scale.shape == (rows // block_rows, cols // block_cols)


def fp8_block_decode(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: tuple[int, int],
    inverse: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode block-scaled FP8 codes into *dtype* in one kernel pass.

    ``codes`` is ``[out, in]`` FP8 and ``scale`` is the per-block scale of
    shape ``[out // block_rows, in // block_cols]`` (the exact divisible
    layout). The result is a new tensor owned by the caller; nothing is cached.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("FP8 block decode: Triton is not available.")
    rows, cols = codes.shape
    block_rows, block_cols = int(block_size[0]), int(block_size[1])
    if rows % block_rows or cols % block_cols:
        raise ValueError("FP8 block decode requires block-divisible geometry.")
    if scale.shape != (rows // block_rows, cols // block_cols):
        raise ValueError("FP8 block decode requires the exact per-block scale shape.")
    out = torch.empty((rows, cols), dtype=dtype, device=codes.device)
    tile_r, tile_c = _decode_tile(block_rows, block_cols)
    grid = (triton.cdiv(rows, tile_r), triton.cdiv(cols, tile_c))
    with _launch_device(codes.device):
        _fp8_block_decode_kernel[grid](
            codes,
            scale,
            out,
            rows,
            cols,
            codes.stride(0),
            codes.stride(1),
            scale.stride(0),
            scale.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=block_rows,
            BLOCK_W=block_cols,
            INVERSE=inverse,
            BLOCK_R=tile_r,
            BLOCK_C=tile_c,
            num_warps=4,
        )
    return out


@lru_cache(maxsize=None)
def _gemv_block_k() -> int:
    return 128


@lru_cache(maxsize=64)
def _gemv_tiles(rows: int, out_features: int) -> tuple[int, int, int, int]:
    """Measured launch tiles for the fused block product.

    ``(block_m, block_n, num_warps, num_stages)``. The kernel is bandwidth-bound
    on the packed payload, so a narrow projection wants a small output tile to
    keep every program reading contiguous codes while a wide one needs a larger
    tile to amortise the launch. Values come from the sweep recorded in
    ``tests/kernels/test_fp8_block_decode_5318.py``.
    """

    if out_features <= 2048:
        block_n, num_warps = 16, 2
    elif out_features <= 8192:
        block_n, num_warps = 32, 4
    else:
        block_n, num_warps = 64, 4
    block_m = 16 if rows <= 16 else 64
    return (block_m, block_n, num_warps, 4)


def fp8_block_gemv_payload_supported(
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    fp8_format: str,
    block_size: tuple[int, int],
    dtype: torch.dtype,
) -> bool:
    """Payload-side preconditions of :func:`fp8_block_gemv`.

    The fused kernel needs the exact divisible block layout, an FP16/BF16
    activation dtype, a device whose Triton target can load the payload's FP8
    operand, and a 128-wide scale column block so each reduction step owns
    exactly one scale. Everything else keeps the portable decode. Callers check
    the activation tensor itself with the per-call check below.
    """

    if not TRITON_AVAILABLE:
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if codes.device.type != "cuda" or scale.device != codes.device:
        return False
    if not triton_fp8_supports_format(codes.device, fp8_format):
        return False
    if codes.ndim != 2 or scale.dtype != torch.float32 or not scale.is_contiguous():
        return False
    out_features, in_features = codes.shape
    block_rows, block_cols = int(block_size[0]), int(block_size[1])
    if block_rows <= 0 or block_cols <= 0 or block_cols != _gemv_block_k():
        return False
    if out_features % block_rows or in_features % block_cols:
        return False
    return scale.shape == (out_features // block_rows, in_features // block_cols)


def fp8_block_gemv(
    x: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: tuple[int, int],
    inverse: bool,
    bias: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
) -> torch.Tensor:
    """Block-scaled W8A16 product that never materialises a decoded weight.

    A supplied bias is added in FP32 inside the epilogue, so the output is
    rounded once, exactly like the wide-bias path of ``matmul_fp32_reduce``.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError("FP8 block GEMV: Triton is not available.")
    rows, in_features = x.shape
    out_features = codes.shape[0]
    block_m, block_n, num_warps, num_stages = _gemv_tiles(rows, out_features)
    out = torch.empty((rows, out_features), dtype=x.dtype, device=x.device)
    with _launch_device(x.device):
        _fp8_block_gemv_kernel[(triton.cdiv(out_features, block_n), triton.cdiv(rows, block_m))](
            x,
            codes,
            scale,
            out,
            bias if bias is not None else scale,
            rows,
            out_features,
            in_features,
            x.stride(0),
            x.stride(1),
            codes.stride(0),
            codes.stride(1),
            scale.stride(0),
            scale.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=int(block_size[0]),
            BLOCK_W=int(block_size[1]),
            INVERSE=inverse,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=_gemv_block_k(),
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


__all__ = [
    "TRITON_AVAILABLE",
    "TRITON_FP8_FORMATS",
    "TRITON_FP8_MIN_CAPABILITY",
    "fp8_block_decode",
    "fp8_block_decode_supported",
    "fp8_block_gemv",
    "fp8_block_gemv_payload_supported",
    "fp8_operand_supported",
    "triton_fp8_available",
    "triton_fp8_supports_format",
]
