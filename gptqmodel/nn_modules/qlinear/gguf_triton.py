# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import weakref
from typing import Any, Callable, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.python import has_gil_disabled
from ...utils.rocm import IS_ROCM
from ._compile_support import COMPILE_LEVEL
from ._runtime_cache import tensor_cache_key
from .gguf import GGUFTorchLinear, _unpack_q4_k_scale_min_torch, gguf_tensor_qtype


log = setup_logger()


try:
    import triton
    import triton.language as tl
    from packaging import version

    from gptqmodel.utils.triton_jit import jit as triton_jit

    from ..triton_utils import custom_autotune

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    custom_autotune = None
    _TRITON_AVAILABLE = False

_CUDA_DEVICE_CAPABILITY_CACHE: dict[int, tuple[int, int]] = {}

# Q1_0 G128 fixed launch configs are plain data consumed by the launch
# selectors below, which are importable (and unit-tested) without Triton.
_Q1_0_G128_SM80_DECODE_NARROW_CONFIG = {
    "BLOCK_SIZE_M": 4,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 2,
}
_Q1_0_G128_SM80_DECODE_WIDE_CONFIG = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 4,
}
_Q1_0_G128_SM89_DECODE_NARROW_CONFIG = {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 32,
    "num_warps": 8,
    "num_stages": 2,
}
_Q1_0_G128_SM89_DECODE_2048_TO_2048_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 2,
}
_Q1_0_G128_SM89_DECODE_2048_TO_6144_CONFIG = {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 64,
    "num_warps": 8,
    "num_stages": 2,
}
_Q1_0_G128_SM89_DECODE_WIDE_CONFIG = {
    "BLOCK_SIZE_M": 2,
    "BLOCK_SIZE_N": 32,
    "num_warps": 8,
    "num_stages": 4,
}
_Q1_0_G128_SM80_PREFILL_NARROW_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 16,
    "num_warps": 4,
    "num_stages": 4,
}
_Q1_0_G128_SM80_PREFILL_WIDE_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 4,
}
_Q1_0_G128_SM89_PREFILL_NARROW_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 16,
    "num_warps": 8,
    "num_stages": 4,
}
_Q1_0_G128_SM89_PREFILL_WIDE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "num_warps": 4,
    "num_stages": 4,
}
_Q1_0_G128_SM80_U32_DECODE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 16,
    "num_warps": 4,
    "num_stages": 4,
}
_Q1_0_G128_SM80_U32_PREFILL_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 4,
}
# These exact-shape decode configs target the dominant post-down-proj A100 decode hotspots.
_Q1_0_G128_SM80_U32_DECODE_2048_TO_1024_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 16,
    "num_warps": 2,
    "num_stages": 2,
}
_Q1_0_G128_SM80_U32_DECODE_2048_TO_2048_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 2,
}
_Q1_0_G128_SM80_U32_DECODE_2048_TO_6144_CONFIG = {
    "BLOCK_SIZE_M": 8,
    "BLOCK_SIZE_N": 32,
    "num_warps": 4,
    "num_stages": 4,
}


# The SM89 fixed Q4_K/Q5_K/Q6_K launch table is plain data read by the module-level
# selectors below, which are importable (and unit-tested) without Triton.
#
# ------------------------------------------------------------------
# SM89 (RTX 4090) fixed launch configs for the Q4_K/Q5_K/Q6_K fused
# kernels.  Measured on one RTX 4090 (torch 2.15.0+cu132, Triton 3.9.0),
# FP16, 4-bit group 128 (KxN), as ``hot / 256 MiB L2-flushed`` medians with
# ``torch.cuda.synchronize`` around every launch, against the kernel's own
# autotuned ``best_config``:
#
#   Q4_K  (1024,128)    blocks=4   rows 1/8/64/127: 1.10x-1.15x
#   Q4_K  (2048,1024)   blocks=8   rows 1/8/64 all faster
#   Q4_K  (2048,2048)   blocks=8   rows 1/8/64/127 all faster
#   Q4_K  (4096,4096)   blocks=16  rows 8/64/127 faster; rows 1 slower
#   Q4_K  (4096,11008)  blocks=16  rows 8/64/127 faster; rows 1 slower
#   Q5_K  (2048,2048)   blocks=8   rows 1/8/64/127 all faster
#   Q5_K  (4096,4096)   blocks=16  rows 8/64/127 faster; rows 1 slower
#   Q5_K  (4096,11008)  blocks=16  rows 8/64/127 faster; rows 1 slower
#   Q6_K  (1024,128)    blocks=4   rows 1/8/64/127: 1.07x-1.13x
#   Q6_K  (2048,1024)   blocks=8   rows 1/8/64/127 all faster
#   Q6_K  (2048,2048)   blocks=8   rows 1/8/64/127 all faster
#   Q6_K  (4096,4096)   blocks=16  rows 8/64/127 faster; rows 1 slower
#   Q6_K  (4096,11008)  blocks=16  rows 8/64/127 faster; rows 1 slower
#
# Keys are the *padded* in_features the dispatcher sees (GGUF blocks are
# 256 wide, so a 896-wide projection is padded to 1024 before the kernel).
# The narrow entries were re-measured on the padded 1024-wide input after
# an unpadded 896-wide probe produced out-of-bounds reads: only the padded
# numbers are claimed (1.07x-1.15x, bit-identical to autotuned).
#
# Only those exact (in_features, out_features) shapes are wired: a fixed
# tile that was not measured on the exact shape could silently lose, and
# the autotuned path is always available.  Rows 2..7 and rows > 127 were
# not swept and stay autotuned.  Where the large-block decode tile lost in
# both hot and flushed medians (rows 1 with >= 16 K blocks: the autotuner
# picks a 2-stage tile that also survives the L2 flush, e.g. Q4_K
# 4096x4096 auto 50.2/65.5 us vs the best fixed 53.2/83.8 us) it is
# deliberately left on the autotuned generic kernel.
_GGUF_SM89_QK_SHAPES = {
    "Q4_K": {
        1: {
            (1024, 128): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 2},
            (2048, 1024): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 2},
            (2048, 2048): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 2},
        },
        "mid": {
            (1024, 128): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 1024): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 2048): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        },
        "large": {
            (1024, 128): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 1024): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 2048): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 2},
        },
    },
    "Q5_K": {
        1: {
            (2048, 2048): {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 2},
        },
        "mid": {
            (2048, 2048): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        },
        "large": {
            (2048, 2048): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        },
    },
    "Q6_K": {
        1: {
            (1024, 128): {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "num_warps": 8, "num_stages": 2},
            (2048, 1024): {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "num_warps": 8, "num_stages": 2},
            (2048, 2048): {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 32, "num_warps": 8, "num_stages": 2},
        },
        "mid": {
            (1024, 128): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 1024): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 2048): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        },
        "large": {
            (1024, 128): {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 1024): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (2048, 2048): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 4096): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
            (4096, 11008): {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 4},
        },
    },
}


def triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:
    @triton_jit
    def _gguf_cast_wide_input_kernel(
        x_ptr, half_ptr, scale_ptr, K: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # rows * K can exceed 2^31 for a very large prefill batch; compute
        # the row offset in 64-bit so the load/store addresses do not wrap.
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK_K)
        values = tl.load(x_ptr + row * K + columns, columns < K, other=0).to(tl.float32)
        maximum = tl.max(tl.abs(values), 0)
        # FP16 cannot represent the full BF16/FP32 activation range. Keep the
        # largest element inside FP16's finite range; for a tiny
        # row, lift it out of the FP16 subnormal/underflow range instead.
        # A power-of-two scale preserves BF16 activation values exactly when
        # they are moved into FP16's exponent range.
        wide_scale = tl.exp2(tl.ceil(tl.log2(maximum / 65504.0)))
        scale = tl.where(maximum > 65504.0, wide_scale,
                         tl.where((maximum > 0) & (maximum < 1.0 / 16384.0), maximum * 2.0, 1.0))
        scale = tl.where((scale > 0) & (scale < 3.4028235e38), scale, 1.0)
        tl.store(half_ptr + row * K + columns, values / scale, columns < K)
        tl.store(scale_ptr + row, scale)

    @triton_jit(do_not_specialize=["N", "ELEMENTS"])
    def _gguf_restore_wide_output_kernel(
        output_ptr, scale_ptr, bias_ptr, result_ptr,
        N, ELEMENTS, HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # `N` and `ELEMENTS` depend on the token count. Keeping them as
        # `tl.constexpr` forced a full recompile for every new M; as runtime
        # ints the same binary serves every batch size.
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < ELEMENTS
        result = tl.load(output_ptr + offsets, mask, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + offsets // N, mask, other=1)
        result = result * scale
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offsets % N, mask, other=0).to(tl.float32)
            result = result + bias
        tl.store(result_ptr + offsets, result, mask)

    _GGUF_TRITON_SMALL_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
    ]

    _GGUF_TRITON_LARGE_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 16,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
            },
            num_stages=2,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 16,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 16,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
            },
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
            },
            num_stages=2,
            num_warps=8,
        ),
    ]
    _GGUF_TRITON_LARGE_NUM_BLOCKS = 16


    @triton_jit
    def _gguf_q1_0_g128_fused_matmul_kernel_impl(
        x_ptr,
        sign_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        NUM_BLOCKS,
        stride_xm,
        stride_xk,
        stride_qb,
        stride_qq,
        stride_qn,
        stride_sb,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        bit_shifts = tl.arange(0, 8)

        for block_idx in range(0, NUM_BLOCKS):
            scale = tl.load(
                scale_ptr + block_idx * stride_sb + offs_n * stride_sn,
                mask=n_mask,
                other=0.0,
            )

            for sign_group in range(0, 4):
                offs_k = block_idx * 128 + sign_group * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                packed = tl.load(
                    sign_ptr
                    + block_idx * stride_qb
                    + (sign_group * 4 + tl.arange(0, 4))[:, None] * stride_qq
                    + offs_n[None, :] * stride_qn,
                    mask=n_mask[None, :],
                    other=0,
                )
                sign_bits = (packed[:, None, :] >> bit_shifts[None, :, None]) & 0x01
                signs = tl.reshape(sign_bits, (32, BLOCK_SIZE_N))
                weight = (tl.cast(signs, tl.float16) * 2.0 - 1.0) * scale[None, :]
                if x_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a, weight.to(tl.float32), input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton_jit
    def _gguf_q1_0_g128_u32_fused_matmul_kernel_impl(
        x_ptr,
        sign_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        NUM_BLOCKS,
        stride_xm,
        stride_xk,
        stride_qb,
        stride_qg,
        stride_qn,
        stride_sb,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        bit_shifts = tl.arange(0, 32)

        for block_idx in range(0, NUM_BLOCKS):
            scale = tl.load(
                scale_ptr + block_idx * stride_sb + offs_n * stride_sn,
                mask=n_mask,
                other=0.0,
            )

            for sign_group in range(0, 4):
                offs_k = block_idx * 128 + sign_group * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                packed = tl.load(
                    sign_ptr + block_idx * stride_qb + sign_group * stride_qg + offs_n * stride_qn,
                    mask=n_mask,
                    other=0,
                )
                packed = tl.cast(packed, tl.uint32)
                sign_bits = (packed[None, :] >> bit_shifts[:, None]) & 0x01
                weight = (tl.cast(sign_bits, tl.float16) * 2.0 - 1.0) * scale[None, :]
                if x_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a, weight.to(tl.float32), input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton_jit
    def _gguf_q1_0_g128_k2048_fused_matmul_kernel_impl(
        x_ptr,
        sign_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        stride_xm,
        stride_xk,
        stride_qb,
        stride_qq,
        stride_qn,
        stride_sb,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        bit_shifts = tl.arange(0, 8)

        for block_idx in range(0, 16):
            scale = tl.load(
                scale_ptr + block_idx * stride_sb + offs_n * stride_sn,
                mask=n_mask,
                other=0.0,
            )

            for sign_group in range(0, 4):
                offs_k = block_idx * 128 + sign_group * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                packed = tl.load(
                    sign_ptr
                    + block_idx * stride_qb
                    + (sign_group * 4 + tl.arange(0, 4))[:, None] * stride_qq
                    + offs_n[None, :] * stride_qn,
                    mask=n_mask[None, :],
                    other=0,
                )
                sign_bits = (packed[:, None, :] >> bit_shifts[None, :, None]) & 0x01
                signs = tl.reshape(sign_bits, (32, BLOCK_SIZE_N))
                weight = (tl.cast(signs, tl.float16) * 2.0 - 1.0) * scale[None, :]
                if x_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a, weight.to(tl.float32), input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton_jit
    def _gguf_q1_0_g128_u32_k2048_fused_matmul_kernel_impl(
        x_ptr,
        sign_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        stride_xm,
        stride_xk,
        stride_qb,
        stride_qg,
        stride_qn,
        stride_sb,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        bit_shifts = tl.arange(0, 32)

        for block_idx in range(0, 16):
            scale = tl.load(
                scale_ptr + block_idx * stride_sb + offs_n * stride_sn,
                mask=n_mask,
                other=0.0,
            )

            for sign_group in range(0, 4):
                offs_k = block_idx * 128 + sign_group * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                packed = tl.load(
                    sign_ptr + block_idx * stride_qb + sign_group * stride_qg + offs_n * stride_qn,
                    mask=n_mask,
                    other=0,
                )
                packed = tl.cast(packed, tl.uint32)
                sign_bits = (packed[None, :] >> bit_shifts[:, None]) & 0x01
                weight = (tl.cast(sign_bits, tl.float16) * 2.0 - 1.0) * scale[None, :]
                if x_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a, weight.to(tl.float32), input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    _gguf_q1_0_g128_fused_matmul_kernel_small = custom_autotune.autotune(
        configs=_GGUF_TRITON_SMALL_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_fused_matmul_kernel_impl)
    _gguf_q1_0_g128_fused_matmul_kernel_large = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N", "NUM_BLOCKS"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_fused_matmul_kernel_impl)
    _gguf_q1_0_g128_u32_fused_matmul_kernel_small = custom_autotune.autotune(
        configs=_GGUF_TRITON_SMALL_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_u32_fused_matmul_kernel_impl)
    _gguf_q1_0_g128_u32_fused_matmul_kernel_large = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N", "NUM_BLOCKS"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_u32_fused_matmul_kernel_impl)
    _gguf_q1_0_g128_k2048_fused_matmul_kernel = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_k2048_fused_matmul_kernel_impl)
    _gguf_q1_0_g128_u32_k2048_fused_matmul_kernel = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q1_0_g128_u32_k2048_fused_matmul_kernel_impl)

    @triton_jit
    def _gguf_q4_k_fused_matmul_kernel_impl(
        x_ptr,
        qs_ptr,
        scale_ptr,
        min_ptr,
        out_ptr,
        M,
        N,
        NUM_BLOCKS,
        stride_xm,
        stride_xk,
        stride_qb,
        stride_qq,
        stride_qn,
        stride_sb,
        stride_ss,
        stride_sn,
        stride_mb,
        stride_ms,
        stride_mn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for block_idx in range(0, NUM_BLOCKS):
            for subblock in range(0, 8):
                offs_k = block_idx * 256 + subblock * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                byte_idx = (subblock // 2) * 32 + tl.arange(0, 32)
                packed = tl.load(
                    qs_ptr + block_idx * stride_qb + byte_idx[:, None] * stride_qq + offs_n[None, :] * stride_qn,
                    mask=n_mask[None, :],
                    other=0,
                )
                if subblock % 2 == 0:
                    q = packed & 0x0F
                else:
                    q = packed >> 4

                scale = tl.load(
                    scale_ptr + block_idx * stride_sb + subblock * stride_ss + offs_n * stride_sn,
                    mask=n_mask,
                    other=0.0,
                )
                min_value = tl.load(
                    min_ptr + block_idx * stride_mb + subblock * stride_ms + offs_n * stride_mn,
                    mask=n_mask,
                    other=0.0,
                )

                weight = tl.cast(q, scale.dtype) * scale[None, :] - min_value[None, :]
                if scale_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a.to(tl.float32), weight, input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    _gguf_q4_k_fused_matmul_kernel_small = custom_autotune.autotune(
        configs=_GGUF_TRITON_SMALL_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q4_k_fused_matmul_kernel_impl)
    _gguf_q4_k_fused_matmul_kernel_large = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N", "NUM_BLOCKS"],
        nearest_power_of_two=True,
    )(_gguf_q4_k_fused_matmul_kernel_impl)

    @triton_jit
    def _gguf_q5_k_fused_matmul_kernel_impl(
        x_ptr,
        qs_ptr,
        qh_ptr,
        scale_ptr,
        min_ptr,
        out_ptr,
        M,
        N,
        NUM_BLOCKS,
        stride_xm,
        stride_xk,
        stride_qsb,
        stride_qsq,
        stride_qsn,
        stride_qhb,
        stride_qhq,
        stride_qhn,
        stride_sb,
        stride_ss,
        stride_sn,
        stride_mb,
        stride_ms,
        stride_mn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for block_idx in range(0, NUM_BLOCKS):
            for subblock in range(0, 8):
                offs_k = block_idx * 256 + subblock * 32 + tl.arange(0, 32)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                byte_idx = (subblock // 2) * 32 + tl.arange(0, 32)
                packed = tl.load(
                    qs_ptr + block_idx * stride_qsb + byte_idx[:, None] * stride_qsq + offs_n[None, :] * stride_qsn,
                    mask=n_mask[None, :],
                    other=0,
                )
                if subblock % 2 == 0:
                    ql = packed & 0x0F
                else:
                    ql = packed >> 4

                qh = tl.load(
                    qh_ptr + block_idx * stride_qhb + tl.arange(0, 32)[:, None] * stride_qhq + offs_n[None, :] * stride_qhn,
                    mask=n_mask[None, :],
                    other=0,
                )
                qh = (qh >> subblock) & 0x01
                q = ql | (qh << 4)

                scale = tl.load(
                    scale_ptr + block_idx * stride_sb + subblock * stride_ss + offs_n * stride_sn,
                    mask=n_mask,
                    other=0.0,
                )
                min_value = tl.load(
                    min_ptr + block_idx * stride_mb + subblock * stride_ms + offs_n * stride_mn,
                    mask=n_mask,
                    other=0.0,
                )

                weight = tl.cast(q, scale.dtype) * scale[None, :] - min_value[None, :]
                if scale_ptr.dtype.element_ty == tl.float32:
                    accumulator += tl.dot(a.to(tl.float32), weight, input_precision="ieee")
                else:
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    _gguf_q5_k_fused_matmul_kernel_small = custom_autotune.autotune(
        configs=_GGUF_TRITON_SMALL_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q5_k_fused_matmul_kernel_impl)
    _gguf_q5_k_fused_matmul_kernel_large = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N", "NUM_BLOCKS"],
        nearest_power_of_two=True,
    )(_gguf_q5_k_fused_matmul_kernel_impl)

    @triton_jit
    def _gguf_q6_k_fused_matmul_kernel_impl(
        x_ptr,
        ql_ptr,
        qh_ptr,
        scale_ptr,
        out_ptr,
        M,
        N,
        NUM_BLOCKS,
        stride_xm,
        stride_xk,
        stride_qlb,
        stride_qlq,
        stride_qln,
        stride_qhb,
        stride_qhq,
        stride_qhn,
        stride_sb,
        stride_ss,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for block_idx in range(0, NUM_BLOCKS):
            for subblock in range(0, 16):
                offs_k = block_idx * 256 + subblock * 16 + tl.arange(0, 16)
                a = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=m_mask[:, None],
                    other=0.0,
                )

                pair_row = subblock // 2
                half = subblock % 2
                row_group = pair_row // 4
                row_in_group = pair_row % 4
                pos32 = half * 16 + tl.arange(0, 16)
                ql_base = row_group * 64 + (32 if row_in_group == 1 or row_in_group == 3 else 0)

                packed_ql = tl.load(
                    ql_ptr + block_idx * stride_qlb + (ql_base + pos32)[:, None] * stride_qlq + offs_n[None, :] * stride_qln,
                    mask=n_mask[None, :],
                    other=0,
                )
                if row_in_group == 0 or row_in_group == 1:
                    low = packed_ql & 0x0F
                else:
                    low = packed_ql >> 4

                packed_qh = tl.load(
                    qh_ptr + block_idx * stride_qhb + (row_group * 32 + pos32)[:, None] * stride_qhq + offs_n[None, :] * stride_qhn,
                    mask=n_mask[None, :],
                    other=0,
                )
                high = (packed_qh >> (row_in_group * 2)) & 0x03
                q = tl.cast(low | (high << 4), tl.int16) - 32

                scale = tl.load(
                    scale_ptr + block_idx * stride_sb + subblock * stride_ss + offs_n * stride_sn,
                    mask=n_mask,
                    other=0.0,
                )

                if scale_ptr.dtype.element_ty == tl.float32:
                    weight = tl.cast(q, tl.float32) * scale[None, :]
                    accumulator += tl.dot(a.to(tl.float32), weight, input_precision="ieee")
                else:
                    weight = tl.cast(q, tl.float16) * scale[None, :]
                    accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    _gguf_q6_k_fused_matmul_kernel_small = custom_autotune.autotune(
        configs=_GGUF_TRITON_SMALL_CONFIGS,
        key=["M", "N"],
        nearest_power_of_two=True,
    )(_gguf_q6_k_fused_matmul_kernel_impl)
    _gguf_q6_k_fused_matmul_kernel_large = custom_autotune.autotune(
        configs=_GGUF_TRITON_LARGE_CONFIGS,
        key=["M", "N", "NUM_BLOCKS"],
        nearest_power_of_two=True,
    )(_gguf_q6_k_fused_matmul_kernel_impl)


def _launch(
    kernel: Callable,
    x: torch.Tensor,
    output: torch.Tensor,
    *args,
) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("GGUF Triton fused matmul requires CUDA/ROCm tensors.")
    if x.shape[0] == 0 or output.shape[1] == 0:
        return output

    def grid(meta):
        return (
            triton.cdiv(x.shape[0], meta["BLOCK_SIZE_M"]),
            triton.cdiv(output.shape[1], meta["BLOCK_SIZE_N"]),
        )

    # Triton launches into the current CUDA context. A layer placed on cuda:1
    # by `device_map` must enter that device first or Triton raises
    # "Pointer argument cannot be accessed from Triton" while cuda:0 is current.
    with torch.cuda.device(x.device):
        kernel[grid](*args)
    return output


def _launch_with_meta(
    kernel: Callable,
    x: torch.Tensor,
    output: torch.Tensor,
    *args,
    **meta,
) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("GGUF Triton fused matmul requires CUDA/ROCm tensors.")
    if x.shape[0] == 0 or output.shape[1] == 0:
        return output

    def grid(meta_args):
        return (
            triton.cdiv(x.shape[0], meta_args["BLOCK_SIZE_M"]),
            triton.cdiv(output.shape[1], meta_args["BLOCK_SIZE_N"]),
        )

    with torch.cuda.device(x.device):
        kernel[grid](*args, **meta)
    return output


def _select_triton_kernel(
    small_kernel: Callable,
    large_kernel: Callable,
    *,
    num_blocks: int,
) -> Callable:
    if num_blocks >= _GGUF_TRITON_LARGE_NUM_BLOCKS:
        return large_kernel
    return small_kernel


def _select_q1_0_g128_fixed_launch_config(
    *,
    capability: tuple[int, int] | None,
    rows: int,
    in_features: int | None = None,
    cols: int,
) -> dict[str, int] | None:
    if capability == (8, 0):
        if rows == 1:
            if cols <= 2048:
                return dict(_Q1_0_G128_SM80_DECODE_NARROW_CONFIG)
            return dict(_Q1_0_G128_SM80_DECODE_WIDE_CONFIG)
        if 8 <= rows <= 128:
            if cols == 2048:
                return dict(_Q1_0_G128_SM80_PREFILL_NARROW_CONFIG)
            if cols == 6144:
                return dict(_Q1_0_G128_SM80_PREFILL_WIDE_CONFIG)
    if capability == (8, 9):
        if rows == 1:
            if cols == 1024:
                return dict(_Q1_0_G128_SM89_DECODE_NARROW_CONFIG)
            if in_features == 2048 and cols == 2048:
                return dict(_Q1_0_G128_SM89_DECODE_2048_TO_2048_CONFIG)
            if in_features == 2048 and cols == 6144:
                return dict(_Q1_0_G128_SM89_DECODE_2048_TO_6144_CONFIG)
            if cols == 2048:
                return dict(_Q1_0_G128_SM89_DECODE_NARROW_CONFIG)
            if cols == 6144:
                return dict(_Q1_0_G128_SM89_DECODE_WIDE_CONFIG)
        if 8 <= rows <= 128:
            if cols == 2048:
                return dict(_Q1_0_G128_SM89_PREFILL_NARROW_CONFIG)
            if cols == 6144:
                return dict(_Q1_0_G128_SM89_PREFILL_WIDE_CONFIG)
    return None


def _select_q1_0_g128_u32_layout(
    *,
    capability: tuple[int, int] | None,
    in_features: int,
    out_features: int,
) -> bool:
    if capability != (8, 0):
        return False
    return (in_features, out_features) in {
        (2048, 1024),
        (2048, 2048),
        (2048, 6144),
        (6144, 2048),
    }


def _select_q1_0_g128_u32_fixed_launch_config(
    *,
    capability: tuple[int, int] | None,
    rows: int,
    in_features: int,
    out_features: int,
) -> dict[str, int] | None:
    if capability != (8, 0):
        return None
    if rows == 1:
        if in_features == 6144 and out_features == 2048:
            return dict(_Q1_0_G128_SM80_U32_DECODE_CONFIG)
        if in_features == 2048 and out_features == 1024:
            return dict(_Q1_0_G128_SM80_U32_DECODE_2048_TO_1024_CONFIG)
        if in_features == 2048 and out_features == 2048:
            return dict(_Q1_0_G128_SM80_U32_DECODE_2048_TO_2048_CONFIG)
        if in_features == 2048 and out_features == 6144:
            return dict(_Q1_0_G128_SM80_U32_DECODE_2048_TO_6144_CONFIG)
        return None
    if 8 <= rows <= 128 and in_features == 6144 and out_features == 2048:
        return dict(_Q1_0_G128_SM80_U32_PREFILL_CONFIG)
    return None


def _use_q1_0_g128_k2048_decode_specialization(
    *,
    rows: int,
    in_features: int,
) -> bool:
    return rows == 1 and in_features == 2048

def _gguf_qk_row_band(rows: int) -> str | int | None:
    """Measured row band for the fixed Q4_K/Q5_K/Q6_K configs on SM89.

    ``1`` is decode, ``"mid"`` is the 8..63 band measured at M=8 and ``"large"``
    is the 64..127 band measured at M=64/127.  Rows 2..7 and rows > 127 were
    not swept and must stay on the autotuned generic kernel.
    """

    if rows == 1:
        return 1
    if 8 <= rows <= 63:
        return "mid"
    if 64 <= rows <= 127:
        return "large"
    return None


def _select_gguf_qk_fixed_launch_config(
    qtype: str,
    *,
    capability: tuple[int, int] | None,
    rows: int,
    in_features: int,
    out_features: int,
) -> dict[str, int] | None:
    """Return a measured SM89 fixed config, or ``None`` for the autotuned path.

    Exact shape match on purpose: the measured table above only backs these
    (K, N) pairs, so an unmeasured projection keeps the kernel's own autotuner
    instead of inheriting a tile that may lose.
    """

    if capability != (8, 9):
        return None
    band = _gguf_qk_row_band(rows)
    if band is None:
        return None
    config = _GGUF_SM89_QK_SHAPES.get(qtype, {}).get(band, {}).get((in_features, out_features))
    return None if config is None else dict(config)


def _select_q4_k_fixed_launch_config(**kwargs) -> dict[str, int] | None:
    return _select_gguf_qk_fixed_launch_config("Q4_K", **kwargs)


def _select_q5_k_fixed_launch_config(**kwargs) -> dict[str, int] | None:
    return _select_gguf_qk_fixed_launch_config("Q5_K", **kwargs)


def _select_q6_k_fixed_launch_config(**kwargs) -> dict[str, int] | None:
    return _select_gguf_qk_fixed_launch_config("Q6_K", **kwargs)


def _cuda_device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    if IS_ROCM:
        # ROCm aliases the CUDA device namespace but reports a meaningless
        # compute capability.  All NVIDIA capability-gated specializations
        # (Q1_0 u32 layout, fixed decode/prefill configs) are therefore
        # skipped; the generic Triton kernels and GGUF_TORCH remain available.
        log.debug.once(
            "GGUF Triton: ROCm host detected; CUDA capability specializations disabled "
            "(generic launch configs only). AUTO still selects GGUFTritonKernel; "
            "GGUF_TORCH remains available as a fallback."
        )
        return None

    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = _CUDA_DEVICE_CAPABILITY_CACHE.get(index)
    if capability is None:
        capability = torch.cuda.get_device_capability(index)
        _CUDA_DEVICE_CAPABILITY_CACHE[index] = capability
    return capability


def fused_q4_k_matmul(
    x: torch.Tensor,
    qs: torch.Tensor,
    scale: torch.Tensor,
    min_value: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q4_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    kernel = _select_triton_kernel(
        _gguf_q4_k_fused_matmul_kernel_small,
        _gguf_q4_k_fused_matmul_kernel_large,
        num_blocks=qs.shape[0],
    )
    return _launch(
        kernel,
        x,
        output,
        x,
        qs,
        scale,
        min_value,
        output,
        x.shape[0],
        output.shape[1],
        qs.shape[0],
        x.stride(0),
        x.stride(1),
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        min_value.stride(0),
        min_value.stride(1),
        min_value.stride(2),
        output.stride(0),
        output.stride(1),
    )


def fused_q1_0_g128_matmul(
    x: torch.Tensor,
    sign_bytes: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    kernel = _select_triton_kernel(
        _gguf_q1_0_g128_fused_matmul_kernel_small,
        _gguf_q1_0_g128_fused_matmul_kernel_large,
        num_blocks=sign_bytes.shape[0],
    )
    return _launch(
        kernel,
        x,
        output,
        x,
        sign_bytes,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        sign_bytes.shape[0],
        x.stride(0),
        x.stride(1),
        sign_bytes.stride(0),
        sign_bytes.stride(1),
        sign_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
    )


def fused_q1_0_g128_k2048_matmul(
    x: torch.Tensor,
    sign_bytes: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch(
        _gguf_q1_0_g128_k2048_fused_matmul_kernel,
        x,
        output,
        x,
        sign_bytes,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        x.stride(0),
        x.stride(1),
        sign_bytes.stride(0),
        sign_bytes.stride(1),
        sign_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
    )


def fused_q1_0_g128_u32_matmul(
    x: torch.Tensor,
    sign_words: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    kernel = _select_triton_kernel(
        _gguf_q1_0_g128_u32_fused_matmul_kernel_small,
        _gguf_q1_0_g128_u32_fused_matmul_kernel_large,
        num_blocks=sign_words.shape[0],
    )
    return _launch(
        kernel,
        x,
        output,
        x,
        sign_words,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        sign_words.shape[0],
        x.stride(0),
        x.stride(1),
        sign_words.stride(0),
        sign_words.stride(1),
        sign_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
    )


def fused_q1_0_g128_u32_k2048_matmul(
    x: torch.Tensor,
    sign_words: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch(
        _gguf_q1_0_g128_u32_k2048_fused_matmul_kernel,
        x,
        output,
        x,
        sign_words,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        x.stride(0),
        x.stride(1),
        sign_words.stride(0),
        sign_words.stride(1),
        sign_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
    )


def _launch_q1_0_g128_u32_fixed_matmul(
    x: torch.Tensor,
    sign_words: torch.Tensor,
    scale: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q1_0_g128_u32_fused_matmul_kernel_impl,
        x,
        output,
        x,
        sign_words,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        sign_words.shape[0],
        x.stride(0),
        x.stride(1),
        sign_words.stride(0),
        sign_words.stride(1),
        sign_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q1_0_g128_u32_k2048_fixed_matmul(
    x: torch.Tensor,
    sign_words: torch.Tensor,
    scale: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q1_0_g128_u32_k2048_fused_matmul_kernel_impl,
        x,
        output,
        x,
        sign_words,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        x.stride(0),
        x.stride(1),
        sign_words.stride(0),
        sign_words.stride(1),
        sign_words.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q1_0_g128_fixed_matmul(
    x: torch.Tensor,
    sign_bytes: torch.Tensor,
    scale: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q1_0_g128_fused_matmul_kernel_impl,
        x,
        output,
        x,
        sign_bytes,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        sign_bytes.shape[0],
        x.stride(0),
        x.stride(1),
        sign_bytes.stride(0),
        sign_bytes.stride(1),
        sign_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q1_0_g128_k2048_fixed_matmul(
    x: torch.Tensor,
    sign_bytes: torch.Tensor,
    scale: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q1_0_g128_k2048_fused_matmul_kernel_impl,
        x,
        output,
        x,
        sign_bytes,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        x.stride(0),
        x.stride(1),
        sign_bytes.stride(0),
        sign_bytes.stride(1),
        sign_bytes.stride(2),
        scale.stride(0),
        scale.stride(1),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q4_k_fixed_matmul(
    x: torch.Tensor,
    qs: torch.Tensor,
    scale: torch.Tensor,
    min_value: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q4_k_fused_matmul_kernel_impl,
        x,
        output,
        x,
        qs,
        scale,
        min_value,
        output,
        x.shape[0],
        output.shape[1],
        qs.shape[0],
        x.stride(0),
        x.stride(1),
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        min_value.stride(0),
        min_value.stride(1),
        min_value.stride(2),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q5_k_fixed_matmul(
    x: torch.Tensor,
    qs: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    min_value: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q5_k_fused_matmul_kernel_impl,
        x,
        output,
        x,
        qs,
        qh,
        scale,
        min_value,
        output,
        x.shape[0],
        output.shape[1],
        qs.shape[0],
        x.stride(0),
        x.stride(1),
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        qh.stride(0),
        qh.stride(1),
        qh.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        min_value.stride(0),
        min_value.stride(1),
        min_value.stride(2),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def _launch_q6_k_fixed_matmul(
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    *,
    fixed_config: dict[str, int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    return _launch_with_meta(
        _gguf_q6_k_fused_matmul_kernel_impl,
        x,
        output,
        x,
        ql,
        qh,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        ql.shape[0],
        x.stride(0),
        x.stride(1),
        ql.stride(0),
        ql.stride(1),
        ql.stride(2),
        qh.stride(0),
        qh.stride(1),
        qh.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(0),
        output.stride(1),
        **fixed_config,
    )


def fused_q5_k_matmul(
    x: torch.Tensor,
    qs: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    min_value: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q5_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    kernel = _select_triton_kernel(
        _gguf_q5_k_fused_matmul_kernel_small,
        _gguf_q5_k_fused_matmul_kernel_large,
        num_blocks=qs.shape[0],
    )
    return _launch(
        kernel,
        x,
        output,
        x,
        qs,
        qh,
        scale,
        min_value,
        output,
        x.shape[0],
        output.shape[1],
        qs.shape[0],
        x.stride(0),
        x.stride(1),
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        qh.stride(0),
        qh.stride(1),
        qh.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        min_value.stride(0),
        min_value.stride(1),
        min_value.stride(2),
        output.stride(0),
        output.stride(1),
    )


def fused_q6_k_matmul(
    x: torch.Tensor,
    ql: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q6_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=output_dtype or x.dtype)
    kernel = _select_triton_kernel(
        _gguf_q6_k_fused_matmul_kernel_small,
        _gguf_q6_k_fused_matmul_kernel_large,
        num_blocks=ql.shape[0],
    )
    return _launch(
        kernel,
        x,
        output,
        x,
        ql,
        qh,
        scale,
        output,
        x.shape[0],
        output.shape[1],
        ql.shape[0],
        x.stride(0),
        x.stride(1),
        ql.stride(0),
        ql.stride(1),
        ql.stride(2),
        qh.stride(0),
        qh.stride(1),
        qh.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        output.stride(0),
        output.stride(1),
    )


def _publish_triton_cache_ready(cache: dict[str, Any], device: torch.device) -> None:
    """Order a freshly built cache against later forwards on other streams.

    ``_build_triton_cache`` only *enqueues* the re-layout kernels that fill the
    cache, and the dict is published as soon as it returns. A first forward of
    the same module on another stream would then launch the fused matmul
    against tensors that the building stream has not written yet, reading
    uninitialized memory (#4797). Recording the building stream and an event
    lets a consumer on a different stream wait for exactly that work.
    """

    # Snapshot the tensors a consumer stream must keep alive.  The forward path
    # must never iterate the cache dict itself: a peer thread may retire the
    # handshake (a ``dict.pop``) while this thread is iterating it, which raises
    # "dictionary changed size during iteration" (#4962).  The cache is not
    # visible to other threads until this returns, so the snapshot is safe here.
    cache["tensors"] = tuple(
        value for value in cache.values() if isinstance(value, torch.Tensor)
    )
    try:
        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
    except Exception:
        # No usable stream (CPU/meta placement or a runtime that cannot record
        # events): nothing can be enqueued out of order either.
        return
    cache["ready_stream"] = stream
    cache["ready_event"] = event


def _is_current_stream_capturing() -> bool:
    """True when the active stream is inside a CUDA graph capture."""

    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


_RAW_CURRENT_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream_key(device: torch.device) -> int | None:
    """Identity of the active CUDA stream, without building a Python wrapper.

    ``torch.cuda.current_stream()`` allocates a ``Stream`` object on every call
    (~2 us on the hosts we measure), which is a large fraction of a decode-sized
    launch; the raw handle is the same identity the kernel sees.  Mirrors
    ``gptqmodel.utils.marlin.marlin_stream_key`` (#4956).
    """

    try:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        if _RAW_CURRENT_STREAM is not None:
            return int(_RAW_CURRENT_STREAM(index))
        return int(torch.cuda.current_stream(index).cuda_stream)
    except Exception:
        return None


def _retire_triton_cache_ready(cache: dict[str, Any]) -> None:
    """Drop the build handshake: nothing is left to order."""

    cache.pop("ready_event", None)
    cache.pop("ready_stream", None)


def _wait_for_triton_cache_ready(cache: dict[str, Any], device: torch.device) -> None:
    """Make the caller's stream wait for the cache builder, when they differ."""

    event = cache.get("ready_event")
    if event is None:
        # Retired: the build is complete and the cache is immutable from then
        # on, so every later forward is a plain kernel launch.  Return before
        # touching any stream API; this is the steady-state path (#4956).
        return
    if _is_current_stream_capturing():
        # A captured stream may only wait on an event recorded inside the same
        # capture, so an explicit ``torch.cuda.graph`` can never consume this
        # one: the driver fails the capture with
        # ``CUDA_ERROR_STREAM_CAPTURE_ISOLATION`` (#4867). Even ``Event.query()``
        # is rejected while capturing, and ``torch.cuda.graph`` synchronizes the
        # device before capture starts, so the build is already complete here:
        # retire instead of waiting or querying.
        _retire_triton_cache_ready(cache)
        return
    if event.query():
        # The build finished, whichever stream is asking.  Retire the handshake
        # so this stream - the builder included - pays nothing from now on
        # (#4956).
        _retire_triton_cache_ready(cache)
        return
    builder = cache.get("ready_stream")
    if builder is not None and _current_stream_key(device) == builder.cuda_stream:
        # Same stream: the build is already ordered before this forward, so the
        # common single-stream path stays free of any bookkeeping.
        return
    try:
        current = torch.cuda.current_stream(device)
    except Exception:
        return
    current.wait_event(event)
    # The cached tensors outlive every forward, but the allocator still has to
    # know this stream reads them.  Iterate the publish-time snapshot, never the
    # dict: a peer thread may retire the handshake concurrently (#4962).
    for value in cache.get("tensors", ()):
        value.record_stream(current)


# One lock for every module's first-forward re-layout. The permute itself is a
# single enqueue plus a short stream wait, so serializing it across modules is
# cheap, and it makes "already permuted" a decision no two threads can race on
# (#4879).
_PACKED_WEIGHT_PERMUTE_LOCK = threading.Lock()

# One cold-build lock per module. A published cache entry owns the
# ``scale``/``min`` tensors that every consumer stream reads, so replacing it
# while a consumer is still running drops the last reference and hands those
# blocks back to the allocator under a pending kernel (#5008). Serializing the
# cold build makes the entry a once-per-module decision, and it also removes
# the N redundant builds of an N-thread cold burst.
#
# The lock lives in a weak map keyed by the module instead of on the module:
# a ``threading.Lock`` attribute would make ``copy.deepcopy``/``torch.save`` of
# a quantized model raise ``cannot pickle '_thread.lock' object``, while a
# deep-copied replica (data-parallel forward) still gets its own lock here.
_TRITON_CACHE_BUILD_LOCKS: "weakref.WeakKeyDictionary[torch.nn.Module, threading.Lock]" = (
    weakref.WeakKeyDictionary()
)
_TRITON_CACHE_BUILD_LOCKS_GUARD = threading.Lock()
# Fallback for a module that cannot be a weak key; never
# ``_PACKED_WEIGHT_PERMUTE_LOCK``, which the build itself acquires.
_TRITON_CACHE_BUILD_FALLBACK_LOCK = threading.Lock()


def _triton_cache_build_lock(module: torch.nn.Module) -> threading.Lock:
    """Return the cold-build lock that belongs to ``module``."""

    try:
        with _TRITON_CACHE_BUILD_LOCKS_GUARD:
            lock = _TRITON_CACHE_BUILD_LOCKS.get(module)
            if lock is None:
                lock = threading.Lock()
                _TRITON_CACHE_BUILD_LOCKS[module] = lock
            return lock
    except TypeError:
        # Not weak-referenceable (a test double built without ``__init__``):
        # fall back to one lock for the process rather than fail the forward.
        return _TRITON_CACHE_BUILD_FALLBACK_LOCK


class GGUFTritonKernel(GGUFTorchLinear):
    # Measured on GB10 (sm_121), torch 2.11.0+cu130: eager runs and an explicit
    # torch.cuda.graph capture succeeds, but dynamo breaks on
    # `tensor.is_inference()` plus the raw Triton launches: EAGER only.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = True

    SUPPORTS_BACKENDS = [BACKEND.GGUF_TRITON]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 45}
    SUPPORTS_BITS = [1, 4, 5, 6]
    # The fused Triton kernels only implement these ggml tensor layouts. The
    # bit width is not enough to identify one (Q4_0 and Q4_K are both 4-bit),
    # so AUTO selection must consult the resolved qtype.
    SUPPORTS_GGUF_QTYPES = frozenset({"Q1_0", "Q1_0_g128", "Q4_K", "Q5_K", "Q6_K"})

    #: GGUF tensor types whose fused Triton forward is safe under an explicit
    #: ``torch.cuda.CUDAGraph`` capture (#5591). The fused kernels are pure
    #: device work (``tl.load``/``tl.dot``) with no host copy, so every layout
    #: the constructor accepts is capturable. The inherited
    #: ``GGUFTorchLinear.GRAPH_CAPTURE_QTYPES`` is *not* this kernel's answer: it
    #: lists Q4_0/Q8_0, which this constructor rejects, and omits the 1-bit
    #: formats it does build, so the per-instance answer under-claimed them.
    #: Measured on sm_120 (RTX 5090, torch 2.15.0+cu132, triton 3.9.0): capture,
    #: replay and a changed-input replay are bitwise identical to eager for all
    #: five layouts, including the Q6_K wide-scale path.
    GRAPH_CAPTURE_QTYPES = frozenset({"Q1_0", "Q1_0_g128", "Q4_K", "Q5_K", "Q6_K"})
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    # ROCm is supported: the fused kernels use only tl.load/tl.dot and every
    # NVIDIA capability specialization below is already skipped there (see
    # ``_cuda_device_capability``, which returns None on ROCm), so ROCm runs the
    # generic launch configs.  Measured 5-13x faster than GGUF_TORCH on gfx1201
    # at both decode and prefill, so AUTO selects this kernel on ROCm; see
    # docs/rocm-rdna4.md for the numbers.
    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    AUTOTUNE = False

    QUANT_TYPE = "gguf"

    def __init__(
        self,
        bits,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.GGUF_TRITON)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        if self.gguf_tensor_qtype not in self.SUPPORTS_GGUF_QTYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports fused GGUF Triton formats "
                f"(Q1_0, Q4_K, Q5_K, Q6_K). Actual GGUF qtype: {self.gguf_tensor_qtype}. "
                "Use BACKEND.GGUF_TORCH for unsupported GGUF formats."
            )
        self._gguf_triton_cache: dict[tuple[int, str, bool], dict[str, Any]] = {}

    @classmethod
    def supports_gguf_qtype(cls, bits) -> bool:
        return gguf_tensor_qtype(bits) in cls.SUPPORTS_GGUF_QTYPES

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _TRITON_AVAILABLE:
            return False, ModuleNotFoundError("GGUFTritonKernel requires `triton` to be installed.")

        triton_v = version.parse(triton.__version__)
        if triton_v < version.parse("2.0.0"):
            return False, ImportError(f"triton version must be >= 2.0.0: actual = {triton.__version__}")

        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            return False, Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        if IS_ROCM:
            # NVIDIA capability specializations are disabled on ROCm (generic
            # launch configs only), which is what the RDNA4 measurements below
            # were taken with.
            log.debug.once(
                "GGUF Triton: ROCm host; generic launch configs (no NVIDIA "
                "capability specializations)."
            )

        return True, None

    def clear_weight_cache(self) -> None:
        # Mutate the cache only under the build lock: a concurrent cold build
        # would otherwise re-publish an entry into the dict this just cleared.
        with _triton_cache_build_lock(self):
            self._gguf_triton_cache.clear()
        return super().clear_weight_cache()

    def iter_runtime_weight_tensors(self):
        """Yield the fused-kernel metadata tensors this module owns (#5810).

        ``_gguf_triton_cache`` is a plain dict, so a budget walk that only
        reads ``parameters()`` and ``buffers()`` misses the derived scale and
        min planes the kernels read. The packed code bytes in each entry are
        strided views of ``qweight``; the budget counts their storage once
        through the registered buffer.
        """

        for entry in tuple(self._gguf_triton_cache.values()):
            for value in entry.values():
                if isinstance(value, torch.Tensor):
                    yield value

    def _packed_weight_is_permuted(self) -> bool:
        """True when ``qweight`` holds the transposed runtime storage (#4811)."""
        qweight = self.qweight
        return (
            qweight.numel() > 0
            and qweight.stride(0) == 1
            and qweight.stride(1) == qweight.shape[0]
        )

    def _permute_packed_weight(self) -> None:
        """Re-lay out ``qweight`` once so the fused kernels read it coalesced.

        ``qweight`` is stored output-feature-major, so the permuted
        ``(k_blocks, bytes, out_features)`` code arrays the fused kernels want
        are strided views whose ``offs_n`` axis moves by a whole packed row.
        Materializing them keeps a second full copy of the weights resident
        (#4748); reading the views uncoalesced costs up to +66% prefill
        (#4811).  Transpose the storage once instead and expose the same
        logical ``qweight`` as a transposed view, so there is still exactly one
        resident copy and the code views become row-contiguous.  The extra
        buffer is one module, not the whole model, and is released as soon as
        the copy completes.
        """
        # Two cold builds of the same module can run concurrently on different
        # streams (multi-threaded serving), so the re-layout is serialized and
        # published only once the copy has actually run (#4879).
        with _PACKED_WEIGHT_PERMUTE_LOCK:
            qweight = self.qweight
            if qweight.numel() == 0 or self._packed_weight_is_permuted():
                return
            # The first cache build runs under the caller's ``inference_mode``.
            # ``qweight`` is a public buffer that callers may still mutate in place,
            # so keep its storage a normal (non-inference) tensor.
            with torch.inference_mode(False):
                storage = torch.empty(
                    (qweight.shape[1], qweight.shape[0]),
                    dtype=qweight.dtype,
                    device=qweight.device,
                )
            storage.copy_(qweight.t())
            # ``copy_`` only *enqueues* the transpose on the calling stream. The
            # permuted view is published below, and a second cold build on another
            # stream would otherwise read bytes this copy has not written yet and
            # bake them into its own cache -- silently wrong output (including
            # NaN) for the rest of the session (#4879). Wait for this stream, so
            # publishing ``self.qweight`` is the point of visibility. A cold build
            # cannot happen inside a capture (allocations are not capturable), but
            # ``synchronize`` would be illegal there, so keep the guard.
            if storage.device.type == "cuda" and not _is_current_stream_capturing():
                torch.cuda.current_stream(storage.device).synchronize()
            self.qweight = storage.t()

    def _reshape_blocks(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        target_device = self.qweight.device if device is None else torch.device(device)
        qweight = self.qweight if self.qweight.device == target_device else self.qweight.to(device=target_device)
        rows = qweight.shape[0]
        num_blocks = qweight.shape[1] // self.gguf_type_size
        if self._packed_weight_is_permuted():
            # Permuted storage (#4811): the canonical block view is a strided
            # view of the same bytes, so no copy is needed.
            blocks = qweight.as_strided(
                (rows, num_blocks, self.gguf_type_size),
                (1, self.gguf_type_size * rows, rows),
            )
        else:
            blocks = qweight.contiguous().view(rows, num_blocks, self.gguf_type_size)
        return blocks, rows, num_blocks

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        key = prefix + "qweight"
        if key in destination and not destination[key].is_contiguous():
            # The permuted storage (#4811) is an internal runtime layout; saved
            # checkpoints must stay in the canonical GGUF order.
            destination[key] = destination[key].contiguous()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "qweight"
        if key in state_dict and self._packed_weight_is_permuted():
            # Checkpoints are stored in canonical order, but the live buffer may
            # still hold the permuted runtime layout (#4811).  The default
            # implementation copies element-wise in logical order, so restore a
            # canonical buffer first; the next cache build re-permutes it.
            self.qweight = self.qweight.contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _triton_cache_key(self, device: torch.device, wide_dot: bool = False) -> tuple[int, str]:
        # ``wide_dot`` is fixed by the module's locked activation dtype
        # (contracts/kernel/forward-dtype.md), so one entry per device suffices.
        del wide_dot
        return (device.index if device.index is not None else -1, self.gguf_tensor_qtype)

    def _build_triton_cache(self, device: torch.device, wide_dot: bool = False) -> dict[str, Any]:
        self._permute_packed_weight()
        blocks, _, _ = self._reshape_blocks(device=device)

        if self.gguf_tensor_qtype in {"Q1_0", "Q1_0_g128"}:
            scale = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).permute(1, 0).contiguous()
            capability = _cuda_device_capability(device)
            if _select_q1_0_g128_u32_layout(
                capability=capability,
                in_features=self.padded_in_features,
                out_features=scale.shape[1],
            ):
                # Strided view of the packed sign bits (no second copy, #4748).
                sign_bytes = blocks[..., 2:].permute(1, 2, 0)
                sign_words_src = sign_bytes.to(torch.int32)
                sign_words = (
                    sign_words_src[:, 0::4, :]
                    | torch.bitwise_left_shift(sign_words_src[:, 1::4, :], 8)
                    | torch.bitwise_left_shift(sign_words_src[:, 2::4, :], 16)
                    | torch.bitwise_left_shift(sign_words_src[:, 3::4, :], 24)
                ).contiguous()
                return {
                    "fixed_decode_config": None,
                    "sign_bytes": sign_bytes,
                    "sign_words": sign_words,
                    "scale": scale,
                    "use_u32": True,
                }

            # Strided view of the packed sign bits (no second copy, #4748).
            sign_bytes = blocks[..., 2:].permute(1, 2, 0)
            return {
                "fixed_decode_config": _select_q1_0_g128_fixed_launch_config(
                    capability=capability,
                    rows=1,
                    in_features=self.padded_in_features,
                    cols=scale.shape[1],
                ),
                "sign_bytes": sign_bytes,
                "scale": scale,
                "use_u32": False,
            }

        if self.gguf_tensor_qtype == "Q4_K":
            scale_dtype = torch.float32 if wide_dot or self._k_scale_overflows(blocks, torch.float16) else torch.float16
            d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).to(scale_dtype)
            dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1).to(scale_dtype)
            sc, mins = _unpack_q4_k_scale_min_torch(blocks[..., 4:16])
            scale = (d.unsqueeze(-1) * sc.to(scale_dtype)).permute(1, 2, 0).contiguous()
            min_value = (dmin.unsqueeze(-1) * mins.to(scale_dtype)).permute(1, 2, 0).contiguous()
            # Packed code bytes are a strided view into `qweight`; the
            # fused kernels index the block layout through the strides
            # passed to them. Materializing this used to keep a second
            # full copy of every packed weight resident (#4748).
            qs = blocks[..., 16:].permute(1, 2, 0)
            return {
                "qs": qs,
                "scale": scale,
                "min": min_value,
            }

        if self.gguf_tensor_qtype == "Q5_K":
            scale_dtype = torch.float32 if wide_dot or self._k_scale_overflows(blocks, torch.float16) else torch.float16
            d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).to(scale_dtype)
            dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1).to(scale_dtype)
            sc, mins = _unpack_q4_k_scale_min_torch(blocks[..., 4:16])
            scale = (d.unsqueeze(-1) * sc.to(scale_dtype)).permute(1, 2, 0).contiguous()
            min_value = (dmin.unsqueeze(-1) * mins.to(scale_dtype)).permute(1, 2, 0).contiguous()
            # Packed code bytes are a strided view into `qweight`; the
            # fused kernels index the block layout through the strides
            # passed to them. Materializing this used to keep a second
            # full copy of every packed weight resident (#4748).
            qh = blocks[..., 16:48].permute(1, 2, 0)
            qs = blocks[..., 48:].permute(1, 2, 0)
            return {
                "qs": qs,
                "qh": qh,
                "scale": scale,
                "min": min_value,
            }

        if self.gguf_tensor_qtype == "Q6_K":
            # Packed code bytes are a strided view into `qweight`; the
            # fused kernels index the block layout through the strides
            # passed to them. Materializing this used to keep a second
            # full copy of every packed weight resident (#4748).
            ql = blocks[..., :128].permute(1, 2, 0)
            qh = blocks[..., 128:192].permute(1, 2, 0)
            subscales = blocks[..., 192:208].contiguous().view(torch.int8).to(torch.float32)
            d = blocks[..., 208:210].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
            scale = d.unsqueeze(-1) * subscales
            # Q6_K codes are signed [-32, 31]. A finite matmul can require
            # weights outside FP16 even when activations are FP16.
            scale_dtype = (
                torch.float32
                if wide_dot or bool((scale.abs() * 32 > torch.finfo(torch.float16).max).any())
                else torch.float16
            )
            scale = scale.to(scale_dtype).permute(1, 2, 0).contiguous()
            return {
                "ql": ql,
                "qh": qh,
                "scale": scale,
            }

        raise NotImplementedError(f"Unsupported GGUF Triton qtype: {self.gguf_tensor_qtype}")

    def _get_triton_cache(self, device: torch.device, wide_dot: bool = False) -> dict[str, Any]:
        """Return this module's published metadata cache, building it once.

        The hit path is the steady state: a dict read, a key compare and a
        retired handshake (#4956). A miss (cold build, or an invalidation that
        changed ``qweight_key``) is serialized per module so a concurrent burst
        cannot replace the entry a peer is already reading (#5008).
        """

        key = self._triton_cache_key(device, wide_dot)
        qweight_key = tensor_cache_key(self.qweight)
        cached = self._gguf_triton_cache.get(key)
        if cached is not None and cached.get("qweight_key") == qweight_key:
            _wait_for_triton_cache_ready(cached, device)
            return cached

        # Cold build, or a genuine invalidation: a weight write, a reload or a
        # device/dtype move changed ``qweight_key``. Serialize it per module so
        # two threads can never both build: the second store used to replace the
        # entry a peer had already picked up, dropping the last reference to that
        # entry's ``scale``/``min`` while a consumer stream was still reading
        # them -- NaN/garbage from the first concurrent forward (#5008).
        with _triton_cache_build_lock(self):
            # Double-check under the lock: a peer may have published while this
            # thread waited, and the first build re-lays out ``qweight`` (#4811),
            # which changes the key it was resolved against above.
            qweight_key = tensor_cache_key(self.qweight)
            cached = self._gguf_triton_cache.get(key)
            if cached is not None and cached.get("qweight_key") == qweight_key:
                _wait_for_triton_cache_ready(cached, device)
                return cached

            cached = self._build_triton_cache(device, wide_dot)
            # The first build re-lays out ``qweight`` (#4811), so key the cache by
            # the tensor that is actually resident now or every forward rebuilds it.
            cached["qweight_key"] = tensor_cache_key(self.qweight)
            _publish_triton_cache_ready(cached, device)
            self._gguf_triton_cache[key] = cached
            return cached

    def _forward_triton(
        self, x_flat: torch.Tensor, output_dtype: torch.dtype | None = None,
        wide_dot: bool = False,
    ) -> torch.Tensor:
        # HIP reuses the "cuda" device namespace, so this admits CUDA and ROCm
        # while still refusing CPU/MPS/XPU placements.
        if x_flat.device.type != "cuda":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CUDA/ROCm inference. "
                "Load GGUF models on CUDA/ROCm or use BACKEND.GGUF_TORCH for the torch fallback."
            )

        if x_flat.shape[-1] != self.padded_in_features:
            x_work = torch.nn.functional.pad(x_flat, (0, self.padded_in_features - x_flat.shape[-1])).contiguous()
        else:
            x_work = x_flat.contiguous()

        cache = self._get_triton_cache(x_work.device, wide_dot=wide_dot)
        output_kwargs = {} if output_dtype is None else {"output_dtype": output_dtype}

        if self.gguf_tensor_qtype in {"Q1_0", "Q1_0_g128"}:
            if cache.get("use_u32"):
                fixed_u32_config = _select_q1_0_g128_u32_fixed_launch_config(
                    capability=_cuda_device_capability(x_work.device),
                    rows=x_work.shape[0],
                    in_features=self.padded_in_features,
                    out_features=cache["scale"].shape[1],
                )
                if fixed_u32_config is not None:
                    if _use_q1_0_g128_k2048_decode_specialization(
                        rows=x_work.shape[0],
                        in_features=self.padded_in_features,
                    ):
                        return _launch_q1_0_g128_u32_k2048_fixed_matmul(
                            x_work,
                            cache["sign_words"],
                            cache["scale"],
                            fixed_config=fixed_u32_config,
                            **output_kwargs,
                        )
                    return _launch_q1_0_g128_u32_fixed_matmul(
                        x_work,
                        cache["sign_words"],
                        cache["scale"],
                        fixed_config=fixed_u32_config,
                        **output_kwargs,
                    )
                return fused_q1_0_g128_matmul(
                    x_work, cache["sign_bytes"], cache["scale"], **output_kwargs,
                )
            fixed_decode_config = cache.get("fixed_decode_config")
            if fixed_decode_config is not None and x_work.shape[0] == 1:
                if _use_q1_0_g128_k2048_decode_specialization(
                    rows=x_work.shape[0],
                    in_features=self.padded_in_features,
                ):
                    return _launch_q1_0_g128_k2048_fixed_matmul(
                        x_work,
                        cache["sign_bytes"],
                        cache["scale"],
                        fixed_config=fixed_decode_config,
                        **output_kwargs,
                    )
                return _launch_q1_0_g128_fixed_matmul(
                    x_work,
                    cache["sign_bytes"],
                    cache["scale"],
                    fixed_config=fixed_decode_config,
                    **output_kwargs,
                )
            if _use_q1_0_g128_k2048_decode_specialization(
                rows=x_work.shape[0],
                in_features=self.padded_in_features,
            ):
                return fused_q1_0_g128_k2048_matmul(
                    x_work, cache["sign_bytes"], cache["scale"], **output_kwargs,
                )
            return fused_q1_0_g128_matmul(
                x_work, cache["sign_bytes"], cache["scale"], **output_kwargs,
            )
        # SM89 (RTX 4090) exact-shape fixed tiles, measured against the
        # autotuned generic kernel.  ``None`` keeps the autotuned path, which
        # stays the fallback for every unmeasured shape and every other GPU.
        capability = _cuda_device_capability(x_work.device)
        if self.gguf_tensor_qtype == "Q4_K":
            fixed = _select_q4_k_fixed_launch_config(
                capability=capability,
                rows=x_work.shape[0],
                in_features=self.padded_in_features,
                out_features=cache["scale"].shape[2],
            )
            if fixed is not None:
                return _launch_q4_k_fixed_matmul(
                    x_work, cache["qs"], cache["scale"], cache["min"],
                    fixed_config=fixed, **output_kwargs,
                )
            return fused_q4_k_matmul(
                x_work, cache["qs"], cache["scale"], cache["min"], **output_kwargs,
            )
        if self.gguf_tensor_qtype == "Q5_K":
            fixed = _select_q5_k_fixed_launch_config(
                capability=capability,
                rows=x_work.shape[0],
                in_features=self.padded_in_features,
                out_features=cache["scale"].shape[2],
            )
            if fixed is not None:
                return _launch_q5_k_fixed_matmul(
                    x_work, cache["qs"], cache["qh"], cache["scale"], cache["min"],
                    fixed_config=fixed, **output_kwargs,
                )
            return fused_q5_k_matmul(
                x_work, cache["qs"], cache["qh"], cache["scale"], cache["min"],
                **output_kwargs,
            )
        if self.gguf_tensor_qtype == "Q6_K":
            fixed = _select_q6_k_fixed_launch_config(
                capability=capability,
                rows=x_work.shape[0],
                in_features=self.padded_in_features,
                out_features=cache["scale"].shape[2],
            )
            if fixed is not None:
                return _launch_q6_k_fixed_matmul(
                    x_work, cache["ql"], cache["qh"], cache["scale"],
                    fixed_config=fixed, **output_kwargs,
                )
            return fused_q6_k_matmul(
                x_work, cache["ql"], cache["qh"], cache["scale"], **output_kwargs,
            )

        raise NotImplementedError(f"Unsupported GGUF Triton qtype: {self.gguf_tensor_qtype}")

    def forward(self, x: torch.Tensor):
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            actual = "scalar" if x.ndim == 0 else x.shape[-1]
            raise ValueError(
                f"{self.__class__.__name__} expected input width {self.in_features}, "
                f"got {actual}."
            )
        if x.device.type != "cuda":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CUDA/ROCm inference. "
                "Load GGUF models on CUDA/ROCm or use BACKEND.GGUF_TORCH for the torch fallback."
            )
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        # The fused Triton matmuls are raw launches with no backward. Use the
        # differentiable GGUF torch path when autograd must flow through the
        # activation or the bias (#4072).
        if torch.is_grad_enabled() and (
            x.requires_grad or (self.bias is not None and self.bias.requires_grad)
        ):
            return super().forward(x)

        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        input_dtype = x_flat.dtype
        wide_dot = input_dtype != torch.float16 and self.bias is not None
        row_scales = None
        if wide_dot:
            # Bias can nearly cancel a large dot. In that case even a small
            # FP16 weight reconstruction error dominates the visible result.
            # Keep both activation and compact scale metadata in FP32.
            x_work = x_flat.float().contiguous()
        elif input_dtype != torch.float16:
            # A plain .half() can turn a finite BF16/FP32 input into inf or
            # zero. The unbiased fused matmul still receives FP16 activations,
            # with a per-row scale restored before output rounding.
            x_dense = x_flat.contiguous()
            x_work = torch.empty(x_dense.shape, device=x_dense.device, dtype=torch.float16)
            row_scales = torch.empty((x_dense.shape[0],), device=x_dense.device, dtype=torch.float32)
            with torch.cuda.device(x_dense.device):
                _gguf_cast_wide_input_kernel[(x_dense.shape[0],)](
                    x_dense, x_work, row_scales, x_dense.shape[-1],
                    triton.next_power_of_2(x_dense.shape[-1]),
                    num_warps=4 if x_dense.shape[-1] <= 4096 else 8,
                )
        else:
            x_work = x_flat

        output = self._forward_triton(
            x_work, output_dtype=torch.float32 if row_scales is not None or self.bias is not None else None,
            wide_dot=wide_dot,
        )

        if row_scales is not None:
            result = torch.empty(output.shape, device=output.device, dtype=input_dtype)
            bias = None if self.bias is None else self.bias.to(device=output.device, dtype=torch.float32).contiguous()
            with torch.cuda.device(output.device):
                _gguf_restore_wide_output_kernel[(triton.cdiv(output.numel(), 256),)](
                    output, row_scales, output if bias is None else bias, result,
                    output.shape[-1], output.numel(), bias is not None, 256,
                )
            output = result
        elif self.bias is not None:
            bias = self.bias.to(device=output.device, dtype=torch.float32)
            output = output + bias

        if input_dtype != output.dtype:
            output = output.to(dtype=input_dtype)

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = [
    "GGUFTritonKernel",
    "_select_gguf_qk_fixed_launch_config",
    "_select_q4_k_fixed_launch_config",
    "_select_q5_k_fixed_launch_config",
    "_select_q6_k_fixed_launch_config",
    "_select_q1_0_g128_fixed_launch_config",
    "_select_q1_0_g128_u32_fixed_launch_config",
    "_select_q1_0_g128_u32_layout",
    "_use_q1_0_g128_k2048_decode_specialization",
    "fused_q1_0_g128_matmul",
    "fused_q1_0_g128_k2048_matmul",
    "fused_q1_0_g128_u32_matmul",
    "fused_q1_0_g128_u32_k2048_matmul",
    "fused_q4_k_matmul",
    "fused_q5_k_matmul",
    "fused_q6_k_matmul",
    "triton_available",
]
