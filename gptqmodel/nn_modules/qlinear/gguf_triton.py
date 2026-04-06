# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.python import has_gil_disabled
from .gguf import GGUFTorchLinear, _unpack_q4_k_scale_min_torch


try:
    import triton
    import triton.language as tl
    from packaging import version

    from ..triton_utils import custom_autotune

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    custom_autotune = None
    _TRITON_AVAILABLE = False

_CUDA_DEVICE_CAPABILITY_CACHE: dict[int, tuple[int, int]] = {}


def triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:
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

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
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

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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

                weight = tl.cast(q, tl.float16) * scale[None, :] - min_value[None, :]
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
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

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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

                weight = tl.cast(q, tl.float16) * scale[None, :] - min_value[None, :]
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
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

    @triton.jit
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

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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

                weight = tl.cast(q, tl.float16) * scale[None, :]
                accumulator += tl.dot(a, weight)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            tl.cast(accumulator, tl.float16),
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
    def grid(meta):
        return (
            triton.cdiv(x.shape[0], meta["BLOCK_SIZE_M"]),
            triton.cdiv(output.shape[1], meta["BLOCK_SIZE_N"]),
        )

    kernel[grid](*args)
    return output


def _launch_with_meta(
    kernel: Callable,
    x: torch.Tensor,
    output: torch.Tensor,
    *args,
    **meta,
) -> torch.Tensor:
    def grid(meta_args):
        return (
            triton.cdiv(x.shape[0], meta_args["BLOCK_SIZE_M"]),
            triton.cdiv(output.shape[1], meta_args["BLOCK_SIZE_N"]),
        )

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


def _cuda_device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q4_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q1_0_g128 fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    output = torch.empty((x.shape[0], scale.shape[1]), device=x.device, dtype=x.dtype)
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


def fused_q5_k_matmul(
    x: torch.Tensor,
    qs: torch.Tensor,
    qh: torch.Tensor,
    scale: torch.Tensor,
    min_value: torch.Tensor,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q5_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=x.dtype)
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
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GGUF Q6_K fused matmul.")

    output = torch.empty((x.shape[0], scale.shape[2]), device=x.device, dtype=x.dtype)
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


class GGUFTritonKernel(GGUFTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_TRITON]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 45}
    SUPPORTS_BITS = [1, 4, 5, 6]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
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
        if self.gguf_tensor_qtype not in {"Q1_0_g128", "Q4_K", "Q5_K", "Q6_K"}:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports fused GGUF Triton formats "
                f"(Q1_0_g128, Q4_K, Q5_K, Q6_K). Actual GGUF qtype: {self.gguf_tensor_qtype}. "
                "Use BACKEND.GGUF_TORCH for unsupported GGUF formats."
            )
        self._gguf_triton_cache: dict[tuple[int, str], dict[str, Any]] = {}

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _TRITON_AVAILABLE:
            return False, ModuleNotFoundError("GGUFTritonKernel requires `triton` to be installed.")

        triton_v = version.parse(triton.__version__)
        if triton_v < version.parse("2.0.0"):
            return False, ImportError(f"triton version must be >= 2.0.0: actual = {triton.__version__}")

        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            return False, Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        return True, None

    def clear_weight_cache(self) -> None:
        self._gguf_triton_cache.clear()
        return super().clear_weight_cache()

    def _triton_cache_key(self, device: torch.device) -> tuple[int, str]:
        return (device.index if device.index is not None else -1, self.gguf_tensor_qtype)

    def _build_triton_cache(self, device: torch.device) -> dict[str, Any]:
        blocks, _, _ = self._reshape_blocks(device=device)

        if self.gguf_tensor_qtype == "Q1_0_g128":
            scale = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).permute(1, 0).contiguous()
            capability = _cuda_device_capability(device)
            if _select_q1_0_g128_u32_layout(
                capability=capability,
                in_features=self.padded_in_features,
                out_features=scale.shape[1],
            ):
                sign_bytes = blocks[..., 2:].permute(1, 2, 0).contiguous()
                sign_words_src = sign_bytes.to(torch.int32)
                sign_words = (
                    sign_words_src[:, 0::4, :]
                    | torch.bitwise_left_shift(sign_words_src[:, 1::4, :], 8)
                    | torch.bitwise_left_shift(sign_words_src[:, 2::4, :], 16)
                    | torch.bitwise_left_shift(sign_words_src[:, 3::4, :], 24)
                ).contiguous()
                return {
                    "fixed_decode_config": None,
                    "qweight_ptr": self.qweight.data_ptr(),
                    "sign_bytes": sign_bytes,
                    "sign_words": sign_words,
                    "scale": scale,
                    "use_u32": True,
                }

            sign_bytes = blocks[..., 2:].permute(1, 2, 0).contiguous()
            return {
                "fixed_decode_config": _select_q1_0_g128_fixed_launch_config(
                    capability=capability,
                    rows=1,
                    in_features=self.padded_in_features,
                    cols=scale.shape[1],
                ),
                "qweight_ptr": self.qweight.data_ptr(),
                "sign_bytes": sign_bytes,
                "scale": scale,
                "use_u32": False,
            }

        if self.gguf_tensor_qtype == "Q4_K":
            d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
            dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1)
            sc, mins = _unpack_q4_k_scale_min_torch(blocks[..., 4:16])
            scale = (d.unsqueeze(-1) * sc.to(torch.float16)).permute(1, 2, 0).contiguous()
            min_value = (dmin.unsqueeze(-1) * mins.to(torch.float16)).permute(1, 2, 0).contiguous()
            qs = blocks[..., 16:].permute(1, 2, 0).contiguous()
            return {
                "qweight_ptr": self.qweight.data_ptr(),
                "qs": qs,
                "scale": scale,
                "min": min_value,
            }

        if self.gguf_tensor_qtype == "Q5_K":
            d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1)
            dmin = blocks[..., 2:4].contiguous().view(torch.float16).squeeze(-1)
            sc, mins = _unpack_q4_k_scale_min_torch(blocks[..., 4:16])
            scale = (d.unsqueeze(-1) * sc.to(torch.float16)).permute(1, 2, 0).contiguous()
            min_value = (dmin.unsqueeze(-1) * mins.to(torch.float16)).permute(1, 2, 0).contiguous()
            qh = blocks[..., 16:48].permute(1, 2, 0).contiguous()
            qs = blocks[..., 48:].permute(1, 2, 0).contiguous()
            return {
                "qweight_ptr": self.qweight.data_ptr(),
                "qs": qs,
                "qh": qh,
                "scale": scale,
                "min": min_value,
            }

        if self.gguf_tensor_qtype == "Q6_K":
            ql = blocks[..., :128].permute(1, 2, 0).contiguous()
            qh = blocks[..., 128:192].permute(1, 2, 0).contiguous()
            scale = blocks[..., 192:208].contiguous().view(torch.int8).to(torch.float16)
            d = blocks[..., 208:210].contiguous().view(torch.float16).squeeze(-1)
            scale = (d.unsqueeze(-1) * scale).permute(1, 2, 0).contiguous()
            return {
                "qweight_ptr": self.qweight.data_ptr(),
                "ql": ql,
                "qh": qh,
                "scale": scale,
            }

        raise NotImplementedError(f"Unsupported GGUF Triton qtype: {self.gguf_tensor_qtype}")

    def _get_triton_cache(self, device: torch.device) -> dict[str, Any]:
        key = self._triton_cache_key(device)
        cached = self._gguf_triton_cache.get(key)
        if cached is not None and cached.get("qweight_ptr") == self.qweight.data_ptr():
            return cached

        cached = self._build_triton_cache(device)
        self._gguf_triton_cache[key] = cached
        return cached

    def _forward_triton(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.device.type != "cuda":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CUDA inference. "
                "Load GGUF models on CUDA or use BACKEND.GGUF_TORCH for the torch fallback."
            )

        if x_flat.shape[-1] != self.padded_in_features:
            x_work = torch.nn.functional.pad(x_flat, (0, self.padded_in_features - x_flat.shape[-1])).contiguous()
        else:
            x_work = x_flat.contiguous()

        cache = self._get_triton_cache(x_work.device)

        if self.gguf_tensor_qtype == "Q1_0_g128":
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
                        )
                    return _launch_q1_0_g128_u32_fixed_matmul(
                        x_work,
                        cache["sign_words"],
                        cache["scale"],
                        fixed_config=fixed_u32_config,
                    )
                return fused_q1_0_g128_matmul(x_work, cache["sign_bytes"], cache["scale"])
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
                    )
                return _launch_q1_0_g128_fixed_matmul(
                    x_work,
                    cache["sign_bytes"],
                    cache["scale"],
                    fixed_config=fixed_decode_config,
                )
            if _use_q1_0_g128_k2048_decode_specialization(
                rows=x_work.shape[0],
                in_features=self.padded_in_features,
            ):
                return fused_q1_0_g128_k2048_matmul(x_work, cache["sign_bytes"], cache["scale"])
            return fused_q1_0_g128_matmul(x_work, cache["sign_bytes"], cache["scale"])
        if self.gguf_tensor_qtype == "Q4_K":
            return fused_q4_k_matmul(x_work, cache["qs"], cache["scale"], cache["min"])
        if self.gguf_tensor_qtype == "Q5_K":
            return fused_q5_k_matmul(x_work, cache["qs"], cache["qh"], cache["scale"], cache["min"])
        if self.gguf_tensor_qtype == "Q6_K":
            return fused_q6_k_matmul(x_work, cache["ql"], cache["qh"], cache["scale"])

        raise NotImplementedError(f"Unsupported GGUF Triton qtype: {self.gguf_tensor_qtype}")

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])

        input_dtype = x_flat.dtype
        if input_dtype != torch.float16:
            x_work = x_flat.to(dtype=torch.float16)
        else:
            x_work = x_flat

        output = self._forward_triton(x_work)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if input_dtype != output.dtype:
            output = output.to(dtype=input_dtype)

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = [
    "GGUFTritonKernel",
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
