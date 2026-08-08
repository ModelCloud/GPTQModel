# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton fused GPTQ block kernel.

One kernel launch processes a full 128-column GPTQ block for all output rows.
It performs the serial column loop inside the kernel, fusing the
quantize / diff / error / trailing weight-update steps that the eager path
issues as many separate small kernels.
"""

from __future__ import annotations

import os
import threading

import torch

from ..utils.logger import setup_logger

log = setup_logger()

# Triton's Python JIT launcher mutates per-function device/binder/kernel caches.
# CPython's GIL normally serializes that state, but Python 3.14t allows the
# ThreadX CUDA workers to enter it concurrently. Keep the host-side cache/launch
# operation atomic; kernels remain asynchronous and continue to overlap on the
# GPUs after the lock is released.
_TRITON_LAUNCH_LOCK = threading.Lock()
_CUDA_CRASH_PROBE = os.environ.get("GPTQMODEL_CUDA_CRASH_PROBE", "0") != "0"


def _triton_available() -> bool:
    try:
        import triton

        return triton is not None and torch.cuda.is_available()
    except Exception:
        return False


if _triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _round_half_to_even(x):
        """Round-to-nearest-even for float32; matches torch.round."""
        s = tl.where(x >= 0.0, 1.0, -1.0)
        ax = x * s
        y = tl.floor(ax)
        frac = ax - y
        is_odd = (tl.cast(y, tl.int32) & 1) != 0
        inc = (frac > 0.5) | ((frac == 0.5) & is_odd)
        y = y + tl.where(inc, 1.0, 0.0)
        return s * y

    @triton.jit
    def _gptq_block_kernel(
        w_ptr,
        q_ptr,
        err_ptr,
        hinv_ptr,
        scale_ptr,
        zero_ptr,
        rows,
        count,
        group_size,
        maxq: tl.float32,
        groupwise: tl.int32,
        stride_w0,
        stride_w1,
        stride_q0,
        stride_q1,
        stride_e0,
        stride_e1,
        stride_h0,
        stride_h1,
        stride_s0,
        stride_s1,
        stride_z0,
        stride_z1,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid
        if row >= rows:
            return

        # Column indices within the block.
        offs = tl.arange(0, BLOCK_SIZE)
        col_mask = offs < count

        # Load the full working row.
        w_row = tl.load(
            w_ptr + row * stride_w0 + offs * stride_w1,
            mask=col_mask,
            other=0.0,
        )

        for i in range(count):
            # Extract the i-th column from the register-held row. Triton does
            # not support dynamic scalar indexing of block tensors, so use a
            # masked sum over the row.
            w_val = tl.sum(tl.where(offs == i, w_row, 0.0))

            # Per-group scale and zero (float32, from Quantizer.find_params).
            g = i // group_size
            scale_f32 = tl.load(scale_ptr + row * stride_s0 + g * stride_s1)
            zero_f32 = tl.load(zero_ptr + row * stride_z0 + g * stride_z1)

            # RTN quantize matching eager quantizer.quantize() for the default
            # (non-groupwise) path. Use tl.div_rn to match torch.div rounding.
            if groupwise != 0:
                ratio = tl.div_rn(w_val, scale_f32)
                q_int = _round_half_to_even(ratio)
                q_int = tl.clamp(q_int, -maxq, maxq)
                q_val = scale_f32 * q_int
            else:
                ratio = tl.div_rn(w_val, scale_f32)
                q_int = _round_half_to_even(ratio) + zero_f32
                q_int = tl.clamp(q_int, 0.0, maxq)
                q_val = scale_f32 * (q_int - zero_f32)

            # Quantization error for this column.
            d = tl.load(hinv_ptr + i * stride_h0 + i * stride_h1)
            err = tl.div_rn(w_val - q_val, d)

            # Store outputs for this column.
            tl.store(q_ptr + row * stride_q0 + i * stride_q1, q_val)
            tl.store(err_ptr + row * stride_e0 + i * stride_e1, err)

            # Load the i-th row of Hinv (only j >= i needed).
            hinv_offs = tl.arange(0, BLOCK_SIZE)
            hinv_mask = (hinv_offs >= i) & (hinv_offs < count)
            hinv_row = tl.load(
                hinv_ptr + i * stride_h0 + hinv_offs * stride_h1,
                mask=hinv_mask,
                other=0.0,
            )

            # Apply the outer-product update to the remainder of the row.
            w_row = w_row - err * hinv_row

        # The updated working row is only needed while processing this block;
        # the caller does not use W1 after the kernel, so skip the final writeback.

    def gptq_block_triton(
        W1: torch.Tensor,
        Q1: torch.Tensor,
        Err1: torch.Tensor,
        Hinv1: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        maxq: int,
        group_size: int,
        groupwise: bool = False,
    ) -> None:
        """Fused GPTQ block step for one contiguous column block.

        Args:
            W1: working weights, shape (rows, count), float32, in-place updated.
            Q1: quantized weights, shape (rows, count), float32.
            Err1: per-column errors, shape (rows, count), float32.
            Hinv1: inverse Hessian block, shape (count, count), float32.
            scale: per-row per-group scales, shape (rows, count // group_size).
            zero: per-row per-group zeros, shape (rows, count // group_size).
            maxq: integer clipping bound (e.g. 15 for 4-bit).
            group_size: columns per group (must divide count).
            groupwise: use groupwise symmetric formula (scale * round(w / scale)).
        """
        rows, count = W1.shape
        if count > 128:
            raise ValueError(f"Triton block kernel supports count <= 128, got {count}")
        if count % group_size != 0:
            raise ValueError(f"group_size {group_size} must divide count {count}")
        if W1.dtype != torch.float32 or Q1.dtype != torch.float32 or Err1.dtype != torch.float32:
            raise TypeError("Triton GPTQ block kernel expects float32 W1/Q1/Err1")

        Hinv1 = Hinv1.contiguous()
        scale = scale.contiguous()
        zero = zero.contiguous()

        operands = (W1, Q1, Err1, Hinv1, scale, zero)
        devices = {tensor.device for tensor in operands}
        if len(devices) != 1:
            raise ValueError(f"Triton GPTQ block tensors must share one device, got {sorted(map(str, devices))}")
        if not W1.is_cuda:
            raise ValueError(f"Triton GPTQ block tensors must be CUDA tensors, got {W1.device}")
        if Hinv1.dtype != torch.float32 or scale.dtype != torch.float32 or zero.dtype != torch.float32:
            raise TypeError("Triton GPTQ block kernel expects float32 Hinv1/scale/zero")
        expected_groups = count // group_size
        if Hinv1.shape != (count, count):
            raise ValueError(f"Hinv1 must have shape {(count, count)}, got {tuple(Hinv1.shape)}")
        if scale.shape != (rows, expected_groups) or zero.shape != (rows, expected_groups):
            raise ValueError(
                f"scale/zero must have shape {(rows, expected_groups)}, got "
                f"{tuple(scale.shape)}/{tuple(zero.shape)}"
            )

        BLOCK_SIZE = 128
        grid = (rows,)
        stream = torch.cuda.current_stream(W1.device)
        with _TRITON_LAUNCH_LOCK:
            _gptq_block_kernel[grid](
                W1,
                Q1,
                Err1,
                Hinv1,
                scale,
                zero,
                rows,
                count,
                group_size,
                float(maxq),
                int(groupwise),
                W1.stride(0),
                W1.stride(1),
                Q1.stride(0),
                Q1.stride(1),
                Err1.stride(0),
                Err1.stride(1),
                Hinv1.stride(0),
                Hinv1.stride(1),
                scale.stride(0),
                scale.stride(1),
                zero.stride(0),
                zero.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
        # Triton launches asynchronously outside the PyTorch dispatcher. Some
        # operands above may be local contiguous copies, and worker tasks can
        # release their final Python reference immediately after returning.
        # Tell the caching allocator which stream still owns every operand so
        # another free-threaded quantization worker cannot recycle the storage
        # while the kernel is running.
        for tensor in operands:
            tensor.record_stream(stream)
        # Keep crash localization opt-in: a device synchronization here is far
        # too expensive for normal quantization, but it attributes asynchronous
        # CUDA faults to this launch instead of an unrelated later torch call.
        if _CUDA_CRASH_PROBE:
            torch.cuda.synchronize(W1.device)

else:
    gptq_block_triton = None
