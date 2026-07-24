# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton fused activation scale-search kernel for Quantizer.find_params_batched.

This kernel is a drop-in fast path for the activation-method scale search.  It
loops over shrink candidates inside a single Triton kernel, computing the
weighted MSE for each (row, group) tile and returning the best scale/zero per
tile.  The kernel replicates the arithmetic of the eager PyTorch path,
including round-half-to-even and the same clamp/dequant sequence, so its
outputs should match the reference scale/zero exactly for the tested group
sizes.
"""

from typing import Tuple

import torch

from ..utils.logger import setup_logger

log = setup_logger()


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
    def _scale_search_activation_kernel(
        x_ptr,
        xmin_ptr,
        xmax_ptr,
        importance_ptr,
        scale_out_ptr,
        zero_out_ptr,
        rows,
        num_groups,
        group_size,
        maxq: tl.float32,
        grid: tl.int32,
        candidate_count: tl.int32,
        sym: tl.int32,
        x_stride_r: tl.int32,
        x_stride_g: tl.int32,
        min_stride_r: tl.int32,
        imp_stride_g: tl.int32,
        out_stride_r: tl.int32,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_blocks = tl.cdiv(rows, BLOCK_ROW)
        g = pid // row_blocks
        rb = pid % row_blocks
        row_start = rb * BLOCK_ROW

        row_idx = row_start + tl.arange(0, BLOCK_ROW)
        col_idx = tl.arange(0, BLOCK_COL)
        row_mask = row_idx < rows
        col_mask = col_idx < group_size
        mask_2d = row_mask[:, None] & col_mask[None, :]

        # x[rows, num_groups, group_size]
        x = tl.load(
            x_ptr + row_idx[:, None] * x_stride_r + g * x_stride_g + col_idx[None, :],
            mask=mask_2d,
            other=0.0,
        )
        x = tl.cast(x, tl.float32)

        # importance[num_groups, group_size]
        importance = tl.load(
            importance_ptr + g * imp_stride_g + col_idx,
            mask=col_mask,
            other=0.0,
        )

        # xmin/xmax[rows, num_groups]
        min_offsets = row_idx * min_stride_r + g
        xmin = tl.load(xmin_ptr + min_offsets, mask=row_mask, other=-1.0)
        xmax = tl.load(xmax_ptr + min_offsets, mask=row_mask, other=1.0)

        best_loss = tl.full((BLOCK_ROW,), value=1e30, dtype=tl.float32)
        best_scale = tl.zeros((BLOCK_ROW,), dtype=tl.float32)
        best_zero = tl.zeros((BLOCK_ROW,), dtype=tl.float32)

        grid_f = tl.cast(grid, tl.float32)
        maxq_f = maxq
        sym_b = sym != 0
        const_zero_sym = (maxq_f + 1.0) / 2.0

        for c in tl.range(0, candidate_count):
            p = 1.0 - tl.cast(c, tl.float32) / grid_f
            xmin_p = p * xmin
            xmax_p = p * xmax
            scale_c = (xmax_p - xmin_p) / maxq_f

            zero_c = tl.where(
                sym_b,
                const_zero_sym,
                _round_half_to_even(-xmin_p / scale_c),
            )

            q_raw = _round_half_to_even(x / scale_c[:, None])
            q_int = q_raw + zero_c[:, None]
            q_int = tl.clamp(q_int, 0.0, maxq_f)
            dequant = (q_int - zero_c[:, None]) * scale_c[:, None]
            error = dequant - x
            weighted_sq = error * error * importance
            weighted_sq = tl.where(mask_2d, weighted_sq, 0.0)
            loss = tl.sum(weighted_sq, axis=1)

            is_better = loss < best_loss
            best_loss = tl.where(is_better, loss, best_loss)
            best_scale = tl.where(is_better, scale_c, best_scale)
            best_zero = tl.where(is_better, zero_c, best_zero)

        out_offsets = row_idx * out_stride_r + g
        tl.store(scale_out_ptr + out_offsets, best_scale, mask=row_mask)
        tl.store(zero_out_ptr + out_offsets, best_zero, mask=row_mask)

    def _triton_find_params_batched_activation(
        x: torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
        importance: torch.Tensor,
        grid: int,
        maxshrink: float,
        maxq: float,
        sym: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, num_groups, group_size = x.shape
        device = x.device
        scale_out = torch.empty((rows, num_groups), dtype=torch.float32, device=device)
        zero_out = torch.empty((rows, num_groups), dtype=torch.float32, device=device)

        candidate_count = int(maxshrink * grid)
        if candidate_count <= 0:
            # Degenerate search; fall back to the initial scale/zero.
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale) if not sym else torch.full_like(scale, (maxq + 1.0) / 2.0)
            return scale, zero

        # Trion kernel expects row-major tensors and group_size <= BLOCK_COL.
        BLOCK_ROW = 8
        BLOCK_COL = 128
        row_blocks = (rows + BLOCK_ROW - 1) // BLOCK_ROW
        total_programs = num_groups * row_blocks
        if total_programs == 0:
            return scale_out, zero_out

        _scale_search_activation_kernel[(total_programs,)](
            x,
            xmin,
            xmax,
            importance,
            scale_out,
            zero_out,
            rows,
            num_groups,
            group_size,
            float(maxq),
            int(grid),
            int(candidate_count),
            int(sym),
            x.stride(0),
            x.stride(1),
            xmin.stride(0),
            importance.stride(0),
            scale_out.stride(0),
            BLOCK_ROW=BLOCK_ROW,
            BLOCK_COL=BLOCK_COL,
        )
        return scale_out, zero_out

else:
    _triton_find_params_batched_activation = None
