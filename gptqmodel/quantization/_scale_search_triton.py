# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton fused activation scale-search kernel for Quantizer.find_params_batched.

This kernel is a fast path for the activation-method scale search. It loops over
shrink candidates inside a single Triton kernel, computing the weighted MSE for
each (row, group, candidate) and returns the full loss tensor. The Python wrapper
then selects the top-k candidates from the Triton loss, recomputes their losses
with the exact PyTorch arithmetic used by the eager reference, and picks the
first minimizer. This keeps the speed benefit of device-side candidate iteration
while guaranteeing bit-exact agreement with the existing Python scale search.
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
        loss_out_ptr,
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
        loss_stride_r: tl.int32,
        loss_stride_g: tl.int32,
        loss_stride_c: tl.int32,
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

        x = tl.load(
            x_ptr + row_idx[:, None] * x_stride_r + g * x_stride_g + col_idx[None, :],
            mask=mask_2d,
            other=0.0,
        )
        x = tl.cast(x, tl.float32)

        importance = tl.load(
            importance_ptr + g * imp_stride_g + col_idx,
            mask=col_mask,
            other=0.0,
        )

        min_offsets = row_idx * min_stride_r + g
        xmin = tl.load(xmin_ptr + min_offsets, mask=row_mask, other=-1.0)
        xmax = tl.load(xmax_ptr + min_offsets, mask=row_mask, other=1.0)

        grid_f = tl.cast(grid, tl.float32)
        maxq_f = maxq
        sym_b = sym != 0
        const_zero_sym = (maxq_f + 1.0) / 2.0

        # loss base for this (row block, group) over all candidates
        loss_base = loss_out_ptr + g * loss_stride_g + row_idx * loss_stride_r
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

            tl.store(loss_base + c * loss_stride_c, loss, mask=row_mask)

    def _triton_find_params_batched_activation(
        x: torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
        importance: torch.Tensor,
        scale_all: torch.Tensor,
        zero_all: torch.Tensor,
        grid: int,
        maxshrink: float,
        maxq: float,
        sym: bool,
        candidate_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, num_groups, group_size = x.shape
        device = x.device

        if candidate_count <= 0:
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale) if not sym else torch.full_like(scale, (maxq + 1.0) / 2.0)
            return scale, zero

        BLOCK_ROW = 8
        BLOCK_COL = 128
        row_blocks = (rows + BLOCK_ROW - 1) // BLOCK_ROW
        total_programs = num_groups * row_blocks
        if total_programs == 0:
            return (
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
            )

        loss_out = torch.empty((rows, num_groups, candidate_count), dtype=torch.float32, device=device)

        _scale_search_activation_kernel[(total_programs,)](
            x,
            xmin,
            xmax,
            importance,
            loss_out,
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
            loss_out.stride(0),
            loss_out.stride(1),
            loss_out.stride(2),
            BLOCK_ROW=BLOCK_ROW,
            BLOCK_COL=BLOCK_COL,
        )

        # The approximate Triton loss is only used to narrow the search to a
        # small top-k. We then recompute the exact PyTorch loss for those
        # candidates and select the first minimizer, matching the eager path's
        # torch.min tie-breaking behaviour. Sorting the top-k indices ascending
        # before recomputing ensures ties resolve to the smallest candidate
        # index, just like the reference.
        k = min(candidate_count, 20)
        topk_values, topk_indices = loss_out.topk(k, dim=-1, largest=False, sorted=False)
        topk_indices, sort_order = topk_indices.sort(dim=-1)
        topk_values = topk_values.gather(2, sort_order)

        # scale_all/zero_all are already in [candidate_count, rows, num_groups]
        # from the eager precompute, so scale_perm matches the reference values.
        scale_perm = scale_all.permute(1, 2, 0)  # [rows, num_groups, candidate_count]
        zero_perm = zero_all.permute(1, 2, 0)
        scale_k = scale_perm.gather(2, topk_indices)
        zero_k = zero_perm.gather(2, topk_indices)

        # Recompute exact loss for the top-k candidates in one vectorized batch.
        # x: [rows, num_groups, group_size]
        # scale_k/zero_k: [rows, num_groups, k]
        # importance: [num_groups, group_size]
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1)  # [rows, num_groups, k, group_size]
        scale_k_exp = scale_k.unsqueeze(-1)
        zero_k_exp = zero_k.unsqueeze(-1)
        q = torch.clamp(torch.round(x_exp / scale_k_exp) + zero_k_exp, 0.0, maxq)
        dequant = (q - zero_k_exp) * scale_k_exp
        error = dequant - x_exp
        importance_exp = importance.unsqueeze(0).unsqueeze(2)  # [1, num_groups, 1, group_size]
        losses_k = (error * error * importance_exp).sum(dim=-1)  # [rows, num_groups, k]

        best_local = losses_k.argmin(dim=-1)
        best_idx = topk_indices.gather(2, best_local.unsqueeze(-1)).squeeze(-1)

        scale_out = scale_perm.gather(2, best_idx.unsqueeze(-1)).squeeze(-1)
        zero_out = zero_perm.gather(2, best_idx.unsqueeze(-1)).squeeze(-1)
        return scale_out, zero_out

else:
    _triton_find_params_batched_activation = None
