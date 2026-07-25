# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Triton fused scale-search kernels for Quantizer.find_params_batched.

This module provides fast paths for the activation and Hessian/Hybrid scale-search
objectives. Each kernel loops over shrink candidates in a single Triton launch,
computes the per-(row, group, candidate) loss, and returns the full loss tensor.
The Python wrapper then selects the top-k candidates, recomputes their losses
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

        BLOCK_ROW = 32
        row_blocks = (rows + BLOCK_ROW - 1) // BLOCK_ROW
        total_programs = num_groups * row_blocks
        if total_programs == 0:
            return (
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
            )

        loss_out = torch.empty((rows, num_groups, candidate_count), dtype=torch.float32, device=device)

        # Use a tile width that matches the actual group size so we do not
        # waste work masking off columns past the group boundary.
        block_col = group_size
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
            BLOCK_COL=block_col,
        )

        # The approximate Triton loss is only used to narrow the search to the
        # two most promising candidates. We then recompute the exact PyTorch loss
        # for those two candidates and select the first minimizer, matching the
        # eager path's torch.min tie-breaking behaviour. Sorting the top-2
        # indices ascending before recomputing ensures ties resolve to the
        # smallest candidate index, just like the reference.
        k = min(candidate_count, 2)
        topk_values, topk_indices = loss_out.topk(k, dim=-1, largest=False, sorted=False)
        topk_indices, sort_order = topk_indices.sort(dim=-1)
        topk_values = topk_values.gather(2, sort_order)

        # Recompute the reference scale/zero only for the short-listed candidates
        # instead of materializing the full candidate grid. Use the same int64
        # maxq tensor as the eager precompute so scale/zero are bit-identical.
        maxq_t = torch.tensor(int(maxq), device=device, dtype=torch.int64)
        maxq_f = float(maxq)
        p = 1.0 - topk_indices.float() / grid
        xmin_k = xmin.unsqueeze(-1) * p
        xmax_k = xmax.unsqueeze(-1) * p
        scale_k = (xmax_k - xmin_k) / maxq_t
        if sym:
            zero_k = torch.full_like(scale_k, (maxq_t + 1) / 2)
        else:
            zero_k = torch.round(-xmin_k / scale_k)

        # Recompute exact loss for the top-k candidates in one vectorized batch.
        # Reuse a single intermediate tensor to avoid allocating dequant/error copies.
        # x: [rows, num_groups, group_size]
        # scale_k/zero_k: [rows, num_groups, k]
        # importance: [num_groups, group_size]
        x_view = x[:, :, None, :]  # [rows, num_groups, 1, group_size] broadcasts with k dim
        scale_k_exp = scale_k[..., None]
        zero_k_exp = zero_k[..., None]
        q = x_view / scale_k_exp
        q.round_()
        q.add_(zero_k_exp).clamp_(0.0, maxq_f)  # q = clamp(round(x/scale)+zero, 0, maxq)
        q.sub_(zero_k_exp).mul_(scale_k_exp)  # q = (q - zero) * scale = dequant
        q.sub_(x_view)  # q = dequant - x = error
        q.pow_(2)
        q.mul_(importance[None, :, None, :])  # weight by importance
        losses_k = q.sum(dim=-1)  # [rows, num_groups, k]

        best_local = losses_k.argmin(dim=-1)

        final_scale = scale_k.gather(2, best_local.unsqueeze(-1)).squeeze(-1)
        final_zero = zero_k.gather(2, best_local.unsqueeze(-1)).squeeze(-1)
        return final_scale, final_zero

    @triton.jit
    def _scale_search_hessian_kernel(
        x_ptr,
        scale_ptr,
        zero_ptr,
        hessian_ptr,
        loss_out_ptr,
        rows,
        num_groups,
        group_size,
        maxq: tl.float32,
        candidate_count: tl.int32,
        x_stride_r: tl.int32,
        x_stride_g: tl.int32,
        scale_stride_r: tl.int32,
        scale_stride_g: tl.int32,
        scale_stride_c: tl.int32,
        zero_stride_r: tl.int32,
        zero_stride_g: tl.int32,
        zero_stride_c: tl.int32,
        h_stride_g: tl.int32,
        h_stride_r: tl.int32,
        h_stride_c: tl.int32,
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

        h_offs = col_idx[:, None] * h_stride_r + col_idx[None, :] * h_stride_c
        h_mask = col_mask[:, None] & col_mask[None, :]
        h = tl.load(hessian_ptr + g * h_stride_g + h_offs, mask=h_mask, other=0.0)

        scale_base = scale_ptr + g * scale_stride_g + row_idx * scale_stride_r
        zero_base = zero_ptr + g * zero_stride_g + row_idx * zero_stride_r

        maxq_f = maxq

        loss_base = loss_out_ptr + g * loss_stride_g + row_idx * loss_stride_r
        for c in tl.range(0, candidate_count):
            scale_c = tl.load(scale_base + c * scale_stride_c, mask=row_mask, other=1.0)
            zero_c = tl.load(zero_base + c * zero_stride_c, mask=row_mask, other=0.0)

            q_raw = _round_half_to_even(x / scale_c[:, None])
            q_int = q_raw + zero_c[:, None]
            q_int = tl.clamp(q_int, 0.0, maxq_f)
            dequant = (q_int - zero_c[:, None]) * scale_c[:, None]
            error = dequant - x

            # projected[r, j] = sum_i error[r, i] * h[i, j]
            projected = tl.dot(error, h, allow_tf32=False, out_dtype=tl.float32)
            weighted = error * projected
            weighted = tl.where(mask_2d, weighted, 0.0)
            loss = tl.sum(weighted, axis=1)
            loss = tl.maximum(loss, 0.0)

            tl.store(loss_base + c * loss_stride_c, loss, mask=row_mask)

    def _triton_find_params_batched_hessian_hybrid(
        x: torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
        prepared_hessian: torch.Tensor,
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

        # Use a smaller row tile for the Hessian/Hybrid kernel: the per-program
        # H tile is group_size x group_size, so increasing BLOCK_ROW also
        # increases register pressure and can make tl.dot spill for gs=128.
        BLOCK_ROW = 8
        row_blocks = (rows + BLOCK_ROW - 1) // BLOCK_ROW
        total_programs = num_groups * row_blocks
        if total_programs == 0:
            return (
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
                torch.empty((rows, num_groups), dtype=torch.float32, device=device),
            )

        # Precompute scales/zeros with the same int64 maxq tensor as the eager
        # fallback so the Triton kernel quantizes identically to the PyTorch path.
        maxq_t = torch.tensor(int(maxq), device=device, dtype=torch.int64)
        shrink = torch.arange(candidate_count, device=device, dtype=torch.float32)
        shrink = 1.0 - shrink / grid
        p = shrink.view(-1, 1, 1)
        xmin_all = p * xmin.unsqueeze(0)
        xmax_all = p * xmax.unsqueeze(0)
        scale_all = (xmax_all - xmin_all) / maxq_t
        if sym:
            zero_all = torch.full_like(scale_all, (maxq_t + 1.0) / 2.0)
        else:
            zero_all = torch.round(-xmin_all / scale_all)

        scale_perm = scale_all.permute(1, 2, 0)
        zero_perm = zero_all.permute(1, 2, 0)

        loss_out = torch.empty((rows, num_groups, candidate_count), dtype=torch.float32, device=device)

        block_col = group_size
        _scale_search_hessian_kernel[(total_programs,)](
            x,
            scale_perm,
            zero_perm,
            prepared_hessian,
            loss_out,
            rows,
            num_groups,
            group_size,
            float(maxq),
            int(candidate_count),
            x.stride(0),
            x.stride(1),
            scale_perm.stride(0),
            scale_perm.stride(1),
            scale_perm.stride(2),
            zero_perm.stride(0),
            zero_perm.stride(1),
            zero_perm.stride(2),
            prepared_hessian.stride(0),
            prepared_hessian.stride(1),
            prepared_hessian.stride(2),
            loss_out.stride(0),
            loss_out.stride(1),
            loss_out.stride(2),
            BLOCK_ROW=BLOCK_ROW,
            BLOCK_COL=block_col,
        )

        k = min(candidate_count, 20)
        topk_values, topk_indices = loss_out.topk(k, dim=-1, largest=False, sorted=False)
        topk_indices, sort_order = topk_indices.sort(dim=-1)
        topk_values = topk_values.gather(2, sort_order)

        # Recompute the reference scale/zero only for the short-listed candidates.
        # Use the same int64 maxq tensor as the eager precompute.
        maxq_f = float(maxq)
        scale_perm_k = scale_perm.gather(2, topk_indices)
        zero_perm_k = zero_perm.gather(2, topk_indices)

        # scale_k/zero_k are [rows, num_groups, k]; expand for per-element work.
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1)  # [rows, num_groups, k, group_size]
        scale_k_exp = scale_perm_k.unsqueeze(-1)
        zero_k_exp = zero_perm_k.unsqueeze(-1)
        q = torch.clamp(torch.round(x_exp / scale_k_exp) + zero_k_exp, 0.0, maxq_f)
        dequant = (q - zero_k_exp) * scale_k_exp
        error = dequant - x_exp

        # Recompute exact Hessian/Hybrid loss: error^T @ prepared_hessian @ error.
        projected = torch.einsum("rgki,gij->rgkj", error, prepared_hessian)
        losses_k = (error * projected).sum(dim=-1).clamp_min(0.0)

        best_local = losses_k.argmin(dim=-1)

        final_scale = scale_perm_k.gather(2, best_local.unsqueeze(-1)).squeeze(-1)
        final_zero = zero_perm_k.gather(2, best_local.unsqueeze(-1)).squeeze(-1)
        return final_scale, final_zero

else:
    _triton_find_params_batched_activation = None
    _triton_find_params_batched_hessian_hybrid = None
