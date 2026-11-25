# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""G2PTQ: GPTQ with generalized (exact) gradient compensation.

Clean-room port of "Improving LLM Post-Training Quantization with Generalized
Gradient Compensation" (Liu et al., arXiv:2609.31009).

The solver keeps GPTQ's column-wise second-order compensation and adds the
first-order term the paper derives from a block-wise objective. For one linear
layer the update is

    dW = dW_gptq + beta * dW_grad
    dW_gptq[:, t:] = ((w_hat_t - w_t) / Hinv[t, t]) * Hinv[t, :]
    dW_grad[:, t:] = ((g Hinv e_t) / Hinv[t, t]) * Hinv[t, :] - g Hinv

with ``Hinv = H^-1`` and a per-row trust-region factor ``beta`` in [0, 1]
(Theorem 3). The recursion is evaluated with the Cholesky factor
``Hinv = U^T U`` in ``O(max{C^3, R C^2})``, matching GPTQ, and it keeps GPTQ's
lazy-batch update so the remaining columns move once per block.

The paper's ``L`` satisfies ``Hinv = L L^T``. This port stores the upper
factor ``U = L^T``, so ``M = (G @ U^T) * beta``, ``F = M @ U`` and
``E_t = (W_t - Q_t - F_t) / U[t, t]``.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..utils.fallback import should_use_fallback
from ..utils.logger import setup_logger
from ..utils.p2p import safe_move_to
from ..utils.torch import torch_sync
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .gptq import GPTQ, _move_static_groups, _static_group_of_column


log = setup_logger()


def trust_region_beta(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    gradient: torch.Tensor,
    fake_quantized: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return the row-wise trust-region factor from Eq. 13/14 and Eq. 56.

    Args:
        weight: ``[groups, rows, columns]`` working weights.
        inverse_hessian: ``[groups, columns, columns]`` dense ``H^-1``.
        gradient: ``[groups, rows, columns]`` weight gradient ``G``.
        fake_quantized: ``[groups, rows, columns]`` fake-quantized ``W``.
        alpha: Trust-region budget; ``alpha == 0`` disables the gradient step.

    Returns:
        ``[groups, rows, 1]`` factors in ``[0, 1]``. A zero budget, zero
        gradient or zero diagonal yields zeros rather than NaNs.
    """

    if weight.shape != gradient.shape or weight.shape != fake_quantized.shape:
        raise ValueError("G2PTQ trust region: weight, gradient and fake-quantized shapes must match.")
    if weight.ndim != 3:
        raise ValueError("G2PTQ trust region: expected [groups, rows, columns] tensors.")
    if alpha <= 0.0:
        return torch.zeros(
            (weight.shape[0], weight.shape[1], 1),
            dtype=weight.dtype,
            device=weight.device,
        )

    # ``g H^-1`` for every row (Eq. 14). The paper's cross term is
    # ``+(w_hat - w) * (g H^-1 e_t)``; ``quant_err`` is ``w_hat - w``.
    ghinv = torch.bmm(gradient, inverse_hessian)
    quant_err = fake_quantized - weight
    diag = torch.diagonal(inverse_hessian, dim1=-2, dim2=-1).unsqueeze(1)
    tiny = torch.finfo(torch.float32).tiny

    # ``diag`` can only be zero for a dead channel; the caller neutralizes
    # those before this point. Keep the guard so a direct caller cannot emit
    # inf/NaN factors.
    safe_diag = torch.where(diag.abs() > tiny, diag, torch.ones_like(diag))
    loss_gptq = ((quant_err * ghinv + 0.5 * quant_err.square()) / safe_diag).abs()
    loss_gptq = torch.where(diag.abs() > tiny, loss_gptq, torch.zeros_like(loss_gptq))

    c = (gradient * ghinv).sum(dim=-1, keepdim=True) - ghinv.square() / safe_diag
    c = torch.where(diag.abs() > tiny, c, torch.zeros_like(c))
    # Eq. 13 saturates at one when c <= 2 * alpha * |dL_gptq|.
    budget = 2.0 * alpha * loss_gptq
    fraction = budget / c.clamp_min(budget).clamp_min(tiny)
    # Rationalize 1 - sqrt(1 - x) to retain small positive budgets in FP32.
    beta = fraction / (1.0 + torch.sqrt(1.0 - fraction))
    return beta.mean(dim=-1, keepdim=True)


def g2ptq_compensate(
    weight: torch.Tensor,
    factor: torch.Tensor,
    gradient: torch.Tensor,
    beta: torch.Tensor,
    quantize_column: Callable[[int], torch.Tensor],
    *,
    blocksize: int,
    loss_terms: Optional[torch.Tensor] = None,
    dynamic_group_size: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the G2PTQ column loop over a batch of channel groups.

    ``weight`` is mutated in place, exactly like GPTQ's block pass. Returns the
    quantized weights ``Q`` and the per-column errors ``E``. A positive
    ``dynamic_group_size`` gives the width read by a group grid refit.
    """

    if weight.ndim != 3:
        raise ValueError("G2PTQ: `weight` must be [groups, rows, columns].")
    groups, rows, columns = weight.shape
    if factor.shape != (groups, columns, columns):
        raise ValueError("G2PTQ: `factor` must be [groups, columns, columns].")
    if gradient.shape != weight.shape:
        raise ValueError("G2PTQ: `gradient` must match `weight`.")
    if beta.shape != (groups, rows, 1):
        raise ValueError("G2PTQ: `beta` must be [groups, rows, 1].")
    if blocksize < 1:
        raise ValueError("G2PTQ: `blocksize` must be positive.")

    # Eq. 17: M = (beta . G) L and F = M L^T. With this port's upper factor
    # ``U = L^T`` that is ``M = (G @ U^T) * beta`` and ``F = M @ U``.
    m = torch.bmm(gradient, factor.transpose(1, 2)) * beta
    f = torch.bmm(m, factor)

    q = torch.zeros_like(weight)
    error = torch.zeros_like(weight)
    offsets = torch.arange(
        blocksize, dtype=torch.long, device=weight.device
    )
    lazy_diag = (blocksize - 1 - offsets).to(weight.dtype)

    block_start = 0
    while block_start < columns:
        block_end = min(block_start + blocksize, columns)
        if dynamic_group_size > 0 and block_end < columns:
            # A refit reads the whole group. Flush pending updates before
            # starting a group whose tail would lie outside this block.
            boundary = block_end - block_end % dynamic_group_size
            if boundary > block_start:
                block_end = boundary
        count = block_end - block_start

        for column in range(block_start, block_end):
            w = weight[:, :, column]
            quantized = quantize_column(column)
            if quantized.shape != w.shape:
                raise ValueError("G2PTQ: `quantize_column` returned the wrong shape.")
            q[:, :, column] = quantized
            # Eq. 18: E_t = (W_t - Q_t - F_t) / U[t, t].
            column_error = (
                w - quantized - f[:, :, column]
            ) / factor[:, column, column].unsqueeze(-1)
            error[:, :, column] = column_error
            if loss_terms is not None:
                loss_terms[:, :, column] = (
                    (w - quantized).square()
                    / factor[:, column, column].unsqueeze(-1).square()
                )
            if column + 1 < block_end:
                # Eq. 16/17 inside the block. W uses F before line 14 moves it.
                weight[:, :, column + 1:block_end] -= (
                    column_error.unsqueeze(-1)
                    * factor[:, column:column + 1, column + 1:block_end]
                    + f[:, :, column + 1:block_end]
                )
                f[:, :, column + 1:block_end] -= (
                    m[:, :, column:column + 1]
                    * factor[:, column:column + 1, column + 1:block_end]
                )

        if block_end < columns:
            # Eq. 19. D = diag(B-1, ..., 0) for the block actually solved.
            block_factor = factor[:, block_start:block_end, block_end:]
            weight[:, :, block_end:] -= (
                error[:, :, block_start:block_end] @ block_factor
                + count * f[:, :, block_end:]
                - (m[:, :, block_start:block_end] * lazy_diag[-count:]) @ block_factor
            )
            f[:, :, block_end:] -= m[:, :, block_start:block_end] @ block_factor

        block_start = block_end

    return q, error


class G2PTQ(GPTQ):
    """GPTQ solver with grouped Fisher Hessians and exact gradient compensation."""

    # The gradient pass lives in the processor; this task never reads the
    # module's forward output through the capture hook.
    needs_module_output = False

    def __init__(self, module: nn.Module, qcfg=None, region_timer=None):
        super().__init__(module, qcfg, region_timer=region_timer)
        # ``[num_groups, columns, columns]`` grouped Fisher Hessians.
        self.group_hessians: Optional[torch.Tensor] = None
        # ``[rows, columns]`` exact weight gradient from the block objective.
        self.weight_gradient: Optional[torch.Tensor] = None
        # ``[columns]`` accumulated squared module input; only its ordering
        # is read, by ``desc_act`` and ``act_group_aware``.
        self.activation_square: Optional[torch.Tensor] = None
        # ``None`` means "use ``alpha_mse``"; the processor sets an
        # explicit value per subset (MSE blocks vs. the final KL block).
        self._g2ptq_trust_alpha: Optional[float] = None

    # ------------------------------------------------------------------
    # statistics plumbing
    # ------------------------------------------------------------------
    def set_block_statistics(
        self,
        *,
        group_hessians: torch.Tensor,
        weight_gradient: torch.Tensor,
        activation_square: Optional[torch.Tensor] = None,
        trust_alpha: Optional[float] = None,
    ) -> None:
        """Install one block's refreshed first- and second-order statistics."""

        if group_hessians.ndim != 3:
            raise ValueError("G2PTQ: `group_hessians` must be [groups, columns, columns].")
        groups, columns, columns_again = group_hessians.shape
        if groups == 0 or self.rows % groups:
            raise ValueError("G2PTQ: Hessian groups must evenly divide the module's output rows.")
        if columns != columns_again:
            raise ValueError(
                f"G2PTQ: Hessian is not square ({columns}x{columns_again})."
            )
        if weight_gradient.shape != (self.rows, columns):
            raise ValueError(
                f"G2PTQ: gradient is {tuple(weight_gradient.shape)}, "
                f"module expects {(self.rows, columns)}."
            )
        if activation_square is not None and activation_square.shape != (columns,):
            raise ValueError("G2PTQ: `activation_square` must be [columns].")
        if columns > self.columns:
            raise ValueError(
                f"G2PTQ: Hessian has {columns} columns, module expects at most {self.columns}."
            )
        if not torch.isfinite(group_hessians).all() or not torch.isfinite(weight_gradient).all():
            raise ValueError("G2PTQ: Hessian and weight gradient must be finite.")
        if columns < self.columns:
            # Tensor-parallel runs pad the column axis. The padding is never
            # calibrated, so it enters with zero signal, exactly like GPTQ's
            # padded Hessian.
            pad = self.columns - columns
            group_hessians = torch.nn.functional.pad(group_hessians, (0, pad, 0, pad))
            weight_gradient = torch.nn.functional.pad(weight_gradient, (0, pad))
            if activation_square is not None:
                activation_square = torch.nn.functional.pad(activation_square, (0, pad))

        self.group_hessians = group_hessians.detach().to(
            device=group_hessians.device, dtype=torch.float32
        )
        self.weight_gradient = weight_gradient.detach().to(dtype=torch.float32)
        self.activation_square = (
            None
            if activation_square is None
            else activation_square.detach().to(dtype=torch.float32)
        )
        self._g2ptq_trust_alpha = None if trust_alpha is None else float(trust_alpha)

    def clear_block_statistics(self) -> None:
        self.group_hessians = None
        self.weight_gradient = None
        self.activation_square = None
        self._g2ptq_trust_alpha = None

    # ------------------------------------------------------------------
    # quantization
    # ------------------------------------------------------------------
    def _g2ptq_config(self):
        config = getattr(self.qcfg, "g2ptq", None)
        if config is None:
            raise ValueError("G2PTQ requires a `g2ptq` configuration.")
        return config

    @torch.inference_mode()
    def quantize(self, blocksize: int = 128):
        try:
            if (
                self.group_hessians is None
                or self.weight_gradient is None
                or self.nsamples == 0
                or self.qcfg.mock_quantization
                or isinstance(self.module, nn.Embedding)
                or should_use_fallback(self.fallback, float(self.nsamples), self.expected_nsamples)
            ):
                # Keep GPTQ's zero-sample and SmoothMSE fallback contracts.
                return super().quantize(blocksize=blocksize)
            return self._g2ptq_quantize(blocksize=blocksize)
        finally:
            self.clear_block_statistics()

    def _resolve_group_count(self) -> int:
        config = self._g2ptq_config()
        groups = int(config.num_groups)
        if self.rows % groups != 0:
            # lm_head vocabularies and some fused projections are not
            # divisible by the configured group count. Sharing one Hessian
            # keeps the solver valid instead of dropping rows.
            log.warn(
                "G2PTQ: Module `%s` has %s output channels, not divisible by "
                "`num_groups=%s`; using one group.",
                self.name,
                self.rows,
                groups,
            )
            return 1
        return groups

    def _g2ptq_quantize(self, blocksize: int = 128):
        config = self._g2ptq_config()
        start = time.time()
        result_device = torch.device(self.module.weight.data.device)

        hessians = self.group_hessians
        gradient = self.weight_gradient
        activation_square = self.activation_square
        num_groups = self._resolve_group_count()
        rows_per_group = self.rows // num_groups

        if num_groups != hessians.shape[0]:
            # Reconcile a configured group count that changed between capture
            # and solve (dynamic overrides). A single captured group is the
            # average over every output channel, so broadcasting it is exact;
            # collapsing several groups keeps the same Fisher estimate.
            captured_groups = hessians.shape[0]
            if captured_groups == 1:
                hessians = hessians.expand(num_groups, -1, -1)
            elif num_groups == 1:
                hessians = hessians.mean(dim=0, keepdim=True)
            else:
                raise ValueError(
                    f"G2PTQ: captured {captured_groups} channel groups but the "
                    f"module is configured for {num_groups}."
                )
        gradient = gradient.reshape(num_groups, rows_per_group, self.columns)

        compute_device = self._select_hessian_target_device(getattr(self.module, "target_device", None))
        hessians = safe_move_to(hessians, compute_device, dtype=torch.float32)
        gradient = safe_move_to(gradient, compute_device, dtype=torch.float32)

        if self.module_copy is None:
            W = self.clone_module(device=compute_device)
        else:
            W = safe_move_to(self.module_copy, compute_device)
            self.module_copy = None

        # Capture is deferred into device-local GPTQ partials. Its diagonal
        # distinguishes dead inputs from live columns with zero Fisher weight.
        input_energy = self._captured_hessian_diagonal(compute_device)
        if input_energy is None:
            input_energy = activation_square
            if input_energy is None:
                input_energy = hessians.diagonal(dim1=-2, dim2=-1).mean(dim=0)
            input_energy = safe_move_to(input_energy, compute_device)
        self.release_hessian_state()

        # Symmetrize before writes, since the grouped input may be a shared
        # expansion or a tensor held by the processor.
        hessians = 0.5 * (hessians + hessians.transpose(1, 2))
        self.group_hessians = None
        dead = input_energy == 0
        W.masked_fill_(dead, 0)
        gradient = gradient.masked_fill(dead, 0)
        self.weight_gradient = None
        hessians.masked_fill_(dead, 0)
        hessians.masked_fill_(dead.unsqueeze(1), 0)

        diagonal = hessians.diagonal(dim1=-2, dim2=-1)
        live = diagonal > 0
        live_count = live.sum(dim=-1, keepdim=True)
        live_mean = diagonal.sum(dim=-1, keepdim=True) / live_count.clamp_min(1)
        # A zero Fisher diagonal has no correlations. Give it a positive
        # diagonal for Cholesky while keeping any calibration-live weight.
        diagonal.copy_(torch.where(live, diagonal, torch.where(live_count > 0, live_mean, 1.0)))
        gradient.masked_fill_(live_count.unsqueeze(-1) == 0, 0)
        del diagonal

        if self.qcfg.group_size == -1:
            self._fit_weight_params(W)

        scale = []
        zero = []
        now_idx = 1
        groups = []
        static_groups = self.qcfg.static_groups and self.qcfg.group_size != -1
        if static_groups:
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                self._fit_weight_params(
                    W[:, i:i + self.qcfg.group_size], quantizer
                )
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        perm = None
        invperm = None
        final_perm = None
        global_perm = None
        if self.qcfg.desc_act:
            ordering_source = input_energy
            perm = torch.argsort(ordering_source, descending=True)
            W = W[:, perm]
            hessians = hessians[:, perm][:, :, perm]
            gradient = gradient[:, :, perm]
            invperm = torch.argsort(perm)
        elif self.qcfg.act_group_aware:
            ordering_source = input_energy
            local_perms, local_values = compute_local_perms(
                ordering_source, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                ordering_source,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(
                local_perms, global_perm, self.qcfg.group_size
            )
            final_perm = extend_perm_with_tail(final_perm, self.columns)
            W = W[:, final_perm]
            hessians = hessians[:, final_perm][:, :, final_perm]
            gradient = gradient[:, :, final_perm]

        static_group_perm = None
        if static_groups:
            static_group_perm = perm if self.qcfg.desc_act else final_perm
        static_group_of_column = _static_group_of_column(
            static_group_perm, self.qcfg.group_size
        )

        weight_groups = W.reshape(num_groups, rows_per_group, self.columns)
        del W

        def _invert_groups():
            factors, damps = [], []
            for group_index in range(num_groups):
                group_factor, group_damp = self.hessian_inverse(hessians[group_index])
                if group_factor is None:
                    return None, 1.0
                factors.append(group_factor)
                damps.append(group_damp)
            return torch.stack(factors), max(damps)

        def _prepare_cpu(cpu_device):
            nonlocal hessians, gradient, weight_groups, scale, zero
            hessians = hessians.to(device=cpu_device)
            gradient = gradient.to(device=cpu_device)
            # Keep the pre-inversion activation ordering and dead-column mask.
            weight_groups = weight_groups.to(device=cpu_device)
            if static_groups:
                scale, zero = _move_static_groups(groups, cpu_device)
            elif self.qcfg.group_size == -1:
                self._fit_weight_params(weight_groups.reshape(-1, self.columns))

        factor, damp = self._run_with_accelerator_oom_cpu_fallback(
            _invert_groups,
            source_device=compute_device,
            stage="G2PTQ Hessian inverse",
            prepare_cpu=_prepare_cpu,
        )[0]
        if factor is None:
            return self._rtn_fallback_after_hessian_failure(blocksize)

        alpha = (
            float(config.alpha_mse)
            if self._g2ptq_trust_alpha is None
            else float(self._g2ptq_trust_alpha)
        )

        if alpha > 0:
            fake = self._fake_quantize_groups(
                weight_groups, groups, static_group_of_column
            )
            inverse = factor.transpose(1, 2) @ factor
            beta = trust_region_beta(
                weight_groups, inverse, gradient, fake, alpha
            )
            del fake, inverse
        else:
            beta = weight_groups.new_zeros((num_groups, rows_per_group, 1))

        # Release the full Hessian: only the factor and the gradient are read
        # from here on.
        # ``hessians`` is a closure variable of ``_invert_groups``; clear the
        # cell instead of deleting it so a stale reference cannot survive.
        hessians = None
        loss_terms = torch.empty_like(weight_groups)

        def _quantize_column(column: int) -> torch.Tensor:
            nonlocal now_idx
            index = column
            if self.qcfg.group_size != -1:
                if not static_groups:
                    if index % self.qcfg.group_size == 0:
                        # Fit on the *current* stacked weights: the solver
                        # mutates them column by column, exactly like GPTQ.
                        self._fit_weight_params(
                            weight_groups.reshape(-1, self.columns)[
                                :, index:index + self.qcfg.group_size
                            ]
                        )
                    if index // self.qcfg.group_size - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                else:
                    if static_group_of_column is not None:
                        self.quantizer = groups[static_group_of_column[index]]
                    else:
                        self.quantizer = groups[index // self.qcfg.group_size]
            w = weight_groups[:, :, column].reshape(-1, 1)
            return self.quantizer.quantize(w).flatten().view(num_groups, rows_per_group)

        q, _error = g2ptq_compensate(
            weight_groups,
            factor,
            gradient,
            beta,
            _quantize_column,
            blocksize=blocksize,
            loss_terms=loss_terms,
            dynamic_group_size=-1 if static_groups else self.qcfg.group_size,
        )

        # ``gradient`` and ``weight_groups`` are closure variables of
        # ``_invert_groups``/``_quantize_column``. Clearing the cells releases
        # the tensors at the same point as ``del`` would.
        del factor, beta, _error
        gradient = None
        weight_groups = None

        torch_sync()
        avg_loss = (loss_terms.sum() / (2.0 * self.nsamples)).item()
        del loss_terms
        if math.isnan(avg_loss):
            raise ValueError(
                f"Quantization: Failed due to `NaN` loss for `{self.name}`"
            )

        Q = q.reshape(self.rows, self.columns)
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if static_groups and self.qcfg.desc_act:
            g_idx = (perm // group_size).to(dtype=torch.int32, device=Q.device)
        else:
            g_idx = torch.arange(
                self.columns, dtype=torch.int32, device=Q.device
            ) // group_size

        if self.qcfg.desc_act:
            invperm = invperm.to(device=Q.device)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        elif self.qcfg.act_group_aware:
            inv_final = invert_perm(final_perm).to(device=Q.device)
            Q = Q[:, inv_final]
            if not static_groups:
                inv_global = invert_perm(global_perm).tolist()
                reordered = len(inv_global)
                scale = [scale[i] for i in inv_global] + scale[reordered:]
                zero = [zero[i] for i in inv_global] + zero[reordered:]

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        Q = self._narrow_reconstruction(Q)

        if not scale:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)

        scale, zero = self._sanitize_scale_and_log(scale, zero)
        Q = safe_move_to(Q, result_device)

        return (
            Q,
            scale,
            zero,
            g_idx,
            time.time() - start,
            avg_loss,
            damp,
            self.nsamples,
        )

    def _fake_quantize_groups(
        self,
        weight_groups: torch.Tensor,
        groups,
        static_group_of_column,
    ) -> torch.Tensor:
        """Quantize the current weights with the active grid, without GPTQ updates."""

        num_groups, rows_per_group, columns = weight_groups.shape
        flat = weight_groups.reshape(-1, columns)
        output = torch.empty_like(flat)
        group_size = self.qcfg.group_size
        if group_size == -1:
            output.copy_(self.quantizer.quantize(flat))
        elif self.qcfg.static_groups:
            for column in range(columns):
                if static_group_of_column is not None:
                    quantizer = groups[static_group_of_column[column]]
                else:
                    quantizer = groups[column // group_size]
                output[:, column:column + 1] = quantizer.quantize(
                    flat[:, column:column + 1]
                )
        else:
            for start in range(0, columns, group_size):
                end = min(start + group_size, columns)
                self._fit_weight_params(flat[:, start:end])
                output[:, start:end] = self.quantizer.quantize(flat[:, start:end])
        return output.view(num_groups, rows_per_group, columns)

    def free(self):
        super().free()
        self.clear_block_statistics()


__all__ = ["G2PTQ", "g2ptq_compensate", "trust_region_beta"]
