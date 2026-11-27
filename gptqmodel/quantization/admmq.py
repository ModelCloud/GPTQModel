# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ADMM-Q weight-only quantizer.

Clean-room implementation of the layer-wise solver introduced by Ryan Lucas,
Mehdi Makni, Xiang Meng, Adam Deng, and Rahul Mazumder in "ADMM-Q: An Improved
Hessian-based Weight Quantizer for Post-Training Quantization of Large Language
Models" (COLM 2026, arXiv:2605.11222; reference implementation:
https://github.com/mazumder-lab/admm-q).  No reference code is copied.

The layer objective is the same Hessian-weighted reconstruction error GPTQ
minimizes::

    1/2 * Tr((W - Wq) H (W - Wq)^T),   H = X^T X + damp * mean(diag(X^T X)) * I

ADMM-Q replaces GPTQ's greedy column sweep with an ADMM split between the
continuous Hessian-weighted update and the discrete quantization projection.
The Hessian is diagonally preconditioned once, factorized once, and each
iteration is two matrix products; a guarded grid refresh and a pair-swap local
search refine the discrete solution.  The returned tensors use the ordinary
GPTQ checkpoint contract, so packing, saving and inference are unchanged.
"""

from __future__ import annotations

import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from ..utils.p2p import safe_move_to
from .config import ADMMQConfig, BaseQuantizeConfig, FallbackStrategy
from .gptq import GPTQ
from .quantizer import _neutralize_dead_columns, quantize


log = setup_logger()


class _AdmmNumericalFailure(Exception):
    """Internal signal that one ADMM layer solve cannot produce a finite result.

    The caller keeps the GPTQ family's RTN quantization for that layer instead
    of aborting the whole model. ``reason`` is recorded in
    ``ADMMQ.admmq_diagnostics`` so a fallback layer stays distinguishable from
    a layer where the ADMM iterate simply did not beat its RTN baseline.
    """

    def __init__(
        self, reason: str, detail: str = "", *, iterations: Optional[int] = None, diagnostics: Optional[dict] = None
    ):
        self.reason = reason
        self.iterations = iterations
        self.diagnostics = diagnostics
        super().__init__(f"{reason}: {detail}" if detail else reason)


#: Penalty growth is capped so a very long run cannot overflow the schedule.
RHO_CAP = 1e6

#: Preconditioned condition-number thresholds for the paper's adaptive
#: penalty schedule. The reference implementation uses the same 5e4 gate for
#: PairSwap.
ILL_CONDITIONED_THRESHOLD = 5e4
ILL_CONDITIONED_SEVERE_THRESHOLD = 3e5

#: Penalty settings used for ill-conditioned layers.
ILL_CONDITIONED_RHO = 0.05
ILL_CONDITIONED_SEVERE_RHO = 0.01
ILL_CONDITIONED_RHO_UPDATE = 1.05

#: Row chunk used for the float64-scaled update; bounds the transient buffer.
_WORK_CHUNK_BYTES = 64 << 20

#: Runtime device types that provide the float64 ``torch.linalg.eigh`` used by
#: the ADMM-Q factorization: the host and the CUDA namespace (NVIDIA CUDA and
#: HIP). Every other family runs the solve on the host.
_SOLVE_DEVICE_TYPES = frozenset({"cpu", "cuda"})


def admm_rho_at(iteration: int, rho0: float, gamma: float, update_iter: int, cap: float = RHO_CAP) -> float:
    """Return the penalty used by ADMM iteration ``iteration`` (0-based).

    The penalty grows by ``gamma`` after every ``update_iter`` iterations,
    matching the paper's ``rho_{t+1} = update(rho_t)`` schedule.
    """

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if update_iter < 1:
        raise ValueError("update_iter must be positive")
    exponent = iteration // update_iter
    if exponent == 0 or gamma == 1.0:
        return min(rho0, cap)
    log_rho = math.log(rho0)
    log_gamma = math.log(gamma)
    if log_gamma > 0.0 and exponent >= (math.log(cap) - log_rho) / log_gamma:
        return cap
    try:
        return min(rho0 * gamma**exponent, cap)
    except OverflowError:
        # The power can overflow even while its product with rho0 is finite.
        return min(math.exp(log_rho + exponent * log_gamma), cap)


def admm_penalty_schedule(
    condition_number: float,
    rho: float,
    rho_update: float,
    *,
    enabled: bool = True,
) -> Tuple[float, float]:
    """Return the initial penalty and growth for one layer.

    ADMM-Q's adaptive schedule replaces the configured values when the
    preconditioned Hessian is ill-conditioned: above ``5e4`` it starts at
    ``0.05`` and above ``3e5`` at ``0.01``, both growing by ``1.05``. The
    reference implementation applies this unconditionally; ``enabled=False``
    keeps the configured schedule for an ablation.
    """

    if not enabled or not condition_number > ILL_CONDITIONED_THRESHOLD:
        return float(rho), float(rho_update)
    initial = (
        ILL_CONDITIONED_SEVERE_RHO
        if condition_number > ILL_CONDITIONED_SEVERE_THRESHOLD
        else ILL_CONDITIONED_RHO
    )
    return float(initial), ILL_CONDITIONED_RHO_UPDATE


def projection_objective(target: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance of one projection (the D-update objective)."""

    return (target - projected).square().sum()


def refresh_accepts(new: float, old: float, rel_margin: float = 0.0) -> bool:
    """Whether a refreshed grid's projection is a strict enough improvement."""

    if rel_margin < 0.0:
        raise ValueError("rel_margin must be non-negative")
    return float(new) < float(old) * (1.0 - float(rel_margin))


def pairswap_delta(
    g1: torch.Tensor,
    g2: torch.Tensor,
    h11: torch.Tensor,
    h22: torch.Tensor,
    h12: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form loss change for the four sign combinations of one pair.

    ``g1``/``g2`` are the gradient entries of the two shifted rows (columns of
    the weight in the paper's layout), ``h`` the three Hessian entries and
    ``d1``/``d2`` the per-output-channel grid steps.  The returned values are
    changes of the same half-scaled objective the solver minimizes, in the
    order ``(+,+)``, ``(+,-)``, ``(-,+)`` and ``(-,-)``.
    """

    one_dim = g1.dim() == 1
    if one_dim:
        g1 = g1.reshape(-1, 1)
        g2 = g2.reshape(-1, 1)
        d1 = d1.reshape(-1, 1)
        d2 = d2.reshape(-1, 1)
    elif g1.dim() != 2:
        raise ValueError("pair-swap gradient entries must be 1-D or 2-D")

    d1_sq = d1 * d1
    d2_sq = d2 * d2
    cross = d1 * d2 * h12
    deltas: List[torch.Tensor] = []
    for sign1, sign2 in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        deltas.append(
            -sign1 * d1 * g1
            + 0.5 * d1_sq * h11
            - sign2 * d2 * g2
            + 0.5 * d2_sq * h22
            + sign1 * sign2 * cross
        )
    if one_dim:
        deltas = [delta.reshape(-1) for delta in deltas]
    return tuple(deltas)


def _diagnostics_stats(values: Sequence[Any]) -> Dict[str, Optional[float]]:
    """Return min/median/max over the finite numeric values only."""

    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not finite:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(finite),
        "median": float(statistics.median(finite)),
        "max": max(finite),
    }


def summarize_admmq_diagnostics(
    layers: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-module ADMM-Q diagnostics into one run summary.

    The summary names the modules that early-stopped, exhausted ``max_iter``,
    fell back to RTN, or accepted a grid refresh. Condition numbers and
    iteration counts are reported as finite min/median/max distributions; an
    infinite condition number is counted separately instead of poisoning the
    statistics.
    """

    names = sorted(layers)
    solved = [
        name for name in names
        if layers[name].get("status", "solved") == "solved"
    ]
    # Classify by ``stop_reason`` rather than ``converged``: a numerical
    # failure records a fallback without a normal stop reason and must not be
    # counted as an exhausted iteration budget.
    early_stop = [
        name for name in solved
        if layers[name].get("stop_reason") == "early_stop"
    ]
    max_iter = [
        name for name in solved
        if layers[name].get("stop_reason") == "max_iter"
    ]
    rtn_fallback = [name for name in names if bool(layers[name].get("fallback_to_rtn"))]

    fallback_reasons: Dict[str, int] = {}
    for name in rtn_fallback:
        reason = layers[name].get("fallback_reason") or "unspecified"
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    refresh_attempted = [
        name for name in names
        if int(layers[name].get("refresh_attempts", 0) or 0) > 0
    ]
    refresh_accepted = [
        name for name in refresh_attempted
        if bool(layers[name].get("refresh_accepted"))
    ]
    pairswap_layers = [
        name for name in names
        if bool(layers[name].get("pairswap_ran"))
    ]
    pairswap_with_flips = [
        name for name in names
        if int(layers[name].get("pairswap_flips", 0) or 0) > 0
    ]

    condition_numbers = [layers[name].get("condition_number") for name in solved]
    iterations = [layers[name].get("iterations") for name in solved]
    nonfinite_condition_numbers = sum(
        1
        for value in condition_numbers
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    )

    return {
        "layers": len(names),
        "solved": len(solved),
        "skipped": len(names) - len(solved),
        "early_stop_layers": early_stop,
        "max_iter_layers": max_iter,
        "rtn_fallback_layers": rtn_fallback,
        "rtn_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "refresh_attempted": len(refresh_attempted),
        "refresh_accepted": len(refresh_accepted),
        "refresh_accepted_layers": refresh_accepted,
        "pairswap_layers": len(pairswap_layers),
        "pairswap_flips": sum(
            int(layers[name].get("pairswap_flips", 0) or 0) for name in names
        ),
        "pairswap_layers_with_flips": pairswap_with_flips,
        "condition_number": _diagnostics_stats(condition_numbers),
        "condition_number_nonfinite": nonfinite_condition_numbers,
        "iterations": _diagnostics_stats(iterations),
    }


def admmq_diagnostics_alert(summary: Mapping[str, Any]) -> Optional[str]:
    """Return a one-line alert when a layer's solve degraded, else ``None``.

    ADMM-Q normally spends the whole iteration budget, so ``max_iter`` alone is
    not worth a log line. Only skipped modules, RTN fallbacks and non-finite
    condition numbers signal a result the caller should inspect; a clean run
    logs nothing and leaves the full record on ``model.admmq_diagnostics``.
    """

    skipped = int(summary.get("skipped") or 0)
    fallback = len(summary.get("rtn_fallback_layers") or ())
    nonfinite = int(summary.get("condition_number_nonfinite") or 0)
    if not (skipped or fallback or nonfinite):
        return None

    reasons = summary.get("rtn_fallback_reasons") or {}
    parts = [
        f"skipped={skipped}",
        f"rtn_fallback={fallback}",
        f"condition_number_nonfinite={nonfinite}",
    ]
    if reasons:
        parts.append(
            "reasons="
            + ",".join(f"{name}:{count}" for name, count in sorted(reasons.items()))
        )
    return " ".join(parts)


@dataclass
class _Grid:
    """One fitted quantization grid (per output channel) for a column range."""

    start: int
    end: int
    scale: torch.Tensor
    zero: torch.Tensor
    code_lo: Optional[torch.Tensor]
    code_hi: Optional[torch.Tensor]
    maxq: int


class ADMMQ(GPTQ):
    """ADMM-Q layer solver; a GPTQ-compatible weight quantizer.

    It reuses GPTQ's activation capture, damping/dead-column handling, tensor
    parallel padding, dtype narrowing and replay contract, and replaces only
    the weight solve.
    """

    def __init__(self, module: nn.Module, qcfg: Optional[BaseQuantizeConfig] = None, region_timer=None):
        if qcfg is None:
            qcfg = ADMMQConfig()
        if not isinstance(qcfg, ADMMQConfig):
            raise ValueError(
                "ADMMQ requires an ADMMQConfig; "
                f"got `{type(qcfg).__name__}`. Use `QuantizeConfig(method='admmq', ...)`."
            )
        super().__init__(module, qcfg=qcfg, region_timer=region_timer)
        self.admmq_diagnostics: Optional[dict] = None

    # ------------------------------------------------------------------
    # grid fitting / projection
    # ------------------------------------------------------------------

    def _group_ranges(self) -> List[Tuple[int, int]]:
        group_size = self.qcfg.group_size
        if group_size is None or int(group_size) <= 0:
            return [(0, self.columns)]
        return [(start, min(start + int(group_size), self.columns)) for start in range(0, self.columns, int(group_size))]

    def _fit_grids(self, weight: torch.Tensor) -> List[_Grid]:
        """Fit one ``(scale, zero)`` grid per quantization group on ``weight``."""

        grids: List[_Grid] = []
        probe = copy.deepcopy(self.quantizer)
        if int(getattr(probe, "_maxq_value", 0)) <= 0:
            probe.configure(perchannel=True)
        for start, end in self._group_ranges():
            self._fit_weight_params(weight[:, start:end], probe)
            grids.append(
                _Grid(
                    start=start,
                    end=end,
                    scale=probe.scale.detach().clone(),
                    zero=probe.zero.detach().clone(),
                    code_lo=None if probe.code_lo is None else probe.code_lo.detach().clone(),
                    code_hi=None if probe.code_hi is None else probe.code_hi.detach().clone(),
                    maxq=int(probe._maxq_value),
                )
            )
        return grids

    @staticmethod
    def _project(weight: torch.Tensor, grids: Sequence[_Grid]) -> torch.Tensor:
        projected = torch.empty_like(weight)
        for grid in grids:
            projected[:, grid.start:grid.end] = quantize(
                weight[:, grid.start:grid.end],
                grid.scale,
                grid.zero,
                grid.maxq,
                False,
                code_lo=grid.code_lo,
                code_hi=grid.code_hi,
            )
        return projected

    def _objective(self, weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
        delta = weight - quantized
        return 0.5 * (delta * (delta @ hessian)).sum()

    # ------------------------------------------------------------------
    # pair-swap local search
    # ------------------------------------------------------------------

    def _pairswap(self, weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor, grids: Sequence[_Grid]) -> Tuple[torch.Tensor, int]:
        """Refine the projected solution with a greedy local search over column pairs.

        Each batch scores ``pairswap_pairs`` disjoint pairs, keeps the single
        pair with the largest total improvement, and applies its
        per-output-channel improving sign combination.  ``pairswap_batch``
        bounds the number of batches per round; the gradient is maintained
        with the rank-2 correction instead of being recomputed.
        """

        cfg = self.qcfg
        rounds = int(cfg.pairswap_rounds)
        device = quantized.device
        scale = torch.cat([grid.scale for grid in grids], dim=1)
        zero = torch.cat([grid.zero for grid in grids], dim=1)
        group_size = self.columns if int(cfg.group_size) <= 0 else int(cfg.group_size)
        group_of_column = torch.arange(self.columns, device=device) // group_size
        column_scale = scale[:, group_of_column]
        column_zero = zero[:, group_of_column]
        code_lo = torch.cat(
            [
                grid.code_lo.expand(-1, grid.end - grid.start)
                if grid.code_lo is not None
                else torch.zeros(
                    (quantized.shape[0], grid.end - grid.start),
                    device=device,
                    dtype=quantized.dtype,
                )
                for grid in grids
            ],
            dim=1,
        )
        code_hi = torch.cat(
            [
                grid.code_hi.expand(-1, grid.end - grid.start)
                if grid.code_hi is not None
                else torch.full(
                    (quantized.shape[0], grid.end - grid.start),
                    float(grid.maxq),
                    device=device,
                    dtype=quantized.dtype,
                )
                for grid in grids
            ],
            dim=1,
        )
        code = torch.round(quantized / column_scale) + column_zero

        generator = torch.Generator(device="cpu").manual_seed(int(cfg.pairswap_seed))
        gradient = (weight - quantized) @ hessian
        pairs_per_batch = max(1, int(cfg.pairswap_pairs))
        batches_per_round = max(1, int(cfg.pairswap_batch))
        flips = 0

        for _ in range(rounds):
            applied_any = False
            for _ in range(batches_per_round):
                available = self.columns // 2
                if available < 1:
                    break
                count = min(pairs_per_batch, available)
                order = torch.randperm(self.columns, generator=generator)[: 2 * count]
                pairs = order.reshape(count, 2).to(device)
                first = pairs[:, 0]
                second = pairs[:, 1]

                deltas = pairswap_delta(
                    gradient[:, first],
                    gradient[:, second],
                    hessian[first, first].unsqueeze(0),
                    hessian[second, second].unsqueeze(0),
                    hessian[first, second].unsqueeze(0),
                    column_scale[:, first],
                    column_scale[:, second],
                )
                up_first = code[:, first] < code_hi[:, first]
                down_first = code[:, first] > code_lo[:, first]
                up_second = code[:, second] < code_hi[:, second]
                down_second = code[:, second] > code_lo[:, second]
                valid = (
                    up_first & up_second,
                    up_first & down_second,
                    down_first & up_second,
                    down_first & down_second,
                )
                zero_delta = torch.zeros_like(deltas[0])
                masked = [torch.where(ok, delta, zero_delta) for ok, delta in zip(valid, deltas)]
                best_delta, best_combo = torch.stack(masked, dim=0).min(dim=0)
                gain = (-best_delta).relu().sum(dim=0)
                chosen_pair = int(gain.argmax().item())
                if float(gain[chosen_pair]) <= 0.0:
                    continue

                improving = best_delta[:, chosen_pair] < 0.0
                combo = best_combo[:, chosen_pair]
                sign_first = torch.where((combo == 0) | (combo == 1), 1.0, -1.0)
                sign_second = torch.where((combo == 0) | (combo == 2), 1.0, -1.0)
                first_column = int(first[chosen_pair].item())
                second_column = int(second[chosen_pair].item())
                # Decode the updated integer codes. Repeatedly adding the
                # FP32 grid step would accumulate error in the replay weight.
                code[:, first_column] += torch.where(improving, sign_first, 0.0)
                code[:, second_column] += torch.where(improving, sign_second, 0.0)
                new_first = column_scale[:, first_column] * (code[:, first_column] - column_zero[:, first_column])
                new_second = column_scale[:, second_column] * (code[:, second_column] - column_zero[:, second_column])
                shift_first = new_first - quantized[:, first_column]
                shift_second = new_second - quantized[:, second_column]
                quantized[:, first_column] = new_first
                quantized[:, second_column] = new_second
                gradient.addr_(shift_first, hessian[first_column], alpha=-1.0)
                gradient.addr_(shift_second, hessian[second_column], alpha=-1.0)
                flips += int(improving.sum().item())
                applied_any = True
            if not applied_any:
                break

        return quantized, flips

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def _condition_number(self, eigenvalues: torch.Tensor) -> float:
        if eigenvalues.numel() == 0:
            return float("inf")
        # eigh returns the complete spectrum in ascending order. Dropping
        # null directions reports the condition of only the nonzero subspace
        # and can make a rank-deficient Hessian look perfectly conditioned.
        smallest = float(eigenvalues[0].item())
        if smallest <= 0.0:
            return float("inf")
        # Divide as Python doubles so a small positive eigenvalue is neither
        # floored nor subject to FP32 ratio overflow.
        return float(eigenvalues[-1].item()) / smallest

    def _should_pairswap(self, condition_number: float) -> bool:
        cfg = self.qcfg
        mode = cfg.pairswap_mode
        if mode == "off" or int(cfg.pairswap_rounds) <= 0 or self.columns < 2:
            return False
        if mode == "always":
            return True
        return condition_number > float(cfg.pairswap_condition_threshold)

    def _admm_weight_solve(self, weight: torch.Tensor, hessian: torch.Tensor) -> Tuple[torch.Tensor, List[_Grid]]:
        """Run Algorithm 1 on one already-prepared ``[out, in]`` weight matrix."""

        cfg = self.qcfg
        rows, columns = weight.shape
        dtype = weight.dtype

        diagonal = torch.diagonal(hessian)
        # Match the GPTQ family's damped objective: the solve owns this
        # Hessian copy, so damp it in place exactly once.
        diagonal.add_(float(cfg.damp_percent) * diagonal.mean())
        safe_diagonal = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal))
        sigma = safe_diagonal.rsqrt()
        column_sigma = sigma.unsqueeze(0)

        # Diagonal preconditioning: W' = W * Sigma^-1 and H' = Sigma H Sigma.
        hessian_scaled = hessian * sigma.unsqueeze(0) * sigma.unsqueeze(1)
        weight_scaled = weight / column_sigma
        hessian_weight = weight_scaled @ hessian_scaled

        # One factorization for the whole solve.  Float64 keeps the
        # preconditioned spectrum accurate; the iteration runs in the weight
        # dtype, like the reference schedule.
        try:
            eigenvalues64, eigenvectors64 = torch.linalg.eigh(hessian_scaled.to(torch.float64))
        except torch._C._LinAlgError as exc:
            raise _AdmmNumericalFailure("eigh_failure", str(exc), iterations=0) from exc
        del hessian_scaled
        if not bool(torch.isfinite(eigenvalues64).all()):
            raise _AdmmNumericalFailure("non_finite_eigenvalues", iterations=0)
        eigenvalues = eigenvalues64.clamp_min(0).to(dtype)
        eigenvectors = eigenvectors64.to(dtype)
        del eigenvalues64, eigenvectors64
        condition_number = self._condition_number(eigenvalues)
        rho_initial, rho_update = admm_penalty_schedule(
            condition_number,
            float(cfg.rho),
            float(cfg.rho_update),
            enabled=bool(cfg.adaptive_penalty),
        )

        grids = self._fit_grids(weight)
        rtn = self._project(weight, grids)
        rtn_grids = grids
        initial_objective = self._objective(weight, rtn, hessian)

        discrete = rtn / column_sigma
        dual = torch.zeros_like(weight_scaled)
        previous = discrete.clone()
        denominator = discrete.square().sum().clamp_min(torch.finfo(dtype).tiny)

        chunk_rows = max(1, int(_WORK_CHUNK_BYTES // max(1, columns * weight.element_size())))
        # A non-finite iterate cannot recover, but scanning every iteration
        # would add a full-tensor reduction to the healthy path. Check on a
        # bounded cadence; the final objective check is the backstop for a
        # NaN introduced between checks.
        check_every = max(1, min(int(cfg.update_iter), 8))
        refresh_attempted = False
        refresh_accepted = False
        refresh_improvement = 0.0
        iterations = 0
        error_history: List[float] = []
        rho = rho_initial
        stop_reason = "max_iter"
        max_iter = int(cfg.max_iter)

        for step in range(max_iter):
            rho = admm_rho_at(step, rho_initial, rho_update, int(cfg.update_iter))
            right_hand_side = hessian_weight + rho * discrete - dual
            continuous = torch.empty_like(weight_scaled)
            check_iterate = step == 0 or (step + 1) % check_every == 0
            for start in range(0, rows, chunk_rows):
                end = min(start + chunk_rows, rows)
                block = right_hand_side[start:end] @ eigenvectors
                block = block / (eigenvalues + rho)
                block = block @ eigenvectors.t()
                # A NaN/Inf iterate makes every later projection and the final
                # objective meaningless. Stop here and use the RTN fallback
                # instead of running the remaining iterations.
                if check_iterate and not bool(torch.isfinite(block).all()):
                    raise _AdmmNumericalFailure(
                        "non_finite_iterate",
                        f"iteration {step + 1}",
                        iterations=step + 1,
                        diagnostics={
                            "condition_number": condition_number,
                            "ill_conditioned": condition_number > ILL_CONDITIONED_THRESHOLD,
                            "rho_initial": rho_initial,
                            "rho_update": rho_update,
                            "rho_final": rho,
                            "refresh_attempts": int(refresh_attempted),
                            "refresh_attempted": refresh_attempted,
                            "refresh_accepted": refresh_accepted,
                            "refresh_improvement": refresh_improvement,
                            "initial_objective": float(initial_objective.item()),
                        },
                    )
                continuous[start:end] = block
            del right_hand_side

            # The projection target is W + V / rho, not the continuous iterate.
            target_scaled = continuous + dual / rho
            target_weight = target_scaled * column_sigma
            projected = self._project(target_weight, grids)
            discrete = projected / column_sigma
            dual = dual + rho * (continuous - discrete)

            iterations = step + 1

            # Stagnation is measured on the normalized reconstruction error,
            # not on the iterate change: a shrinking change is convergence,
            # while a flat error means the fixed grid has stopped helping.
            # Sample it once per penalty update to bound the extra matmul.
            if (step + 1) % int(cfg.update_iter) == 0:
                current_error = self._objective(weight, projected, hessian)
                error_scale = float(initial_objective.item())
                error_rel = (
                    float(current_error.item()) / error_scale if error_scale > 0.0 else float(current_error.item())
                )
                error_history.append(error_rel)

            window = max(2, int(cfg.stagnation_window))
            if (
                cfg.refresh_on_stagnation
                and not refresh_attempted
                and len(error_history) >= window
            ):
                window_values = error_history[-window:]
                first_error = abs(window_values[0])
                spread = window_values[0] - min(window_values)
                flat = first_error == 0.0 or spread / first_error < float(cfg.stagnation_tol)
                if flat:
                    refresh_attempted = True
                    continuous_weight = continuous * column_sigma
                    refreshed = self._fit_grids(continuous_weight)
                    refreshed_projection = self._project(target_weight, refreshed)
                    old_value = float(projection_objective(target_scaled, discrete).item())
                    new_value = float(
                        projection_objective(target_scaled, refreshed_projection / column_sigma).item()
                    )
                    if refresh_accepts(new_value, old_value, float(cfg.refresh_accept_rel)):
                        refresh_accepted = True
                        refresh_improvement = 0.0 if old_value == 0.0 else (old_value - new_value) / old_value
                        refreshed_discrete = refreshed_projection / column_sigma
                        dual = dual + rho * (discrete - refreshed_discrete)
                        discrete = refreshed_discrete
                        grids = refreshed

            # Measure convergence on the iterate that will carry into the
            # next solve, including any accepted grid refresh.
            d_rel = float(((discrete - previous).square().sum() / denominator).item())
            previous = discrete
            if int(cfg.switch_iter) <= iterations < max_iter and d_rel < float(cfg.early_stop_tol):
                stop_reason = "early_stop"
                break

        quantized = discrete * column_sigma
        # Land exactly on the final grid so packing recovers integer codes.
        quantized = self._project(quantized, grids)

        pairswap_ran = self._should_pairswap(condition_number)
        if pairswap_ran:
            quantized, flips = self._pairswap(weight, quantized, hessian, grids)
        else:
            flips = 0

        final_objective = self._objective(weight, quantized, hessian)
        fallback_to_rtn = False
        fallback_reason: Optional[str] = None
        if not bool(torch.isfinite(final_objective)) or final_objective > initial_objective:
            finite_final = bool(torch.isfinite(final_objective))
            fallback_reason = "non_finite_objective" if not finite_final else "worse_than_rtn"
            log.warn(
                "ADMM-Q: module `%s` did not improve on its RTN initialization "
                "(initial=%s final=%s); keeping the RTN projection.",
                self.name,
                float(initial_objective.item()),
                float(final_objective.item()) if finite_final else "nan",
            )
            quantized = rtn
            grids = rtn_grids
            final_objective = initial_objective
            flips = 0
            fallback_to_rtn = True

        self.admmq_diagnostics = {
            "status": "solved",
            "converged": stop_reason == "early_stop",
            "stop_reason": stop_reason,
            "iterations": iterations,
            "max_iter": max_iter,
            "condition_number": condition_number,
            "ill_conditioned": condition_number > ILL_CONDITIONED_THRESHOLD,
            "rho_initial": rho_initial,
            "rho_update": rho_update,
            "rho_final": rho,
            "refresh_attempts": 1 if refresh_attempted else 0,
            "refresh_attempted": refresh_attempted,
            "refresh_accepted": refresh_accepted,
            "refresh_improvement": refresh_improvement,
            "pairswap_ran": pairswap_ran,
            "pairswap_flips": flips,
            "fallback_to_rtn": fallback_to_rtn,
            "fallback_strategy": FallbackStrategy.RTN.value if fallback_to_rtn else None,
            "fallback_reason": fallback_reason,
            "initial_objective": float(initial_objective.item()),
            "final_objective": float(final_objective.item()),
        }
        return quantized, grids

    def _solve_device(self, source_device: torch.device) -> torch.device:
        """Return the device that runs one layer's ADMM-Q solve.

        The solve factorizes the preconditioned Hessian with
        ``torch.linalg.eigh`` in float64. The host and the CUDA namespace
        (NVIDIA CUDA and HIP) provide it; MPS has neither float64 nor ``eigh``,
        and XPU/NPU are not qualified for that factorization, so those layers
        run the solve on the host.
        """

        if source_device.type in _SOLVE_DEVICE_TYPES:
            return source_device
        return torch.device("cpu")

    def _set_skipped_diagnostics(self, reason: str) -> None:
        """Record a module that did not run the ADMM solve."""

        self.admmq_diagnostics = {
            "status": "skipped",
            "reason": reason,
            "converged": False,
            "stop_reason": "skipped",
            "iterations": 0,
            "max_iter": int(self.qcfg.max_iter),
            "condition_number": None,
            "rho_final": None,
            "refresh_attempts": 0,
            "refresh_attempted": False,
            "refresh_accepted": False,
            "refresh_improvement": 0.0,
            "pairswap_ran": False,
            "pairswap_flips": 0,
            "fallback_to_rtn": False,
            "fallback_strategy": None,
            "fallback_reason": reason if reason == "insufficient_calibration" else None,
            "initial_objective": None,
            "final_objective": None,
        }

    def _fallback_quantize(
        self,
        strategy: FallbackStrategy,
        blocksize: int,
        target_device: Optional[torch.device] = None,
        hessian_diag: Optional[torch.Tensor] = None,
    ):
        result = super()._fallback_quantize(
            strategy, blocksize, target_device=target_device, hessian_diag=hessian_diag
        )
        diagnostics = self.admmq_diagnostics
        if diagnostics is not None:
            diagnostics["fallback_to_rtn"] = strategy is FallbackStrategy.RTN
            diagnostics["fallback_strategy"] = strategy.value
        return result

    # ------------------------------------------------------------------
    # GPTQ entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _quantize_impl(self, blocksize: int = 128):
        if isinstance(self.module, nn.Embedding):
            # The embedding Hessian is a token-frequency diagonal; the ADMM
            # split has nothing to gain over the exact per-group RTN projection.
            self._set_skipped_diagnostics("embedding")
            return self._quantize_embedding(blocksize=blocksize)

        from ..utils.fallback import should_use_fallback

        if self.nsamples == 0 or should_use_fallback(
            self.fallback, float(self.nsamples), self.expected_nsamples
        ):
            # GPTQ selects the fallback strategy from the calibration coverage.
            self._set_skipped_diagnostics("insufficient_calibration")
            return super()._quantize_impl(blocksize=blocksize)

        start = time.time()
        target_device = getattr(self.module, "target_device", None)
        result_device = torch.device(self.module.weight.data.device)
        self.finalize_hessian(target_device=target_device)
        self._release_deferred_hessian_sources()
        if self.H is None:
            raise RuntimeError(f"ADMM-Q: module `{self.name}` has no captured Hessian.")

        source_device = self.H.device
        solve_device = self._solve_device(source_device)
        module_copy = self.module_copy
        self.module_copy = None
        inverse_permutation = None
        original_diagonal = None

        def solve():
            nonlocal inverse_permutation, original_diagonal
            # Each attempt owns its diagnostics, including the CPU retry.
            self.admmq_diagnostics = None
            hessian = self.H
            if hessian.device.type == "cuda":
                # Only the diagonal is changed by dead-column handling and
                # damping. Save it once so an allocation retry starts undamped.
                original_diagonal = hessian.diagonal().clone()
            weight = (
                self.clone_module(device=hessian.device)
                if module_copy is None
                else safe_move_to(module_copy, hessian.device, copy=True)
            )
            _neutralize_dead_columns(hessian, weight)
            if self.qcfg.desc_act:
                permutation = torch.argsort(hessian.diagonal(), descending=True)
                hessian = hessian[permutation[:, None], permutation]
                weight = weight[:, permutation]
                inverse_permutation = torch.argsort(permutation)

            quantized, grids = self._admm_weight_solve(weight, hessian)
            loss = self._objective(weight, quantized, hessian) / max(1, self.nsamples)
            if not bool(torch.isfinite(loss)):
                iterations = (
                    None
                    if self.admmq_diagnostics is None
                    else self.admmq_diagnostics.get("iterations")
                )
                raise _AdmmNumericalFailure(
                    "non_finite_loss",
                    f"module `{self.name}` produced a non-finite reconstruction loss",
                    iterations=iterations,
                )
            return quantized, grids, float(loss.item())

        def prepare_cpu(device):
            self.H = safe_move_to(self.H, device)
            if original_diagonal is not None:
                self.H.diagonal().copy_(original_diagonal)

        try:
            if solve_device != source_device:
                log.warn(
                    "Quantization: Module `%s` -> ADMM-Q factorizes the Hessian with a "
                    "float64 eigendecomposition that is not qualified on `%s`; running "
                    "the solve on CPU. Due to this fallback, the calculation may take "
                    "much longer than normal.",
                    self.name,
                    source_device,
                )
                self.H = safe_move_to(self.H, solve_device)
                source_device = solve_device
            try:
                (quantized, grids, avg_loss), _ = self._run_with_accelerator_oom_cpu_fallback(
                    solve, source_device=source_device, stage="ADMM-Q solve", prepare_cpu=prepare_cpu
                )
            except (_AdmmNumericalFailure, torch._C._LinAlgError) as exc:
                # A failed factorization, a non-finite iterate or a non-finite
                # reconstruction loss is a property of this layer's Hessian or
                # weights. Keep the GPTQ family's RTN quantization for the
                # layer instead of aborting the whole model.
                reason = exc.reason if isinstance(exc, _AdmmNumericalFailure) else "eigh_failure"
                log.warn(
                    "ADMM-Q: module `%s` -> %s; using the GPTQ RTN fallback instead of aborting.",
                    self.name,
                    exc,
                )
                if self.admmq_diagnostics is None:
                    self.admmq_diagnostics = {
                        "status": "solved",
                        "converged": False,
                        "stop_reason": "fallback",
                        "iterations": (
                            exc.iterations if isinstance(exc, _AdmmNumericalFailure) else None
                        ),
                        "max_iter": int(self.qcfg.max_iter),
                        "condition_number": None,
                        "rho_final": None,
                        "refresh_attempts": 0,
                        "refresh_attempted": False,
                        "refresh_accepted": False,
                        "refresh_improvement": None,
                        "pairswap_ran": False,
                        "pairswap_flips": 0,
                        "fallback_reason": reason,
                        "initial_objective": None,
                        "final_objective": None,
                    }
                    if isinstance(exc, _AdmmNumericalFailure) and exc.diagnostics is not None:
                        self.admmq_diagnostics.update(exc.diagnostics)
                else:
                    self.admmq_diagnostics["fallback_reason"] = reason
                    self.admmq_diagnostics["converged"] = False
                    self.admmq_diagnostics["stop_reason"] = "fallback"
                result = self._rtn_fallback_after_hessian_failure(blocksize)
                if not bool(torch.isfinite(result[0]).all()):
                    raise ValueError(
                        f"ADMM-Q: module `{self.name}` -> the RTN fallback also produced "
                        "non-finite weights; the captured Hessian or source weights are not finite."
                    ) from exc
                return result
        finally:
            self.H = None

        if inverse_permutation is not None:
            quantized = quantized[:, inverse_permutation]

        if self._tp_pad_cols:
            quantized = quantized[:, : self._original_columns]

        quantized = self._narrow_reconstruction(quantized)

        scale = torch.cat([grid.scale for grid in grids], dim=1)
        zero = torch.cat([grid.zero for grid in grids], dim=1)
        effective_group = self.columns if int(self.qcfg.group_size) <= 0 else int(self.qcfg.group_size)
        g_idx = torch.arange(self.columns, dtype=torch.int32, device=quantized.device) // effective_group
        if inverse_permutation is not None:
            g_idx = g_idx[inverse_permutation]
        if self._tp_pad_cols:
            g_idx = g_idx[: self._original_columns]
            scale = self.truncate_last_dim(scale, self._original_columns)
            zero = self.truncate_last_dim(zero, self._original_columns)

        scale, zero = self._sanitize_scale_and_log(scale, zero)

        quantized = safe_move_to(quantized, result_device)
        return (
            quantized,
            scale,
            zero,
            g_idx,
            time.time() - start,
            avg_loss,
            float(self.qcfg.damp_percent),
            self.nsamples,
        )


__all__ = [
    "ADMMQ",
    "ILL_CONDITIONED_RHO",
    "ILL_CONDITIONED_RHO_UPDATE",
    "ILL_CONDITIONED_SEVERE_RHO",
    "ILL_CONDITIONED_SEVERE_THRESHOLD",
    "ILL_CONDITIONED_THRESHOLD",
    "admm_penalty_schedule",
    "admm_rho_at",
    "pairswap_delta",
    "projection_objective",
    "refresh_accepts",
]
