# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""JointQ: joint scale-and-integer refinement of a GPTQ solution.

Clean-room implementation of the method described in

    Keiji Kimura and Yuma Ichikawa,
    "JointQ: Joint Scale-and-Integer Quantization for Large Language Models",
    arXiv:2611.20384 (2026), Fujitsu Limited.

No code was copied from the authors' reference implementation. The
implementation keeps the GPTQ checkpoint format: only the centered integers
and the per-group scales change, so packing, loading and inference kernels are
untouched.

The refinement keeps two warm-start candidates (a clipping-based solution and
the GPTQ solution itself), refits every candidate's group scales jointly with
its integers fixed, selects the better candidate per output row, and then runs
a block-coordinate local search. A candidate move changes one integer and is
scored with the closed-form best-response scale of the affected group, while
the other groups enter the score through the full Hessian.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule
from ..utils.fallback import resolve_fallback_strategy, should_use_fallback
from ..utils.logger import setup_logger
from ..utils.p2p import safe_move_to
from .config import QuantizeConfig
from .gptq import GPTQ
from .quantizer import SCALE_STORAGE_FLOOR, _neutralize_dead_columns


log = setup_logger()


#: Layer-type-specific regularization strengths from Table 6 of the paper.
#: ``None`` as the second key component means per-channel quantization.
_PAPER_REGULARIZATION = {
    (4, 128): {
        "q_proj": 0.0,
        "k_proj": 0.0,
        "v_proj": 0.01,
        "o_proj": 0.0,
        "gate_proj": 0.01,
        "up_proj": 0.01,
        "down_proj": 0.0,
    },
    (4, None): {
        "q_proj": 0.0,
        "k_proj": 0.0,
        "v_proj": 0.15,
        "o_proj": 0.0,
        "gate_proj": 0.15,
        "up_proj": 0.15,
        "down_proj": 0.01,
    },
    (3, 128): {
        "q_proj": 0.01,
        "k_proj": 0.01,
        "v_proj": 0.01,
        "o_proj": 0.0,
        "gate_proj": 0.01,
        "up_proj": 0.01,
        "down_proj": 0.0,
    },
    (3, None): {
        "q_proj": 0.15,
        "k_proj": 0.15,
        "v_proj": 0.15,
        "o_proj": 0.15,
        "gate_proj": 0.15,
        "up_proj": 0.15,
        "down_proj": 0.01,
    },
    (2, 32): {
        "q_proj": 0.1,
        "k_proj": 0.1,
        "v_proj": 0.1,
        "o_proj": 0.0,
        "gate_proj": 0.0,
        "up_proj": 0.0,
        "down_proj": 0.0,
    },
}

#: Largest FP16 magnitude, used for the paper's scale bound ``65504 / L``.
_FP16_MAX = 65504.0

#: Smallest positive scale the GPTQ checkpoint layout can store.
_STORAGE_FLOOR = SCALE_STORAGE_FLOOR


def paper_regularization(
    bits: int,
    group_size: int,
    columns: int,
    layer_name: str,
) -> float:
    """Return the paper's layer-type strength, or ``0`` when untuned."""

    per_channel = group_size == -1 or group_size >= columns
    key = (bits, None) if per_channel else (bits, group_size)
    table = _PAPER_REGULARIZATION.get(key)
    if table is None:
        return 0.0
    return table.get(layer_name.rsplit(".", 1)[-1], 0.0)


@dataclass(frozen=True)
class _GroupPlan:
    """Column layout of one layer's quantization groups."""

    columns: Tuple[torch.Tensor, ...]
    g_idx: torch.Tensor
    num_groups: int
    size: int

    @property
    def device(self) -> torch.device:
        return self.g_idx.device

    def expand(self, scales: torch.Tensor) -> torch.Tensor:
        """Broadcast one scale per group to one scale per input column."""

        return scales.index_select(1, self.g_idx)


def _build_group_plan(g_idx: torch.Tensor, num_groups: int) -> _GroupPlan:
    """Group columns by their ``g_idx`` value.

    Grouping by the returned ``g_idx`` (instead of assuming contiguous
    natural-order groups) keeps activation ordering and group-aware ordering
    correct without replaying either permutation.
    """

    if g_idx.dim() != 1:
        raise ValueError("JointQ: `g_idx` must be one-dimensional.")
    if num_groups < 1:
        raise ValueError("JointQ: the quantizer returned no groups.")
    if g_idx.numel() == 0:
        raise ValueError("JointQ: the quantizer returned an empty column layout.")
    g_idx = g_idx.to(torch.int64)
    if int(g_idx.min()) < 0 or int(g_idx.max()) >= num_groups:
        raise ValueError(
            f"JointQ: `g_idx` values must lie in [0, {num_groups}); "
            f"observed [{int(g_idx.min())}, {int(g_idx.max())}]."
        )
    counts = torch.bincount(g_idx, minlength=num_groups)
    if int(counts.sum()) != g_idx.numel():
        raise ValueError("JointQ: `g_idx` does not cover every input column.")
    if int((counts == 0).sum()) != 0:
        raise ValueError("JointQ: `g_idx` leaves an empty group.")
    order = torch.argsort(g_idx, stable=True)
    offsets = torch.cumsum(counts, 0) - counts
    columns = tuple(
        order[int(offset): int(offset) + int(count)]
        for offset, count in zip(offsets.tolist(), counts.tolist())
    )
    return _GroupPlan(
        columns=columns,
        g_idx=g_idx,
        num_groups=num_groups,
        size=int(counts.max()),
    )


def _codes_from_reconstruction(
    reconstruction: torch.Tensor,
    scales: torch.Tensor,
    zero: float,
    plan: _GroupPlan,
    max_code: int,
) -> torch.Tensor:
    """Recover the integer codes the packer will store.

    The packer never sees the quantizer's integers: it rebuilds them from the
    narrowed reconstruction with ``round(W / scale + zero)``. Recovering them
    the same way, from the same stored scales, keeps the JointQ result and the
    packed checkpoint consistent.
    """

    scale_cols = plan.expand(scales)
    # A zero scale carries no information; pin its lane to the fixed zero
    # point so the reconstruction stays exactly zero.
    zero_scale = scale_cols == 0
    safe_scale = torch.where(zero_scale, torch.ones_like(scale_cols), scale_cols)
    # The packer rebuilds integers from ``(W + zero * scale) / scale`` in
    # FP32, not from ``W / scale + zero``. Mirror that arithmetic so a value
    # sitting on a rounding boundary cannot land on a different code in the
    # checkpoint than in the replay weight.
    codes = torch.round(
        (reconstruction.float() + zero * safe_scale) / safe_scale
    )
    codes = codes.clamp_(0, max_code).to(torch.int64)
    return torch.where(zero_scale, torch.full_like(codes, int(zero)), codes)


def _dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zero: float,
    plan: _GroupPlan,
) -> torch.Tensor:
    centered = codes.to(scales.dtype) - zero
    return centered * plan.expand(scales)


def _row_objectives(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    hessian: torch.Tensor,
    *,
    rows: Optional[torch.Tensor] = None,
    chunk_rows: int = 512,
) -> torch.Tensor:
    """Return ``phi_i = (w_i - w_hat_i) H (w_i - w_hat_i)^T`` per row."""

    if rows is not None:
        weight = weight.index_select(0, rows)
        reconstruction = reconstruction.index_select(0, rows)
    total = weight.shape[0]
    out = torch.empty(total, dtype=weight.dtype, device=weight.device)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        diff = weight[start:stop] - reconstruction[start:stop]
        out[start:stop] = (diff @ hessian).mul_(diff).sum(dim=1)
    return out


def _integer_search(
    group_energy: torch.Tensor,
    codes: torch.Tensor,
    hessian_block: torch.Tensor,
    *,
    zero: float,
    max_code: int,
    tolerance: float,
    stall_fraction: float,
    iteration_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best-improvement ``+-1`` search on one group for every row.

    ``codes`` are raw codes in ``[0, max_code]``; the paper works with the
    centered integers ``q = code - zero``. ``group_energy`` is the group slice
    of ``H (w - w_hat) + H_t w_hat`` during the layer search, or ``H_t w`` for
    the group-local warm start. Every candidate move is scored with the
    closed-form best-response scale of the group.
    """

    dtype = group_energy.dtype
    rows = codes.shape[0]
    size = codes.shape[1]
    work = codes.to(dtype) - zero
    cache = work @ hessian_block
    beta = (work * group_energy).sum(dim=1)
    gamma = (work * cache).sum(dim=1)
    kappa = torch.where(gamma > tolerance, beta.square() / gamma, torch.zeros_like(beta))
    diag = hessian_block.diagonal()
    cap = int(iteration_cap) if iteration_cap else 10 * size
    for _ in range(cap):
        beta_plus = beta.unsqueeze(1) + group_energy
        gamma_plus = gamma.unsqueeze(1) + 2.0 * cache + diag.unsqueeze(0)
        ok_plus = (codes < max_code) & (gamma_plus > tolerance)
        kappa_plus = torch.where(
            ok_plus,
            beta_plus.square() / gamma_plus.clamp_min(tolerance),
            torch.zeros_like(beta_plus),
        )
        beta_minus = beta.unsqueeze(1) - group_energy
        gamma_minus = gamma.unsqueeze(1) - 2.0 * cache + diag.unsqueeze(0)
        ok_minus = (codes > 0) & (gamma_minus > tolerance)
        kappa_minus = torch.where(
            ok_minus,
            beta_minus.square() / gamma_minus.clamp_min(tolerance),
            torch.zeros_like(beta_minus),
        )
        best_plus, index_plus = kappa_plus.max(dim=1)
        best_minus, index_minus = kappa_minus.max(dim=1)
        # Ties prefer the +1 move, as Algorithm 2 does.
        take_plus = best_plus >= best_minus
        best = torch.where(take_plus, best_plus, best_minus)
        improvement = best - kappa
        if int((improvement < tolerance).sum()) > stall_fraction * rows:
            break
        move = improvement > tolerance
        if not bool(move.any()):
            break
        move_rows = move.nonzero(as_tuple=False).squeeze(1)
        index = torch.where(take_plus, index_plus, index_minus).index_select(0, move_rows)
        delta = torch.where(take_plus, 1, -1).to(torch.int64).index_select(0, move_rows)
        codes[move_rows, index] += delta
        chosen_beta = torch.where(
            take_plus,
            beta_plus.gather(1, index_plus.unsqueeze(1)).squeeze(1),
            beta_minus.gather(1, index_minus.unsqueeze(1)).squeeze(1),
        )
        chosen_gamma = torch.where(
            take_plus,
            gamma_plus.gather(1, index_plus.unsqueeze(1)).squeeze(1),
            gamma_minus.gather(1, index_minus.unsqueeze(1)).squeeze(1),
        )
        chosen_kappa = torch.where(
            take_plus,
            kappa_plus.gather(1, index_plus.unsqueeze(1)).squeeze(1),
            kappa_minus.gather(1, index_minus.unsqueeze(1)).squeeze(1),
        )
        beta[move_rows] = chosen_beta.index_select(0, move_rows)
        gamma[move_rows] = chosen_gamma.index_select(0, move_rows)
        kappa[move_rows] = chosen_kappa.index_select(0, move_rows)
        cache[move_rows] += (
            delta.to(dtype).unsqueeze(1)
            * hessian_block.index_select(1, index).T
        )
    # Recompute the accumulated scalars from the final integers, as the paper
    # does, so incremental rounding cannot drift the returned scale.
    work = codes.to(dtype) - zero
    cache = work @ hessian_block
    beta = (work * group_energy).sum(dim=1)
    gamma = (work * cache).sum(dim=1)
    sigma = torch.where(gamma > tolerance, beta / gamma, torch.zeros_like(beta))
    return codes, sigma


def _solve_scales(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    *,
    scale_bound: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the per-row group-scale systems with the paper's damping fallback.

    Rows that factor cleanly keep their undamped solution; rows that are
    singular or violate the FP16 scale bound are re-solved with the paper's
    diagonal damping. Rows that still fail keep the caller's previous scale,
    which the returned ``solved`` mask reports.
    """

    size = gram.shape[-1]
    rows = gram.shape[0]
    eye = torch.eye(size, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    failed_all = torch.zeros(rows, dtype=torch.bool, device=gram.device)

    def solve(matrix: torch.Tensor):
        try:
            solution, info = torch.linalg.solve_ex(matrix, rhs.unsqueeze(-1))
            solution = solution.squeeze(-1)
            ok = info == 0
        except AttributeError:  # older torch without solve_ex
            solution = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
            ok = torch.ones(rows, dtype=torch.bool, device=gram.device)
        except RuntimeError:
            return None, failed_all
        ok = (
            ok
            & torch.isfinite(solution).all(dim=1)
            & (solution.abs() <= scale_bound).all(dim=1)
        )
        return solution, ok

    solution, solved = solve(gram)
    if solution is None:
        solution = torch.zeros_like(rhs)
    if not bool(solved.all()):
        damping = (0.01 * gram.diagonal(dim1=1, dim2=2).amax(dim=1)).clamp_min(1e-6)
        damped = gram + damping.unsqueeze(1).unsqueeze(2) * eye
        damped_solution, damped_solved = solve(damped)
        if damped_solution is not None:
            solution = torch.where(solved.unsqueeze(1), solution, damped_solution)
            solved = solved | damped_solved
    return solution, solved


def _joint_scale_solve(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    hessian: torch.Tensor,
    plan: _GroupPlan,
    zero: float,
    *,
    tolerance: float,
    nsamples: int,
    scale_bound: float,
    row_chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-solve all group scales of the given rows with their integers fixed."""

    dtype = weight.dtype
    total = weight.shape[0]
    num_groups = plan.num_groups
    columns = plan.columns
    out_scales = scales.clone()
    out_phi = torch.empty(total, dtype=dtype, device=weight.device)
    for start in range(0, total, row_chunk):
        stop = min(start + row_chunk, total)
        weight_chunk = weight[start:stop]
        codes_chunk = codes[start:stop]
        scale_chunk = scales[start:stop]
        centered = codes_chunk.to(dtype) - zero
        centered_groups = [centered.index_select(1, cols) for cols in columns]
        reconstruction = _dequantize(codes_chunk, scale_chunk, zero, plan)
        phi_old = _row_objectives(weight_chunk, reconstruction, hessian)
        hessian_weight = weight_chunk @ hessian  # [rows, p]
        rhs = torch.empty((stop - start, num_groups), dtype=dtype, device=weight.device)
        gram = torch.empty(
            (stop - start, num_groups, num_groups), dtype=dtype, device=weight.device
        )
        for group_index, group_columns in enumerate(columns):
            rhs[:, group_index] = (
                centered_groups[group_index]
                * hessian_weight.index_select(1, group_columns)
            ).sum(dim=1)
            projection = centered_groups[group_index] @ hessian.index_select(0, group_columns)
            for other_index, other_columns in enumerate(columns):
                gram[:, group_index, other_index] = (
                    projection.index_select(1, other_columns)
                    * centered_groups[other_index]
                ).sum(dim=1)
        active = torch.stack(
            [group.ne(0).any(dim=1) for group in centered_groups], dim=1
        )
        gram = gram * active.unsqueeze(2) * active.unsqueeze(1)
        gram = gram + torch.diag_embed((~active).to(dtype))
        rhs = rhs * active.to(dtype)
        solution, solved = _solve_scales(gram, rhs, scale_bound=scale_bound)
        candidate = torch.where(active & solved.unsqueeze(1), solution, scale_chunk)
        # An all-zero group has no reconstruction regardless of its scale;
        # zero it here and let the storage pass install the representable
        # placeholder.
        candidate = torch.where(active, candidate, torch.zeros_like(candidate))
        candidate_reconstruction = _dequantize(codes_chunk, candidate, zero, plan)
        phi_new = _row_objectives(weight_chunk, candidate_reconstruction, hessian)
        accept = phi_new < phi_old - nsamples * tolerance
        out_scales[start:stop] = torch.where(accept.unsqueeze(1), candidate, scale_chunk)
        out_phi[start:stop] = torch.where(accept, phi_new, phi_old)
    return out_scales, out_phi


def _clip_candidate(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    plan: _GroupPlan,
    zero: float,
    max_code: int,
    *,
    tolerance: float,
    stall_fraction: float,
    iteration_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's Clip-Optimize candidate, one group at a time."""

    dtype = weight.dtype
    rows, columns = weight.shape
    codes = torch.zeros((rows, columns), dtype=torch.int64, device=weight.device)
    scales = torch.zeros((rows, plan.num_groups), dtype=dtype, device=weight.device)
    for group_index, group_columns in enumerate(plan.columns):
        group_weight = weight.index_select(1, group_columns)
        scale = group_weight.abs().amax(dim=1) / max(zero - 1.0, 1.0)
        scale = scale.clamp_min(1e-8)
        initial = torch.round(group_weight / scale.unsqueeze(1)).clamp_(-zero, zero - 1)
        initial = (initial + zero).to(torch.int64)
        hessian_block = hessian.index_select(0, group_columns).index_select(1, group_columns)
        energy = group_weight @ hessian_block.T
        group_codes, group_scale = _integer_search(
            energy,
            initial,
            hessian_block,
            zero=zero,
            max_code=max_code,
            tolerance=tolerance,
            stall_fraction=stall_fraction,
            iteration_cap=iteration_cap,
        )
        codes[:, group_columns] = group_codes
        scales[:, group_index] = group_scale
    return codes, scales


def _local_search(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    plan: _GroupPlan,
    zero: float,
    max_code: int,
    *,
    tolerance: float,
    stall_fraction: float,
    active_fraction: float,
    re_solve_low: float,
    re_solve_high: float,
    iteration_cap: int,
    max_passes: int,
    nsamples: int,
    scale_bound: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run the paper's block-coordinate layer search."""

    rows = weight.shape[0]
    num_groups = plan.num_groups
    reconstruction = _dequantize(codes, scales, zero, plan)
    phi = _row_objectives(weight, reconstruction, hessian)
    active = torch.ones(rows, dtype=torch.bool, device=weight.device)
    passes = 0
    for _ in range(max_passes):
        passes += 1
        previous_phi = phi.clone()
        mu = torch.zeros(rows, dtype=torch.int64, device=weight.device)
        for group_index, group_columns in enumerate(plan.columns):
            hessian_block = hessian.index_select(0, group_columns).index_select(1, group_columns)
            reconstruction = _dequantize(codes, scales, zero, plan)
            difference = weight - reconstruction
            energy = difference @ hessian.index_select(1, group_columns)
            energy = energy + reconstruction.index_select(1, group_columns) @ hessian_block
            group_codes, group_scale = _integer_search(
                energy,
                codes.index_select(1, group_columns),
                hessian_block,
                zero=zero,
                max_code=max_code,
                tolerance=tolerance,
                stall_fraction=stall_fraction,
                iteration_cap=iteration_cap,
            )
            current = codes.index_select(1, group_columns)
            # Algorithm 1 installs the returned pair and counts the row when
            # either member changes, so a scale-only update still contributes
            # to the joint re-solve selection.
            changed = (group_codes != current).any(dim=1) | (
                group_scale != scales[:, group_index]
            )
            if bool(changed.any()):
                changed_rows = changed.nonzero(as_tuple=False).squeeze(1)
                codes[changed_rows.unsqueeze(1), group_columns.unsqueeze(0)] = (
                    group_codes.index_select(0, changed_rows)
                )
                scales[changed_rows, group_index] = group_scale.index_select(0, changed_rows)
                mu[changed_rows] += 1
                updated = _dequantize(
                    codes.index_select(0, changed_rows),
                    scales.index_select(0, changed_rows),
                    zero,
                    plan,
                )
                phi[changed_rows] = _row_objectives(
                    weight.index_select(0, changed_rows), updated, hessian
                )
        selected = active & (
            (mu > re_solve_high * num_groups) | (mu < re_solve_low * num_groups)
        )
        if bool(selected.any()):
            selected_rows = selected.nonzero(as_tuple=False).squeeze(1)
            new_scales, new_phi = _joint_scale_solve(
                weight.index_select(0, selected_rows),
                codes.index_select(0, selected_rows),
                scales.index_select(0, selected_rows),
                hessian,
                plan,
                zero,
                tolerance=tolerance,
                nsamples=nsamples,
                scale_bound=scale_bound,
            )
            scales[selected_rows] = new_scales
            phi[selected_rows] = new_phi
        active = phi < previous_phi - nsamples * tolerance
        if int(active.sum()) < active_fraction * rows:
            break
    return codes, scales, phi, passes


class JointQ(GPTQ):
    """GPTQ refinement that jointly optimizes group scales and integers."""

    needs_module_output = False

    def __init__(
        self,
        module: NamedModule,
        qcfg: Optional[QuantizeConfig] = None,
        region_timer=None,
    ):
        super().__init__(module, qcfg, region_timer=region_timer)
        config = self.qcfg.jointq
        if config is None:
            raise ValueError("JointQ requires `jointq` in the quantization config.")
        if self.qcfg.gptaq is not None or self.qcfg.foem is not None:
            raise ValueError("JointQ cannot be combined with GPTAQ or FOEM.")
        if isinstance(self.module, nn.Embedding):
            raise ValueError("JointQ supports linear projections only.")
        if self._tp_pad_cols:
            raise ValueError("JointQ does not support tensor-parallel padded projections.")
        if self.qcfg.static_groups:
            raise ValueError("JointQ does not support `static_groups=True`.")
        if not self.qcfg.sym:
            raise ValueError("JointQ supports symmetric quantization only (`sym=True`).")
        self.jointq_diagnostics = {}

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def _paper_setting(self) -> Tuple[int, int, bool]:
        bits = int(self.qcfg.bits)
        group_size = int(self.qcfg.group_size)
        per_channel = group_size == -1 or group_size >= self.columns
        return bits, group_size, per_channel

    def _resolve_regularization(self) -> float:
        setting = self.qcfg.jointq.regularization
        if not isinstance(setting, str):
            return float(setting)
        bits, group_size, _ = self._paper_setting()
        return paper_regularization(bits, group_size, self.columns, self.name)

    def _retry_enabled(self, regularization: float) -> bool:
        config = self.qcfg.jointq
        if regularization != 0.0:
            return False
        if config.adaptive_retry is not None:
            return bool(config.adaptive_retry)
        # The paper retries every layer assigned ``lambda == 0`` and disables
        # the retry for its 2-bit setting.
        bits, _, _ = self._paper_setting()
        return bits >= 3

    @staticmethod
    def _work_dtype(config, device: torch.device) -> torch.dtype:
        if config.dtype == "float32":
            return torch.float32
        # Allocation alone does not prove that the device implements FP64
        # linear algebra, so probe the operation the solver actually calls.
        probe = torch.eye(2, dtype=torch.float64, device=device)
        try:
            torch.linalg.solve(probe, probe)
        except (RuntimeError, TypeError, NotImplementedError):
            log.warn(
                "Quantization: JointQ -> float64 linear algebra is unavailable on `%s`; using float32.",
                device,
            )
            return torch.float32
        return torch.float64

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def quantize(self, blocksize: int = 128):
        config = self.qcfg.jointq
        start = time.time()
        module_device = getattr(self.module, "target_device", None)
        hessian = self.finalize_hessian(target_device=module_device)
        below_threshold = should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        if self.nsamples == 0 or hessian is None or below_threshold:
            strategy = resolve_fallback_strategy(self.fallback)
            log.warn(
                "Quantization: Module `%s` -> JointQ has no usable Hessian; "
                "using `%s` fallback quantization.",
                self.name,
                strategy.value,
            )
            return self._fallback_quantize(strategy, blocksize)

        weight = self.clone_module(device=hessian.device)
        dead_columns = _neutralize_dead_columns(hessian, weight)
        regularization = self._resolve_regularization()
        result = self._run(
            weight,
            hessian,
            regularization,
            blocksize,
            start,
            config,
        )
        if self._retry_enabled(regularization):
            relative_error = self._relative_weight_error(result[0], weight)
            self.jointq_diagnostics["relative_error"] = relative_error
            if relative_error >= config.retry_ratio:
                log.info(
                    "Quantization: Module `%s` -> JointQ retry with regularization %s "
                    "(relative weight error %.4f).",
                    self.name,
                    config.retry_regularization,
                    relative_error,
                )
                result = self._run(
                    weight,
                    hessian,
                    float(config.retry_regularization),
                    blocksize,
                    start,
                    config,
                )
                self.jointq_diagnostics["retried"] = True
        self.jointq_diagnostics["dead_columns"] = int(dead_columns.sum())

        quantized, scales, zeros, g_idx, duration, avg_loss, damp, nsamples = result
        quantized = safe_move_to(quantized, torch.device(self.module.weight.data.device))
        # Match GPTQ's return contract: the grid tensors live on the compute
        # device, while the replay weight lives on the module device.
        grid_device = hessian.device
        scales = safe_move_to(scales, grid_device)
        zeros = safe_move_to(zeros, grid_device)
        g_idx = safe_move_to(g_idx, grid_device)
        return quantized, scales, zeros, g_idx, duration, avg_loss, damp, nsamples

    def _run(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        regularization: float,
        blocksize: int,
        start: float,
        config,
    ):
        module_device = hessian.device
        requested = config.device
        if isinstance(requested, str) and requested == "auto":
            solver_device = module_device
        else:
            solver_device = torch.device(requested)

        # The GPTQ candidate consumes the regularized Hessian with its own
        # damping. GPTQ mutates the matrix it is given, so it gets a private
        # float32 copy on the module's device. The neutralized weight is
        # handed over so the dead columns stay zero through the block pass.
        gptq_hessian = self._regularized_hessian(hessian, regularization, torch.float32)
        self.H = gptq_hessian
        # GPTQ error-propagates into the weight it is given, so the warm start
        # gets its own copy: the refinement and any adaptive retry must keep
        # the pristine reference weight.
        self.module_copy = weight.clone()
        try:
            candidate = super().quantize(blocksize=blocksize)
        finally:
            self.H = None
            self.module_copy = None

        work_dtype = self._work_dtype(config, solver_device)
        solver_hessian = self._regularized_hessian(hessian, regularization, work_dtype)
        if solver_device != solver_hessian.device:
            solver_hessian = safe_move_to(solver_hessian, solver_device, copy=True)
        solver_weight = weight.to(dtype=work_dtype, device=solver_device)
        return self._refine(
            solver_weight,
            hessian,
            solver_hessian,
            candidate,
            regularization,
            start,
        )

    def _regularized_hessian(
        self,
        hessian: torch.Tensor,
        regularization: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        config = self.qcfg.jointq
        result = hessian.to(dtype=dtype)
        if result.data_ptr() == hessian.data_ptr():
            result = result.clone()
        if regularization > 0.0:
            diagonal = hessian.diagonal()
            mean = diagonal.mean()
            if float(mean) > 0.0:
                rho = (diagonal / mean).clamp_min(0.0).pow(config.eta)
                result.diagonal().add_(
                    rho.to(dtype) * (float(self.nsamples) * float(regularization))
                )
        return result

    def _refine(
        self,
        weight: torch.Tensor,
        raw_hessian: torch.Tensor,
        hessian: torch.Tensor,
        candidate,
        regularization: float,
        start: float,
    ):
        config = self.qcfg.jointq
        reconstruction_gptq, scales_gptq, zeros, g_idx, _, _, damp, nsamples = candidate
        bits = int(self.qcfg.bits)
        max_code = (1 << bits) - 1
        zero = float(1 << (bits - 1))
        device = weight.device
        if not torch.allclose(
            zeros.to(device=device),
            torch.full_like(zeros.to(device=device), zero),
        ):
            raise ValueError("JointQ requires the symmetric fixed zero point.")

        g_idx = g_idx.to(device=device)
        num_groups = int(scales_gptq.shape[1])
        plan = _build_group_plan(g_idx, num_groups)
        work_dtype = hessian.dtype

        # The packer rebuilds the integers from the returned FP32 scales and
        # only then casts them to FP16 for the checkpoint, so the warm start
        # keeps GPTQ's computed scales and merely verifies that they can be
        # stored.
        stored_gptq = scales_gptq.to(device=device).to(torch.float32)
        gptq_stored_fp16 = stored_gptq.to(torch.float16)
        if (
            not bool(torch.isfinite(stored_gptq).all())
            or not bool(torch.isfinite(gptq_stored_fp16).all())
            or bool(((gptq_stored_fp16 == 0) & (stored_gptq != 0)).any())
        ):
            raise ValueError(
                f"JointQ: module `{self.name}` returned GPTQ scales that cannot be "
                "stored in the FP16 checkpoint layout."
            )
        codes_gptq = _codes_from_reconstruction(
            reconstruction_gptq.to(device=device), stored_gptq, zero, plan, max_code
        )

        tolerance = float(config.tolerance)
        iteration_cap = (
            int(config.max_iterations_per_group)
            if config.max_iterations_per_group
            else 10 * plan.size
        )
        scale_bound = _FP16_MAX / zero

        codes_clip, scales_clip = _clip_candidate(
            weight,
            hessian,
            plan,
            zero,
            max_code,
            tolerance=tolerance,
            stall_fraction=float(config.inner_stall_fraction),
            iteration_cap=iteration_cap,
        )
        scales_gptq, phi_gptq = _joint_scale_solve(
            weight,
            codes_gptq,
            stored_gptq.to(dtype=work_dtype),
            hessian,
            plan,
            zero,
            tolerance=tolerance,
            nsamples=nsamples,
            scale_bound=scale_bound,
        )
        scales_clip, phi_clip = _joint_scale_solve(
            weight,
            codes_clip,
            scales_clip,
            hessian,
            plan,
            zero,
            tolerance=tolerance,
            nsamples=nsamples,
            scale_bound=scale_bound,
        )
        # Prefer the clipping candidate on ties, as the paper does.
        choose_clip = phi_clip <= phi_gptq
        codes = torch.where(choose_clip.unsqueeze(1), codes_clip, codes_gptq)
        scales = torch.where(choose_clip.unsqueeze(1), scales_clip, scales_gptq)

        codes, scales, _, passes = _local_search(
            weight,
            hessian,
            codes,
            scales,
            plan,
            zero,
            max_code,
            tolerance=tolerance,
            stall_fraction=float(config.inner_stall_fraction),
            active_fraction=float(config.outer_active_fraction),
            re_solve_low=float(config.re_solve_low),
            re_solve_high=float(config.re_solve_high),
            iteration_cap=iteration_cap,
            max_passes=int(config.max_passes),
            nsamples=nsamples,
            scale_bound=scale_bound,
        )

        codes, final_scales = self._finalize_storage(
            codes,
            scales,
            codes_gptq,
            stored_gptq,
            plan,
            zero,
            max_code,
        )

        reconstruction = _dequantize(codes, final_scales, zero, plan)
        quantized = reconstruction.to(dtype=self.module.weight.dtype)
        if not bool(torch.isfinite(quantized).all()):
            raise ValueError(
                f"JointQ: module `{self.name}` produced a reconstruction that is not "
                f"finite in `{self.module.weight.dtype}`."
            )
        # Adopt exactly the codes the packer will recover from the stored
        # weights so replay and the checkpoint cannot disagree.
        recovered = _codes_from_reconstruction(quantized, final_scales, zero, plan, max_code)
        if not torch.equal(recovered, codes):
            codes = recovered
            reconstruction = _dequantize(codes, final_scales, zero, plan)
            quantized = reconstruction.to(dtype=self.module.weight.dtype)

        if isinstance(self.module, transformers.Conv1D):
            quantized = quantized.t()
        if quantized.shape != self.module.weight.shape:
            quantized = quantized.reshape(self.module.weight.shape)

        raw_device = raw_hessian.device
        phi_raw = _row_objectives(
            weight.to(device=raw_device, dtype=torch.float32),
            quantized.to(device=raw_device, dtype=torch.float32),
            raw_hessian.to(torch.float32),
        )
        avg_loss = float(0.5 * phi_raw.sum().item() / max(int(nsamples), 1))
        self.jointq_diagnostics.update(
            {
                "regularization": float(regularization),
                "passes": passes,
                "candidate": "clip" if bool(choose_clip.any()) else "gptq",
                "clip_rows": int(choose_clip.sum()),
                "avg_loss": avg_loss,
            }
        )
        duration = time.time() - start
        return (
            quantized,
            final_scales,
            zeros.to(device=device),
            g_idx,
            duration,
            avg_loss,
            damp,
            nsamples,
        )

    @staticmethod
    def _finalize_storage(
        codes: torch.Tensor,
        scales: torch.Tensor,
        fallback_codes: torch.Tensor,
        fallback_scales: torch.Tensor,
        plan: _GroupPlan,
        zero: float,
        max_code: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Install FP16-representable scales without desynchronizing codes.

        A group whose scale is zero has no reconstruction; its codes are
        pinned to the fixed zero point so the placeholder scale installed
        below still decodes to exactly zero. A group whose refined scale
        cannot be stored falls back to the GPTQ candidate for that row and
        group, which is always representable because GPTQ wrote it.

        The returned scales stay in their computed FP32 precision. The
        packer recovers the integers from these values and only then casts
        the scales to FP16 for the checkpoint, so rounding here would make
        the solver optimize a different grid than the one that is stored.
        """

        codes = codes.clone()
        zero_scale = scales == 0
        if bool(zero_scale.any()):
            for group_index, group_columns in enumerate(plan.columns):
                rows = zero_scale[:, group_index].nonzero(as_tuple=False).squeeze(1)
                if rows.numel() == 0:
                    continue
                codes[rows.unsqueeze(1), group_columns.unsqueeze(0)] = int(zero)
            scales = scales.clone()
            scales[zero_scale] = _STORAGE_FLOOR

        def unstorable(values: torch.Tensor) -> torch.Tensor:
            stored = values.to(torch.float16)
            return (
                ~torch.isfinite(values)
                | ~torch.isfinite(stored)
                | ((stored == 0) & (values != 0))
            )

        bad = unstorable(scales)
        if bool(bad.any()):
            for group_index, group_columns in enumerate(plan.columns):
                rows = bad[:, group_index].nonzero(as_tuple=False).squeeze(1)
                if rows.numel() == 0:
                    continue
                scales[rows, group_index] = fallback_scales[rows, group_index]
                codes[rows.unsqueeze(1), group_columns.unsqueeze(0)] = fallback_codes[
                    rows.unsqueeze(1), group_columns.unsqueeze(0)
                ]
            if bool(unstorable(scales).any()):
                raise ValueError(
                    "JointQ: the refined and fallback scales cannot be stored in "
                    "the FP16 checkpoint layout."
                )
        return codes, scales

    @staticmethod
    def _relative_weight_error(quantized: torch.Tensor, weight: torch.Tensor) -> float:
        difference = quantized.to(torch.float32) - weight.to(torch.float32)
        numerator = difference.square().sum()
        denominator = weight.to(torch.float32).square().sum().clamp_min(1e-12)
        return float((numerator / denominator).item())
