# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Experimental whole-model AdjacentExact hybridization for GPTQ research."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
import math
import os
import sys
import threading
import time
from typing import Any

import torch
from torch import Tensor

from .adjacent import (
    adjacent_branch_bound_cuda,
    adjacent_dequantize,
    build_adjacent_rounding_qubo,
)


@dataclass
class AdjacentModelConfig:
    """Control the bounded whole-model AdjacentExact research path and collect its module statistics."""

    coordinate_starts: tuple[str, ...] = ("nearest", "zero", "one", "linear")
    max_coordinate_flips: int = 32
    coordinate_rebase_interval: int = 8
    batch_coordinate_starts_on_cuda: bool = True
    executor: str = "cuda"
    row_chunk_size: int = 2048
    cpu_row_chunk_size: int = 512
    cpu_workers: int = 64
    cpu_min_row_groups: int = 393_216
    objective_row_chunk_size: int = 256
    native_refinements_per_module: int = 4
    native_split_depth: int = 6
    native_max_nodes_per_worker: int = 500
    certificate_tolerance: float = 1e-12
    selection_tolerance: float = 1e-7
    _stats: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _stats_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        allowed_starts = {"nearest", "zero", "one", "linear"}
        unknown = set(self.coordinate_starts) - allowed_starts
        if unknown:
            raise ValueError(f"Unknown adjacent coordinate starts: {sorted(unknown)}.")
        if not self.coordinate_starts:
            raise ValueError("coordinate_starts must contain at least one start.")
        if self.executor not in {"auto", "cpu", "cuda"}:
            raise ValueError("executor must be 'auto', 'cpu', or 'cuda'.")
        for name in (
            "max_coordinate_flips",
            "coordinate_rebase_interval",
            "row_chunk_size",
            "cpu_row_chunk_size",
            "cpu_workers",
            "cpu_min_row_groups",
            "objective_row_chunk_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.native_refinements_per_module < 0:
            raise ValueError("native_refinements_per_module must be non-negative.")
        if not 0 <= self.native_split_depth <= 20:
            raise ValueError("native_split_depth must be in [0, 20].")
        if self.native_max_nodes_per_worker < 0:
            raise ValueError("native_max_nodes_per_worker must be non-negative.")
        if not math.isfinite(self.certificate_tolerance) or self.certificate_tolerance < 0:
            raise ValueError("certificate_tolerance must be finite and non-negative.")
        if not math.isfinite(self.selection_tolerance) or self.selection_tolerance < 0:
            raise ValueError("selection_tolerance must be finite and non-negative.")

    def __deepcopy__(self, memo: dict[int, Any]) -> AdjacentModelConfig:
        """Share one immutable run configuration and statistics sink across per-module qcfg clones."""

        memo[id(self)] = self
        return self

    def record(self, module_stats: dict[str, Any]) -> None:
        """Append one JSON-compatible module result without racing parallel quantization workers."""

        with self._stats_lock:
            self._stats.append(copy.deepcopy(module_stats))

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a stable copy of all module results collected so far."""

        with self._stats_lock:
            return copy.deepcopy(self._stats)


def _coordinate_start(
    name: str,
    *,
    nearest: Tensor,
    linear: Tensor,
    active: Tensor,
) -> Tensor:
    if name == "nearest":
        state = nearest.clone()
    elif name == "zero":
        state = torch.zeros_like(nearest)
    elif name == "one":
        state = torch.ones_like(nearest)
    elif name == "linear":
        state = linear.lt(0).to(torch.float64)
    else:  # pragma: no cover - AdjacentModelConfig validates this contract.
        raise AssertionError(f"Unsupported coordinate start: {name}")
    return torch.where(active, state, torch.zeros_like(state))


def _coordinate_descent_batch_reference(
    *,
    residual: Tensor,
    steps: Tensor,
    hessian: Tensor,
    initial_state: Tensor,
    max_flips: int,
    rebase_interval: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = initial_state.clone()
    active_decisions = steps.ne(0)
    error = residual - steps * state
    hessian_error_gradient = error @ hessian
    signed_steps = steps * (1.0 - 2.0 * state)
    step_hessian_diagonal = steps.square() * hessian.diagonal()
    flips = torch.zeros(state.shape[0], dtype=torch.int32, device=state.device)

    for iteration in range(max_flips):
        deltas = -2.0 * signed_steps * hessian_error_gradient + step_hessian_diagonal
        deltas.masked_fill_(~active_decisions, torch.inf)
        best_delta, best_index = deltas.min(dim=1)
        improving = best_delta < -1e-14
        if not bool(improving.any()):
            break

        rows = torch.nonzero(improving, as_tuple=False).flatten()
        indices = best_index.index_select(0, rows)
        amplitude = signed_steps[rows, indices]
        state[rows, indices] = 1.0 - state[rows, indices]
        signed_steps[rows, indices] = -amplitude
        error[rows, indices] -= amplitude
        hessian_error_gradient[rows] -= amplitude.unsqueeze(1) * hessian.index_select(0, indices)
        flips.index_add_(0, rows, torch.ones_like(rows, dtype=flips.dtype))

        if (iteration + 1) % rebase_interval == 0:
            hessian_error_gradient = error @ hessian

    hessian_error_gradient = error @ hessian
    final_deltas = -2.0 * signed_steps * hessian_error_gradient + step_hessian_diagonal
    final_deltas.masked_fill_(~active_decisions, torch.inf)
    converged = final_deltas.min(dim=1).values >= -1e-14
    costs = (error * hessian_error_gradient).sum(dim=1)
    return state, costs, converged, flips


def _coordinate_descent_batch_cuda_masked(
    *,
    residual: Tensor,
    steps: Tensor,
    hessian: Tensor,
    initial_state: Tensor,
    max_flips: int,
    rebase_interval: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run fixed-iteration CUDA descent without data-dependent host synchronization."""

    state = initial_state.clone()
    active_decisions = steps.ne(0)
    error = residual - steps * state
    hessian_error_gradient = error @ hessian
    signed_steps = steps * (1.0 - 2.0 * state)
    step_hessian_diagonal = steps.square() * hessian.diagonal()
    flips = torch.zeros(state.shape[0], dtype=torch.int32, device=state.device)
    rows = torch.arange(state.shape[0], device=state.device)

    for iteration in range(max_flips):
        deltas = -2.0 * signed_steps * hessian_error_gradient + step_hessian_diagonal
        deltas.masked_fill_(~active_decisions, torch.inf)
        best_delta, best_index = deltas.min(dim=1)
        improving = best_delta < -1e-14

        selected_amplitude = signed_steps[rows, best_index]
        amplitude = torch.where(improving, selected_amplitude, torch.zeros_like(selected_amplitude))
        selected_state = state[rows, best_index]
        state[rows, best_index] = torch.where(improving, 1.0 - selected_state, selected_state)
        signed_steps[rows, best_index] = torch.where(improving, -selected_amplitude, selected_amplitude)
        error[rows, best_index] -= amplitude
        hessian_error_gradient -= amplitude.unsqueeze(1) * hessian.index_select(0, best_index)
        flips.add_(improving.to(flips.dtype))

        if (iteration + 1) % rebase_interval == 0:
            hessian_error_gradient = error @ hessian

    hessian_error_gradient = error @ hessian
    final_deltas = -2.0 * signed_steps * hessian_error_gradient + step_hessian_diagonal
    final_deltas.masked_fill_(~active_decisions, torch.inf)
    converged = final_deltas.min(dim=1).values >= -1e-14
    costs = (error * hessian_error_gradient).sum(dim=1)
    return state, costs, converged, flips


def _coordinate_descent_batch(
    *,
    residual: Tensor,
    steps: Tensor,
    hessian: Tensor,
    initial_state: Tensor,
    max_flips: int,
    rebase_interval: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    implementation = (
        _coordinate_descent_batch_cuda_masked
        if residual.device.type == "cuda"
        else _coordinate_descent_batch_reference
    )
    return implementation(
        residual=residual,
        steps=steps,
        hessian=hessian,
        initial_state=initial_state,
        max_flips=max_flips,
        rebase_interval=rebase_interval,
    )


def _adjacent_group_candidate(
    *,
    weight: Tensor,
    hessian: Tensor,
    scale: Tensor,
    zero: Tensor,
    bits: int,
    config: AdjacentModelConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    source_codes = weight / scale + zero
    work_weight = weight.to(torch.float64)
    work_scale = scale.to(torch.float64)
    work_zero = zero.to(torch.float64)
    real_codes = work_weight / work_scale + work_zero
    maxq = (1 << bits) - 1
    lower_codes = torch.floor(real_codes).clamp_(0, maxq)
    upper_codes = torch.ceil(real_codes).clamp_(0, maxq)
    nearest_codes = torch.round(source_codes).clamp_(0, maxq).to(torch.float64)
    steps = work_scale * (upper_codes - lower_codes)
    lower_values = work_scale * (lower_codes - work_zero)
    residual = work_weight - lower_values
    work_hessian = hessian.to(torch.float64)
    hessian_times_residual = residual @ work_hessian
    linear = steps.square() * work_hessian.diagonal() - 2.0 * steps * hessian_times_residual
    nearest_state = (nearest_codes - lower_codes).clamp_(0, 1)
    active = steps.ne(0)

    best_state = torch.zeros_like(work_weight)
    best_cost = torch.full(
        (weight.shape[0],),
        torch.inf,
        dtype=torch.float64,
        device=weight.device,
    )
    best_converged = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    best_flips = torch.zeros(weight.shape[0], dtype=torch.int32, device=weight.device)
    initial_states = [
        _coordinate_start(
            start_name,
            nearest=nearest_state,
            linear=linear,
            active=active,
        )
        for start_name in config.coordinate_starts
    ]
    if (
        weight.device.type == "cuda"
        and config.batch_coordinate_starts_on_cuda
        and len(initial_states) > 1
    ):
        start_count = len(initial_states)
        state, costs, converged, flips = _coordinate_descent_batch_cuda_masked(
            residual=residual.repeat(start_count, 1),
            steps=steps.repeat(start_count, 1),
            hessian=work_hessian,
            initial_state=torch.cat(initial_states, dim=0),
            max_flips=config.max_coordinate_flips,
            rebase_interval=config.coordinate_rebase_interval,
        )
        rows = torch.arange(weight.shape[0], device=weight.device)
        state = state.view(start_count, weight.shape[0], weight.shape[1])
        costs = costs.view(start_count, weight.shape[0])
        converged = converged.view(start_count, weight.shape[0])
        flips = flips.view(start_count, weight.shape[0])
        best_start = costs.argmin(dim=0)
        best_state = state[best_start, rows]
        best_cost = costs[best_start, rows]
        best_converged = converged[best_start, rows]
        best_flips = flips[best_start, rows]
    else:
        for initial_state in initial_states:
            state, costs, converged, flips = _coordinate_descent_batch(
                residual=residual,
                steps=steps,
                hessian=work_hessian,
                initial_state=initial_state,
                max_flips=config.max_coordinate_flips,
                rebase_interval=config.coordinate_rebase_interval,
            )
            improved = costs < best_cost
            best_cost = torch.where(improved, costs, best_cost)
            best_state[improved] = state[improved]
            best_converged = torch.where(improved, converged, best_converged)
            best_flips = torch.where(improved, flips, best_flips)

    candidate = lower_values + steps * best_state
    if not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(best_cost).all()):
        raise FloatingPointError("Adjacent whole-model candidate produced non-finite values.")
    return candidate.to(torch.float32), best_cost, best_converged, best_flips


@dataclass(frozen=True)
class _CandidateTask:
    group_index: int
    group_start: int
    group_stop: int
    row_start: int
    row_stop: int


def _resolve_executor(
    config: AdjacentModelConfig,
    *,
    row_group_count: int,
    bits: int,
    group_size: int,
    gil_enabled: bool | None = None,
    available_cpu_count: int | None = None,
) -> str:
    if config.executor != "auto":
        return config.executor
    if gil_enabled is None:
        gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    if available_cpu_count is None:
        available_cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
    use_cpu = (
        not gil_enabled
        and available_cpu_count >= config.cpu_workers
        and row_group_count >= config.cpu_min_row_groups
        and bits == 4
        and group_size == 128
    )
    return "cpu" if use_cpu else "cuda"


def _candidate_tasks(
    *,
    rows: int,
    columns: int,
    group_size: int,
    row_chunk_size: int,
) -> list[_CandidateTask]:
    return [
        _CandidateTask(
            group_index=group_index,
            group_start=group_start,
            group_stop=min(group_start + group_size, columns),
            row_start=row_start,
            row_stop=min(row_start + row_chunk_size, rows),
        )
        for group_index, group_start in enumerate(range(0, columns, group_size))
        for row_start in range(0, rows, row_chunk_size)
    ]


def _cuda_candidate_results(
    *,
    weight: Tensor,
    hessian: Tensor,
    scales: Tensor,
    zeros: Tensor,
    bits: int,
    group_size: int,
    config: AdjacentModelConfig,
) -> Iterator[tuple[_CandidateTask, tuple[Tensor, Tensor, Tensor, Tensor]]]:
    tasks = _candidate_tasks(
        rows=weight.shape[0],
        columns=weight.shape[1],
        group_size=group_size,
        row_chunk_size=config.row_chunk_size,
    )
    for task in tasks:
        yield task, _adjacent_group_candidate(
            weight=weight[task.row_start : task.row_stop, task.group_start : task.group_stop],
            hessian=hessian[task.group_start : task.group_stop, task.group_start : task.group_stop],
            scale=scales[task.row_start : task.row_stop, task.group_index].unsqueeze(1),
            zero=zeros[task.row_start : task.row_stop, task.group_index].unsqueeze(1),
            bits=bits,
            config=config,
        )


def _cpu_candidate_results(
    *,
    weight: Tensor,
    hessian: Tensor,
    scales: Tensor,
    zeros: Tensor,
    bits: int,
    group_size: int,
    config: AdjacentModelConfig,
) -> Iterator[tuple[_CandidateTask, tuple[Tensor, Tensor, Tensor, Tensor]]]:
    tasks = _candidate_tasks(
        rows=weight.shape[0],
        columns=weight.shape[1],
        group_size=group_size,
        row_chunk_size=config.cpu_row_chunk_size,
    )
    cpu_weight = weight.detach().to(device="cpu", copy=True).contiguous()
    cpu_scales = scales.detach().to(device="cpu", copy=True).contiguous()
    cpu_zeros = zeros.detach().to(device="cpu", copy=True).contiguous()
    cpu_hessians = tuple(
        hessian[group_start : min(group_start + group_size, weight.shape[1]), group_start : group_start + group_size]
        .detach()
        .to(device="cpu", copy=True)
        .contiguous()
        for group_start in range(0, weight.shape[1], group_size)
    )

    def solve(task: _CandidateTask) -> tuple[_CandidateTask, tuple[Tensor, Tensor, Tensor, Tensor]]:
        with torch.inference_mode():
            result = _adjacent_group_candidate(
                weight=cpu_weight[task.row_start : task.row_stop, task.group_start : task.group_stop],
                hessian=cpu_hessians[task.group_index],
                scale=cpu_scales[task.row_start : task.row_stop, task.group_index].unsqueeze(1),
                zero=cpu_zeros[task.row_start : task.row_stop, task.group_index].unsqueeze(1),
                bits=bits,
                config=config,
            )
        return task, result

    with ThreadPoolExecutor(
        max_workers=config.cpu_workers,
        thread_name_prefix="adjacent-candidate",
    ) as executor:
        yield from executor.map(solve, tasks)


def _row_hessian_costs(
    *,
    weight: Tensor,
    quantized: Tensor,
    hessian: Tensor,
    row_chunk_size: int,
) -> Tensor:
    costs = torch.empty(weight.shape[0], dtype=torch.float64, device=weight.device)
    work_hessian = hessian.to(torch.float32)
    for start in range(0, weight.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, weight.shape[0])
        error = weight[start:stop].to(torch.float32) - quantized[start:stop].to(torch.float32)
        chunk_cost = (error @ work_hessian * error).sum(dim=1)
        costs[start:stop] = chunk_cost.to(torch.float64)
    return costs


def _native_refine_candidates(
    *,
    weight: Tensor,
    hessian: Tensor,
    adjacent: Tensor,
    scales: Tensor,
    zeros: Tensor,
    group_size: int,
    bits: int,
    candidates: list[tuple[float, int, int]],
    config: AdjacentModelConfig,
) -> dict[str, int | float]:
    attempted = 0
    certified = 0
    improved = 0
    nodes = 0
    improvement = 0.0
    for _priority, group_index, row_index in sorted(candidates, reverse=True)[
        : config.native_refinements_per_module
    ]:
        start = group_index * group_size
        stop = min(start + group_size, weight.shape[1])
        problem = build_adjacent_rounding_qubo(
            weight[row_index, start:stop],
            hessian[start:stop, start:stop],
            scale=scales[row_index, group_index],
            zero=zeros[row_index, group_index],
            bits=bits,
        )
        scale = problem.scale
        zero = problem.zero
        current_codes = torch.round(adjacent[row_index, start:stop].to(torch.float64) / scale + zero)
        initial_state = (current_codes - problem.lower_codes).clamp_(0, 1).to(torch.float64)
        current_error = problem.weight - adjacent[row_index, start:stop].to(torch.float64)
        current_cost = float((current_error @ problem.hessian @ current_error).item())
        result = adjacent_branch_bound_cuda(
            problem,
            initial_state=initial_state,
            split_depth=min(config.native_split_depth, problem.active_decisions),
            max_nodes_per_worker=config.native_max_nodes_per_worker,
            certificate_tolerance=config.certificate_tolerance,
            require_optimal=False,
        )
        attempted += 1
        certified += int(result.optimal)
        nodes += result.nodes_visited
        if result.cost < current_cost - 1e-14:
            refined = adjacent_dequantize(problem, result.state, dtype=torch.float32)
            adjacent[row_index, start:stop] = refined
            improved += 1
            improvement += current_cost - result.cost
    return {
        "native_refinements_attempted": attempted,
        "native_refinements_certified": certified,
        "native_refinements_improved": improved,
        "native_nodes_visited": nodes,
        "native_group_error_reduction": improvement,
    }


@torch.inference_mode()
def apply_adjacent_model_hybrid(
    *,
    module_name: str,
    weight: Tensor,
    hessian: Tensor,
    classic_quantized: Tensor,
    scale_parts: list[Tensor],
    zero_parts: list[Tensor],
    bits: int,
    group_size: int,
    config: AdjacentModelConfig,
) -> tuple[Tensor, dict[str, Any]]:
    """Return a pack-compatible whole-module hybrid that never exceeds Classic GPTQ row Hessian cost."""

    start_time = time.perf_counter()
    if weight.device.type != "cuda" or hessian.device.type != "cuda":
        raise ValueError("Whole-model AdjacentExact requires CUDA-resident weights and Hessians.")
    if bits not in (2, 3, 4, 8):
        raise ValueError("Whole-model AdjacentExact supports 2, 3, 4, or 8 bits.")
    if not 1 <= group_size <= 128:
        raise ValueError("Whole-model AdjacentExact requires group_size in [1, 128].")
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("weight and hessian shapes are incompatible.")
    if classic_quantized.shape != weight.shape:
        raise ValueError("classic_quantized must match weight.")

    group_count = math.ceil(weight.shape[1] / group_size)
    if len(scale_parts) != group_count or len(zero_parts) != group_count:
        raise ValueError(
            "Adjacent whole-model scale/zero groups do not match the weight columns: "
            f"expected {group_count}, found {len(scale_parts)}/{len(zero_parts)}."
        )
    scales = torch.cat(scale_parts, dim=1)
    zeros = torch.cat(zero_parts, dim=1)
    if scales.shape != (weight.shape[0], group_count) or zeros.shape != scales.shape:
        raise ValueError(
            "Adjacent whole-model scale/zero tensors must have shape "
            f"{(weight.shape[0], group_count)}, found {tuple(scales.shape)}/{tuple(zeros.shape)}."
        )

    row_group_count = weight.shape[0] * group_count
    resolved_executor = _resolve_executor(
        config,
        row_group_count=row_group_count,
        bits=bits,
        group_size=group_size,
    )
    candidate_phase_start = time.perf_counter()
    candidate_results = (
        _cpu_candidate_results
        if resolved_executor == "cpu"
        else _cuda_candidate_results
    )(
        weight=weight,
        hessian=hessian,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        config=config,
    )
    candidate_output_device = torch.device("cpu") if resolved_executor == "cpu" else weight.device
    adjacent = torch.empty(weight.shape, dtype=torch.float32, device=candidate_output_device)
    refinement_candidates: list[tuple[float, int, int]] = []
    converged_row_groups = 0
    total_coordinate_flips = 0
    max_coordinate_flips = 0
    local_group_error = 0.0

    for task, result in candidate_results:
        candidate, costs, converged, flips = result
        adjacent[task.row_start : task.row_stop, task.group_start : task.group_stop] = candidate
        converged_row_groups += int(converged.sum().item())
        total_coordinate_flips += int(flips.sum().item())
        max_coordinate_flips = max(max_coordinate_flips, int(flips.max().item()))
        local_group_error += float(costs.sum().item())

        if config.native_refinements_per_module:
            top_count = min(config.native_refinements_per_module, costs.numel())
            top_costs, top_rows = torch.topk(costs, k=top_count)
            refinement_candidates.extend(
                (
                    float(cost),
                    task.group_index,
                    task.row_start + int(row),
                )
                for cost, row in zip(top_costs.tolist(), top_rows.tolist(), strict=True)
            )

    if adjacent.device != weight.device:
        adjacent = adjacent.to(weight.device)
    candidate_phase_wall_seconds = time.perf_counter() - candidate_phase_start

    native_stats = _native_refine_candidates(
        weight=weight,
        hessian=hessian,
        adjacent=adjacent,
        scales=scales,
        zeros=zeros,
        group_size=group_size,
        bits=bits,
        candidates=refinement_candidates,
        config=config,
    )

    adjacent = adjacent.to(classic_quantized.dtype).to(torch.float32)
    classic = classic_quantized.to(torch.float32)
    classic_costs = _row_hessian_costs(
        weight=weight,
        quantized=classic,
        hessian=hessian,
        row_chunk_size=config.objective_row_chunk_size,
    )
    adjacent_costs = _row_hessian_costs(
        weight=weight,
        quantized=adjacent,
        hessian=hessian,
        row_chunk_size=config.objective_row_chunk_size,
    )
    threshold = config.selection_tolerance * (1.0 + classic_costs.abs())
    select_adjacent = adjacent_costs < classic_costs - threshold
    hybrid = classic.clone()
    hybrid[select_adjacent] = adjacent[select_adjacent]
    hybrid_costs = torch.where(select_adjacent, adjacent_costs, classic_costs)
    if bool((hybrid_costs > classic_costs + threshold).any()):
        raise AssertionError("Adjacent hybrid row selection exceeded the Classic GPTQ objective.")

    classic_total = float(classic_costs.sum().item())
    adjacent_total = float(adjacent_costs.sum().item())
    hybrid_total = float(hybrid_costs.sum().item())
    stats: dict[str, Any] = {
        "module": module_name,
        "rows": weight.shape[0],
        "columns": weight.shape[1],
        "bits": bits,
        "group_size": group_size,
        "group_count": group_count,
        "row_group_count": row_group_count,
        "coordinate_starts": list(config.coordinate_starts),
        "executor_requested": config.executor,
        "executor": resolved_executor,
        "candidate_row_chunk_size": (
            config.cpu_row_chunk_size
            if resolved_executor == "cpu"
            else config.row_chunk_size
        ),
        "candidate_workers": config.cpu_workers if resolved_executor == "cpu" else 1,
        "candidate_phase_wall_seconds": candidate_phase_wall_seconds,
        "coordinate_converged_row_groups": converged_row_groups,
        "coordinate_capped_row_groups": row_group_count - converged_row_groups,
        "coordinate_total_flips": total_coordinate_flips,
        "coordinate_max_flips": max_coordinate_flips,
        "local_group_hessian_error": local_group_error,
        "classic_full_hessian_error": classic_total,
        "adjacent_full_hessian_error": adjacent_total,
        "hybrid_full_hessian_error": hybrid_total,
        "hybrid_full_hessian_error_reduction": classic_total - hybrid_total,
        "hybrid_full_hessian_error_reduction_pct": (
            100.0 * (classic_total - hybrid_total) / classic_total
            if classic_total > 0
            else 0.0
        ),
        "hybrid_selected_rows": int(select_adjacent.sum().item()),
        "hybrid_total_rows": weight.shape[0],
        "adjacent_model_wall_seconds": time.perf_counter() - start_time,
        **native_stats,
    }
    return hybrid.to(classic_quantized.dtype), stats


__all__ = [
    "AdjacentModelConfig",
    "apply_adjacent_model_hybrid",
]
