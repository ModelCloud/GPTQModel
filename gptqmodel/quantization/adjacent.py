# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Solver-agnostic Hessian-weighted adjacent rounding for GPTQ weight groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdjacentRoundingQUBO:
    """One fixed-codebook binary rounding problem for one GPTQ weight row."""

    bits: int
    weight: Tensor
    hessian: Tensor
    scale: Tensor
    zero: Tensor
    lower_codes: Tensor
    upper_codes: Tensor
    nearest_codes: Tensor
    lower_values: Tensor
    steps: Tensor
    constant: Tensor
    linear: Tensor
    pair: Tensor

    @property
    def size(self) -> int:
        return int(self.weight.numel())

    @property
    def active_decisions(self) -> int:
        return int(torch.count_nonzero(self.steps).item())


@dataclass(frozen=True)
class AdjacentIsing:
    """Ising coefficients for z=(1-Z)/2."""

    constant: Tensor
    linear_z: Tensor
    pair_zz: Tensor


@dataclass(frozen=True)
class AdjacentRoundingResult:
    state: Tensor
    cost: float
    states_checked: int
    optimal: bool = True
    lower_bound: float | None = None
    nodes_visited: int = 0


class AdjacentExactIncompleteError(RuntimeError):
    """Raised when a bounded native search cannot certify its candidate."""

    def __init__(self, result: AdjacentRoundingResult):
        self.result = result
        gap = (
            result.cost - result.lower_bound
            if result.lower_bound is not None
            else float("inf")
        )
        super().__init__(
            "AdjacentExact CUDA exhausted its node budget without an optimality "
            f"certificate: best={result.cost:.17g}, "
            f"lower_bound={result.lower_bound}, gap={gap:.6g}, "
            f"nodes={result.nodes_visited}."
        )


AdjacentSolver = Callable[[AdjacentRoundingQUBO], Tensor]


def _scalar(value: Tensor | float, *, name: str, device: torch.device) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=torch.float64)
    if result.numel() != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    result = result.reshape(())
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def build_adjacent_rounding_qubo(
    weight: Tensor,
    hessian: Tensor,
    *,
    scale: Tensor | float,
    zero: Tensor | float,
    bits: int,
) -> AdjacentRoundingQUBO:
    """Build the exact QUBO for adjacent lower/upper codes at fixed scale and zero."""

    if bits not in (2, 3, 4, 8):
        raise ValueError(
            "Adjacent rounding supports 2-bit, 3-bit, 4-bit, and 8-bit GPTQ."
        )
    if weight.ndim != 1 or weight.numel() < 1:
        raise ValueError("weight must be a non-empty rank-one tensor.")
    if hessian.shape != (weight.numel(), weight.numel()):
        raise ValueError("hessian shape must be [weight.numel(), weight.numel()].")

    device = weight.device
    work_weight = weight.detach().to(device=device, dtype=torch.float64)
    work_hessian = hessian.detach().to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(work_weight).all()) or not bool(
        torch.isfinite(work_hessian).all()
    ):
        raise ValueError("weight and hessian must contain only finite values.")
    if not torch.allclose(work_hessian, work_hessian.mT, rtol=0.0, atol=1e-10):
        raise ValueError("hessian must be symmetric.")

    work_scale = _scalar(scale, name="scale", device=device)
    work_zero = _scalar(zero, name="zero", device=device)
    if not bool(work_scale > 0):
        raise ValueError("scale must be positive.")

    maxq = (1 << bits) - 1
    # Preserve the source arithmetic used by Quantizer.quantize for the RTN
    # baseline. In particular, a float32 value can land on an exact half-code
    # even when the same division in float64 lies infinitesimally to one side.
    source_scale = torch.as_tensor(scale, device=device)
    source_zero = torch.as_tensor(zero, device=device)
    nearest_codes = torch.round(weight.detach() / source_scale + source_zero)
    nearest_codes = nearest_codes.clamp_(0, maxq).to(torch.int64)
    real_codes = work_weight / work_scale + work_zero
    lower_codes = torch.floor(real_codes).clamp_(0, maxq).to(torch.int64)
    upper_codes = torch.ceil(real_codes).clamp_(0, maxq).to(torch.int64)
    code_steps = upper_codes - lower_codes
    if not bool(((code_steps == 0) | (code_steps == 1)).all()):
        raise AssertionError("Adjacent code endpoints must differ by zero or one.")

    lower_values = work_scale * (lower_codes.to(torch.float64) - work_zero)
    steps = work_scale * code_steps.to(torch.float64)
    residual = work_weight - lower_values
    hessian_times_residual = work_hessian @ residual
    constant = residual @ hessian_times_residual
    linear = (
        steps.square() * work_hessian.diagonal() - 2.0 * steps * hessian_times_residual
    )
    pair = torch.triu(2.0 * torch.outer(steps, steps) * work_hessian, diagonal=1)
    return AdjacentRoundingQUBO(
        bits=bits,
        weight=work_weight,
        hessian=work_hessian,
        scale=work_scale,
        zero=work_zero,
        lower_codes=lower_codes,
        upper_codes=upper_codes,
        nearest_codes=nearest_codes,
        lower_values=lower_values,
        steps=steps,
        constant=constant,
        linear=linear,
        pair=pair,
    )


def _states(problem: AdjacentRoundingQUBO, states: Tensor) -> Tensor:
    states = torch.as_tensor(states, device=problem.weight.device, dtype=torch.float64)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.ndim != 2 or states.shape[1] != problem.size:
        raise ValueError("states must have shape [samples, problem.size].")
    if not bool(((states == 0) | (states == 1)).all()):
        raise ValueError("states must be binary.")
    return states


def adjacent_dequantize(
    problem: AdjacentRoundingQUBO, state: Tensor, *, dtype: torch.dtype | None = None
) -> Tensor:
    state = _states(problem, state)
    if state.shape[0] != 1:
        raise ValueError("adjacent_dequantize accepts exactly one state.")
    result = problem.lower_values + problem.steps * state[0]
    return result.to(dtype=dtype) if dtype is not None else result


def adjacent_hessian_error(problem: AdjacentRoundingQUBO, states: Tensor) -> Tensor:
    states = _states(problem, states)
    error = (
        problem.weight.unsqueeze(0)
        - problem.lower_values.unsqueeze(0)
        - states * problem.steps.unsqueeze(0)
    )
    return torch.einsum("bi,ij,bj->b", error, problem.hessian, error)


def adjacent_qubo_energy(problem: AdjacentRoundingQUBO, states: Tensor) -> Tensor:
    states = _states(problem, states)
    linear = states @ problem.linear
    pair = torch.einsum("bi,ij,bj->b", states, problem.pair, states)
    return problem.constant + linear + pair


def adjacent_qubo_to_ising(problem: AdjacentRoundingQUBO) -> AdjacentIsing:
    incident_pairs = problem.pair.sum(dim=0) + problem.pair.sum(dim=1)
    constant = problem.constant + 0.5 * problem.linear.sum() + 0.25 * problem.pair.sum()
    linear_z = -0.5 * problem.linear - 0.25 * incident_pairs
    pair_zz = 0.25 * problem.pair
    return AdjacentIsing(constant=constant, linear_z=linear_z, pair_zz=pair_zz)


def adjacent_ising_energy(problem: AdjacentIsing, states: Tensor) -> Tensor:
    states = torch.as_tensor(
        states, device=problem.linear_z.device, dtype=torch.float64
    )
    if states.ndim == 1:
        states = states.unsqueeze(0)
    spins = 1.0 - 2.0 * states
    linear = spins @ problem.linear_z
    pair = torch.einsum("bi,ij,bj->b", spins, problem.pair_zz, spins)
    return problem.constant + linear + pair


def adjacent_round_to_nearest_state(problem: AdjacentRoundingQUBO) -> Tensor:
    state = torch.zeros(problem.size, dtype=torch.float64, device=problem.weight.device)
    active = problem.upper_codes != problem.lower_codes
    state[active] = (problem.nearest_codes[active] - problem.lower_codes[active]).to(
        torch.float64
    )
    return state


def adjacent_coordinate_descent(
    problem: AdjacentRoundingQUBO,
    initial_state: Tensor | None = None,
) -> AdjacentRoundingResult:
    """Run deterministic best-improvement flips until reaching a one-flip local optimum."""

    state = (
        adjacent_round_to_nearest_state(problem)
        if initial_state is None
        else _states(problem, initial_state)[0].clone()
    )
    current_cost = float(adjacent_hessian_error(problem, state)[0].item())
    states_checked = 1
    while True:
        best_cost = current_cost
        best_index: int | None = None
        for index in torch.nonzero(problem.steps, as_tuple=False).flatten().tolist():
            candidate = state.clone()
            candidate[index] = 1.0 - candidate[index]
            candidate_cost = float(adjacent_hessian_error(problem, candidate)[0].item())
            states_checked += 1
            if candidate_cost < best_cost - 1e-14:
                best_cost = candidate_cost
                best_index = int(index)
        if best_index is None:
            return AdjacentRoundingResult(
                state=state, cost=current_cost, states_checked=states_checked
            )
        state[best_index] = 1.0 - state[best_index]
        current_cost = best_cost


def enumerate_adjacent_states(
    size: int, *, device: torch.device | str = "cpu"
) -> Tensor:
    if not 1 <= size <= 20:
        raise ValueError("Exact enumeration size must be in [1, 20].")
    integers = torch.arange(1 << size, device=device, dtype=torch.int64)
    shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.int64)
    return ((integers.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(torch.float64)


def adjacent_exact(problem: AdjacentRoundingQUBO) -> AdjacentRoundingResult:
    states = enumerate_adjacent_states(problem.size, device=problem.weight.device)
    costs = adjacent_hessian_error(problem, states)
    index = int(torch.argmin(costs).item())
    return AdjacentRoundingResult(
        state=states[index],
        cost=float(costs[index].item()),
        states_checked=int(states.shape[0]),
    )


def _exact_interaction_components(interaction: Tensor) -> list[Tensor]:
    """Return exact connected components of a symmetric nonzero interaction graph."""

    size = int(interaction.shape[0])
    adjacency = interaction.detach().ne(0).cpu()
    remaining = set(range(size))
    components: list[Tensor] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: list[int] = []
        remaining.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = (
                torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
            )
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(
            torch.tensor(
                sorted(component), device=interaction.device, dtype=torch.int64
            )
        )
    return components


def _automatic_branch_split_depth(device: torch.device, decisions: int) -> int:
    properties = torch.cuda.get_device_properties(device)
    target_workers = max(1, int(properties.multi_processor_count) * 32)
    return min(decisions, 12, (target_workers - 1).bit_length())


def _unpack_branch_bound_state(words: Tensor, size: int) -> Tensor:
    indices = torch.arange(size, device=words.device, dtype=torch.int64)
    selected_words = words.index_select(0, torch.div(indices, 64, rounding_mode="floor"))
    shifts = torch.remainder(indices, 64)
    return ((selected_words >> shifts) & 1).to(torch.float64)


def _gauge_and_order_qubo(
    linear: Tensor,
    interaction: Tensor,
    incumbent: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    incumbent = incumbent.to(device=linear.device, dtype=torch.float64)
    signs = 1.0 - 2.0 * incumbent
    constant = (
        incumbent @ linear + 0.5 * incumbent @ interaction @ incumbent
    ).reshape(())
    gauged_linear = signs * (linear + interaction @ incumbent)
    gauged_interaction = interaction * torch.outer(signs, signs)
    impact = gauged_linear.abs() + 0.5 * gauged_interaction.abs().sum(dim=1)
    order = torch.argsort(impact, descending=True, stable=True)
    return (
        constant,
        gauged_linear.index_select(0, order),
        gauged_interaction.index_select(0, order).index_select(1, order),
        order,
    )


def _best_native_incumbent(
    problem: AdjacentRoundingQUBO,
    initial_state: Tensor | None,
) -> Tensor:
    if initial_state is not None:
        return _states(problem, initial_state)[0].clone()

    starts = [
        adjacent_round_to_nearest_state(problem),
        torch.zeros(problem.size, dtype=torch.float64, device=problem.weight.device),
        torch.ones(problem.size, dtype=torch.float64, device=problem.weight.device),
        problem.linear.lt(0).to(torch.float64),
    ]
    indices = torch.arange(
        problem.size, dtype=torch.int64, device=problem.weight.device
    )
    for block_width in (1, 2, 4, 8, 16, 32):
        pattern = torch.remainder(
            torch.div(indices, block_width, rounding_mode="floor"), 2
        ).to(torch.float64)
        starts.extend((pattern, 1.0 - pattern))
    interaction = problem.pair + problem.pair.mT
    active = problem.steps.ne(0)
    best_state: Tensor | None = None
    best_cost = float("inf")
    for start in starts:
        state = start.clone()
        field = problem.linear + interaction @ state
        while True:
            deltas = (1.0 - 2.0 * state) * field
            deltas = torch.where(active, deltas, torch.full_like(deltas, torch.inf))
            index = torch.argmin(deltas)
            best_delta = float(deltas[index].item())
            if best_delta >= -1e-14:
                break
            direction = float((1.0 - 2.0 * state[index]).item())
            state[index] = 1.0 - state[index]
            field.add_(interaction[:, index], alpha=direction)
        cost = float(adjacent_hessian_error(problem, state)[0].item())
        if cost < best_cost:
            best_cost = cost
            best_state = state
    if best_state is None:
        raise AssertionError("At least one native incumbent must be evaluated.")
    return best_state


def _native_branch_bound_component(
    linear: Tensor,
    interaction: Tensor,
    incumbent: Tensor,
    *,
    split_depth: int | None,
    max_nodes_per_worker: int,
    certificate_tolerance: float,
) -> tuple[Tensor, float, float, int, bool]:
    decisions = int(linear.numel())
    if not 1 <= decisions <= 128:
        raise ValueError("Native branch-and-bound supports 1 to 128 decisions.")
    resolved_split_depth = (
        _automatic_branch_split_depth(linear.device, decisions)
        if split_depth is None
        else split_depth
    )

    constant, ordered_linear, ordered_interaction, order = _gauge_and_order_qubo(
        linear, interaction, incumbent
    )
    from gptqmodel.utils.adjacent_exact import adjacent_branch_bound_candidates

    candidates = adjacent_branch_bound_candidates(
        constant,
        ordered_linear,
        ordered_interaction,
        split_depth=resolved_split_depth,
        max_nodes_per_worker=max_nodes_per_worker,
        certificate_tolerance=certificate_tolerance,
    )
    best_index = torch.argmin(candidates.costs)
    ordered_flips = _unpack_branch_bound_state(
        candidates.states[best_index], decisions
    )
    flips = torch.empty_like(ordered_flips)
    flips.index_copy_(0, order, ordered_flips)
    state = torch.remainder(incumbent + flips, 2.0)
    cost = float(
        (
            state @ linear + 0.5 * state @ interaction @ state
        ).item()
    )
    complete = bool(candidates.completed.all().item())
    lower_bound = (
        cost
        if complete
        else float(candidates.root_lower_bounds.min().item())
    )
    nodes_visited = int(candidates.nodes_visited.sum().item())
    return state, cost, min(lower_bound, cost), nodes_visited, complete


def _hessian_has_nonnegative_objective(problem: AdjacentRoundingQUBO) -> bool:
    eigenvalues = torch.linalg.eigvalsh(problem.hessian)
    scale = max(1.0, float(eigenvalues.abs().max().item()))
    return float(eigenvalues[0].item()) >= -1e-12 * scale


def adjacent_branch_bound_cuda(
    problem: AdjacentRoundingQUBO,
    *,
    initial_state: Tensor | None = None,
    split_depth: int | None = None,
    max_nodes_per_worker: int = 0,
    certificate_tolerance: float = 1e-12,
    require_optimal: bool = True,
) -> AdjacentRoundingResult:
    """Natively solve one dense active component of up to 128 decisions on CUDA."""

    if problem.weight.device.type != "cuda":
        raise ValueError("adjacent_branch_bound_cuda requires a CUDA-resident problem.")
    if max_nodes_per_worker < 0:
        raise ValueError(
            "max_nodes_per_worker must be zero (unlimited) or positive."
        )

    active = torch.nonzero(problem.steps, as_tuple=False).flatten()
    state = torch.zeros(problem.size, dtype=torch.float64, device=problem.weight.device)
    if active.numel() == 0:
        return AdjacentRoundingResult(
            state=state,
            cost=float(problem.constant.item()),
            states_checked=1,
            lower_bound=float(problem.constant.item()),
            nodes_visited=1,
        )
    if active.numel() > 128:
        raise ValueError(
            "Native branch-and-bound supports at most 128 active decisions; "
            f"found {int(active.numel())}."
        )

    incumbent = _best_native_incumbent(problem, initial_state)
    local_incumbent = incumbent.index_select(0, active)
    local_pair = problem.pair.index_select(0, active).index_select(1, active)
    local_interaction = local_pair + local_pair.mT
    local_state, _, local_lower, nodes, complete = _native_branch_bound_component(
        problem.linear.index_select(0, active),
        local_interaction,
        local_incumbent,
        split_depth=split_depth,
        max_nodes_per_worker=max_nodes_per_worker,
        certificate_tolerance=certificate_tolerance,
    )
    state.index_copy_(0, active, local_state)
    cost = float(adjacent_hessian_error(problem, state)[0].item())
    lower_bound = min(float(problem.constant.item()) + local_lower, cost)
    if not complete and _hessian_has_nonnegative_objective(problem):
        lower_bound = max(0.0, lower_bound)
    result = AdjacentRoundingResult(
        state=state,
        cost=cost,
        states_checked=nodes,
        optimal=complete,
        lower_bound=cost if complete else lower_bound,
        nodes_visited=nodes,
    )
    if require_optimal and not complete:
        raise AdjacentExactIncompleteError(result)
    return result


def adjacent_exact_cuda(
    problem: AdjacentRoundingQUBO,
    *,
    warps: int = 0,
    decompose: bool = True,
    initial_state: Tensor | None = None,
    split_depth: int | None = None,
    max_nodes_per_worker: int = 0,
    certificate_tolerance: float = 1e-12,
    require_optimal: bool = True,
) -> AdjacentRoundingResult:
    """Solve adjacent rounding exactly using exhaustive or native branch-and-bound CUDA."""

    if problem.weight.device.type != "cuda":
        raise ValueError("adjacent_exact_cuda requires a CUDA-resident problem.")
    if warps < 0:
        raise ValueError("warps must be zero (automatic) or positive.")

    active = torch.nonzero(problem.steps, as_tuple=False).flatten()
    state = torch.zeros(problem.size, dtype=torch.float64, device=problem.weight.device)
    if active.numel() == 0:
        return AdjacentRoundingResult(
            state=state,
            cost=float(problem.constant.item()),
            states_checked=1,
        )

    active_pair = problem.pair.index_select(0, active).index_select(1, active)
    active_interaction = active_pair + active_pair.mT
    components = (
        _exact_interaction_components(active_interaction)
        if decompose
        else [torch.arange(active.numel(), device=active.device)]
    )
    if any(component.numel() > 128 for component in components):
        largest = max(int(component.numel()) for component in components)
        raise ValueError(
            "AdjacentExact CUDA supports at most 128 coupled active decisions; "
            f"found {largest}."
        )

    from gptqmodel.utils.adjacent_exact import adjacent_exact_candidates

    states_checked = 0
    nodes_visited = 0
    all_complete = True
    component_lower_bounds = []
    zero = problem.constant.new_zeros(())
    native_incumbent = (
        _best_native_incumbent(problem, initial_state)
        if any(component.numel() > 32 for component in components)
        else None
    )
    for component in components:
        component_active = active.index_select(0, component)
        local_linear = problem.linear.index_select(0, component_active)
        local_pair = problem.pair.index_select(0, component_active).index_select(
            1, component_active
        )
        local_interaction = local_pair + local_pair.mT
        if component.numel() <= 32:
            candidate_masks, candidate_costs = adjacent_exact_candidates(
                zero,
                local_linear,
                local_interaction,
                warps=warps,
            )
            best_mask = candidate_masks[torch.argmin(candidate_costs)]
            shifts = torch.arange(
                component.numel(), device=state.device, dtype=torch.int64
            )
            local_state = ((best_mask >> shifts) & 1).to(torch.float64)
            checked = 1 << int(component.numel())
            local_lower = float(
                (
                    local_state @ local_linear
                    + 0.5 * local_state @ local_interaction @ local_state
                ).item()
            )
            complete = True
            native_nodes = 0
        else:
            if native_incumbent is None:
                raise AssertionError("Native components require an incumbent.")
            (
                local_state,
                _,
                local_lower,
                native_nodes,
                complete,
            ) = _native_branch_bound_component(
                local_linear,
                local_interaction,
                native_incumbent.index_select(0, component_active),
                split_depth=split_depth,
                max_nodes_per_worker=max_nodes_per_worker,
                certificate_tolerance=certificate_tolerance,
            )
            checked = native_nodes
        state.index_copy_(0, component_active, local_state)
        states_checked += checked
        nodes_visited += native_nodes
        all_complete &= complete
        component_lower_bounds.append(local_lower)

    cost = float(adjacent_hessian_error(problem, state)[0].item())
    lower_bound = min(
        float(problem.constant.item()) + sum(component_lower_bounds), cost
    )
    if not all_complete and _hessian_has_nonnegative_objective(problem):
        lower_bound = max(0.0, lower_bound)
    result = AdjacentRoundingResult(
        state=state,
        cost=cost,
        states_checked=states_checked,
        optimal=all_complete,
        lower_bound=cost if all_complete else lower_bound,
        nodes_visited=nodes_visited,
    )
    if require_optimal and not all_complete:
        raise AdjacentExactIncompleteError(result)
    return result


def adjacent_exact_blocks(
    problem: AdjacentRoundingQUBO,
    *,
    block_size: int = 8,
    coupling_tolerance: float = 1e-12,
) -> AdjacentRoundingResult:
    """Certify a block-diagonal QUBO by exhaustively minimizing every block."""

    if problem.size % block_size:
        raise ValueError("problem size must be divisible by block_size.")
    block_ids = torch.arange(problem.size, device=problem.weight.device) // block_size
    cross_block = block_ids.unsqueeze(1) != block_ids.unsqueeze(0)
    if bool((problem.hessian[cross_block].abs() > coupling_tolerance).any()):
        raise ValueError("adjacent_exact_blocks requires a block-diagonal Hessian.")

    local_states = enumerate_adjacent_states(block_size, device=problem.weight.device)
    state = torch.zeros(problem.size, dtype=torch.float64, device=problem.weight.device)
    states_checked = 0
    for start in range(0, problem.size, block_size):
        stop = start + block_size
        error = (
            problem.weight[start:stop].unsqueeze(0)
            - problem.lower_values[start:stop].unsqueeze(0)
            - local_states * problem.steps[start:stop].unsqueeze(0)
        )
        costs = torch.einsum(
            "bi,ij,bj->b",
            error,
            problem.hessian[start:stop, start:stop],
            error,
        )
        state[start:stop] = local_states[int(torch.argmin(costs).item())]
        states_checked += int(local_states.shape[0])
    return AdjacentRoundingResult(
        state=state,
        cost=float(adjacent_hessian_error(problem, state)[0].item()),
        states_checked=states_checked,
    )


def quantize_adjacent_rows(
    weight: Tensor,
    hessian: Tensor,
    *,
    scales: Tensor,
    zeros: Tensor,
    bits: int,
    solver: AdjacentSolver,
) -> tuple[Tensor, Tensor]:
    """Quantize every row in one GPTQ group using a caller-supplied binary solver."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape [rows, group_size].")
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("hessian must have shape [group_size, group_size].")
    flat_scales = torch.as_tensor(scales, device=weight.device).reshape(-1)
    flat_zeros = torch.as_tensor(zeros, device=weight.device).reshape(-1)
    if flat_scales.numel() != weight.shape[0] or flat_zeros.numel() != weight.shape[0]:
        raise ValueError("scales and zeros must contain one value per weight row.")

    quantized = torch.empty_like(weight)
    states = torch.empty_like(weight, dtype=torch.float64)
    for row in range(weight.shape[0]):
        problem = build_adjacent_rounding_qubo(
            weight[row],
            hessian,
            scale=flat_scales[row],
            zero=flat_zeros[row],
            bits=bits,
        )
        state = _states(problem, solver(problem))[0]
        states[row].copy_(state)
        quantized[row].copy_(adjacent_dequantize(problem, state, dtype=weight.dtype))
    return quantized, states


def adjacent_problem_payload(problem: AdjacentRoundingQUBO) -> dict:
    """Return a JSON-compatible problem for an out-of-process quantum solver."""

    def values(tensor: Tensor):
        return tensor.detach().cpu().tolist()

    return {
        "schema": "gptqmodel-adjacent-qubo-v1",
        "bits": problem.bits,
        "size": problem.size,
        "active_decisions": problem.active_decisions,
        "weight": values(problem.weight),
        "hessian": values(problem.hessian),
        "scale": float(problem.scale.item()),
        "zero": float(problem.zero.item()),
        "lower_codes": values(problem.lower_codes),
        "upper_codes": values(problem.upper_codes),
        "qubo_constant": float(problem.constant.item()),
        "qubo_linear": values(problem.linear),
        "qubo_pair": values(problem.pair),
    }


__all__ = [
    "AdjacentExactIncompleteError",
    "AdjacentIsing",
    "AdjacentRoundingQUBO",
    "AdjacentRoundingResult",
    "AdjacentSolver",
    "adjacent_branch_bound_cuda",
    "adjacent_coordinate_descent",
    "adjacent_dequantize",
    "adjacent_exact",
    "adjacent_exact_blocks",
    "adjacent_exact_cuda",
    "adjacent_hessian_error",
    "adjacent_ising_energy",
    "adjacent_problem_payload",
    "adjacent_qubo_energy",
    "adjacent_qubo_to_ising",
    "adjacent_round_to_nearest_state",
    "build_adjacent_rounding_qubo",
    "enumerate_adjacent_states",
    "quantize_adjacent_rows",
]
