# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Non-uniform bitwidth allocation from Section 3.3 of arXiv:2605.02404.

Implements linear and Shapley sensitivity estimation plus an ILP-based bitwidth
allocator. The outputs can be converted into a ``QuantizeConfig.dynamic`` dict
so that existing GPT-QModel processors quantize each layer at its assigned
bitwidth.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .gamma import quantize_uniform


def reconstruction_error(
    weight: torch.Tensor,
    bits: int,
    *,
    symmetric: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalized squared reconstruction error ``||W - Q(W)||_F^2 / ||W||_F^2``.

    Args:
        weight: Weight tensor.
        bits: Target bitwidth.
        symmetric: Use symmetric quantization grid.
        eps: Numerical stability constant.

    Returns:
        Scalar error tensor.
    """

    wq = quantize_uniform(weight, bits, symmetric=symmetric)
    denom = weight.norm(p="fro").square().clamp(min=eps)
    return (weight - wq).norm(p="fro").square() / denom


def linear_sensitivity(
    weights: list[torch.Tensor],
    bitwidths: list[int],
    *,
    symmetric: bool = True,
    importance: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Linear sensitivity database from reconstruction error.

    Returns a cost matrix ``C[m, b]`` where ``C[m, b]`` is the predicted
    degradation of assigning group ``m`` to bitwidth ``bitwidths[b]``. If
    ``importance`` is supplied, each row is scaled by the group importance.

    Args:
        weights: List of per-group weight tensors.
        bitwidths: Candidate bitwidths (must include the highest reference).
        symmetric: Use symmetric grids for reconstruction error.
        importance: Optional per-group scalar importances ``alpha_m``.

    Returns:
        ``(M, B)`` numpy cost array.
    """

    bitwidths = list(bitwidths)
    costs = np.zeros((len(weights), len(bitwidths)), dtype=np.float64)
    if importance is None:
        importance = np.ones(len(weights), dtype=np.float64)
    else:
        importance = np.asarray(importance, dtype=np.float64)

    for m, w in enumerate(weights):
        for b, bits in enumerate(bitwidths):
            costs[m, b] = importance[m] * reconstruction_error(w, bits, symmetric=symmetric).item()

    return costs


def _default_bmax(bitwidths: list[int]) -> int:
    return max(bitwidths)


def shapley_sensitivity(
    evaluate_fn: Callable[[dict[str, int]], dict[str, float]],
    groups: list[str],
    bitwidths: list[int],
    *,
    permutations: int = 30,
    metrics: tuple[str, ...] = ("kl", "ear"),
) -> dict[str, np.ndarray]:
    """Multi-bitwidth Shapley sensitivity estimation (Algorithm 1).

    For every target bitwidth ``b*`` (all bitwidths except the maximum), the
    function samples ``permutations`` orderings, starts every group at ``bmax``,
    and switches one group at a time to ``b*``. The average marginal metric
    change is the Shapley value for that group-bitwidth pair.

    Args:
        evaluate_fn: Function accepting a ``{group_name: bits}`` dict and
            returning a ``{metric_name: float}`` dict. ``kl`` and ``ear`` are
            expected as positive costs (larger = worse).
        groups: Group names.
        bitwidths: Candidate bitwidths (must include the maximum reference).
        permutations: Number of random permutations to sample.
        metrics: Metric names to compute Shapley values for.

    Returns:
        Dictionary mapping metric name to ``(M, B)`` numpy cost array. The
        column for ``bmax`` is all zeros because costs are relative to the
        reference bitwidth.
    """

    bitwidths = sorted(set(bitwidths))
    bmax = bitwidths[-1]
    targets = bitwidths[:-1]
    m = len(groups)
    b = len(bitwidths)
    index = {bits: i for i, bits in enumerate(bitwidths)}

    raw: dict[str, np.ndarray] = {name: np.zeros((m, b), dtype=np.float64) for name in metrics}
    rng = np.random.default_rng(42)

    for target_bits in targets:
        target_idx = index[target_bits]
        marginals = {name: [] for name in metrics}

        for _ in range(permutations):
            order = rng.permutation(m)
            assignment = {g: bmax for g in groups}
            base_metrics = evaluate_fn(assignment)

            for j in order:
                g = groups[j]
                assignment[g] = target_bits
                new_metrics = evaluate_fn(assignment)
                for name in metrics:
                    delta = new_metrics[name] - base_metrics[name]
                    marginals[name].append((j, delta))
                base_metrics = new_metrics

        for name in metrics:
            sums = np.zeros(m, dtype=np.float64)
            counts = np.zeros(m, dtype=np.int64)
            for (j, delta) in marginals[name]:
                sums[j] += delta
                counts[j] += 1
            shapley = sums / np.maximum(counts, 1)
            raw[name][:, target_idx] = shapley

    # Costs for bmax are zero (the reference).
    for name in metrics:
        raw[name][:, index[bmax]] = 0.0

    return raw


def _allocate_greedy(
    costs: np.ndarray,
    bitwidths: np.ndarray,
    weights: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Greedy fallback when SciPy is unavailable.

    Assigns each group the bitwidth with lowest cost while respecting the
    weighted average bitwidth budget.
    """

    m, b = costs.shape
    assignment = np.full(m, b - 1, dtype=np.int64)
    current_bits = bitwidths[assignment]
    total_weight = weights.sum()

    # Sort candidate (group, bitwidth) improvements by cost reduction per bit increase.
    improvements = []
    for i in range(m):
        for j in range(b):
            if j == assignment[i]:
                continue
            delta_cost = costs[i, j] - costs[i, assignment[i]]
            delta_bits = bitwidths[j] - bitwidths[assignment[i]]
            if delta_bits < 0:
                improvements.append((i, j, delta_cost, delta_bits))

    # Greedily lower bitwidth where cost increase is smallest per bit saved.
    improvements.sort(key=lambda x: x[2] / max(-x[3], 1e-12))
    for i, j, _, delta_bits in improvements:
        new_bits = current_bits.copy()
        new_bits[i] = bitwidths[j]
        new_avg = (new_bits * weights).sum() / total_weight
        if new_avg <= budget:
            assignment[i] = j
            current_bits = new_bits

    return assignment


def allocate_bitwidth_ilp(
    costs: np.ndarray,
    bitwidths: list[int] | np.ndarray,
    *,
    weights: list[float] | np.ndarray | None = None,
    budget: float | None = None,
    minimize_bits: bool = False,
    max_cost: float | None = None,
    time_limit: float = 60.0,
) -> np.ndarray:
    """Solve the multiple-choice bitwidth allocation problem.

    Minimizes total predicted cost subject to a weighted-average bitwidth
    budget, or minimizes bits subject to a cost bound.

    Args:
        costs: ``(M, B)`` numpy array of predicted degradation.
        bitwidths: Candidate bitwidths (one per column of ``costs``).
        weights: Optional per-group parameter counts or sizes (default ones).
        budget: Maximum weighted-average bitwidth (cost-minimization mode).
        minimize_bits: If True, minimize average bitwidth subject to
            ``total_cost <= max_cost``.
        max_cost: Cost bound when ``minimize_bits`` is True.
        time_limit: MILP solver time limit in seconds.

    Returns:
        ``(M,)`` integer array of selected column indices.
    """

    costs = np.asarray(costs, dtype=np.float64)
    bitwidths = np.asarray(bitwidths, dtype=np.int64)
    m, b = costs.shape

    if weights is None:
        weights = np.ones(m, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    total_weight = weights.sum()

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        if budget is None:
            budget = bitwidths.max()
        return _allocate_greedy(costs, bitwidths, weights, budget)

    # Variables are flattened x[m, b] in {0, 1}.
    c = costs.ravel()
    integrality = np.ones(m * b, dtype=np.int64)
    bounds = Bounds(lb=np.zeros(m * b, dtype=np.float64), ub=np.ones(m * b, dtype=np.float64))

    # Each group picks exactly one bitwidth.
    A_eq = np.zeros((m, m * b), dtype=np.float64)
    for i in range(m):
        A_eq[i, i * b:(i + 1) * b] = 1.0
    group_constraint = LinearConstraint(A_eq, 1.0, 1.0)

    constraints = [group_constraint]

    if budget is not None and not minimize_bits:
        A_budget = np.zeros((1, m * b), dtype=np.float64)
        for i in range(m):
            for j in range(b):
                A_budget[0, i * b + j] = bitwidths[j] * weights[i] / total_weight
        constraints.append(LinearConstraint(A_budget, -np.inf, budget))

    if minimize_bits:
        # Objective: weighted average bitwidth.
        c = (bitwidths[None, :] * weights[:, None] / total_weight).ravel()
        if max_cost is not None:
            A_cost = np.zeros((1, m * b), dtype=np.float64)
            for i in range(m):
                A_cost[0, i * b:(i + 1) * b] = costs[i]
            constraints.append(LinearConstraint(A_cost, -np.inf, max_cost))
        elif budget is not None:
            A_budget = np.zeros((1, m * b), dtype=np.float64)
            for i in range(m):
                for j in range(b):
                    A_budget[0, i * b + j] = bitwidths[j] * weights[i] / total_weight
            constraints.append(LinearConstraint(A_budget, -np.inf, budget))

    res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality, options={"time_limit": time_limit})
    if not res.success:
        raise RuntimeError(f"Bitwidth ILP failed: {res.message}")

    x = res.x.reshape(m, b)
    assignment = x.argmax(axis=1)
    return assignment.astype(np.int64)


def binary_search_budget(
    costs: np.ndarray,
    bitwidths: list[int] | np.ndarray,
    predicate_fn: Callable[[np.ndarray, float], bool],
    *,
    weights: list[float] | np.ndarray | None = None,
    low: float | None = None,
    high: float | None = None,
    tolerance: float = 0.05,
    max_iter: int = 20,
) -> tuple[float, np.ndarray]:
    """Binary search over the average bitwidth budget.

    For each candidate budget the function calls ``allocate_bitwidth_ilp`` and
    then ``predicate_fn(assignment, budget)``. The search returns the smallest
    budget whose assignment satisfies the predicate.

    Args:
        costs: ``(M, B)`` numpy cost matrix.
        bitwidths: Candidate bitwidths.
        predicate_fn: Returns True if the assignment meets the quality target.
        weights: Optional per-group weights.
        low: Lower bound for average bits (default min bitwidth).
        high: Upper bound for average bits (default max bitwidth).
        tolerance: Search termination tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of ``(best_budget, best_assignment)``.
    """

    bitwidths = np.asarray(bitwidths)
    if low is None:
        low = float(bitwidths.min())
    if high is None:
        high = float(bitwidths.max())

    best_assignment = allocate_bitwidth_ilp(costs, bitwidths, weights=weights, budget=high)
    best_budget = high

    for _ in range(max_iter):
        if high - low <= tolerance:
            break
        mid = (low + high) / 2.0
        assignment = allocate_bitwidth_ilp(costs, bitwidths, weights=weights, budget=mid)
        if predicate_fn(assignment, mid):
            best_budget = mid
            best_assignment = assignment
            high = mid
        else:
            low = mid

    return best_budget, best_assignment
