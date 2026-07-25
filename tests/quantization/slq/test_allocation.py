# GPU=-1
import numpy as np
import pytest
import torch

from gptqmodel.quantization.slq.allocation import (
    allocate_bitwidth_ilp,
    binary_search_budget,
    linear_sensitivity,
    reconstruction_error,
    shapley_sensitivity,
)


def test_reconstruction_error_decreases_with_bits():
    torch.manual_seed(0)
    w = torch.randn(256) * 0.5 + 0.1
    errors = [reconstruction_error(w, b, symmetric=False).item() for b in [2, 3, 4, 5, 6]]
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]


def test_linear_sensitivity_shape():
    weights = [torch.randn(64), torch.randn(128), torch.randn(32)]
    bitwidths = [2, 3, 4]
    costs = linear_sensitivity(weights, bitwidths, symmetric=False)
    assert costs.shape == (3, 3)
    assert np.all(costs >= 0)
    # Higher bitwidth -> lower or equal error.
    assert np.all(np.diff(costs, axis=1) <= 1e-6)


def test_allocate_bitwidth_ilp_one_per_group():
    costs = np.array([[10.0, 1.0, 0.1], [5.0, 0.5, 0.05], [2.0, 0.2, 0.02]])
    bitwidths = [2, 4, 8]
    assignment = allocate_bitwidth_ilp(costs, bitwidths, budget=3.5)
    assert len(assignment) == costs.shape[0]
    assert all(0 <= a < len(bitwidths) for a in assignment)
    # Budget 3.5 with weights=1 forces mostly 2-bit choices.
    avg = sum(bitwidths[a] for a in assignment) / len(assignment)
    assert avg <= 3.5 + 1e-6


def test_allocate_bitwidth_ilp_respects_weights():
    costs = np.array([[10.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
    bitwidths = [2, 4, 8]
    weights = np.array([1.0, 10.0])
    # Budget 2.5 should put heavy group at 2-bit and light group at higher.
    assignment = allocate_bitwidth_ilp(costs, bitwidths, weights=weights, budget=2.6)
    avg = sum(bitwidths[a] * weights[i] for i, a in enumerate(assignment)) / weights.sum()
    assert avg <= 2.6 + 1e-6


def test_allocate_bitwidth_minimize_bits():
    costs = np.array([[10.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
    bitwidths = [2, 4, 8]
    assignment = allocate_bitwidth_ilp(costs, bitwidths, minimize_bits=True, max_cost=1.5)
    # Should choose the lowest bitwidth that keeps total cost <= 1.5.
    total_cost = sum(costs[i, a] for i, a in enumerate(assignment))
    assert total_cost <= 1.5 + 1e-6
    assert all(bitwidths[a] >= 2 for a in assignment)


def test_binary_search_budget():
    np.random.seed(0)
    weights = [torch.randn(128) for _ in range(4)]
    bitwidths = [2, 3, 4, 5, 6, 7, 8]
    costs = linear_sensitivity(weights, bitwidths, symmetric=False)

    def predicate(assignment, budget):
        total = sum(costs[i, a] for i, a in enumerate(assignment))
        return total <= 0.05

    best_budget, best_assignment = binary_search_budget(
        costs, bitwidths, predicate, tolerance=0.1, max_iter=10
    )
    assert best_budget >= 2.0
    assert best_budget <= 8.0
    assert len(best_assignment) == costs.shape[0]
    total = sum(costs[i, a] for i, a in enumerate(best_assignment))
    assert total <= 0.05


def test_shapley_sensitivity_shape():
    groups = ["a", "b", "c"]
    bitwidths = [4, 6, 8]

    def evaluate_fn(assignment):
        # Synthetic cost: groups with lower bitwidths increase 'kl' cost.
        kl = 0.0
        for g, b in assignment.items():
            idx = groups.index(g)
            kl += (8 - b) * (idx + 1) * 0.01
        return {"kl": kl, "ear": 1.0 - max(0.0, 1.0 - kl)}

    result = shapley_sensitivity(evaluate_fn, groups, bitwidths, permutations=10)
    assert result["kl"].shape == (3, 3)
    # Reference bitwidth (max) has zero cost.
    assert np.allclose(result["kl"][:, -1], 0.0)
    # Lower bitwidths should generally show higher cost.
    for i in range(len(groups)):
        assert np.all(np.diff(result["kl"][i]) <= 1e-6)


def test_allocate_raises_when_infeasible():
    costs = np.array([[10.0, 5.0], [10.0, 5.0]])
    bitwidths = [4, 8]
    # Budget below minimum bitwidth should be infeasible.
    with pytest.raises((RuntimeError, ValueError)):
        allocate_bitwidth_ilp(costs, bitwidths, budget=3.0)
