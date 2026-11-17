# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch
from test_format_reference import scalar_decode, scalar_encode

from gptqmodel.quantization.format_reference import FORMATS, quantize_rtn
from gptqmodel.quantization.format_solver_reference import (
    SolverResult,
    allocate_rco,
    reconstruction_error,
    solve_awq,
    solve_gptq,
    solve_gsq,
)
from scripts.prove_format_solvers import problem


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def independent_gptq(weight, inputs, grid, act_order):
    """Solve the remaining normal equations afresh after every fixed column.

    No Cholesky inverse-Hessian recurrence from the implementation is reused.
    """
    hessian = inputs.T @ inputs / len(inputs)
    order = hessian.diagonal().argsort(descending=True, stable=True) if act_order else torch.arange(weight.shape[1])
    hessian += .01 * hessian.diagonal().mean() * torch.eye(weight.shape[1], dtype=torch.float64)
    hessian = hessian[order][:, order]
    original = weight[:, order]
    quantized = torch.empty_like(original)
    codes = torch.empty_like(weight, dtype=torch.uint8)
    scales, zeros = grid.expanded()[:, order], grid.zero_grid()[:, order]
    for column in range(weight.shape[1]):
        remaining = original[:, column:].clone()
        if column:
            fixed_error = quantized[:, :column] - original[:, :column]
            rhs = -fixed_error @ hessian[:column, column:]
            remaining += torch.linalg.solve(hessian[column:, column:], rhs.T).T
        for row in range(weight.shape[0]):
            normalized = float(remaining[row, 0] / scales[row, column] + zeros[row, column])
            if grid.spec.element == "int4":
                code = max(0, min(15, round(normalized)))
            else:
                code = scalar_encode(normalized, grid.spec.element)
            codes[row, order[column]] = code
            quantized[row, column] = (scalar_decode(code, grid.spec.element) - zeros[row, column]) * scales[row, column]
    return codes


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_matches_independent_constrained_least_squares(spec, act_order):
    weight, inputs, _ = problem(13)
    weight = weight[:2]
    actual = solve_gptq(weight, inputs, spec.name, act_order=act_order)
    expected = independent_gptq(weight, inputs, actual.weight.grid, act_order)
    assert torch.equal(actual.weight.codes(), expected)
    baseline = quantize_rtn(weight, spec.name)
    assert torch.equal(actual.weight.grid.scales, baseline.grid.scales)
    assert not torch.equal(actual.weight.payload, baseline.payload)


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
@pytest.mark.parametrize("solver", [solve_gptq, solve_awq, solve_gsq], ids=["gptq", "awq", "gsq"])
def test_solver_improves_real_hard_target_payload_not_just_soft_loss(spec, solver):
    weight, calibration, heldout = problem(0)
    before_weight, before_inputs = weight.clone(), calibration.clone()
    baseline = SolverResult(quantize_rtn(weight, spec.name))
    actual = solver(weight, calibration, spec.name)
    assert reconstruction_error(actual, weight, calibration) < .99 * reconstruction_error(baseline, weight, calibration)
    assert reconstruction_error(actual, weight, heldout) < reconstruction_error(baseline, weight, heldout)
    assert torch.equal(weight, before_weight)
    assert torch.equal(calibration, before_inputs)
    # The next consumer uses the same hard packed solution without recomputing
    # scales or fitting a second quantizer.
    copied = SolverResult(actual.weight.grid.pack(actual.weight.codes()), actual.input_scale)
    torch.testing.assert_close(copied.forward(heldout), actual.forward(heldout), rtol=0, atol=0)
    assert set(vars(actual)) == {"weight", "input_scale", "history"}


@pytest.mark.parametrize("spec", FORMATS[1:], ids=lambda s: s.name)
def test_solving_target_is_not_int4_solving_followed_by_conversion(spec):
    weight, calibration, heldout = problem(0)
    direct = solve_gptq(weight, calibration, spec.name)
    int4 = solve_gptq(weight, calibration, "int4")
    converted = SolverResult(quantize_rtn(int4.weight.dequantize(), spec.name))
    assert not torch.equal(direct.weight.payload, converted.weight.payload)
    assert reconstruction_error(direct, weight, heldout) < reconstruction_error(converted, weight, heldout)


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
def test_awq_keeps_transform_and_optimizes_target_format_objective(spec):
    weight, calibration, heldout = problem(0)
    result = solve_awq(weight, calibration, spec.name, grid_steps=5)
    scores, scales = [], []
    for alpha in (0., .25, .5, .75, 1.):
        scale = calibration.abs().mean(0).clamp(min=1e-12).pow(alpha)
        scale = (scale / (scale.min() * scale.max()).sqrt()).float()
        hard = quantize_rtn(weight * scale.double(), spec.name).dequantize()
        target = calibration @ weight.T
        score = ((calibration / scale.double()) @ hard.T - target).square().sum() / target.square().sum()
        scores.append(float(score))
        scales.append(scale)
    assert result.input_scale is not None
    assert torch.equal(result.input_scale, scales[scores.index(min(scores))])
    assert reconstruction_error(result, weight, calibration) == pytest.approx(min(scores), abs=1e-14)
    explicit = (heldout / result.input_scale.double()) @ result.weight.dequantize().T
    torch.testing.assert_close(result.forward(heldout), explicit, rtol=0, atol=0)
    wrong = SolverResult(result.weight)
    assert reconstruction_error(wrong, weight, heldout) > reconstruction_error(result, weight, heldout)
    assert result.storage_bits == result.weight.storage_bits + 32 * weight.shape[1]


@pytest.mark.parametrize("spec", FORMATS, ids=lambda s: s.name)
def test_gsq_frozen_scales_hard_guard_and_rng_ownership(spec):
    weight, calibration, _ = problem(0)
    rng_state = torch.random.get_rng_state().clone()
    first = solve_gsq(weight, calibration, spec.name)
    second = solve_gsq(weight, calibration, spec.name)
    baseline = quantize_rtn(weight, spec.name)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(first.weight.payload, second.weight.payload)
    assert torch.equal(first.weight.grid.scales, baseline.grid.scales)
    assert first.history[-1] < first.history[0]
    assert all(a >= b for a, b in zip(first.history, first.history[1:]))
    assert first.history[-1] == reconstruction_error(first, weight, calibration)


def test_rco_exact_storage_budget_matches_exhaustive_format_allocation():
    candidates, scores = [], []
    for seed in range(3):
        weight, calibration, _ = problem(seed)
        row = [solve_gptq(weight, calibration, spec.name) for spec in FORMATS]
        candidates.append(row)
        scores.append([reconstruction_error(c, weight, calibration) for c in row])
    errors = torch.tensor(scores, dtype=torch.float64)
    costs = [[c.storage_bits for c in row] for row in candidates]
    low, high = sum(min(row) for row in costs), sum(max(row) for row in costs)
    for budget in (low, (low + high) // 2, high):
        result = allocate_rco(candidates, errors, budget_bits=budget).tolist()
        assert sum(costs[i][j] for i, j in enumerate(result)) <= budget
        feasible = [choice for choice in itertools.product(range(len(FORMATS)), repeat=3)
                    if sum(costs[i][j] for i, j in enumerate(choice)) <= budget]
        oracle = min(sum(scores[i][j] for i, j in enumerate(choice)) for choice in feasible)
        assert sum(scores[i][j] for i, j in enumerate(result)) == pytest.approx(oracle, abs=1e-14)
    with pytest.raises(RuntimeError, match="no feasible"):
        allocate_rco(candidates, errors, budget_bits=low - 1)


def test_rco_counts_awq_transform_and_rejects_malformed_request():
    weight, inputs, _ = problem(0)
    plain = SolverResult(quantize_rtn(weight, "mxfp4"))
    transformed = solve_awq(weight, inputs, "mxfp4")
    assert transformed.input_scale is not None
    choice = allocate_rco([[plain, transformed]], torch.tensor([[1., 0.]]), budget_bits=plain.storage_bits)
    assert choice.tolist() == [0]
    with pytest.raises(ValueError, match="rectangular"):
        allocate_rco([[plain], [plain, transformed]], torch.ones(2, 1), budget_bits=100)
    with pytest.raises(ValueError, match="nonnegative"):
        allocate_rco([[plain]], torch.tensor([[-1.]]), budget_bits=100)


def test_rco_manifold_math_accepts_actual_hardware_format_costs():
    from gptqmodel.quantization.rco import budget_normal, project_gradient, retraction

    weight, inputs, _ = problem(0)
    costs = torch.tensor([quantize_rtn(weight, s.name).storage_bits / weight.numel() for s in FORMATS],
                         dtype=torch.float64)
    logits = torch.linspace(-.3, .5, 18, dtype=torch.float64).reshape(3, 6).requires_grad_()
    weights = torch.tensor([.2, .3, .5], dtype=torch.float64)
    expected = (logits.softmax(-1) * costs).sum(-1) @ weights
    oracle_normal = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(budget_normal(logits, costs, weights), oracle_normal, rtol=1e-12, atol=1e-12)
    logits.grad = torch.arange(18, dtype=torch.float64).reshape(3, 6) / 17
    project_gradient(logits, costs, weights)
    assert abs(float((logits.grad * oracle_normal).sum())) < 1e-10
    target = float((costs.min() + costs.max()) / 2)
    retraction(logits, costs, target, weights, tol=1e-10)
    independent_cost = float((logits.detach().softmax(-1) * costs).sum(-1) @ weights)
    assert independent_cost == pytest.approx(target, abs=1e-10)


@pytest.mark.parametrize("solver", [solve_gptq, solve_awq, solve_gsq])
def test_zero_calibration_and_representable_solution(solver):
    weight = torch.zeros(2, 64, dtype=torch.float64)
    zero_inputs = torch.zeros(10, 64, dtype=torch.float64)
    result = solver(weight, zero_inputs, "mxfp4")
    assert not result.weight.dequantize().any()
    assert reconstruction_error(result, weight, zero_inputs) == 0
    # Zero calibration has no curvature; returning a legal baseline is enough.
    nonzero = torch.linspace(-2, 2, 128, dtype=torch.float64).reshape(2, 64)
    result = solver(nonzero, zero_inputs, "mxfp4")
    assert torch.isfinite(result.weight.dequantize()).all()


def test_gsq_does_not_invent_improvement_for_identity_metric():
    weight, _, _ = problem(0)
    inputs = torch.eye(64, dtype=torch.float64)
    result = solve_gsq(weight, inputs, "mxfp4", steps=10)
    baseline = quantize_rtn(weight, "mxfp4")
    assert torch.equal(result.weight.payload, baseline.payload)
    assert result.history[0] == result.history[-1]


@pytest.mark.parametrize("solver", [solve_gptq, solve_awq, solve_gsq])
def test_invalid_calibration_and_nonfinite_weights_fail(solver):
    weight, inputs, _ = problem(0)
    with pytest.raises(ValueError, match="input dimension"):
        solver(weight, inputs[:, :-1], "mxfp4")
    with pytest.raises(ValueError, match="finite nonempty"):
        solver(weight * float("nan"), inputs, "mxfp4")
