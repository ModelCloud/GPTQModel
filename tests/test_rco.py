# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU numerical oracles for the standalone RCO allocator."""

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch


# This module only depends on Torch/NumPy; do not load inference backends to
# exercise the allocator on a CPU-only development machine.
_SPEC = importlib.util.spec_from_file_location(
    'rco_under_test', Path(__file__).parents[1] / 'gptqmodel/quantization/rco.py')
rco = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rco)


def exhaustive_assignment(scores, weights, costs, target, resolution=500):
    """Independent FP64 Torch enumeration of the specified integer-grid problem."""
    scores = scores.double()
    weights, costs = weights.double(), costs.double()
    assignments = torch.tensor(list(itertools.product(range(len(costs)), repeat=len(weights))))
    units = torch.round(weights[:, None] * costs / weights.sum() * resolution).long()
    expenses = units[torch.arange(len(weights)), assignments].sum(-1)
    values = scores[torch.arange(len(weights)), assignments].sum(-1)
    feasible = expenses <= round(target * resolution)
    best = values[feasible].max()
    return assignments[feasible][values[feasible] == best], best


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('temperature', [1., .05, .01, .005])
def test_saturated_scores_preserve_budget_tradeoff(dtype, temperature):
    scores = torch.tensor([[0., 1.], [0., 2.]], dtype=dtype) / temperature
    weights, costs = torch.ones(2), torch.tensor([2., 3.])
    actual = rco.budget_constrained_argmax(scores, weights, costs, 2.5)
    expected, _ = exhaustive_assignment(scores, weights, costs, 2.5)
    assert expected.tolist() == [[0, 1]]
    assert torch.equal(actual, expected[0])
    if temperature <= .01:
        # Pin why routing the scores through probabilities is not equivalent.
        legacy = rco.round_with_budget(scores.softmax(-1), weights, costs, 2.5)
        assert legacy.tolist() == [1, 0]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_budget_constraint_rounding_handles_scores_on_mps(dtype):
    if dtype == torch.float64:
        pytest.skip("MPS does not provide FP64 storage")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    scores = torch.tensor([[0., 1.], [0., 2.]], device=device, dtype=dtype)
    actual = rco.budget_constrained_argmax(
        scores, torch.ones(2, device=device), torch.tensor([2., 3.], device=device), 2.5)
    expected, _ = exhaustive_assignment(scores.cpu(), torch.ones(2), torch.tensor([2., 3.]), 2.5)
    assert any(torch.equal(actual.cpu(), candidate) for candidate in expected)
    assert actual.device.type == "mps"


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_score_tie_and_representable_neighbors(dtype):
    one = torch.tensor(1., dtype=dtype)
    for second in (torch.nextafter(one, torch.tensor(0., dtype=dtype)), one,
                   torch.nextafter(one, torch.tensor(2., dtype=dtype))):
        scores = torch.stack((torch.stack((one * 0, one)), torch.stack((one * 0, second))))
        actual = rco.budget_constrained_argmax(scores, torch.ones(2), torch.tensor([2., 3.]), 2.5)
        expected, _ = exhaustive_assignment(scores, torch.ones(2), torch.tensor([2., 3.]), 2.5)
        assert any(torch.equal(actual, candidate) for candidate in expected)
        if second == one:
            assert actual.tolist() == [1, 0]  # retain the existing DP tie rule


@pytest.mark.parametrize('target', [2., 2.5, 3., 4.])
def test_weighted_multi_option_assignment_matches_exhaustive_oracle(target):
    generator = torch.Generator().manual_seed(23)
    weights, costs = torch.tensor([1., 2., 3., 4.]), torch.tensor([2., 3., 4.])
    for _ in range(8):
        scores = torch.randn(4, 3, generator=generator, dtype=torch.float64) * 200
        actual = rco.budget_constrained_argmax(scores, weights, costs, target)
        expected, _ = exhaustive_assignment(scores, weights, costs, target)
        assert any(torch.equal(actual, candidate) for candidate in expected)
        shifted = scores + torch.tensor([[1000.], [-300.], [0.], [80.]])
        assert torch.equal(actual, rco.budget_constrained_argmax(shifted, weights, costs, target))


def test_assignment_can_under_spend_and_legacy_api_is_preserved():
    scores = torch.tensor([[0., -2.], [0., -1.]])
    weights, costs = torch.ones(2), torch.tensor([2., 3.])
    expected = torch.zeros(2, dtype=torch.long)
    assert torch.equal(rco.budget_constrained_argmax(scores, weights, costs, 3.), expected)
    assert torch.equal(rco.round_with_budget(scores.softmax(-1), weights, costs, 3.), expected)
    with pytest.raises(RuntimeError, match='no feasible'):
        rco.budget_constrained_argmax(scores, weights, costs, 1.)


def test_achieved_bits_gathers_per_group_costs_and_uses_fp64():
    assignment = torch.tensor([0, 1, 1, 0])
    weights = torch.tensor([1., 1., 1., 1.])
    shared = torch.tensor([2., 4.])
    per_group = torch.tensor([[2., 4.], [2., 4.], [2., 3.], [2., 4.]])
    assert rco.achieved_bits(assignment, weights, shared) == 3.0
    assert rco.achieved_bits(assignment, weights, per_group) == 2.75
    # Large equal counts whose FP32 sum loses the fractional part.
    wide = torch.full((2,), 16777217.0)
    exact = rco.achieved_bits(torch.tensor([0, 1]), wide, torch.tensor([2., 4.]))
    assert exact == 3.0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_scores_are_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        rco.budget_constrained_argmax(torch.tensor([[0., value]]), torch.ones(1),
                                    torch.tensor([2., 3.]), 3.)


@pytest.mark.parametrize('groups', [64, 448, 800])
def test_compiled_and_torch_dp_agree_on_large_equal_group_grids(groups):
    """The C++ kernel is a drop-in for the Torch fallback, not a policy choice."""
    if rco._rco_dp_compiled() is None:
        pytest.skip("the compiled RCO DP needs a C++ toolchain")
    scores = torch.zeros(groups, 2, dtype=torch.float64)
    scores[:, 1] = 1.0
    weights = torch.ones(groups, dtype=torch.float64)
    costs = torch.tensor([2.0, 4.0], dtype=torch.float64)
    budget = round(2.5 * 500)
    units = torch.round(costs * weights[:, None] / weights.sum() * 500).long()
    kernel = rco._budget_dp(scores, units, budget, groups, 2)
    fallback = rco._budget_dp_torch(scores, units.tolist(), budget, groups, 2)
    assert torch.equal(kernel, fallback)
    assert torch.equal(
        rco.budget_constrained_argmax(scores, weights, costs, 2.5),
        rco.budget_constrained_argmax(scores, weights, costs, 2.5),
    )


def _interpolated_linear(candidates, *, dtype, device='cpu'):
    """Build a real one-input Linear and install the production parametrization."""
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(1, 1, bias=False).to(device=device, dtype=dtype)
    options = [candidate.to(dtype=dtype, device=device) for candidate in candidates]
    model.proj.weight.data = options[-1].clone()
    costs = torch.arange(2., 2. + len(options))
    owner = rco.InterpolatedLlamaLayers(
        model, {'proj': [option.clone() for option in options]},
        torch.ones(2), costs, control_device=device).setup()
    return model, owner


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('values', [[1., 1000.], [-60000., 60000.], [.25, -.75]])
@pytest.mark.parametrize('search_forward', [False, True])
def test_one_hot_forward_reproduces_the_selected_candidate(dtype, values, search_forward):
    """A hard assignment must evaluate exactly the candidate it selects.

    Deltas materialized in the model dtype round the difference, so
    ``reference + delta`` need not return the candidate, and two finite
    candidates can differ by more than the dtype's range. The search forward
    fed to the objective must match the selected candidate exactly, including
    the straight-through ``onehot - soft.detach() + soft`` form.
    """
    candidates = torch.tensor(values, dtype=dtype).reshape(-1, 1, 1)
    model, owner = _interpolated_linear(list(candidates), dtype=dtype)
    values_in = torch.ones(1, 1, dtype=dtype)
    soft = torch.linspace(.2, .8, len(values))
    try:
        for index in range(len(values)):
            onehot = torch.zeros(1, len(values))
            onehot[0, index] = 1
            owner.set_probs(onehot - soft + soft if search_forward else onehot)
            with torch.no_grad():
                actual = model.proj(values_in)
            assert actual.dtype == dtype
            assert bool(torch.isfinite(actual).all())
            assert actual.item() == candidates[index].to(dtype).item()
    finally:
        owner.cleanup()


def test_soft_interpolation_gradient_matches_wide_oracle():
    """Gradients keep the reference form ``dL/dp_k = <G, W_k - W_ref>``."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    candidates = [torch.randn(3, 2).to(dtype) * 40, torch.randn(3, 2).to(dtype) * 40,
                  torch.randn(3, 2).to(dtype) * 40]
    model, owner = _interpolated_linear(candidates, dtype=dtype)
    owner.alpha.data = torch.tensor([[.3, -.2, .1]])
    values_in = torch.randn(4, 2, dtype=dtype)
    upstream = torch.randn(4, 3, dtype=dtype)
    try:
        loss = (model.proj(values_in) * upstream).sum()
        alpha_gradient, = torch.autograd.grad(loss, owner.alpha)
        # Independent FP64 oracle: the same convex mix, softmax and chain rule.
        alpha64 = owner.alpha.detach().double()
        probs64 = torch.softmax(alpha64[0], dim=-1)
        weights64 = torch.stack([candidate.double() for candidate in candidates])
        grad_mixed = upstream.double().T @ values_in.double()
        grad_probs = torch.stack([(grad_mixed * weight).sum() for weight in weights64])
        jacobian = torch.diag(probs64) - probs64[:, None] * probs64[None, :]
        expected = jacobian.T @ grad_probs
        assert bool(torch.isfinite(alpha_gradient).all())
        torch.testing.assert_close(alpha_gradient.double()[0], expected, rtol=5e-3, atol=5e-3)
    finally:
        owner.cleanup()


def test_hard_selection_gradient_stays_finite_for_out_of_range_candidates():
    """Finite candidates that differ beyond the dtype range must not poison grads.

    The removed delta form materialized ``-inf`` for this pair, and the
    reference selection then produced ``nan``. The straight-through search
    gradient must instead use the exact wide difference of the two stored
    candidates.
    """
    dtype = torch.float16
    candidates = [torch.tensor([[-60000.]]), torch.tensor([[60000.]])]
    model, owner = _interpolated_linear(candidates, dtype=dtype)
    owner.alpha.data = torch.tensor([[.4, -.3]])
    values_in = torch.ones(1, 1, dtype=dtype)
    try:
        for index in range(len(candidates)):
            onehot = torch.zeros(1, len(candidates))
            onehot[0, index] = 1
            # Same straight-through form the search builds from ``alpha``.
            soft = torch.softmax(owner.alpha, dim=-1)
            owner.set_probs(onehot - soft.detach() + soft)
            loss = model.proj(values_in).sum()
            gradient, = torch.autograd.grad(loss, owner.alpha)
            assert bool(torch.isfinite(gradient).all())
            # ``values_in`` is all ones, so dL/dp is the column sum of each
            # candidate difference: -120000 for the first option and zero for
            # the reference, whose coefficient is the leftover mass. Chain
            # through the softmax of the same probabilities.
            probabilities = torch.softmax(owner.alpha.double()[0], dim=-1)
            jacobian = torch.diag(probabilities) - probabilities[:, None] * probabilities[None, :]
            expected = jacobian.T @ torch.tensor([-120000., 0.], dtype=torch.float64)
            torch.testing.assert_close(gradient.double()[0], expected, rtol=5e-3, atol=5e-3)
            with torch.no_grad():
                owner.clear_probs()
                owner.set_probs(onehot)
                assert model.proj(values_in).item() == (candidates[index] if index == 0 else 60000.)
    finally:
        owner.cleanup()


def test_grouped_interpolation_shares_one_decision_and_cleanup_restores():
    dtype, names = torch.float16, ['a', 'b']
    model = torch.nn.Module()
    sources = {}
    for name in names:
        setattr(model, name, torch.nn.Linear(2, 2, bias=False).to(dtype))
        sources[name] = getattr(model, name).weight.detach().clone()
    options = {name: [sources[name] * .25, sources[name]] for name in names}
    owner = rco.InterpolatedLlamaLayers(
        model, options, torch.ones(2), torch.tensor([2., 3.]),
        control_device='cpu', groups=[names]).setup()
    assert [owner.index_of(name) for name in names] == [0, 0]
    owner.set_probs(torch.tensor([[0., 1.]]))
    try:
        with torch.no_grad():
            for name in names:
                assert torch.equal(getattr(model, name).weight, sources[name])
                output = getattr(model, name)(torch.ones(1, 2, dtype=dtype))
                assert torch.equal(output, torch.ones(1, 2, dtype=dtype) @ sources[name].T)
    finally:
        owner.cleanup()
    for name in names:
        assert isinstance(getattr(model, name).weight, torch.nn.Parameter)
        assert torch.equal(getattr(model, name).weight, sources[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs an accelerator')
def test_interpolation_follows_a_device_move_and_cleanup_restores_it():
    """Candidates and the cleanup reference must follow ``model.to(device)``."""
    candidates = [torch.tensor([[1.]]), torch.tensor([[1000.]])]
    model, owner = _interpolated_linear(candidates, dtype=torch.bfloat16)
    model.to('cuda')
    owner.set_probs(torch.tensor([[1., 0.]], device='cuda'))
    try:
        with torch.no_grad():
            actual = model.proj(torch.ones(1, 1, dtype=torch.bfloat16, device='cuda'))
        assert actual.item() == 1. and actual.is_cuda
        # A dtype move must carry the candidates too: the mix stays in the
        # module dtype the contract requires.
        model.to(torch.float16)
        owner.clear_probs()
        owner.set_probs(torch.tensor([[0., 1.]], device='cuda'))
        with torch.no_grad():
            later = model.proj(torch.ones(1, 1, dtype=torch.float16, device='cuda'))
        assert later.item() == 1000. and later.dtype == torch.float16
        owner.set_probs(torch.tensor([[1., 0.]], device='cuda'))
        with torch.no_grad():
            first = model.proj(torch.ones(1, 1, dtype=torch.float16, device='cuda'))
        assert first.item() == 1.
    finally:
        owner.cleanup()
    assert model.proj.weight.is_cuda
    assert model.proj.weight.item() == 1000.


def test_tiled_gradient_matches_the_untiled_reduction(monkeypatch):
    """The bounded backward scratch must match the untiled reduction."""
    torch.manual_seed(3)
    candidates = [torch.randn(6, 4, dtype=torch.float32) * 30,
                  torch.randn(6, 4, dtype=torch.float32) * 30,
                  torch.randn(6, 4, dtype=torch.float32) * 30]
    model, owner = _interpolated_linear(candidates, dtype=torch.float16)
    owner.alpha.data = torch.tensor([[.4, -.1, .2]])
    values = torch.randn(2, 4, dtype=torch.float16)
    upstream = torch.randn(2, 6, dtype=torch.float16)
    try:
        loss = (model.proj(values) * upstream).sum()
        untiled_gradient, = torch.autograd.grad(loss, owner.alpha)
        # ``raising=False``: the whole mix is bounded per tile, so a revision
        # without the knob simply runs the untiled reduction twice.
        monkeypatch.setattr(rco, '_INTERPOLATION_CHUNK_ELEMENTS', 4, raising=False)
        tiled_loss = (model.proj(values) * upstream).sum()
        tiled_gradient, = torch.autograd.grad(tiled_loss, owner.alpha)
        # The tiled backward sums each tile separately, so only the FP32
        # summation order may differ from the untiled reduction.
        torch.testing.assert_close(tiled_gradient, untiled_gradient, rtol=1e-6, atol=1e-6)
        assert bool(torch.isfinite(tiled_gradient).all())
    finally:
        owner.cleanup()
