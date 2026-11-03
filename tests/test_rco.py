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


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_scores_are_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        rco.budget_constrained_argmax(torch.tensor([[0., value]]), torch.ones(1),
                                    torch.tensor([2., 3.]), 3.)
