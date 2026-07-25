# GPU=-1
import math

import pytest
import torch

from gptqmodel.quantization.slq.metrics import (
    expected_acceptance_rate,
    kl_divergence_topk,
    next_token_distributions,
)


def test_ear_identical_distributions():
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = p.clone()
    ear = expected_acceptance_rate(p, q)
    assert torch.allclose(ear, torch.tensor(1.0))


def test_ear_disjoint_top2():
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.1, 0.2, 0.7]])
    # top-2 under p are tokens 0 and 1; overlap = min(0.7, 0.1) + min(0.2, 0.2) = 0.3
    # p mass in top-2 = 0.9, so EAR = 0.3 / 0.9
    ear = expected_acceptance_rate(p, q, top_k=2)
    expected = 0.3 / 0.9
    assert math.isclose(ear.item(), expected, rel_tol=1e-5)


def test_ear_topk_restricted_full_mass():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.4, 0.3, 0.3]])
    ear_top2 = expected_acceptance_rate(p, q, top_k=2)
    ear_all = expected_acceptance_rate(p, q, top_k=None)
    assert ear_all >= ear_top2
    assert 0.0 <= ear_top2 <= 1.0
    assert 0.0 <= ear_all <= 1.0


def test_kl_topk_zero_for_identical():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    kl = kl_divergence_topk(p, p, top_k=2)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_topk_matches_analytic():
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.5, 0.3, 0.2]])
    kl = kl_divergence_topk(p, q, top_k=2)
    # top-2 under p are tokens 0,1; mass = 0.9
    # KL = sum p_i log(p_i/q_i) for i=0,1 over p mass
    expected = (0.7 * math.log(0.7 / 0.5) + 0.2 * math.log(0.2 / 0.3)) / 0.9
    assert math.isclose(kl.item(), expected, rel_tol=1e-5)


def test_next_token_distributions_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    p, q = next_token_distributions(logits, logits, temperature=0.5)
    assert torch.allclose(p, q)
    assert p.shape == logits.shape
    assert torch.allclose(p.sum(dim=-1), torch.tensor(1.0))


def test_ear_batch_mean():
    p = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
    q = p.clone()
    ear = expected_acceptance_rate(p, q)
    assert torch.allclose(ear, torch.tensor(1.0))


def test_ear_handles_numerical_inaccuracy():
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.6999, 0.2001, 0.1]])
    ear = expected_acceptance_rate(p, q)
    assert 0.99 < ear.item() <= 1.0


@pytest.mark.parametrize("top_k", [1, 2, 3, None])
def test_ear_topk_bounds(top_k):
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.2, 0.3, 0.5]])
    ear = expected_acceptance_rate(p, q, top_k=top_k)
    assert 0.0 <= ear.item() <= 1.0
