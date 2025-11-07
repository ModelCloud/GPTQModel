# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _build_gptq(damp_percent: float, damp_auto_increment: float) -> GPTQ:
    module = nn.Linear(2, 2, bias=False)
    qcfg = QuantizeConfig(damp_percent=damp_percent, damp_auto_increment=damp_auto_increment)
    return GPTQ(module, qcfg=qcfg)


def _damped_hessian(base: torch.Tensor, used_damp: float) -> torch.Tensor:
    """Reconstruct the damped matrix the solver actually inverted."""
    damped = base.clone()
    diag_view = damped.diagonal()
    mean = torch.mean(diag_view)
    diag_view.add_(used_damp * mean)
    return damped


def test_hessian_inverse_handles_rank_deficiency():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.05)
    device = gptq.module.target_device
    hessian = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device)

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert hessian_inv.shape == hessian.shape
    assert 0 < damp < 1
    assert torch.allclose(hessian_inv, torch.triu(hessian_inv))
    # Accuracy sanity check: recovered triangular factor should match the inverse of the damped matrix.
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(_damped_hessian(hessian, damp))
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-5, rtol=1e-4)


def test_hessian_inverse_returns_none_for_indefinite_matrix():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.25)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32, device=device)
    original = hessian.clone()

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is None
    assert damp == 1.0
    # The diagonal should reflect the final floor attempt.
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.1, device=device))
    # Off-diagonals must remain untouched.
    assert torch.allclose(hessian - torch.diag(hessian.diagonal()), original - torch.diag(original.diagonal()))


def test_hessian_inverse_matches_reference_for_positive_definite_matrix():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.05)
    device = gptq.module.target_device
    original = torch.tensor(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.5]],
        dtype=torch.float32,
        device=device,
    )
    hessian = original.clone()

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    # Ensure the solver does not mutate a healthy block.
    assert torch.allclose(hessian, original)

    damped = _damped_hessian(hessian, used_damp)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-6, rtol=1e-5)


def test_hessian_inverse_applies_diagonal_floor_for_semi_definite_input():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.0)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0, 0.01], [0.01, 0.0]], dtype=torch.float32, device=device)

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert used_damp == pytest.approx(gptq.qcfg.damp_percent)
    # Diagonal should be floored to a positive value so later steps see a PD matrix.
    assert torch.all(hessian.diagonal() > 0)
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.01, device=device), atol=1e-7, rtol=0.0)

    damped = _damped_hessian(hessian, used_damp)
    # Should be positive definite after flooring, so Cholesky succeeds.
    torch.linalg.cholesky(damped)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-5, rtol=1e-4)


def test_hessian_inverse_handles_singleton_flooring():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.0)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0]], dtype=torch.float32, device=device)

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert hessian_inv.shape == hessian.shape
    assert torch.allclose(hessian.diagonal(), torch.tensor([1e-6], dtype=torch.float32, device=device))

    damped = _damped_hessian(hessian, used_damp)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-6, rtol=1e-4)
