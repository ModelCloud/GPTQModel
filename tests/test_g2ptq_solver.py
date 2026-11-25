# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Oracle tests for the G2PTQ solver math (no model, no GPU required)."""

import math

import pytest
import torch

from gptqmodel.quantization.g2ptq import (
    g2ptq_compensate,
    trust_region_beta,
)


def _spd(columns: int, groups: int, dtype=torch.float64, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn((groups, columns, columns), generator=generator, dtype=dtype)
    hessian = a @ a.transpose(1, 2) + 0.5 * torch.eye(columns, dtype=dtype)
    return hessian


def _fake_quantize(weight: torch.Tensor, levels: int = 15) -> torch.Tensor:
    scale = weight.abs().amax().clamp_min(1e-8) / levels
    return (weight / scale).round().clamp(-levels, levels) * scale


def _brute_force_columns(weight, hessian, gradient, beta, levels=15):
    """Eq. 16 evaluated eagerly with an explicit Schur complement per step.

    After columns ``0 .. t-1`` are quantized the remaining sub-problem uses
    ``(H_{t:, t:})^-1``, whose upper Cholesky factor is ``U[t:, t:]``. The
    trailing principal block of ``H^-1`` is exactly that inverse, so the
    reference needs no factor and no recursion: it rebuilds ``F^(t)`` from
    scratch at every step.
    """

    groups, rows, columns = weight.shape
    work = weight.clone()
    q = torch.zeros_like(weight)
    for column in range(columns):
        w = work[:, :, column]
        quantized = _fake_quantize(w, levels)
        q[:, :, column] = quantized
        # After columns ``0 .. t-1`` are quantized the remaining sub-problem
        # uses ``(H_{t:, t:})^-1``; Algorithm 1 reaches it through rows of
        # ``U`` instead of materializing it.
        sub = torch.linalg.inv(hessian[:, column:, column:])
        diagonal = sub[:, 0, 0].unsqueeze(-1)
        row = sub[:, 0, :].unsqueeze(1)
        # F^(t) = G[:, t:] (H_{t:, t:})^-1, left padded to the full width.
        f = torch.bmm(gradient[:, :, column:], sub)
        delta = ((quantized - w) / diagonal).unsqueeze(-1) * row
        delta = delta + beta * (f[:, :, 0:1] / diagonal.unsqueeze(-1) * row - f)
        work[:, :, column:] += delta
    return q


def test_g2ptq_matches_eq15_oracle():
    torch.manual_seed(0)
    groups, rows, columns = 2, 6, 8
    weight = torch.randn((groups, rows, columns), dtype=torch.float64)
    hessian = _spd(columns, groups)
    gradient = torch.randn((groups, rows, columns), dtype=torch.float64) * 1e-3
    beta = torch.rand((groups, rows, 1), dtype=torch.float64)

    inverse = torch.linalg.inv(hessian)
    factor = torch.linalg.cholesky(inverse, upper=True)
    assert torch.allclose(factor.transpose(1, 2) @ factor, inverse, atol=1e-10)

    expected = _brute_force_columns(weight, hessian, gradient, beta)

    work = weight.clone()

    def quantize(column):
        return _fake_quantize(work[:, :, column])

    actual, error = g2ptq_compensate(
        work, factor, gradient, beta, quantize, blocksize=columns
    )

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(error).all()


@pytest.mark.parametrize("blocksize", [1, 2, 3, 5, 8])
def test_g2ptq_lazy_batch_matches_per_column(blocksize):
    torch.manual_seed(1)
    groups, rows, columns = 3, 5, 11
    weight = torch.randn((groups, rows, columns), dtype=torch.float64)
    hessian = _spd(columns, groups)
    gradient = torch.randn((groups, rows, columns), dtype=torch.float64) * 1e-2
    beta = torch.rand((groups, rows, 1), dtype=torch.float64)
    inverse = torch.linalg.inv(hessian)
    factor = torch.linalg.cholesky(inverse, upper=True)

    def run(size):
        work = weight.clone()

        def quantize(column):
            return _fake_quantize(work[:, :, column])

        return g2ptq_compensate(work, factor, gradient, beta, quantize, blocksize=size)[0]

    reference = run(columns)
    actual = run(blocksize)
    assert torch.allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_g2ptq_zero_gradient_is_gptq():
    torch.manual_seed(2)
    groups, rows, columns = 2, 4, 7
    weight = torch.randn((groups, rows, columns), dtype=torch.float64)
    hessian = _spd(columns, groups)
    gradient = torch.zeros((groups, rows, columns), dtype=torch.float64)
    beta = torch.zeros((groups, rows, 1), dtype=torch.float64)
    factor = torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True)

    work = weight.clone()

    def quantize(column):
        return _fake_quantize(work[:, :, column])

    actual, _ = g2ptq_compensate(work, factor, gradient, beta, quantize, blocksize=3)
    expected = _brute_force_columns(weight, hessian, gradient, beta)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_trust_region_beta_range_and_edges():
    torch.manual_seed(3)
    groups, rows, columns = 2, 4, 6
    weight = torch.randn((groups, rows, columns), dtype=torch.float64)
    hessian = _spd(columns, groups)
    inverse = torch.linalg.inv(hessian)
    gradient = torch.randn((groups, rows, columns), dtype=torch.float64) * 1e-3
    fake = _fake_quantize(weight)

    beta = trust_region_beta(weight, inverse, gradient, fake, alpha=0.05)
    assert beta.shape == (groups, rows, 1)
    assert torch.isfinite(beta).all()
    assert (beta >= 0).all() and (beta <= 1).all()

    # A zero budget disables the first-order step entirely.
    zero_beta = trust_region_beta(weight, inverse, gradient, fake, alpha=0.0)
    assert torch.equal(zero_beta, torch.zeros_like(zero_beta))

    # A zero gradient leaves no first-order direction. ``c`` collapses to the
    # clamped budget ``2 * alpha * |dL_gptq|``, so every row sits on the
    # boundary and beta saturates at one; the gradient step it scales is zero.
    zero_gradient = trust_region_beta(
        weight, inverse, torch.zeros_like(gradient), fake, alpha=0.05
    )
    assert torch.isfinite(zero_gradient).all()
    assert (zero_gradient >= 0).all() and (zero_gradient <= 1).all()

    # A larger budget can only relax the bound, never tighten it, and every
    # row still saturates at the trust-region ceiling of one.
    large = trust_region_beta(weight, inverse, gradient, fake, alpha=1e6)
    assert (large >= 0).all() and (large <= 1).all()
    assert (large >= beta).all()
    assert torch.isclose(large.max(), torch.ones((), dtype=large.dtype))


def test_trust_region_beta_is_finite_with_zero_diagonal():
    groups, rows, columns = 1, 2, 4
    weight = torch.randn((groups, rows, columns), dtype=torch.float64)
    inverse = torch.eye(columns, dtype=torch.float64).expand(groups, -1, -1).clone()
    inverse[:, 2, 2] = 0.0
    gradient = torch.randn((groups, rows, columns), dtype=torch.float64)
    fake = weight + 0.1
    beta = trust_region_beta(weight, inverse, gradient, fake, alpha=0.1)
    assert torch.isfinite(beta).all()
    assert (beta >= 0).all() and (beta <= 1).all()


def test_trust_region_beta_clamps_c_to_boundary():
    """When ``c <= 2 * alpha * |dL|`` the paper sets beta to one."""

    groups, rows, columns = 1, 1, 3
    inverse = torch.eye(columns, dtype=torch.float64).unsqueeze(0)
    weight = torch.zeros((groups, rows, columns), dtype=torch.float64)
    # A tiny gradient makes c small relative to the quantization loss.
    gradient = torch.full((groups, rows, columns), 1e-8, dtype=torch.float64)
    fake = torch.full((groups, rows, columns), 1.0, dtype=torch.float64)
    beta = trust_region_beta(weight, inverse, gradient, fake, alpha=0.5)
    assert math.isclose(float(beta.item()), 1.0, abs_tol=1e-9)
