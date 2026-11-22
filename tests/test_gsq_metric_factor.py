# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Peak-memory and numerical-equivalence coverage for GSQ's metric factor.

``_metric_factor`` used to run a dense fp64 ``eigh`` on the accelerator
whenever the undamped fp32 Cholesky failed, which happens on real
``down_proj`` Grams (#4812).  These tests pin the cheap damped retry path and
the one-buffer symmetrization.
"""

import pytest
import torch

import gptqmodel.quantization.gsq_scalar as gsq_scalar
from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import _metric_factor, refine_gptq_scalar


def _symmetrized(hessian):
    return (hessian.float() + hessian.float().T) * 0.5


def _reference_factor(hessian):
    """The pre-#4812 fp64 eigendecomposition fallback, evaluated on the host."""
    values, vectors = torch.linalg.eigh(_symmetrized(hessian).double())
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)).float()


def test_metric_factor_matches_cholesky_when_well_conditioned():
    torch.manual_seed(1)
    x = torch.randn(256, 128)
    hessian = x.T @ x + torch.eye(128) * 1e-3
    untouched = hessian.clone()

    factor = _metric_factor(hessian)
    reference = torch.linalg.cholesky(_symmetrized(hessian))

    # The undamped path is unchanged: bit-identical to the explicit Cholesky.
    assert torch.equal(factor, reference)
    assert torch.equal(hessian, untouched)


def test_metric_factor_rank_deficient_matches_reference_metric():
    torch.manual_seed(2)
    inputs = torch.randn(8, 64)
    hessian = inputs.T @ inputs
    with pytest.raises(torch._C._LinAlgError):
        torch.linalg.cholesky(_symmetrized(hessian))

    factor = _metric_factor(hessian)
    reference = _reference_factor(hessian)
    got_metric = factor @ factor.T
    reference_metric = reference @ reference.T
    relative = (got_metric - reference_metric).abs().max() / reference_metric.abs().max()

    # The retry adds at most 1.1e-5 * mean(diag) on this fixture, so the metric
    # differs from the fp64 reference only by that jitter relative to the
    # largest eigenvalue.
    assert relative <= 1e-5, float(relative)


def test_metric_factor_rank_deficient_preserves_gsq_selection(monkeypatch):
    rows, columns = 4, 32
    torch.manual_seed(7)
    target = torch.randn(rows, columns, dtype=torch.float16)
    scales = torch.full((rows, 4), 0.1, dtype=torch.float16)
    zeros = torch.full_like(scales, 8)
    groups = torch.arange(columns, dtype=torch.int32) // 8
    baseline = torch.zeros_like(target)
    inputs = torch.randn(6, columns)
    hessian = inputs.T @ inputs
    config = GSQConfig(
        enabled=True, steps=4, candidates=4, learn_scales=True,
        max_candidate_bytes=rows * 4 * columns * 4,
    )

    reference = _reference_factor(hessian)
    with monkeypatch.context() as patch:
        patch.setattr(gsq_scalar, "_metric_factor", lambda _h: reference.to(_h.device))
        expected = refine_gptq_scalar(
            baseline, scales, zeros, groups, target=target, hessian=hessian,
            bits=4, config=config,
        )
    actual = refine_gptq_scalar(
        baseline, scales, zeros, groups, target=target, hessian=hessian,
        bits=4, config=config,
    )

    # A different factor with the same Gram only rotates the projected
    # objective, so the selected grid codes and scales must be identical.
    assert torch.equal(actual.scales, expected.scales)
    assert torch.equal(actual.weight, expected.weight)
    assert torch.equal(actual.zeros, expected.zeros)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_metric_factor_cuda_peak_on_rank_deficient_gram():
    c = 4096
    torch.manual_seed(0)
    inputs = torch.randn(64, c, device="cuda", dtype=torch.float32)
    hessian = inputs.T @ inputs
    del inputs
    torch.cuda.synchronize()
    with pytest.raises(torch._C._LinAlgError):
        torch.linalg.cholesky(_symmetrized(hessian))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    factor = _metric_factor(hessian)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - baseline
    budget = 3 * c * c * 4

    assert factor.shape == (c, c)
    assert bool(torch.isfinite(factor).all())
    # The pre-#4812 fp64 eigh peaked at ~900 MiB above the Hessian here; the
    # damped Cholesky path keeps the peak to the symmetrized buffer plus one
    # factor buffer (2 fp32 copies, 128 MiB).
    assert extra <= budget, (
        f"_metric_factor peaked {extra / 2**20:.0f} MiB above the Hessian for "
        f"C={c}; budget is {budget / 2**20:.0f} MiB (three fp32 copies)"
    )
