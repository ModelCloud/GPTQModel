# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GSQ's metric factor must not fall back to a dense fp64 ``eigh`` (#4812).

``_metric_factor`` used to go straight from a failed undamped fp32 Cholesky to
``torch.linalg.eigh(h.double())`` on the accelerator. A real calibration Gram is
often only ill-conditioned rather than indefinite -- the last ``down_proj`` of a
1B model fails that Cholesky with 64 C4 samples -- and the fp64 eigen workspace
is about 14x the Hessian, so the run peaked at 4181 MiB on one layer where plain
GPTQ used 1324 MiB.

The factor now retries the fp32 Cholesky with a tiny diagonal jitter, and any
eigendecomposition that is still needed happens off the accelerator.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.gsq_scalar import _metric_factor


def _rank_deficient_gram(size: int, rank: int, device="cpu", seed: int = 4812) -> torch.Tensor:
    """A PSD Gram of rank ``rank < size``: the undamped Cholesky must fail."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn(size, rank, generator=generator, dtype=torch.float32)
    return (basis @ basis.t()).to(device)


def _assert_factor_reconstructs(factor: torch.Tensor, hessian: torch.Tensor) -> None:
    assert factor.shape == hessian.shape
    assert torch.isfinite(factor).all()
    assert torch.allclose(factor, factor.tril())
    rebuilt = factor @ factor.t()
    scale = hessian.abs().max().clamp_min(1e-6)
    # The jittered factor inverts ``H + eps*I``, so allow the jitter itself.
    assert (rebuilt - hessian.float()).abs().max() <= 1e-3 * scale


def test_ill_conditioned_gram_needs_no_eigendecomposition(monkeypatch):
    """The reported real case must stay in fp32 Cholesky."""

    calls = []
    real_eigh = torch.linalg.eigh

    def recording_eigh(matrix, *args, **kwargs):
        calls.append((matrix.dtype, matrix.device.type))
        return real_eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", recording_eigh)

    hessian = _rank_deficient_gram(128, 8)
    factor = _metric_factor(hessian)

    assert calls == [], "an ill-conditioned PSD Gram should not need an eigh"
    _assert_factor_reconstructs(factor, hessian)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")
@pytest.mark.cuda
def test_fallback_keeps_the_fp64_eigen_workspace_off_the_accelerator(monkeypatch):
    """When the jitter cannot save it, the fp64 eigh runs on the CPU."""

    calls = []
    real_eigh = torch.linalg.eigh

    def recording_eigh(matrix, *args, **kwargs):
        calls.append((matrix.dtype, matrix.device.type))
        return real_eigh(matrix, *args, **kwargs)

    def always_fails(matrix, *args, **kwargs):
        factor = torch.zeros_like(matrix)
        return factor, torch.tensor(1, dtype=torch.int32, device=matrix.device)

    monkeypatch.setattr(torch.linalg, "eigh", recording_eigh)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", always_fails)

    hessian = _rank_deficient_gram(256, 32, device="cuda")
    factor = _metric_factor(hessian)

    assert calls, "the forced path must reach the eigendecomposition"
    for dtype, device_type in calls:
        assert dtype is torch.float64
        assert device_type == "cpu", "fp64 eigh must not allocate on the accelerator"
    assert factor.device == hessian.device
    assert torch.isfinite(factor).all()


def test_indefinite_gram_still_raises(monkeypatch):
    """A genuinely indefinite Hessian is still rejected."""

    real_eigh = torch.linalg.eigh

    def always_fails(matrix, *args, **kwargs):
        return torch.zeros_like(matrix), torch.tensor(1, dtype=torch.int32, device=matrix.device)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", always_fails)
    monkeypatch.setattr(torch.linalg, "eigh", real_eigh)

    hessian = torch.eye(32, dtype=torch.float32)
    hessian[0, 0] = -1.0

    with pytest.raises(ValueError, match="positive-semidefinite"):
        _metric_factor(hessian)


def test_well_conditioned_gram_is_unchanged():
    """The fast path still returns the plain Cholesky factor."""

    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(64, 64, generator=generator, dtype=torch.float32)
    hessian = basis @ basis.t() + 64.0 * torch.eye(64)

    factor = _metric_factor(hessian)

    assert torch.allclose(factor, torch.linalg.cholesky(hessian), rtol=0, atol=0)
