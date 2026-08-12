# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPTQ Hessian inverse validation (eac9082e)."""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


MPS_AVAILABLE = torch.backends.mps.is_available()


try:
    import gptqmodel.nn_modules.qlinear.pack_block_ext as pack_block_ext
    _HAS_PACK_BLOCK_EXT = hasattr(pack_block_ext, "hessian_inverse_cholesky_cpu")
except Exception:  # pragma: no cover - extension may not be built
    _HAS_PACK_BLOCK_EXT = False


@pytest.fixture
def gptq():
    return GPTQ(
        nn.Linear(8, 8, bias=False),
        qcfg=QuantizeConfig(bits=4, group_size=8, sym=True, desc_act=False),
    )


def test_hessian_inverse_rejects_non_finite_H(gptq):
    """A Hessian containing NaN/Inf is rejected and returns None."""

    gptq.name = "test"
    H = torch.eye(8)
    H[0, 0] = float("nan")
    Hinv, used_damp = gptq._compute_hessian_inverse_uncached(H)
    assert Hinv is None
    assert used_damp == 1.0


def test_hessian_inverse_succeeds_for_well_conditioned_matrix(gptq):
    """A well-conditioned positive-definite Hessian returns a finite inverse."""

    gptq.name = "test"
    H = torch.eye(8) * 4.0
    Hinv, used_damp = gptq._compute_hessian_inverse_uncached(H)
    assert Hinv is not None
    assert torch.isfinite(Hinv).all()
    assert (Hinv.diagonal() > 0).all()
    assert used_damp < 1.0


@pytest.mark.parametrize("release_input", [False, True])
def test_mock_hessian_inverse_obeys_release_without_claiming_external_ownership(release_input):
    """Mock inversion releases only the instance-owned Hessian when explicitly requested."""

    owned = torch.eye(4)
    gptq = GPTQ(nn.Linear(4, 4, bias=False), qcfg=QuantizeConfig(mock_quantization=True))
    gptq.H = owned
    factor, _ = gptq.mock_hessian_inverse(owned, release_input=release_input)
    assert torch.equal(factor, torch.eye(4))
    assert (gptq.H is None) is release_input

    external = torch.eye(4) * 2
    gptq.H = owned
    gptq.mock_hessian_inverse(external, release_input=True)
    assert gptq.H is owned


@pytest.mark.skipif(not _HAS_PACK_BLOCK_EXT, reason="pack_block_ext not available")
def test_hessian_inverse_rejects_non_finite_inverse_result(gptq, monkeypatch):
    """A Cholesky factor that inverts to NaN triggers the Hinv validity guard."""

    def _fake_hessian_inverse_cholesky_cpu(H, diag_delta):
        return (torch.full_like(H, float("nan")), torch.tensor(True))

    monkeypatch.setattr(
        pack_block_ext,
        "hessian_inverse_cholesky_cpu",
        _fake_hessian_inverse_cholesky_cpu,
    )

    gptq.name = "test"
    H = torch.eye(8)
    Hinv, used_damp = gptq._compute_hessian_inverse_uncached(H)
    assert Hinv is None
    assert used_damp == 1.0


@pytest.mark.skipif(not _HAS_PACK_BLOCK_EXT, reason="pack_block_ext not available")
def test_hessian_inverse_stops_when_damp_increment_is_below_dtype_resolution(monkeypatch):
    """A positive increment that rounds to zero must not create an infinite recovery loop."""
    calls = 0

    def _always_fail_hessian_inverse(H, diag_delta):
        nonlocal calls
        calls += 1
        return torch.empty_like(H), torch.tensor(False)

    monkeypatch.setattr(
        pack_block_ext,
        "hessian_inverse_cholesky_cpu",
        _always_fail_hessian_inverse,
    )
    qcfg = QuantizeConfig(damp_percent=0.05, damp_auto_increment=1e-50)
    quantizer = GPTQ(nn.Linear(2, 1, bias=False), qcfg=qcfg)
    quantizer.name = "tiny-damp-step"

    actual, used_damp = quantizer._compute_hessian_inverse_uncached(torch.eye(2))

    assert actual is None
    assert used_damp == 1.0
    assert calls == 7


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")
@pytest.mark.parametrize(("size", "seed"), [(8, 37), (64, 701), (128, 2027)])
def test_mps_hessian_inverse_is_bitwise_equal_to_canonical_factorization(size, seed):
    torch.manual_seed(seed)
    matrix = torch.randn(size, size, device="mps")
    hessian = matrix.T @ matrix / size + torch.eye(size, device="mps") * 0.1
    original = hessian.clone()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=size,
        damp_percent=0.05,
        damp_auto_increment=0.01,
        offload_to_disk=False,
    )
    quantizer = GPTQ(nn.Linear(size, 4, bias=False, device="mps"), qcfg=qcfg)

    expected_input = hessian.clone()
    expected_input.diagonal().add_(hessian.diagonal().mean() * 0.05)
    lower, info = torch.linalg.cholesky_ex(expected_input, upper=False)
    assert info.item() == 0
    dense_inverse = torch.empty_like(lower)
    torch.cholesky_inverse(lower, upper=False, out=dense_inverse)
    expected = torch.linalg.cholesky(dense_inverse, upper=True, out=dense_inverse)

    actual, used_damp = quantizer._compute_hessian_inverse_uncached(hessian)
    torch.mps.synchronize()

    assert used_damp == pytest.approx(0.05)
    assert torch.equal(actual, expected)
    assert torch.equal(hessian, original)


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is not available")
def test_mps_hessian_inverse_fast_failure_enters_damping_recovery():
    hessian = torch.eye(8, device="mps")
    hessian[:2, :2] = torch.tensor([[1.0, 1.2], [1.2, 1.0]], device="mps")
    original = hessian.clone()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=8,
        damp_percent=0.05,
        damp_auto_increment=0.05,
        offload_to_disk=False,
    )
    quantizer = GPTQ(nn.Linear(8, 4, bias=False, device="mps"), qcfg=qcfg)

    actual, used_damp = quantizer._compute_hessian_inverse_uncached(hessian)
    torch.mps.synchronize()

    assert actual is not None
    assert used_damp == pytest.approx(0.25)
    assert torch.isfinite(actual).all()
    assert (actual.diagonal() > 0).all()
    assert torch.equal(hessian, original)
