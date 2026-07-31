# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPTQ Hessian inverse validation (eac9082e)."""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


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
