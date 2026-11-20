# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Damping recovery must not materialize an unbounded candidate grid (#5790).

``damp_auto_increment`` has no lower bound beyond zero, so the legacy additive
grid grew in proportion to ``1 / increment`` before the first recovery probe.
Recovery almost always succeeds on the next candidate, so probe that value
first and bound the materialized grid.
"""

import tracemalloc
from unittest.mock import patch

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


RANK_ONE = [[1.0, 1.0], [1.0, 1.0]]
# Eigenvalues 1, 1, -1e-3: recovery needs a damp above ~1.5e-3, not the
# second candidate.
NEGATIVE_EIGENVALUE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1e-3]]
INDEFINITE = [[0.0, 1.0], [1.0, 0.0]]


def _build(damp_percent: float, increment: float) -> GPTQ:
    qcfg = QuantizeConfig(
        bits=4,
        group_size=2,
        damp_percent=damp_percent,
        damp_auto_increment=increment,
        offload_to_disk=False,
    )
    return GPTQ(nn.Linear(2, 2, bias=False), qcfg=qcfg)


def _legacy_grid(hessian: torch.Tensor, initial: float, increment: float):
    """The former linear walk: first success in repeated-addition order."""

    damp = initial
    while damp < 1.0:
        damped = hessian.clone()
        damped.diagonal().add_(damp * hessian.diagonal().mean())
        chol, info = torch.linalg.cholesky_ex(damped, check_errors=False)
        if int(info) == 0:
            return torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True), damp
        damp += increment
    return None, 1.0


def _peak_heap_bytes(task: GPTQ, hessian: torch.Tensor) -> int:
    tracemalloc.start()
    try:
        task.hessian_inverse(hessian.clone())
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("increment", (1e-5, 1e-6), ids=("1e-5", "1e-6"))
def test_first_retry_recovers_without_building_the_grid(increment):
    task = _build(1e-10, increment)
    hessian = torch.tensor(RANK_ONE, dtype=torch.float32)
    reference, expected_damp = _legacy_grid(hessian, 1e-10, increment)
    # The fixture must recover on the candidate right after the configured damp.
    assert expected_damp == 1e-10 + increment

    with patch.object(torch.linalg, "cholesky_ex", wraps=torch.linalg.cholesky_ex) as probes:
        factor, damp = task.hessian_inverse(hessian.clone())

    assert damp == expected_damp
    torch.testing.assert_close(factor, reference, atol=0, rtol=0)
    # Initial damp plus the single retry; the legacy grid walked every candidate.
    assert probes.call_count <= 3

    peak = _peak_heap_bytes(task, hessian)
    assert peak < (1 << 20), f"recovery peaked at {peak / (1 << 20):.1f} MiB of Python heap"


def test_recovery_order_still_matches_the_legacy_walk():
    task = _build(1e-3, 5e-4)
    hessian = torch.tensor(NEGATIVE_EIGENVALUE, dtype=torch.float32)
    reference, expected_damp = _legacy_grid(hessian, 1e-3, 5e-4)
    # First success is the third candidate, so the shortcut must not stop early.
    assert expected_damp == 1e-3 + 2 * 5e-4

    factor, damp = task.hessian_inverse(hessian.clone())

    assert damp == expected_damp
    torch.testing.assert_close(factor, reference, atol=0, rtol=0)


def test_grid_beyond_the_work_budget_fails_before_allocating():
    # 1e-6 leaves roughly a million candidates below 1.0: 30+ MiB of Python
    # floats before the fix, while every candidate fails for this matrix.
    task = _build(1e-10, 1e-6)
    hessian = torch.tensor(INDEFINITE, dtype=torch.float32)

    peak = _peak_heap_bytes(task, hessian)

    # Bounded by the work budget (65,536 candidates, ~2 MiB of Python floats)
    # instead of the ~31 MiB the full million-entry grid would need.
    assert peak < (4 << 20), f"recovery peaked at {peak / (1 << 20):.1f} MiB of Python heap"

    factor, damp = task.hessian_inverse(hessian)
    assert factor is None and damp == 1.0
    # Skipping the grid still runs the diagonal-floor fallback to the last step.
    assert torch.all(hessian.diagonal() > 0)
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.1), atol=0, rtol=0)
