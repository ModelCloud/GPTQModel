# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Regression tests for #5692: the GSQ candidate budget must not change the export.

``refine_gptq_scalar`` processes output rows in candidate chunks whose width is
derived from ``max_candidate_bytes``. The scale trial used to be gated on
whether *any* row in the chunk improved its codes, so a row whose codes were
already optimal never tried its closed-form scale unless a neighbour in the
same chunk happened to improve. With an identity Gram the least-squares scale
for the first row below is exactly 1.25 at centered code 1, so a 32-byte budget
(one row per chunk) exported 1.0 while a 64-byte budget (both rows in one
chunk) exported 1.25. The decision must be per row, independent of chunking.
"""

import torch

from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import refine_gptq_scalar


BUDGETS = (32, 64)


def _run(budget, weight, target):
    scales = torch.ones((weight.shape[0], 1))
    config = GSQConfig(
        enabled=True,
        steps=3,
        candidates=4,
        learn_scales=True,
        max_candidate_bytes=budget,
    )
    return refine_gptq_scalar(
        weight,
        scales,
        scales * 8,
        torch.zeros(weight.shape[1], dtype=torch.int32),
        target=target,
        hessian=torch.eye(weight.shape[1]),
        bits=4,
        config=config,
    )


def test_candidate_budget_does_not_change_exported_weights_or_scores():
    weight = torch.ones((2, 2))
    target = torch.tensor([[1.25, 1.25], [2.25, 2.25]])

    narrow = _run(32, weight, target)
    wide = _run(64, weight, target)

    # One row per candidate chunk (32 bytes) must reach the same export as two
    # rows per chunk (64 bytes).
    torch.testing.assert_close(narrow.weight, wide.weight, rtol=0, atol=0)
    torch.testing.assert_close(narrow.scales, wide.scales, rtol=0, atol=0)
    # ``before`` aggregates chunk losses as a weighted average, so the two
    # chunk counts differ only by FP32 re-association (~1e-8 absolute).
    torch.testing.assert_close(torch.tensor(narrow.before), torch.tensor(wide.before),
                               rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(torch.tensor(narrow.after), torch.tensor(wide.after),
                               rtol=1e-6, atol=1e-8)

    # Row 0 cannot improve its centered code, but its scale is exactly 1.25.
    # Row 1 moves its code and refits its scale to 1.125.
    torch.testing.assert_close(narrow.weight, target, rtol=0, atol=0)
    torch.testing.assert_close(narrow.scales, torch.tensor([[1.25], [1.125]]), rtol=0, atol=0)
    assert narrow.after == 0.0
    assert narrow.after < narrow.before


def test_improvable_row_still_learns_its_scale():
    weight = torch.ones((1, 2))
    target = torch.tensor([[2.25, 2.25]])

    result = _run(32, weight, target)

    # The code sweep moves centered code 1 -> 2, then the closed-form refit
    # rescales the group so the exported weight hits the teacher exactly. A
    # per-row gate must not skip the scale trial that follows a code move.
    torch.testing.assert_close(result.weight, target, rtol=0, atol=0)
    torch.testing.assert_close(result.scales, torch.tensor([[1.125]]), rtol=0, atol=0)
    assert result.after == 0.0
    assert result.after < result.before
