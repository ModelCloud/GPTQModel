# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Grouped GPTQ must not fit a whole-projection grid it then discards (#5830).

``GPTQ._quantize_impl`` used to run ``self.quantizer.find_params(W)`` over the
whole projection before the block pass.  That grid is only read by the
per-channel solve (``group_size == -1``); static groups fit every group on a
deep-copied quantizer and dynamic groups refit at each group boundary, so the
whole-projection range search (including its ``mse`` shrink search) was pure
overhead.  The tests pin the number and shapes of the weight range fits so a
regression reintroducing the discarded fit is caught, and check that a grid
fitted immediately before the first group fit cannot change the result.
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import Quantizer


ROWS = 24
COLUMNS = 320
GROUP = 64


def _task(
    *,
    group_size: int = GROUP,
    columns: int = COLUMNS,
    desc_act: bool = False,
    static_groups: bool = False,
    mock_quantization: bool = False,
) -> GPTQ:
    torch.manual_seed(20261120)
    layer = nn.Linear(columns, ROWS, bias=False)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=desc_act,
        static_groups=static_groups,
        mock_quantization=mock_quantization,
        mse=2.4,
        fallback=None,
    )
    task = GPTQ(layer, qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    # A fixed positive diagonal keeps every column calibration-live while
    # giving desc_act/act-group ordering something to reorder.
    task.H = torch.diag(torch.linspace(1.0, 4.0, columns))
    task.nsamples = 16
    return task


def _run(task: GPTQ, *, inject_full_fit: bool = False):
    """Quantize while recording every weight range fit's input shape."""

    fit = Quantizer.find_params
    shapes: Counter = Counter()
    injected = False

    def record(self, x, *args, **kwargs):
        nonlocal injected
        if kwargs.get("weight"):
            shapes[tuple(x.shape)] += 1
            if inject_full_fit and not injected:
                injected = True
                # Emulate the old discarded whole-projection fit right before
                # the first group fit: the group fit must overwrite its grid.
                full = torch.randn(
                    (x.shape[0], task.columns),
                    generator=torch.Generator().manual_seed(7),
                    dtype=x.dtype,
                    device=x.device,
                )
                fit(self, full, weight=True)
        return fit(self, x, *args, **kwargs)

    with patch.object(Quantizer, "find_params", record):
        result = task.quantize()
    return result, shapes


@pytest.mark.parametrize(
    "desc_act,static_groups",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["dynamic", "dynamic-desc-act", "static", "static-desc-act"],
)
def test_grouped_solve_fits_only_group_grids(desc_act, static_groups):
    result, shapes = _run(_task(desc_act=desc_act, static_groups=static_groups))

    # One fit per group; the whole projection is never measured.
    assert shapes == Counter({(ROWS, GROUP): COLUMNS // GROUP})
    assert all(width == GROUP for (_, width) in shapes)
    assert result[0].shape == (ROWS, COLUMNS)


def test_grouped_solve_partial_last_group_is_the_only_wide_fit():
    result, shapes = _run(_task(columns=300))

    assert shapes == Counter({(ROWS, GROUP): 4, (ROWS, 300 - 4 * GROUP): 1})
    assert result[0].shape == (ROWS, 300)


def test_mock_grouped_solve_fits_only_group_grids():
    result, shapes = _run(_task(columns=300, mock_quantization=True))

    assert shapes == Counter({(ROWS, GROUP): 4, (ROWS, 300 - 4 * GROUP): 1})
    assert result[0].shape == (ROWS, 300)


@pytest.mark.parametrize("desc_act", [False, True])
def test_per_channel_solve_still_fits_the_whole_projection_once(desc_act):
    result, shapes = _run(_task(group_size=-1, desc_act=desc_act))

    assert shapes == Counter({(ROWS, COLUMNS): 1})
    assert result[0].shape == (ROWS, COLUMNS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": GROUP},
        {"group_size": GROUP, "desc_act": True},
        {"group_size": GROUP, "static_groups": True},
        {"group_size": GROUP, "static_groups": True, "desc_act": True},
        {"group_size": GROUP, "mock_quantization": True},
    ],
    ids=["dynamic", "dynamic-desc-act", "static", "static-desc-act", "mock"],
)
def test_injected_whole_projection_grid_cannot_change_grouped_result(kwargs):
    reference, _ = _run(_task(**kwargs))
    injected, shapes = _run(_task(**kwargs), inject_full_fit=True)

    # The injected fit is not one of the solve's own measurements.
    assert all(width != COLUMNS for (_, width) in shapes)
    # Skip the solve duration (index 4): it is wall time, not a result.
    for index in (0, 1, 2, 3, 5, 6, 7):
        actual, expected = injected[index], reference[index]
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected), index
        else:
            assert actual == expected, index
