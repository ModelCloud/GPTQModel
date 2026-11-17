# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue #4742 (desc_act sink rows saturate GPTQ columns).

``desc_act=True`` disables ``act_group_aware``, so a couple of
massive-activation ("attention sink") calibration rows dominate ``X^T X``.
Those rows own the largest Hessian diagonal, ``desc_act`` therefore quantizes
their columns first with a tiny ``Hinv[i, i]``, and the resulting error is
amplified into every column correlated with the sink. On Llama-3.1-8B that
saturates ``layers.1.mlp.down_proj`` columns (rel err ~5) and produces garbage
PPL on windows whose first token after BOS is a delimiter.

The fix caps each row's contribution to the Hessian by its norm, and only when
``desc_act`` has switched the group-aware protection off. These tests are
CPU-only and use a synthetic Hessian, so they need no checkpoint.
"""

import pytest
import torch
from torch import nn

from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


try:  # pragma: no cover - the constant only exists once the fix is present
    from gptqmodel.quantization.gptq import SINK_ROW_CAP_PERCENTILE
except ImportError:  # pragma: no cover - keeps the regression test runnable on `main`
    SINK_ROW_CAP_PERCENTILE = 99.9


COLS = 256
ROWS = 64
OUT_FEATURES = 32
# Two sink rows (BOS + the leading delimiter of the real failure), each with
# three saturated channels that are 700x the normal activations.
SINK_ROWS = (0, 1)
SINK_CHANNELS = ((17, 91, 203), (55, 140, 220))
SINK_MULTIPLE = 700.0
NORMAL_ROW_SCALE = 0.1
WEIGHT_SCALE = 0.05


def _fixture(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    activations = torch.randn(ROWS, COLS, generator=generator) * NORMAL_ROW_SCALE
    for row, channels in zip(SINK_ROWS, SINK_CHANNELS):
        activations[row, list(channels)] *= SINK_MULTIPLE
    weight = torch.randn(OUT_FEATURES, COLS, generator=generator) * WEIGHT_SCALE
    return activations, weight


def _task(weight: torch.Tensor, *, desc_act: bool, cap_multiple=None) -> GPTQ:
    module = nn.Linear(COLS, OUT_FEATURES, bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    hessian = HessianConfig()
    if cap_multiple is not None:
        hessian.sink_row_cap_median_multiple = cap_multiple
    task = GPTQ(
        module,
        QuantizeConfig(bits=4, group_size=128, desc_act=desc_act, sym=True, hessian=hessian),
    )
    task.quantizer.configure(perchannel=True)
    return task


def _hessian(weight: torch.Tensor, *, desc_act: bool, cap_multiple=None) -> torch.Tensor:
    activations, _ = _fixture()
    task = _task(weight, desc_act=desc_act, cap_multiple=cap_multiple)
    task.add_batch(activations, None)
    return task.finalize_hessian()


def _column_rel_error(weight: torch.Tensor, *, desc_act: bool, cap_multiple=None) -> torch.Tensor:
    activations, _ = _fixture()
    task = _task(weight, desc_act=desc_act, cap_multiple=cap_multiple)
    task.add_batch(activations, None)
    quantized, *_ = task.quantize(blocksize=128)
    return (quantized - weight).norm(dim=0) / weight.norm(dim=0)


def _median_row_norm(activations: torch.Tensor) -> float:
    norms = torch.linalg.vector_norm(activations.to(torch.float32), dim=1)
    return float(torch.kthvalue(norms, (activations.shape[0] + 1) // 2).values)


def test_descact_caps_sink_row_contribution_to_hessian():
    """The sink channels' Hessian diagonal must be bounded by the row cap.

    On unmodified ``main`` the sink channels carry ~3.5e5 on the diagonal (the
    raw 700x rows), which is what makes ``desc_act`` order them first.
    """

    activations, weight = _fixture()
    capped_task = _task(weight, desc_act=True)
    capped_task.add_batch(activations, None)
    capped = capped_task.finalize_hessian()

    uncapped_task = _task(weight, desc_act=True, cap_multiple=0)
    uncapped_task.add_batch(activations, None)
    uncapped = uncapped_task.finalize_hessian()

    threshold = 4.0 * _median_row_norm(activations)
    sink_channels = [channel for channels in SINK_CHANNELS for channel in channels]

    capped_diag = capped.diagonal()[sink_channels]
    uncapped_diag = uncapped.diagonal()[sink_channels]

    # Two capped sink rows can each contribute at most `threshold ** 2`; allow
    # a little slack for the 62 normal rows. `finalize_hessian` reports
    # `2 * X^T X / rows`, hence the extra factor.
    bound = 2.05 * threshold ** 2 * 2.0 / ROWS
    assert float(capped_diag.max()) <= bound
    assert float(uncapped_diag.max()) > 50.0 * bound

    # The capped Hessian is exactly `X^T X` over the row-clamped activations.
    scale = capped_task.resolve_sink_row_cap_scale(activations, None)
    assert scale is not None
    clamped = (activations * scale.unsqueeze(1)).double()
    expected = 2.0 * (clamped.T @ clamped) / ROWS
    torch.testing.assert_close(capped.double(), expected, rtol=1e-4, atol=1e-4)


def test_descact_sink_row_cap_bounds_worst_column():
    """The capped path keeps every column usable and is much better than raw.

    ``desc_act`` alone leaves a saturated column (worst rel err ~0.78 on this
    fixture, ~5 on the issue's 8B checkpoint); the cap brings it under 0.5 and
    well below the uncapped path.
    """

    _, weight = _fixture()

    capped = _column_rel_error(weight, desc_act=True)
    uncapped = _column_rel_error(weight, desc_act=True, cap_multiple=0)

    assert float(capped.max()) < 0.5
    assert float(capped.max()) < 0.75 * float(uncapped.max())


def test_descact_sink_row_cap_is_not_applied_without_desc_act():
    """Default GPTQ (``act_group_aware``) must stay bit-identical.

    ``resolve_sink_row_cap_scale`` is the single gate: with ``desc_act=False``
    it returns ``None``, so the Hessian and the quantized weights do not change
    even when the cap is configured.
    """

    activations, weight = _fixture()
    task = _task(weight, desc_act=False)
    assert task.qcfg.act_group_aware is True
    assert task.sink_row_cap_median_multiple() is None
    assert task.resolve_sink_row_cap_scale(activations, None) is None

    default = _column_rel_error(weight, desc_act=False)
    disabled = _column_rel_error(weight, desc_act=False, cap_multiple=0)
    assert torch.equal(default, disabled)

    default_hessian = _hessian(weight, desc_act=False)
    disabled_hessian = _hessian(weight, desc_act=False, cap_multiple=0)
    assert torch.equal(default_hessian, disabled_hessian)


def test_descact_sink_row_cap_can_be_disabled():
    """``sink_row_cap_median_multiple=0`` restores the exact raw Hessian."""

    activations, weight = _fixture()
    task = _task(weight, desc_act=True, cap_multiple=0)
    assert task.sink_row_cap_median_multiple() is None
    assert task.resolve_sink_row_cap_scale(activations, None) is None

    task.add_batch(activations, None)
    expected = 2.0 * (activations.to(torch.float64).T @ activations.to(torch.float64)) / ROWS
    torch.testing.assert_close(task.finalize_hessian().double(), expected, rtol=1e-4, atol=1e-4)


def test_descact_sink_row_cap_threshold_uses_percentile_and_median():
    """The cap threshold is ``min(p99.9, N x median)`` and is chunk independent."""

    activations, weight = _fixture()
    task = _task(weight, desc_act=True)
    scale_whole = task.resolve_sink_row_cap_scale(activations, None)
    scale_chunked = task.resolve_sink_row_cap_scale(activations, 7)
    assert scale_whole is not None
    torch.testing.assert_close(scale_whole, scale_chunked, rtol=0, atol=0)

    norms = torch.linalg.vector_norm(activations.to(torch.float32), dim=1)
    median = torch.kthvalue(norms, (ROWS + 1) // 2).values
    rank = max(1, min(ROWS, int(torch.tensor(SINK_ROW_CAP_PERCENTILE / 100.0 * ROWS).ceil())))
    percentile = torch.kthvalue(norms, rank).values
    threshold = torch.minimum(percentile, median * 4.0)

    sink_rows = torch.tensor(SINK_ROWS)
    torch.testing.assert_close(
        scale_whole[sink_rows] * norms[sink_rows],
        threshold.expand(len(SINK_ROWS)),
        rtol=1e-6,
        atol=1e-6,
    )
    # Normal rows are untouched.
    normal = torch.ones(ROWS, dtype=torch.bool)
    normal[sink_rows] = False
    assert bool((scale_whole[normal] == 1.0).all())


def test_descact_sink_row_cap_config_default_and_validation():
    """The new Hessian knob defaults to 4x median and validates its input."""

    assert HessianConfig().sink_row_cap_median_multiple == 4.0
    assert HessianConfig(sink_row_cap_median_multiple=8).sink_row_cap_median_multiple == 8.0
    assert HessianConfig(sink_row_cap_median_multiple=0).sink_row_cap_median_multiple == 0.0
    assert HessianConfig(sink_row_cap_median_multiple=None).sink_row_cap_median_multiple is None

    for bad in (-1, -0.5, "4", True, [4.0]):
        with pytest.raises(ValueError, match="sink_row_cap_median_multiple"):
            HessianConfig(sink_row_cap_median_multiple=bad)


def test_descact_sink_row_cap_survives_serialization():
    """A reload must not silently restore the default cap."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        hessian=HessianConfig(sink_row_cap_median_multiple=2.5),
    )
    payload = qcfg.to_dict()
    assert payload["meta"]["hessian"]["sink_row_cap_median_multiple"] == 2.5
    assert "sink_row_cap_median_multiple" not in payload

    reloaded = QuantizeConfig(**payload)
    assert reloaded.hessian.sink_row_cap_median_multiple == 2.5
