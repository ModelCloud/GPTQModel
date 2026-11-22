# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The ``desc_act`` sink-row cap must also weight the GPTAQ/FOEM capture (#5769).

Plain GPTQ applies ``HessianConfig.sink_row_cap_median_multiple`` inside
``compute_hessian_xtx``. The GPTAQ and FOEM capture variants accumulate their
own Hessian/dXXT and bypassed it, so the option silently did nothing for those
solvers. These tests pin the capped Hessian and the capped native cross term
against an independent float64 oracle.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import (
    FOEMConfig,
    GPTAQConfig,
    HessianConfig,
    QuantizeConfig,
)
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ
from gptqmodel.quantization.gptq import GPTQ


ROWS = 17
COLUMNS = 16
CAP = 4.0
# fp32 accumulation against a float64 oracle; the fixture has 100x rows.
REL_TOL = 1e-5


def _fixture():
    """One outlier row among ``ROWS`` so the cap selects the 4x median bound."""

    generator = torch.Generator().manual_seed(494)
    rows = torch.randn(ROWS, COLUMNS, generator=generator)
    rows[0] *= 100
    weight = torch.randn(8, COLUMNS, generator=generator) * 0.1
    return rows, weight


def _cap_scale(rows):
    """Independent nearest-rank/median cap oracle, in float64."""

    rows64 = rows.double()
    norms = rows64.norm(dim=1)
    threshold = CAP * norms.median()
    return (threshold / norms).clamp(max=1.0)


def _capped(rows):
    return rows.double() * _cap_scale(rows).unsqueeze(1)


def _assert_close(actual, expected):
    """Compare against the fp64 oracle by relative Frobenius error."""

    error = ((actual - expected).norm() / expected.norm()).item()
    assert error < REL_TOL, f"relative Frobenius error {error:.3e} >= {REL_TOL:.1e}"


def _build(method, options, cap, rows, weight, native=None):
    """Construct the task; the constructor consumes the native capture."""

    config = QuantizeConfig(
        bits=4,
        group_size=COLUMNS,
        desc_act=True,
        act_group_aware=False,
        hessian=HessianConfig(sink_row_cap_median_multiple=cap),
        offload_to_disk=False,
        **options,
    )
    layer = nn.Linear(COLUMNS, 8, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    wrapped = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    wrapped.state[NATIVE_INPUTS_STATE_KEY] = [(0, (rows if native is None else native).clone())]
    return wrapped, method(wrapped, config)


def _capture(method, options, cap, rows, weight):
    """Run one capture and return ``(task, hessian, quantized_weight)``."""

    wrapped, task = _build(method, options, cap, rows, weight)
    task.quantizer.configure(perchannel=True)
    task.add_batch(rows, None)
    hessian = (task.finalize_hessian() if method is GPTQ else task.H).double().clone()
    return task, hessian, task.quantize()[0]


# GPTQ normalizes the finished Hessian by its token count; GPTAQ keeps the
# upstream per-request normalizer; FOEM normalizes per token.
def _native_options(method):
    if method is GPTQ:
        return {}, ROWS
    if method is GPTAQ:
        return {"gptaq": GPTAQConfig(alpha=0.25)}, 1
    return {"foem": FOEMConfig(alpha=0.0, beta=0.0)}, ROWS


@pytest.mark.parametrize("method", [GPTQ, GPTAQ, FOEM])
def test_sink_row_cap_weights_the_hessian(method):
    """The capped Hessian matches the oracle and reaches the weights."""

    rows, weight = _fixture()
    options, denominator = _native_options(method)

    capped_task, capped_h, capped_weight = _capture(method, options, CAP, rows, weight)
    raw_task, raw_h, raw_weight = _capture(method, options, 0.0, rows, weight)

    _assert_close(capped_h, (2.0 / denominator) * (_capped(rows).T @ _capped(rows)))
    _assert_close(raw_h, (2.0 / denominator) * (rows.double().T @ rows.double()))
    # The cap must actually reach the quantization result, not just the Hessian.
    assert (capped_weight != raw_weight).sum().item() > 0
    assert capped_task.sink_row_cap_median_multiple() == pytest.approx(CAP)
    assert raw_task.sink_row_cap_median_multiple() is None


@pytest.mark.parametrize("method", [GPTAQ, FOEM])
def test_sink_row_cap_weights_the_native_cross_term(method):
    """``dXXT`` is formed from the capped rows with the same factors."""

    rows, weight = _fixture()
    native = rows + 1.0
    options = (
        {"gptaq": GPTAQConfig(alpha=0.25)}
        if method is GPTAQ
        else {"foem": FOEMConfig(alpha=0.25)}
    )
    denominator = 1 if method is GPTAQ else ROWS

    wrapped, task = _build(method, options, CAP, rows, weight, native)
    task.quantizer.configure(perchannel=True)
    task.add_batch(rows, None)

    scale = _cap_scale(rows).unsqueeze(1)
    capped = rows.double() * scale
    capped_native = native.double() * scale
    expected = (2.0 / denominator) * ((capped_native - capped).T @ capped)
    _assert_close(task.dXXT.double(), expected)


@pytest.mark.parametrize("method", [GPTAQ, FOEM])
def test_sink_row_cap_off_matches_the_raw_oracle(method):
    """``sink_row_cap_median_multiple=0`` restores the uncapped capture."""

    rows, weight = _fixture()
    options = (
        {"gptaq": GPTAQConfig(alpha=0.25)}
        if method is GPTAQ
        else {"foem": FOEMConfig(alpha=0.25)}
    )
    denominator = 1 if method is GPTAQ else ROWS
    native = rows + 1.0

    wrapped, task = _build(method, options, 0.0, rows, weight, native)
    task.quantizer.configure(perchannel=True)
    task.add_batch(rows, None)

    rows64 = rows.double()
    _assert_close(task.H.double(), (2.0 / denominator) * (rows64.T @ rows64))
    _assert_close(
        task.dXXT.double(),
        (2.0 / denominator) * ((native.double() - rows64).T @ rows64),
    )


def test_feature_first_cap_scale_matches_row_first():
    """``feature_dim=0`` is the same policy on a transposed layout."""

    rows, weight = _fixture()
    wrapped, task = _build(GPTQ, {}, CAP, rows, weight)

    row_first = task.resolve_sink_row_cap_scale(rows, None)
    feature_first = task.resolve_sink_row_cap_scale(rows.t(), None, feature_dim=0)

    assert row_first is not None
    assert feature_first is not None
    torch.testing.assert_close(row_first, feature_first)
    # The 100x outlier is the only row the cap must touch.
    assert float(row_first.min()) < 1.0
    assert float(row_first[1:].max()) == pytest.approx(1.0)


def test_sink_row_cap_is_not_applied_without_desc_act():
    """The new capture hook stays a no-op for the default configuration."""

    rows, weight = _fixture()
    for method, options, denominator in (
        (GPTAQ, {"gptaq": GPTAQConfig(alpha=0.25)}, 1),
        (FOEM, {"foem": FOEMConfig(alpha=0.0, beta=0.0)}, ROWS),
    ):
        config = QuantizeConfig(
            bits=4,
            group_size=COLUMNS,
            desc_act=False,
            hessian=HessianConfig(sink_row_cap_median_multiple=CAP),
            offload_to_disk=False,
            **options,
        )
        layer = nn.Linear(COLUMNS, 8, bias=False)
        with torch.no_grad():
            layer.weight.copy_(weight)
        wrapped = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
        wrapped.state[NATIVE_INPUTS_STATE_KEY] = [(0, rows.clone())]
        task = method(wrapped, config)
        assert task.sink_row_cap_median_multiple() is None
        task.quantizer.configure(perchannel=True)
        task.add_batch(rows, None)
        _assert_close(
            task.H.double(),
            (2.0 / denominator) * (rows.double().T @ rows.double()),
        )
