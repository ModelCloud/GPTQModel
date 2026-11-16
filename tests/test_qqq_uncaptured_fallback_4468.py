# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4468: QQQ must not zero uncaptured modules or index a
missing inverse Hessian.

A module that receives no calibration rows (for example an unrouted MoE expert)
used to materialize a zero Hessian. ``_neutralize_dead_columns`` then treated
every input column as calibration-dead, zeroed the weight, and returned an
all-zero module even with RTN fallback configured. A selected RTN fallback
after a failed Hessian inverse had a parallel bug: ``Hinv`` was set to ``None``
but the RTN branch fell through to the block loop and raised
``TypeError: 'NoneType' object is not subscriptable``.
"""

import pytest
import torch
from torch import nn

import gptqmodel.quantization.qqq as qqq_module
from gptqmodel.quantization.config import Fallback, QQQConfig
from gptqmodel.quantization.qqq import QQQ


def _linear(weight: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def _rtn_reference(weight: torch.Tensor, qcfg: QQQConfig) -> torch.Tensor:
    """Quantize ``weight`` with QQQ's RTN quantizer, independent of the task."""

    quantizer = qqq_module.Quantizer(qcfg=qcfg)
    quantizer.configure(
        bits=qcfg.bits,
        perchannel=True,
        sym=qcfg.sym,
        groupsize=qcfg.group_size,
    )
    quantizer.find_params(weight.clone(), weight=True)
    return quantizer.quantize(weight.clone())


@pytest.mark.parametrize(
    "fallback",
    [Fallback(threshold=True), None],
    ids=["configured_rtn", "fallback_disabled"],
)
def test_uncaptured_qqq_module_uses_rtn_instead_of_zero_weights(fallback):
    """No calibration rows must not mean "every column is dead"."""

    torch.manual_seed(4468)
    weight = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    layer = _linear(weight)

    qcfg = QQQConfig(bits=4, group_size=-1, fallback=fallback)
    task = QQQ(layer, qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=-1)
    assert task.nsamples == 0

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _extra, nsamples = task.quantize(blocksize=4)

    assert nsamples == 0
    assert torch.isfinite(quantized).all()
    assert quantized.abs().max().item() > 0
    torch.testing.assert_close(
        quantized,
        _rtn_reference(weight, qcfg),
        rtol=0,
        atol=0,
    )
    assert str(avg_loss).startswith("fallback(rtn)")
    assert task.H is None


def test_qqq_selected_rtn_fallback_does_not_dereference_missing_inverse(monkeypatch):
    """A failed inverse with RTN selected must return RTN, not index ``None``."""

    torch.manual_seed(4468)
    layer = _linear(torch.linspace(-1.0, 1.0, 32).reshape(4, 8))
    qcfg = QQQConfig(
        bits=4,
        group_size=-1,
        damp_percent=1e-10,
        fallback=Fallback(threshold=2),
    )
    task = QQQ(layer, qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=-1)
    task.add_batch(torch.ones(1, 8), None)
    assert task.nsamples == 1

    def fail_cholesky(*_args, **_kwargs):
        raise RuntimeError("Hessian is not positive definite")

    # The sample count (1) is below the configured threshold (2), so QQQ must
    # select RTN. Before the fix it set Hinv=None and then sliced that None.
    monkeypatch.setattr(torch.linalg, "cholesky", fail_cholesky)

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _extra, nsamples = task.quantize(blocksize=4)

    assert nsamples == 1
    assert torch.isfinite(quantized).all()
    assert quantized.abs().max().item() > 0
    assert str(avg_loss).startswith("fallback(rtn)")
