# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4566: QQQ must apply ``Fallback.threshold`` first.

GPTQ resolves the documented low-sample policy before it finalizes or inverts
the Hessian. QQQ used to consult the same policy only inside the ``except``
around the inverse, so a module with ``0 < nsamples < threshold`` (a rarely
routed MoE expert) always ran the full error-propagation loop on a near
rank-deficient Hessian. These tests pin the pre-Hessian contract from both
sides: below the threshold the module must be bitwise equal to RTN, above it
QQQ must run GPTQ, and ``fallback=None`` must keep GPTQ even for a low-sample
module.
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


def _make_task(
    weight: torch.Tensor,
    fallback,
    *,
    samples: int,
    expected_nsamples: float,
    seed: int = 4566,
) -> tuple[QQQ, QQQConfig]:
    qcfg = QQQConfig(bits=4, group_size=-1, fallback=fallback)
    task = QQQ(_linear(weight), qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=-1)
    task.expected_nsamples = expected_nsamples
    torch.manual_seed(seed)
    task.add_batch(torch.randn(samples, weight.shape[1]), None)
    return task, qcfg


@pytest.mark.parametrize(
    "fallback",
    [Fallback(threshold="25%"), Fallback(threshold=16)],
    ids=["percent", "absolute"],
)
def test_below_threshold_module_uses_rtn_fallback(fallback):
    """Fewer captured samples than the threshold must select RTN, like GPTQ."""

    torch.manual_seed(4566)
    weight = torch.randn(8, 16) * 0.1
    # 25% of 64 (or the absolute 16) is 16; only 8 tokens reached this module.
    task, qcfg = _make_task(weight, fallback, samples=8, expected_nsamples=64)
    assert task.nsamples == 8

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _extra, nsamples = task.quantize(blocksize=16)

    assert nsamples == 8
    assert str(avg_loss).startswith("fallback(rtn)")
    assert task.H is None
    torch.testing.assert_close(
        quantized,
        _rtn_reference(weight, qcfg),
        rtol=0,
        atol=0,
    )


def test_above_threshold_module_still_uses_gptq(monkeypatch):
    """At or above the threshold the error-propagation path must still run."""

    torch.manual_seed(4566)
    weight = torch.randn(8, 16) * 0.1
    # 32 captured tokens are above the 16-token threshold.
    task, qcfg = _make_task(weight, Fallback(threshold="25%"), samples=32, expected_nsamples=64)
    assert task.nsamples == 32

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("an above-threshold module must not use fallback")

    monkeypatch.setattr(task, "_fallback_quantize", unexpected_fallback)

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _extra, nsamples = task.quantize(blocksize=16)

    assert nsamples == 32
    assert isinstance(avg_loss, float)
    assert not torch.equal(quantized, _rtn_reference(weight, qcfg))


def test_fallback_none_keeps_gptq_for_low_sample_module(monkeypatch):
    """``fallback=None`` must not synthesize a threshold policy."""

    torch.manual_seed(4566)
    weight = torch.randn(8, 16) * 0.1
    # If a default Fallback leaked in, 0.5% of 65536 would select RTN here.
    task, qcfg = _make_task(weight, None, samples=8, expected_nsamples=65536)
    assert task.fallback is None
    assert task.nsamples == 8

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("fallback=None must not select a fallback")

    monkeypatch.setattr(task, "_fallback_quantize", unexpected_fallback)

    quantized, _scale, _zero, _g_idx, _duration, avg_loss, _damp, _extra, nsamples = task.quantize(blocksize=16)

    assert nsamples == 8
    assert isinstance(avg_loss, float)
    assert not str(avg_loss).startswith("fallback")
    assert not torch.equal(quantized, _rtn_reference(weight, qcfg))
