# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Ungrouped symmetric GPTQ must encode the negative endpoint as code 0 (#5756).

The symmetric INT4 grid stores zero point ``2 ** (bits - 1)`` and spans
``[-8, 7]``.  The mock-quantization branch used to clamp the rounded quotient
to ``[-7, 7]`` before adding the zero point, so the most negative endpoint
became code 1 and the reachable code 0 was never used.  The NaN-loss retry
re-enters the same branch with ``mock_quantization=True``.
"""

import torch
import torch.nn as nn

from gptqmodel.quantization.config import Fallback, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import QQQQuantizer


WEIGHT = torch.tensor([[-15.0, -9.0, -5.0, -1.0, 1.0, 5.0, 9.0, 15.0]])


def _task(weight=WEIGHT, *, group_size=-1, mock=False, fallback=None, quantizer_cls=None):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        mock_quantization=mock,
        fallback=fallback,
    )
    task = GPTQ(layer, qcfg)
    if quantizer_cls is not None:
        task.quantizer = quantizer_cls(qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    task.H = torch.eye(weight.shape[1])
    task.nsamples = weight.shape[1]
    return task


def _codes(weight, scale, zero):
    return weight.div(scale).round().add(zero).to(torch.int32)


def _symmetric_oracle(weight):
    # Symmetric INT4: the range is zero-centered, so scale = 2 * amax / 15
    # and the fixed zero point is 8.
    scale = 2.0 * weight.abs().max() / 15.0
    return torch.clamp(torch.round(weight / scale) + 8, 0, 15).to(torch.int32)


def test_ungrouped_symmetric_mock_encodes_negative_endpoint_as_code_zero():
    task = _task(mock=True)
    quantized, scale, zero, *_ = task.quantize()

    codes = _codes(quantized, scale, zero)

    assert scale.item() == 2.0
    assert zero.item() == 8.0
    assert codes.tolist() == [[0, 4, 6, 8, 8, 10, 12, 15]]
    assert codes[0, 0].item() == 0
    assert codes[0, -1].item() == 15
    assert torch.equal(codes, _symmetric_oracle(WEIGHT))


def test_ungrouped_symmetric_mock_matches_single_group_mock():
    ungrouped = _task(mock=True, group_size=-1)
    single_group = _task(mock=True, group_size=WEIGHT.shape[1])

    q_ungrouped, s_ungrouped, z_ungrouped, *_ = ungrouped.quantize()
    q_grouped, s_grouped, z_grouped, *_ = single_group.quantize()

    torch.testing.assert_close(q_ungrouped, q_grouped, rtol=0, atol=0)
    torch.testing.assert_close(s_ungrouped, s_grouped, rtol=0, atol=0)
    torch.testing.assert_close(z_ungrouped, z_grouped, rtol=0, atol=0)


def test_nan_loss_mock_retry_encodes_negative_endpoint_as_code_zero(monkeypatch):
    task = _task(mock=False, fallback=Fallback(threshold=True))
    original_inverse = GPTQ.hessian_inverse
    calls = {"count": 0}

    def poisoned_inverse(self, hessian):
        factor, damp = original_inverse(self, hessian)
        calls["count"] += 1
        if calls["count"] == 1:
            factor = factor.clone()
            factor[0, 0] = float("nan")
        return factor, damp

    monkeypatch.setattr(GPTQ, "hessian_inverse", poisoned_inverse)

    quantized, scale, zero, *_ = task.quantize()

    assert calls["count"] >= 1
    assert task.qcfg.mock_quantization is True
    codes = _codes(quantized, scale, zero)
    assert codes[0, 0].item() == 0
    assert torch.equal(codes, _symmetric_oracle(WEIGHT))


def test_qqq_signed_grid_keeps_its_own_clamp():
    """A signed grid (QQQ) must not be widened by the GPTQ zero-point fix."""

    task = _task(mock=True, quantizer_cls=QQQQuantizer)
    quantized, scale, zero, *_ = task.quantize()

    assert zero.item() == 0.0
    expected_codes = torch.clamp(torch.round(WEIGHT / scale), -7, 7).to(torch.int32)
    codes = quantized.div(scale).round().to(torch.int32)

    assert expected_codes[0, 0].item() == -7
    assert torch.equal(codes, expected_codes)
    torch.testing.assert_close(quantized, scale * expected_codes, rtol=0, atol=1e-6)
