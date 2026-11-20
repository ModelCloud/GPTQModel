# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dead Hessian columns must not set the per-channel grid (#5611).

A channel absent from every calibration sample carries no signal.  GPTQ and
QQQ zero its weight, but they used to fit the whole-row per-channel scale
first, so the discarded column's (arbitrarily large) value set the grid that
every live column was quantized with.  Changing only that column must leave
the scales and the live quantized weights unchanged.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.config import QQQConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


def _quantize(quantizer_cls, dead_weight: float, *, desc_act: bool):
    layer = torch.nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.weight[:, 0] = dead_weight
    config = (
        QQQConfig(bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None)
        if quantizer_cls is QQQ
        else QuantizeConfig(bits=4, group_size=-1, desc_act=desc_act, sym=True, fallback=None)
    )
    quantizer = quantizer_cls(layer, qcfg=config)
    if quantizer_cls is QQQ:
        quantizer.quantizer.configure(4, perchannel=True, sym=True, groupsize=-1)
    else:
        quantizer.quantizer.configure(perchannel=True)
    # Channel 0 is never activated.
    x = torch.eye(32)[1:].unsqueeze(0)
    quantizer.add_batch(x, layer(x))
    return quantizer.quantize()


@pytest.mark.parametrize("quantizer_cls", [GPTQ, QQQ], ids=["gptq", "qqq"])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("dead_weight", [0.0, 128.0, -128.0])
def test_dead_column_does_not_change_live_quantization(quantizer_cls, desc_act, dead_weight):
    if quantizer_cls is QQQ and desc_act:
        pytest.skip("QQQ does not support desc_act")
    reference = _quantize(quantizer_cls, 0.0, desc_act=desc_act)
    changed = _quantize(quantizer_cls, dead_weight, desc_act=desc_act)

    quantized, scale, zero = reference[0], reference[1], reference[2]
    other_quantized, other_scale, other_zero = changed[0], changed[1], changed[2]

    assert torch.equal(scale.float(), other_scale.float())
    assert torch.equal(zero.float(), other_zero.float())
    assert torch.equal(quantized.float(), other_quantized.float())
    # The discarded column itself is neutralized in both runs.
    assert torch.count_nonzero(quantized[:, 0]) == 0
