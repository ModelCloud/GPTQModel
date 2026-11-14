# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQ/QQQ fallback strategies must keep the group offset.

MIDPOINT/MEAN/MEDIAN/STDCLIP derive an affine zero point and clamp it to
``[0, maxq]``. A group that does not cross zero clamped to the boundary and
shifted every dequantized value by roughly the group minimum (about four code
steps for ``[0.01, 0.05]`` at 4 bits). Extending the fitted range to include
zero keeps the zero point representable and the error within half a step.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import (
    Fallback,
    FallbackStrategy,
    QuantizeConfig,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


STRATEGIES = [
    FallbackStrategy.MIDPOINT,
    FallbackStrategy.MEAN,
    FallbackStrategy.MEDIAN,
    FallbackStrategy.STDCLIP,
]


def _quantize(cls, strategy, values):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        fallback=Fallback(strategy=strategy, threshold=False),
    )
    linear = nn.Linear(128, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(values, dtype=torch.float32).reshape(1, 128))
    quantizer = cls(linear, qcfg)
    if cls is GPTQ:
        result = quantizer._fallback_quantize(strategy, 128)
    else:
        result = quantizer._fallback_quantize(strategy)
    dequant, scale, zero = result[0], result[1], result[2]
    return linear.weight, dequant, scale, zero


@pytest.mark.parametrize("cls", [GPTQ, QQQ], ids=["gptq", "qqq"])
@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.value)
@pytest.mark.parametrize(
    "values",
    [
        [0.01 + 0.04 * i / 127 for i in range(128)],   # positive-only
        [-0.05 + 0.04 * i / 127 for i in range(128)],  # negative-only
        [-0.02 + 0.05 * i / 127 for i in range(128)],  # crosses zero
    ],
    ids=["positive_only", "negative_only", "crosses_zero"],
)
def test_fallback_error_stays_within_half_a_step(cls, strategy, values):
    weight, dequant, scale, zero = _quantize(cls, strategy, values)
    step = float(scale.flatten()[0])
    assert step > 0
    error = (dequant - weight).abs().max().item()
    assert error <= 0.5 * step + 1e-6, (
        f"{cls.__name__}/{strategy.value}: max error {error:.6f} exceeds "
        f"half a step {0.5 * step:.6f} (zero point {float(zero.flatten()[0]):.2f})"
    )
