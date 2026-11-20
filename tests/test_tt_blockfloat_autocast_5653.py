# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""TT block-float's dense fallback must ignore ambient autocast (#5653).

The CPU fallback dequantizes the block-float grid to FP32 and adds the stored
bias there, because the FP16/BF16 grid can overflow before the bias cancels
it.  An ambient BF16 autocast context narrowed those explicitly wide operands
again, so a projection whose true value is ``256 - 257 = -1`` returned ``0``.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.utils.backend import BACKEND


def _layer(weight: float = 1.0, bias: float = -257.0) -> TTBlockfloatLinear:
    source = torch.nn.Linear(16, 1)
    source.weight.data.fill_(weight)
    source.bias.data.fill_(bias)
    return TTBlockfloatLinear(
        bits=4,
        in_features=16,
        out_features=1,
        bias=True,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.float32,
    ).pack(source)


def _cancelling_input() -> torch.Tensor:
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[0, 0] = 256
    return x


@pytest.mark.parametrize("autocast_dtype", [torch.bfloat16, torch.float16])
def test_dense_fallback_ignores_ambient_autocast(autocast_dtype):
    layer = _layer()
    x = _cancelling_input()

    with torch.inference_mode():
        assert layer(x).item() == -1.0
        with torch.autocast("cpu", dtype=autocast_dtype):
            assert layer(x).item() == -1.0


def test_autocast_context_is_restored():
    layer = _layer()
    x = _cancelling_input()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        layer(x)
        # The fallback disables autocast only for its own reduction.
        assert torch.is_autocast_enabled("cpu")
    assert not torch.is_autocast_enabled("cpu")


def test_non_cancelling_values_are_unchanged():
    layer = _layer(weight=1.0, bias=0.5)
    x = torch.ones(1, 16, dtype=torch.bfloat16)

    with torch.inference_mode():
        plain = layer(x).item()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert layer(x).item() == plain == 16.5
