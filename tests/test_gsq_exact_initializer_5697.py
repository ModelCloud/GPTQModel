# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GSQ must not corrupt a GPTQ initializer that is already exact (#5697).

``_refine_rows`` used to recover export codes with FP16 arithmetic:
``zero_point * scale`` for zero-point 8 and scale 8192 is 65536, which
overflows FP16 to ``inf`` before the zero is added.  The recovered code then
clamped to 14 and an exactly representable zero was exported as 49152, while
the reported baseline error stayed at 5 (nonzero) for an initializer whose true
error is exactly zero.

The real ``pack_original`` widens the weight, scale and zero-point to FP32
before multiplying, so GSQ now shares ``affine_codes`` and cannot disagree with
the packer.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.gsq_scalar import refine_gptq_scalar

COLUMNS = 8
SCALE = 8192.0
ZERO = 8
# Every value below sits exactly on the checkpoint grid ``(code - 8) * 8192``.
GRID_VALUES = (0.0, 7 * SCALE, -7 * SCALE)


def _fixture(value):
    g_idx = torch.zeros(COLUMNS, dtype=torch.int32)
    weight = torch.zeros(COLUMNS, COLUMNS, dtype=torch.float16)
    weight[:, 1] = value
    scales = torch.full((COLUMNS, 1), SCALE, dtype=torch.float16)
    zeros = torch.full_like(scales, ZERO)
    return weight, scales, zeros, g_idx


def _packed_code(weight, scales, zeros, g_idx, column=1):
    """Extract one nibble of the packed word through the real GPTQ packer.

    GPTQ packs eight 4-bit input columns into each int32 word, so input column
    ``column`` lives in nibble ``column`` of ``qweight[0, 0]``.
    """

    source = torch.nn.Linear(COLUMNS, COLUMNS, bias=False, dtype=torch.float16)
    source.weight.data.copy_(weight)
    module = TorchLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=COLUMNS,
        out_features=COLUMNS,
        bias=False,
    )
    module.pack_original(source, scales, zeros, g_idx)
    return int((module.qweight[0, 0].item() >> (4 * column)) & 0xF)


@pytest.mark.parametrize("value", GRID_VALUES)
def test_the_initializer_round_trips_through_the_packer(value):
    """The control: each fixture value really is exactly representable."""

    weight, scales, zeros, g_idx = _fixture(value)
    code = _packed_code(weight, scales, zeros, g_idx)
    assert (code - ZERO) * SCALE == value


@pytest.mark.parametrize("value", GRID_VALUES)
def test_gsq_keeps_an_exact_initializer_exact(value):
    weight, scales, zeros, g_idx = _fixture(value)
    before_code = _packed_code(weight, scales, zeros, g_idx)

    result = refine_gptq_scalar(
        weight,
        scales,
        zeros,
        g_idx,
        target=weight,
        hessian=torch.eye(COLUMNS),
        bits=4,
        config=GSQConfig(enabled=True, steps=1, candidates=4, learn_scales=False),
    )

    # The initializer is a fixed point of the objective, so nothing is gained
    # and the row must be retained verbatim.
    assert result.before == 0.0
    assert result.after == 0.0
    assert torch.equal(result.weight[:, 1], weight[:, 1])
    assert _packed_code(result.weight, result.scales, result.zeros, result.g_idx) == before_code
    assert (before_code - ZERO) * SCALE == value


def test_gsq_still_improves_a_row_that_is_not_exact():
    """Guard: the exact-initializer fix must not disable the sweep."""

    weight, scales, zeros, g_idx = _fixture(0.0)
    teacher = weight.clone()
    # Move one column to the next grid point: code 8 -> 9 is a real gain.
    teacher[:, 2] = SCALE

    result = refine_gptq_scalar(
        weight,
        scales,
        zeros,
        g_idx,
        target=teacher,
        hessian=torch.eye(COLUMNS),
        bits=4,
        config=GSQConfig(enabled=True, steps=2, candidates=4, learn_scales=False),
    )

    assert result.before > 0.0
    assert result.after < result.before
    assert torch.equal(result.weight[:, 2], teacher[:, 2])
