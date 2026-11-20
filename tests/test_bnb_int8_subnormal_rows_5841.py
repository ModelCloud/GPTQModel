# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5841: finite FP32 subnormal rows must keep their INT8 codes.

``row_absmax / 127`` underflows to zero for a subnormal row absmax, so the
portable row quantizer selected the unit denominator meant for a genuinely
zero row and wrote all-zero codes. The dense dequantizer divided the row
absmax by 127 first, so it could erase a valid packed checkpoint the same way.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitsandbytes import (
    BitsAndBytesLinear,
    _dequantize_int8_weight,
    _int8_vectorwise_quant_fp32,
)


pytestmark = [pytest.mark.cpu]

_WIDTH = 8
_SUBNORMAL = 2.0**-149  # smallest positive FP32 subnormal
_ROW_ABSMAX = 2.0**-145


def _subnormal_row(width: int = _WIDTH) -> torch.Tensor:
    row = torch.zeros(width, dtype=torch.float32)
    row[0] = _SUBNORMAL
    row[1] = -_ROW_ABSMAX
    return row


def test_row_quantizer_keeps_subnormal_codes():
    """The FP64 reference gives [8, -127]; the pre-fix helper gave [0, 0]."""

    x = _subnormal_row().reshape(1, -1)
    codes, stats, outliers = _int8_vectorwise_quant_fp32(x)

    assert outliers is None
    assert stats[0].item() == _ROW_ABSMAX

    reference = (
        x.double() / stats.double().reshape(-1, 1) * 127.0
    ).round().clamp(-127, 127)
    assert codes.tolist() == reference.to(torch.int8).tolist()
    assert codes[0, :2].tolist() == [8, -127]


def test_dequantizer_keeps_subnormal_scale():
    x = _subnormal_row().reshape(1, -1)
    codes, stats, _ = _int8_vectorwise_quant_fp32(x)
    dequant = _dequantize_int8_weight(codes, stats)

    reference = (
        codes.double() * (stats.double() / 127.0).reshape(-1, 1)
    ).to(torch.float32)
    torch.testing.assert_close(dequant, reference, rtol=0, atol=0)
    assert dequant[0, 0].item() == _SUBNORMAL
    assert dequant[0, 1].item() == -_ROW_ABSMAX


def test_zero_row_still_quantizes_to_zero():
    x = torch.zeros(1, _WIDTH, dtype=torch.float32)
    codes, stats, _ = _int8_vectorwise_quant_fp32(x)
    assert not codes.any()
    assert stats[0].item() == 0.0


def test_public_pack_and_forward_preserve_subnormal_row():
    """The packed layer keeps the row scale and projects the finite value."""

    width = 16
    linear = torch.nn.Linear(width, 4, bias=False)
    layer = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=width,
        out_features=4,
        bias=False,
        dtype=torch.float32,
    )
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, :2] = _subnormal_row(2)
    layer.pack_original(linear, None, None)
    layer.post_init()

    dense_row = linear.weight[0].detach().double()
    stats = dense_row.abs().amax()
    expected_codes = (dense_row / stats * 127).round().clamp(-127, 127).to(torch.int8)
    assert layer.weight_scb[0].item() == stats.item()
    assert layer.weight[0].tolist() == expected_codes.tolist()

    x = torch.zeros(1, width, dtype=torch.float32)
    x[0, 1] = 2.0**127
    with torch.inference_mode():
        output = layer(x)[0, 0].item()
    reference = ((expected_codes.double() / 127 * stats) @ x[0].double()).item()
    assert output == reference
    assert output == -(2.0**-18)
