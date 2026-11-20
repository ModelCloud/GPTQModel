# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""One matrix may mix blocks whose exact grid needs different exponent shifts.

``quantize_blockfloat`` recovers each block's exponent independently, so a row
near ``float32.tiny`` can land one shift below the clamped base while another
row sits on the base.  ``decompose_blockfloat`` used to accept a shift only when
it reproduced *every* block, which rejected such a matrix even though each row
packed on its own.  Regression for #5648.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_blockfloat_quant import (
    BLOCKFLOAT_BLOCK,
    decompose_blockfloat,
    quantize_blockfloat,
    recompose_blockfloat,
)


@pytest.mark.parametrize("bits", [4, 8])
def test_mixed_block_exponents_pack_and_round_trip(bits):
    # Row 0 is a subnormal whose block maximum clamps to float32.tiny, so the
    # quantizer picks exponent -127; row 1 is exactly on the base exponent 0.
    weight = torch.ones(2, BLOCKFLOAT_BLOCK, dtype=torch.float32)
    weight[0].fill_(2.0 ** (-127 - max(0, bits - 2)))
    expected_exponents = [[-127], [0]]

    quantized, exponents = quantize_blockfloat(weight, bits)
    assert torch.equal(quantized, weight)
    assert exponents.to(torch.int64).tolist() == expected_exponents

    codes, recovered = decompose_blockfloat(weight, bits)
    assert recovered.to(torch.int64).tolist() == expected_exponents
    assert torch.equal(recompose_blockfloat(codes, recovered, bits), weight)

    source = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        source.weight.copy_(weight)
    layer = TTBlockfloatLinear(
        bits=bits,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.float32,
    ).pack(source)

    assert layer.exponents.to(torch.int64).tolist() == expected_exponents
    assert torch.equal(layer.dequantize(dtype=torch.float32), weight)
