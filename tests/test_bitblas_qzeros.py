# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""BitBLAS zero-point unpacking has to work for every supported bit width.

``BITBLAS_SUPPORTED_BITS`` is [2, 4, 8], but the unpacker masked each field with
a hardcoded 0xF. That is harmless at bits=2 and bits=4 and wrong at bits=8: a
stored zero point of 127 came back as 15, which silently dequantizes every
weight against the wrong offset.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.bitblas import BITBLAS_SUPPORTED_BITS, unpack_gptq_qzeros


def _pack_zeros(zeros: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack (1, N) zero points into the int32 layout the unpacker expects."""
    per_word = 32 // bits
    n = zeros.shape[1]
    assert n % per_word == 0
    # Word w holds the codes at columns w*per_word .. w*per_word+per_word-1,
    # column j sitting at bit offset bits * (j % per_word).
    groups = zeros.reshape(1, n // per_word, per_word)
    packed = torch.zeros((1, n // per_word), dtype=torch.int32)
    for i in range(per_word):
        packed[0] |= groups[0, :, i].to(torch.int32) << (bits * i)
    return packed


@pytest.mark.parametrize("bits", BITBLAS_SUPPORTED_BITS)
def test_v2_unpack_round_trips_every_value(bits):
    """v2 checkpoints store the logical zero point directly."""
    per_word = 32 // bits
    values = (torch.arange(2 * per_word) % (1 << bits)).to(torch.int32).reshape(1, -1)
    unpacked = unpack_gptq_qzeros(_pack_zeros(values, bits), bits, is_gptq_v2=True)
    assert torch.equal(unpacked.to(torch.int32), values)


@pytest.mark.parametrize("bits", BITBLAS_SUPPORTED_BITS)
def test_v1_unpack_restores_the_offset(bits):
    """v1 checkpoints store zero - 1, so unpacking has to add one back."""
    per_word = 32 // bits
    span = (1 << bits) - 1
    values = ((torch.arange(2 * per_word) % span) + 1).to(torch.int32).reshape(1, -1)
    unpacked = unpack_gptq_qzeros(_pack_zeros(values - 1, bits), bits, is_gptq_v2=False)
    assert torch.equal(unpacked.to(torch.int32), values)


def test_8bit_zero_point_keeps_its_high_nibble():
    """Reproduces the 0xF mask bug: 127 used to unpack as 15."""
    zeros = torch.tensor([[127, 200, 130, 255]], dtype=torch.int32)
    unpacked = unpack_gptq_qzeros(_pack_zeros(zeros, 8), 8, is_gptq_v2=True)
    assert unpacked.reshape(-1).tolist() == [127, 200, 130, 255]
