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


def _all_field_values(bits: int) -> torch.Tensor:
    """Every stored value for this width, padded to whole int32 words."""
    per_word = 32 // bits
    span = 1 << bits
    values = torch.arange(span, dtype=torch.int32).repeat(per_word)
    assert values.numel() % per_word == 0
    return values.reshape(1, -1)


@pytest.mark.parametrize("bits", BITBLAS_SUPPORTED_BITS)
def test_v2_unpack_round_trips_every_stored_value(bits):
    """v2 checkpoints store the logical zero point directly."""
    values = _all_field_values(bits)
    unpacked = unpack_gptq_qzeros(_pack_zeros(values, bits), bits, is_gptq_v2=True)
    assert torch.equal(unpacked.to(torch.int32), values)


@pytest.mark.parametrize("bits", BITBLAS_SUPPORTED_BITS)
def test_v1_unpack_restores_the_offset_for_every_stored_value(bits):
    """v1 checkpoints store zero - 1, so unpacking has to add one back.

    Stored 255 is the top of the 8-bit range and yields 256; the field cannot
    represent that as a zero point, and the caller masks it afterwards. The
    point of covering it is that the unpacker must not wrap it to 0.
    """
    stored = _all_field_values(bits)
    unpacked = unpack_gptq_qzeros(_pack_zeros(stored, bits), bits, is_gptq_v2=False)
    assert torch.equal(unpacked.to(torch.int32), stored + 1)


@pytest.mark.parametrize("bits", BITBLAS_SUPPORTED_BITS)
def test_high_bit_and_max_code_boundaries(bits):
    """The two values a short mask truncates or borrows from the next field."""
    per_word = 32 // bits
    span = 1 << bits
    boundaries = torch.tensor([[span // 2, span - 1] * (per_word // 2)], dtype=torch.int32)
    assert boundaries.numel() % per_word == 0
    unpacked = unpack_gptq_qzeros(_pack_zeros(boundaries, bits), bits, is_gptq_v2=True)
    assert torch.equal(unpacked.to(torch.int32), boundaries)


def test_8bit_zero_point_keeps_its_high_nibble():
    """Reproduces the 0xF mask bug: 127 used to unpack as 15."""
    zeros = torch.tensor([[127, 200, 130, 255]], dtype=torch.int32)
    unpacked = unpack_gptq_qzeros(_pack_zeros(zeros, 8), 8, is_gptq_v2=True)
    assert unpacked.reshape(-1).tolist() == [127, 200, 130, 255]


def test_2bit_zero_point_does_not_borrow_from_the_next_field():
    """Reproduces the other half of the same bug: 0,1,2,3 used to unpack as
    4,9,14,3 because 0xF kept the next field's low two bits."""
    values = torch.tensor([[0, 1, 2, 3] * 4], dtype=torch.int32)
    unpacked = unpack_gptq_qzeros(_pack_zeros(values, 2), 2, is_gptq_v2=True)
    assert unpacked.reshape(-1).tolist() == values.reshape(-1).tolist()
