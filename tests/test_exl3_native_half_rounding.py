# SPDX-License-Identifier: Apache-2.0
import struct

import pytest
import torch

from gptqmodel.nn_modules.exllamav3_torch import _codebook_lut


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_exllamav3_torch_codebook_matches_native_half_law_all_states(codebook):
    # Independent scalar law from quant/codebook.cuh: __hadd or __hfma,
    # then __half2float. Python double exactly represents these operands.
    def half_from_bits(bits):
        return struct.unpack("<e", struct.pack("<H", bits))[0]

    def native_value(state):
        if codebook == "mul1":
            raw = (state * 0x83DCD12D) & 0xFFFFFFFF
            byte_sum = sum((raw >> shift) & 0xFF for shift in (0, 8, 16, 24))
            value = (half_from_bits(0x6400 + byte_sum) * half_from_bits(0x1EEE)
                     + half_from_bits(0xC931))
        else:
            multiplier, offset = ((89226354, 64248484) if codebook == "3inst"
                                  else (0xCBAC1FED, 0))
            raw = (state * multiplier + offset) & 0xFFFFFFFF
            raw = 0x3B603B60 ^ (raw & 0x8FFF8FFF)
            value = half_from_bits(raw & 0xFFFF) + half_from_bits(raw >> 16)
        return struct.unpack("<e", struct.pack("<e", value))[0]

    expected = torch.tensor([native_value(state) for state in range(65536)],
                            dtype=torch.float32, device="cpu")
    actual = _codebook_lut(codebook, "cpu", None)
    assert actual.dtype == torch.float32
    assert actual.shape == (65536,)
    # Compare float32 bits too: native signed zeros are part of the format.
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
