# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
    convert_gptq_v2_to_v1_format_module,
)


class _TestQuantLinear(BaseQuantLinear):
    REQUIRES_FORMAT_V2 = True

    def __init__(self, *, bits: int, pack_dtype: torch.dtype, columns: int) -> None:
        nn.Module.__init__(self)
        self.bits = bits
        self.pack_dtype = pack_dtype
        self.qzeros = nn.Parameter(torch.zeros((1, columns), dtype=pack_dtype), requires_grad=False)
        self._qzeros_format = 1

    def qzero_format(self, format: int | None = None) -> int:
        if format is None:
            return self._qzeros_format
        self._qzeros_format = format
        return self._qzeros_format


def _make_module(bits: int, pack_dtype: torch.dtype) -> _TestQuantLinear:
    columns = 3 if bits == 3 else 1
    return _TestQuantLinear(bits=bits, pack_dtype=pack_dtype, columns=columns)


@pytest.mark.parametrize(
    "bits, pack_dtype",
    (
        (2, torch.int8),
        (3, torch.int32),
        (4, torch.int16),
        (8, torch.int32),
    ),
)
@torch.inference_mode()
def test_qzero_offsets_roundtrip(bits: int, pack_dtype: torch.dtype) -> None:
    module = _make_module(bits=bits, pack_dtype=pack_dtype)
    original = module.qzeros.data.clone()

    convert_gptq_v1_to_v2_format_module(module=module, bits=bits, pack_dtype=pack_dtype)

    convert_gptq_v2_to_v1_format_module(
        module=module,
        quantize_config=SimpleNamespace(
            bits=bits,
            pack_dtype=pack_dtype,
            quant_method=METHOD.GPTQ,
            format=FORMAT.GPTQ,
        ),
    )

    assert torch.equal(module.qzeros.data, original)


@torch.inference_mode()
def test_qzero_offsets_scalar_patterns():
    cases = [
        (2, torch.int8, torch.tensor([[0x55]], dtype=torch.int8)),
        (2, torch.int32, torch.tensor([[0x5555_5555]], dtype=torch.int32)),
        (4, torch.int16, torch.tensor([[0x1111]], dtype=torch.int16)),
        (8, torch.int32, torch.tensor([[0x0101_0101]], dtype=torch.int32)),
    ]

    for bits, pack_dtype, expected in cases:
        module = _make_module(bits=bits, pack_dtype=pack_dtype)
        convert_gptq_v1_to_v2_format_module(module=module, bits=bits, pack_dtype=pack_dtype)
        assert torch.equal(module.qzeros.data, expected)

@torch.inference_mode()
def test_qzero_offsets_triangular_patterns():
    cases = [
        (
            torch.int8,
            # 0x92 == -110 in int8 (two's complement)
            torch.tensor([[0x24, -0x6E, 0x49]], dtype=torch.int8),
        ),
        (
            torch.int32,
            # 0x9249_2492 exceeds int32 max; use its 32-bit two's complement: -0x6DB6_DB6E
            torch.tensor([[0x2492_4924, -0x6DB6_DB6E, 0x4924_9249]], dtype=torch.int32),
        ),
    ]

    for pack_dtype, expected in cases:
        module = _make_module(bits=3, pack_dtype=pack_dtype)
        convert_gptq_v1_to_v2_format_module(module=module, bits=3, pack_dtype=pack_dtype)
        assert torch.equal(module.qzeros.data, expected)
