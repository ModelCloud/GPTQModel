# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.ascend_npu import AscendNPUQuantLinear, unpack_from_int32
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import AUTO_SELECT_BACKEND_ORDER_MAP, SUPPORTS_BACKEND_MAP


def _pack_int32(values: list[int], num_bits: int) -> int:
    pack_factor = 32 // num_bits
    if len(values) != pack_factor:
        raise ValueError(f"Expected {pack_factor} values, got {len(values)}")
    mask = (1 << num_bits) - 1
    packed = 0
    for i, val in enumerate(values):
        packed |= (val & mask) << (num_bits * i)
    return packed


def test_ascend_npu_backend_is_registered() -> None:
    assert BACKEND.ASCEND_NPU.value == "ascend_npu"
    assert DEVICE.NPU.value == "npu"
    assert BACKEND.ASCEND_NPU in AUTO_SELECT_BACKEND_ORDER_MAP[METHOD.GPTQ]
    assert BACKEND.ASCEND_NPU in SUPPORTS_BACKEND_MAP[METHOD.GPTQ][FORMAT.GPTQ]


@pytest.mark.parametrize("packed_dim", (0, 1))
def test_unpack_from_int32_bits4_signed_offset(packed_dim: int) -> None:
    # For 4-bit quantization, unpack maps unsigned [0..15] -> signed [-8..7].
    packed = _pack_int32(list(range(8)), num_bits=4)
    t = torch.tensor([[packed]], dtype=torch.int32)
    unpacked = unpack_from_int32(t, num_bits=4, packed_dim=packed_dim)

    expected = torch.tensor([-8, -7, -6, -5, -4, -3, -2, -1], dtype=torch.int8)
    if packed_dim == 1:
        assert unpacked.shape == (1, 8)
        assert torch.equal(unpacked[0], expected)
    else:
        assert unpacked.shape == (8, 1)
        assert torch.equal(unpacked[:, 0], expected)


def test_validate_once_checks_torch_npu_availability() -> None:
    try:
        importlib.import_module("torch_npu")
        torch_npu_importable = True
    except BaseException:
        torch_npu_importable = False

    ok, err = AscendNPUQuantLinear.validate_once()
    assert ok is torch_npu_importable
    if torch_npu_importable:
        assert err is None
    else:
        assert isinstance(err, ImportError)

