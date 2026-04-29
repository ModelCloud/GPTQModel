# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
    convert_gptq_v2_to_v1_format_module,
)
from gptqmodel.utils.model_dequant import _correct_gptq_v1_qzeros, convert_gptq_file


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


def _pack_rows(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    if values.dtype != torch.int32:
        values = values.to(torch.int32)

    rows, cols = values.shape
    if bits == 3:
        if rows % 32 != 0:
            raise ValueError("3-bit rows must be divisible by 32")

        blocks = rows // 32
        values_i64 = (values.to(torch.int64) & 0x7).reshape(blocks, 32, cols)
        packed = torch.zeros((blocks * 3, cols), dtype=torch.int32)

        for block in range(blocks):
            word0 = torch.zeros((cols,), dtype=torch.int64)
            for idx in range(10):
                word0 |= values_i64[block, idx, :] << (3 * idx)
            word0 |= values_i64[block, 10, :] << 30

            word1 = (values_i64[block, 10, :] >> 2) & 0x1
            for idx in range(10):
                word1 |= values_i64[block, 11 + idx, :] << (1 + 3 * idx)
            word1 |= values_i64[block, 21, :] << 31

            word2 = (values_i64[block, 21, :] >> 1) & 0x3
            for idx in range(10):
                word2 |= values_i64[block, 22 + idx, :] << (2 + 3 * idx)

            packed[block * 3 + 0, :] = word0.to(torch.int32)
            packed[block * 3 + 1, :] = word1.to(torch.int32)
            packed[block * 3 + 2, :] = word2.to(torch.int32)
        return packed

    pack_factor = 32 // bits
    if rows % pack_factor != 0:
        raise ValueError("rows must be divisible by the pack factor")

    packed = torch.zeros(rows // pack_factor, cols, dtype=torch.int32)
    mask = (1 << bits) - 1
    for row in range(rows):
        word = row // pack_factor
        shift = (row % pack_factor) * bits
        packed[word, :] |= (values[row, :] & mask) << shift
    return packed


def _pack_cols(values: torch.Tensor, bits: int = 4) -> torch.Tensor:
    if values.dtype != torch.int32:
        values = values.to(torch.int32)

    rows, cols = values.shape
    if bits == 3:
        if cols % 32 != 0:
            raise ValueError("3-bit columns must be divisible by 32")

        blocks = cols // 32
        values_i64 = (values.to(torch.int64) & 0x7).reshape(rows, blocks, 32)
        packed = torch.zeros((rows, blocks * 3), dtype=torch.int32)

        for block in range(blocks):
            word0 = torch.zeros((rows,), dtype=torch.int64)
            for idx in range(10):
                word0 |= values_i64[:, block, idx] << (3 * idx)
            word0 |= values_i64[:, block, 10] << 30

            word1 = (values_i64[:, block, 10] >> 2) & 0x1
            for idx in range(10):
                word1 |= values_i64[:, block, 11 + idx] << (1 + 3 * idx)
            word1 |= values_i64[:, block, 21] << 31

            word2 = (values_i64[:, block, 21] >> 1) & 0x3
            for idx in range(10):
                word2 |= values_i64[:, block, 22 + idx] << (2 + 3 * idx)

            packed[:, block * 3 + 0] = word0.to(torch.int32)
            packed[:, block * 3 + 1] = word1.to(torch.int32)
            packed[:, block * 3 + 2] = word2.to(torch.int32)
        return packed

    pack_factor = 32 // bits
    if cols % pack_factor != 0:
        raise ValueError("columns must be divisible by the pack factor")

    packed = torch.zeros((rows, cols // pack_factor), dtype=torch.int32)
    mask = (1 << bits) - 1
    for col in range(cols):
        word = col // pack_factor
        shift = (col % pack_factor) * bits
        packed[:, word] |= (values[:, col] & mask) << shift
    return packed


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
def test_qzero_offsets_3bit_int32_match_canonical_pack_order():
    module = _make_module(bits=3, pack_dtype=torch.int32)
    convert_gptq_v1_to_v2_format_module(module=module, bits=3, pack_dtype=torch.int32)

    expected = _pack_cols(torch.ones((1, 32), dtype=torch.int32), bits=3)
    assert torch.equal(module.qzeros.data, expected)


@pytest.mark.parametrize(
    "bits, pack_dtype",
    (
        (2, torch.int8),
        (2, torch.int32),
        (3, torch.int32),
        (4, torch.int16),
        (4, torch.int32),
        (8, torch.int32),
    ),
)
@torch.inference_mode()
def test_model_dequant_qzero_correction_matches_module_conversion(bits: int, pack_dtype: torch.dtype) -> None:
    module = _make_module(bits=bits, pack_dtype=pack_dtype)
    convert_gptq_v1_to_v2_format_module(module=module, bits=bits, pack_dtype=pack_dtype)

    corrected = _correct_gptq_v1_qzeros(torch.zeros_like(module.qzeros.data), bits)
    assert torch.equal(corrected, module.qzeros.data)


@torch.inference_mode()
def test_convert_gptq_file_applies_v1_qzero_correction(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "linear.qweight": _pack_rows(torch.zeros((8, 8), dtype=torch.int32), bits=4),
            "linear.qzeros": torch.zeros((1, 1), dtype=torch.int32),
            "linear.scales": torch.ones((1, 8), dtype=torch.float32),
            "linear.g_idx": torch.zeros((8,), dtype=torch.int32),
        },
        str(path),
    )

    weight = convert_gptq_file(
        path,
        torch.float32,
        {"bits": 4, "checkpoint_format": "gptq"},
        "cpu",
    )["linear.weight"]

    torch.testing.assert_close(weight, torch.full((8, 8), -1.0, dtype=torch.float32))


@torch.inference_mode()
def test_convert_gptq_file_skips_v2_qzero_correction(tmp_path) -> None:
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "linear.qweight": _pack_rows(torch.zeros((8, 8), dtype=torch.int32), bits=4),
            "linear.qzeros": torch.zeros((1, 1), dtype=torch.int32),
            "linear.scales": torch.ones((1, 8), dtype=torch.float32),
            "linear.g_idx": torch.zeros((8,), dtype=torch.int32),
        },
        str(path),
    )

    weight = convert_gptq_file(
        path,
        torch.float32,
        {"bits": 4, "checkpoint_format": "gptq_v2"},
        "cpu",
    )["linear.weight"]

    torch.testing.assert_close(weight, torch.zeros((8, 8), dtype=torch.float32))


@pytest.mark.parametrize("checkpoint_format", ("gptq", "gptq_v2"))
@torch.inference_mode()
def test_convert_gptq_file_3bit_matches_reference_weights(tmp_path, checkpoint_format: str) -> None:
    path = tmp_path / f"model-{checkpoint_format}.safetensors"

    intweight = torch.arange(32 * 32, dtype=torch.int32).reshape(32, 32) % 8
    true_qzeros = ((torch.arange(32, dtype=torch.int32).reshape(1, 32) * 3) + 1) % 8
    stored_qzeros = true_qzeros if checkpoint_format == "gptq_v2" else (true_qzeros - 1) & 0x7
    scales = torch.ones((1, 32), dtype=torch.float32)
    g_idx = torch.zeros((32,), dtype=torch.int32)

    save_file(
        {
            "linear.qweight": _pack_rows(intweight, bits=3),
            "linear.qzeros": _pack_cols(stored_qzeros, bits=3),
            "linear.scales": scales,
            "linear.g_idx": g_idx,
        },
        str(path),
    )

    weight = convert_gptq_file(
        path,
        torch.float32,
        {"bits": 3, "checkpoint_format": checkpoint_format},
        "cpu",
    )["linear.weight"]

    expected = (intweight.to(torch.float32) - true_qzeros.to(torch.float32)).t().contiguous()
    torch.testing.assert_close(weight, expected)
