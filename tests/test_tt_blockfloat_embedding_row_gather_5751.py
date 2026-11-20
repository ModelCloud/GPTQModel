# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5751: the portable block-float embedding gathers rows.

``TTBlockfloatEmbedding.forward`` expanded the whole vocabulary in FP32 and
then gathered. The portable path now slices the packed code and exponent rows
for the requested ids first, so a single-token decode costs one row of scratch
instead of the table. These tests pin the compact path against a packed-byte
oracle and pin every geometry that must keep the whole-table path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding


def _module(vocab=64, dim=32, bits=4, block=16, dtype=torch.float32, block_axis="in"):
    torch.manual_seed(5751)
    module = TTBlockfloatEmbedding(
        vocab, dim, bits=bits, block=block, dtype=dtype, block_axis=block_axis
    )
    rows, cols = (vocab, dim) if block_axis == "in" else (dim, vocab)
    packed = torch.randint(0, 256, (rows * cols * bits // 8,), dtype=torch.uint8)
    exponents = torch.randint(-4, 5, (rows, cols // block), dtype=torch.int8)
    module.set_packed_state(packed, exponents)
    return module


def _count_dequantize(module):
    """Record whole-table decodes so the chosen lookup path is observable."""

    calls = []
    original = module.dequantize

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    module.dequantize = tracked
    return calls


def _oracle(module, ids):
    """Decode the requested rows straight from the packed bytes.

    Deliberately independent of ``_unpack_codes``/``recompose_blockfloat``: the
    codes are extracted from the stored byte fields and scaled with ``ldexp``,
    so a bug shared with the library decoder cannot cancel out.
    """

    assert module.block_axis == "in"
    bits = module.bits
    hidden = module.in_features
    mantissa_bits = max(bits - 2, 0)
    levels = 1 << mantissa_bits
    offset = 2 * levels - 1
    bytes_per_row = hidden * bits // 8

    packed = module.codes.reshape(module.out_features, bytes_per_row)
    flat = ids.reshape(-1).to(torch.int64)
    columns = torch.arange(hidden)
    bit_offsets = columns * bits
    byte = packed[flat[:, None], bit_offsets // 8].to(torch.int32)
    code = (byte >> (bit_offsets % 8)) & ((1 << bits) - 1)
    exponent = module.exponents[flat[:, None], columns // module.block].to(torch.int32)
    values = torch.ldexp((code - offset).to(torch.float32), exponent - mantissa_bits)
    return values.to(module.output_dtype).reshape(*ids.shape, hidden)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_row_gather_matches_the_packed_byte_oracle(bits, dtype):
    module = _module(bits=bits, dtype=dtype)
    ids = torch.tensor([[63, 1, 1, 0, 17]], dtype=torch.int64)
    calls = _count_dequantize(module)

    result = module(ids)

    assert calls == [], "a narrow lookup must not decode the whole table"
    assert result.dtype == dtype
    torch.testing.assert_close(result, _oracle(module, ids), rtol=0, atol=0)
    torch.testing.assert_close(
        result, module.dequantize(dtype=dtype)[ids], rtol=0, atol=0
    )


def test_float32_output_matches_the_dense_table():
    module = _module()
    ids = torch.tensor([5, 0, 63], dtype=torch.int64)
    calls = _count_dequantize(module)

    result = module(ids)

    assert calls == []
    assert result.dtype == torch.float32
    assert torch.equal(result, module.dequantize()[ids])


def test_int32_and_int64_ids_agree():
    module = _module()
    ids = torch.tensor([[7, 3], [63, 0]])
    calls = _count_dequantize(module)

    narrow = module(ids.to(torch.int32))
    wide = module(ids.to(torch.int64))

    assert calls == [], "both integer widths take the compact path"
    assert torch.equal(narrow, wide)
    torch.testing.assert_close(narrow, _oracle(module, ids), rtol=0, atol=0)


def test_repeated_and_unsorted_ids_are_decoded_once_per_lookup():
    module = _module()
    ids = torch.tensor([9, 9, 9, 0, 63, 9])
    calls = _count_dequantize(module)

    result = module(ids)

    assert calls == []
    torch.testing.assert_close(result, _oracle(module, ids), rtol=0, atol=0)


def test_wider_block_geometry_still_gathers():
    # block=32 is a legal storage geometry (the native kernels refuse it, the
    # portable path must not); rows are still byte aligned.
    module = _module(bits=4, block=32)
    ids = torch.tensor([[63, 4, 4]])
    calls = _count_dequantize(module)

    result = module(ids)

    assert calls == []
    torch.testing.assert_close(result, _oracle(module, ids), rtol=0, atol=0)


def test_block_axis_out_keeps_the_whole_table_path():
    # Stored W^T means a token id selects a column of the packed stream, not a
    # contiguous row slice.
    module = _module(block_axis="out")
    ids = torch.tensor([[63, 2]])
    calls = _count_dequantize(module)

    result = module(ids)

    assert len(calls) == 1, "a column gather must fall back to the dense table"
    torch.testing.assert_close(result, module.dequantize()[ids], rtol=0, atol=0)


def test_non_byte_aligned_rows_keep_the_whole_table_path():
    # 6 values at 2 bits is 12 bits per row, so rows do not start on a byte.
    module = _module(vocab=8, dim=6, bits=2, block=2)
    ids = torch.tensor([7, 1])
    calls = _count_dequantize(module)

    result = module(ids)

    assert len(calls) == 1, "unaligned rows must fall back to the dense table"
    torch.testing.assert_close(result, module.dequantize()[ids], rtol=0, atol=0)


def test_query_wider_than_the_table_keeps_the_whole_table_path():
    module = _module(vocab=8)
    ids = torch.arange(16) % 8
    calls = _count_dequantize(module)

    result = module(ids)

    assert len(calls) == 1, "each row is decoded once, not once per lookup"
    torch.testing.assert_close(result, module.dequantize()[ids], rtol=0, atol=0)


@pytest.mark.parametrize("ids", [[64], [0, 64], [-1], [3, -1]])
def test_out_of_range_ids_raise_the_same_error_as_the_dense_path(ids):
    module = _module()
    input_ids = torch.tensor(ids)

    with pytest.raises(IndexError) as dense:
        F.embedding(input_ids, module.dequantize())
    with pytest.raises(IndexError) as gathered:
        module(input_ids)

    assert str(gathered.value) == str(dense.value)


def test_float_ids_raise_the_same_error_as_the_dense_path():
    module = _module()
    input_ids = torch.tensor([1.0, 2.0])

    with pytest.raises(RuntimeError) as dense:
        F.embedding(input_ids, module.dequantize())
    with pytest.raises(RuntimeError) as gathered:
        module(input_ids)

    assert str(gathered.value) == str(dense.value)


def test_multidimensional_ids_preserve_the_query_shape():
    module = _module(dtype=torch.bfloat16)
    ids = torch.arange(24).reshape(2, 3, 4) % 64
    calls = _count_dequantize(module)

    result = module(ids)

    assert calls == []
    assert result.shape == (2, 3, 4, 32)
    torch.testing.assert_close(result, _oracle(module, ids), rtol=0, atol=0)


def test_output_dtype_follows_module_casts():
    module = _module()
    ids = torch.tensor([1, 2, 3])

    module = module.to(torch.float16)
    assert module(ids).dtype == torch.float16
    module = module.to(torch.bfloat16)
    assert module(ids).dtype == torch.bfloat16
