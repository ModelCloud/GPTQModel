# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 codebooks: TurboDerp and ExLlamaV3 contributors.

"""Independent Torch-oracle checks for EXL3 codebook reconstruction on MLX."""

import gc
import struct
import sys
from functools import lru_cache

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3 import exl3_decode_states_mlx


def _half_from_bits(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


@lru_cache(maxsize=3)
def _torch_codebook_oracle(codebook):
    """Generate all EXL3 codebook values with independent Torch arithmetic."""
    states = torch.arange(1 << 16, dtype=torch.int64)
    if codebook == "3inst":
        raw = (states * 89226354 + 64248484) & 0xFFFFFFFF
        raw = 0x3B603B60 ^ (raw & 0x8FFF8FFF)
        halves = torch.stack(
            ((raw & 0xFFFF).to(torch.uint16), ((raw >> 16) & 0xFFFF).to(torch.uint16)),
            dim=-1,
        ).contiguous()
        values = halves.view(torch.float16).to(torch.float64)
        return values.sum(dim=-1).to(torch.float16).to(torch.float32)
    if codebook == "mcg":
        raw = (states * 0xCBAC1FED) & 0xFFFFFFFF
        raw = 0x3B603B60 ^ (raw & 0x8FFF8FFF)
        halves = torch.stack(
            ((raw & 0xFFFF).to(torch.uint16), ((raw >> 16) & 0xFFFF).to(torch.uint16)),
            dim=-1,
        ).contiguous()
        values = halves.view(torch.float16).to(torch.float64)
        return values.sum(dim=-1).to(torch.float16).to(torch.float32)
    if codebook == "mul1":
        raw = (states * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = (
            (raw & 0xFF)
            + ((raw >> 8) & 0xFF)
            + ((raw >> 16) & 0xFF)
            + ((raw >> 24) & 0xFF)
        )
        accum_bits = (byte_sum + 0x6400).to(torch.uint16).contiguous()
        accumulator = accum_bits.view(torch.float16).to(torch.float64)
        result = accumulator * _half_from_bits(0x1EEE) + _half_from_bits(0xC931)
        return result.to(torch.float16).to(torch.float32)
    raise ValueError(codebook)


def _torch_decode_oracle(encoded, codebook):
    indices = torch.from_numpy(np.ascontiguousarray(encoded)).to(torch.int64) & 0xFFFF
    return _torch_codebook_oracle(codebook)[indices]


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_exl3_codebook_exhaustive_states(codebook):
    encoded = np.arange(1 << 16, dtype=np.uint16).view(np.int16).reshape(256, 256)
    expected = _torch_decode_oracle(encoded, codebook).numpy()
    actual = np.asarray(exl3_decode_states_mlx(mx.array(encoded), codebook=codebook))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
def test_exl3_codebook_signed_state_boundaries(codebook):
    boundary = np.array([-32768, -32767, -1, 0, 1, 32766, 32767], dtype=np.int16)
    encoded = np.resize(boundary, (3, 256)).astype(np.int16, copy=False)
    expected = _torch_decode_oracle(encoded, codebook).numpy()
    actual = np.asarray(exl3_decode_states_mlx(mx.array(encoded), codebook=codebook))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_codebook_qwen38_projection_oracle(name, out_features, in_features):
    shape = (in_features // 16, out_features // 16, 256)
    rng = np.random.default_rng(1187 + out_features + in_features)
    encoded = rng.integers(-32768, 32768, shape, dtype=np.int16)
    expected = _torch_decode_oracle(encoded, "mcg").numpy()
    actual = np.asarray(exl3_decode_states_mlx(mx.array(encoded), codebook="mcg"))
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    del encoded, expected, actual
    gc.collect()


def test_exl3_codebook_rejects_invalid_inputs():
    valid = mx.zeros((1, 256), dtype=mx.int16)
    for codebook in ("unknown", 1, None):
        with pytest.raises(ValueError, match="codebook"):
            exl3_decode_states_mlx(valid, codebook=codebook)
    with pytest.raises(ValueError, match="rank"):
        exl3_decode_states_mlx(mx.zeros((256,), dtype=mx.int16))
    with pytest.raises(ValueError, match="256 states"):
        exl3_decode_states_mlx(mx.zeros((1, 255), dtype=mx.int16))
    with pytest.raises(ValueError, match="nonempty"):
        exl3_decode_states_mlx(mx.zeros((0, 256), dtype=mx.int16))
    with pytest.raises(ValueError, match="int16"):
        exl3_decode_states_mlx(mx.zeros((1, 256), dtype=mx.int32))
