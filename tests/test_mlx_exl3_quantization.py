# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 trellis format: TurboDerp and ExLlamaV3 contributors.
# MLX runtime license: MIT, https://github.com/ml-explore/mlx
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.

"""Independent Torch-oracle tests for EXL3 trellis packing on MLX."""

import gc
import sys

import numpy as np
import pytest
import torch

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from gptqmodel.quantization.mlx_exl3 import exl3_pack_trellis_mlx


def _torch_pack_trellis_oracle(encoded, bits, *, chunk_tiles=4096):
    """Pack EXL3 symbols independently with Torch integer arithmetic."""
    source = torch.from_numpy(np.ascontiguousarray(encoded)).reshape(-1, 256)
    outputs = []
    mask = (1 << bits) - 1
    for start in range(0, source.shape[0], chunk_tiles):
        codes = (source[start : start + chunk_tiles].to(torch.int32) & mask).reshape(
            -1, 16, 16
        )
        words = torch.zeros(
            (codes.shape[0], 16, bits), dtype=torch.int32, device=codes.device
        )
        for word in range(bits):
            for output_bit in range(16):
                stream_bit = word * 16 + output_bit
                symbol = stream_bit // bits
                symbol_bit = stream_bit % bits
                bit = (codes[:, :, symbol] >> (bits - 1 - symbol_bit)) & 1
                words[:, :, word] |= bit << (15 - output_bit)
        packed = (
            words.reshape(codes.shape[0], -1, 2)
            .flip(-1)
            .reshape(codes.shape[0], 16 * bits)
        )
        outputs.append(packed.to(torch.int16))
    return torch.cat(outputs).reshape(*encoded.shape[:-1], 16 * bits)


def _consistent_states(symbols, bits):
    """Build full tail-biting states from the low-bit symbols EXL3 stores."""
    symbols = torch.from_numpy(np.asarray(symbols, dtype=np.int64))
    mask = (1 << 16) - 1
    warmup = (16 + bits - 1) // bits - 1
    state = torch.zeros(symbols.shape[:-1], dtype=torch.int64)
    for index in range(256 - warmup, 256):
        state = ((state << bits) | symbols[..., index]) & mask
    encoded = torch.empty_like(symbols)
    for index in range(256):
        state = ((state << bits) | symbols[..., index]) & mask
        encoded[..., index] = state
    return encoded.to(torch.int16).numpy()


@pytest.mark.parametrize("bits", range(1, 9))
def test_exl3_pack_trellis_boundaries(bits):
    values = np.array(
        [
            -32768,
            -1,
            0,
            1,
            (1 << bits) - 1,
            1 << bits,
            32767,
        ],
        dtype=np.int16,
    )
    encoded = np.resize(values, (2, 3, 256)).astype(np.int16, copy=False)
    expected = _torch_pack_trellis_oracle(encoded, bits).numpy()
    actual = np.asarray(exl3_pack_trellis_mlx(mx.array(encoded), bits=bits))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_exl3_pack_trellis_roundtrip_through_existing_decoder(bits, tmp_path):
    from safetensors.torch import load_file, save_file

    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    rng = np.random.default_rng(991 + bits)
    symbols = rng.integers(0, 1 << bits, (8, 8, 256), dtype=np.int16)
    encoded = _consistent_states(symbols, bits)
    packed = exl3_pack_trellis_mlx(mx.array(encoded), bits=bits)
    path = tmp_path / f"exl3_{bits}bit.safetensors"
    save_file({"trellis": torch.from_numpy(np.asarray(packed).copy())}, path)
    tensors = {
        "trellis": load_file(path)["trellis"],
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(128, dtype=torch.float16),
    }
    layer = ExllamaV3TorchLinear.from_tensors(
        in_features=128,
        out_features=128,
        name="linear",
        tensors=tensors,
    )
    np.testing.assert_array_equal(
        layer._unpack_indices().numpy(), encoded.astype(np.int64) & 0xFFFF
    )


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_exl3_pack_trellis_qwen38_projection_oracle(name, out_features, in_features):
    bits = 3
    shape = (in_features // 16, out_features // 16, 256)
    rng = np.random.default_rng(731 + out_features + in_features)
    encoded = rng.integers(-32768, 32768, shape, dtype=np.int16)
    expected = _torch_pack_trellis_oracle(encoded, bits).numpy()
    actual = np.asarray(exl3_pack_trellis_mlx(mx.array(encoded), bits=bits))
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    del encoded, expected, actual
    gc.collect()


def test_exl3_pack_trellis_rejects_invalid_inputs():
    valid = mx.zeros((1, 1, 256), dtype=mx.int16)
    for bits in (0, 9, True, 3.0):
        with pytest.raises(ValueError, match="bits"):
            exl3_pack_trellis_mlx(valid, bits=bits)
    with pytest.raises(ValueError, match="shape"):
        exl3_pack_trellis_mlx(mx.zeros((1, 256), dtype=mx.int16), bits=3)
    with pytest.raises(ValueError, match="shape"):
        exl3_pack_trellis_mlx(mx.zeros((1, 1, 255), dtype=mx.int16), bits=3)
    with pytest.raises(ValueError, match="one tile"):
        exl3_pack_trellis_mlx(mx.zeros((0, 1, 256), dtype=mx.int16), bits=3)
    with pytest.raises(ValueError, match="int16"):
        exl3_pack_trellis_mlx(mx.zeros((1, 1, 256), dtype=mx.int32), bits=3)
