# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=1
"""qzeros pack/unpack helpers: device preservation and bitwise equality.

The GPTQ v1 -> v2 zero-point correction runs ``unpack_cols``/``pack_cols`` and
the dequantizer runs ``unpack_rows``/``unpack_cols`` on whatever device the
checkpoint shard was opened on.  The helpers used to allocate their result on
the host and fill it with a Python loop, which (a) made ``dequantize_model(...,
device="cuda")`` fail with "indices should be either on cpu or on the same
device" once ``g_idx`` was device-resident and (b) moved every qzeros tensor
GPU -> host loop -> GPU on load.

These tests pin the vectorised helpers to an independent loop oracle and cover
the end-to-end conversion on a CUDA shard.
"""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import (
    convert_gptq_file,
    pack_cols,
    unpack_cols,
    unpack_rows,
)


def _oracle_unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows, cols * pack_factor, dtype=torch.int32, device=packed.device)
    for i in range(pack_factor):
        result[:, i::pack_factor] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


def _oracle_pack_cols(values: torch.Tensor, bits: int, pack_dtype: torch.dtype) -> torch.Tensor:
    pack_bits = torch.empty((), dtype=pack_dtype).element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    rows, cols = values.shape
    packed = torch.zeros(rows, cols // pack_factor, dtype=pack_dtype, device=values.device)
    values_uint = values.to(torch.int64) & mask
    for i in range(pack_factor):
        packed |= (values_uint[:, i::pack_factor] << (i * bits)).to(pack_dtype)
    return packed


def _oracle_unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_bits = packed.element_size() * 8
    pack_factor = pack_bits // bits
    mask = (1 << bits) - 1
    packed_uint = packed.to(torch.int64) & ((1 << pack_bits) - 1)
    rows, cols = packed.shape
    result = torch.empty(rows * pack_factor, cols, dtype=torch.int32, device=packed.device)
    for i in range(pack_factor):
        result[i::pack_factor, :] = ((packed_uint >> (i * bits)) & mask).to(torch.int32)
    return result


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
def test_pack_unpack_bitwise_matches_loop_oracle(device, bits, pack_dtype):
    pack_bits = torch.empty((), dtype=pack_dtype).element_size() * 8
    if pack_bits % bits:
        pytest.skip(f"{pack_bits}-bit words do not hold {bits}-bit fields")

    generator = torch.Generator().manual_seed(1234)
    packed = torch.randint(
        -(2 ** (pack_bits - 1)),
        2 ** (pack_bits - 1),
        (9, 12),
        dtype=torch.int64,
        generator=generator,
    ).to(device=device, dtype=pack_dtype)

    unpacked = unpack_cols(packed, bits)
    assert unpacked.device.type == device
    assert unpacked.dtype == torch.int32
    assert torch.equal(unpacked, _oracle_unpack_cols(packed, bits))

    repacked = pack_cols(unpacked, bits, pack_dtype=pack_dtype)
    assert repacked.device.type == device
    assert repacked.dtype == pack_dtype
    assert torch.equal(repacked, _oracle_pack_cols(unpacked, bits, pack_dtype))
    assert torch.equal(repacked, packed)

    rows = unpack_rows(packed, bits)
    assert rows.device.type == device
    assert torch.equal(rows, _oracle_unpack_rows(packed, bits))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_unpacked_qzeros_stay_on_device_for_g_idx_gather():
    """The GPU dequantize_model crash: host zeros indexed by a device g_idx."""

    bits = 4
    in_features, out_features, group_size = 256, 128, 128
    num_groups = in_features // group_size
    qzeros = torch.randint(
        0, 2 ** 31 - 1, (num_groups, out_features // (32 // bits)), dtype=torch.int32, device="cuda"
    )
    g_idx = (torch.arange(in_features, device="cuda", dtype=torch.int32) // group_size)

    zeros = unpack_cols(qzeros, bits)
    assert zeros.device.type == "cuda"
    # This is the exact expression that raised on the host-resident result.
    assert torch.equal(zeros[g_idx.long()], _oracle_unpack_cols(qzeros, bits)[g_idx.long()])


def _write_gptq_shard(path: Path, bits: int) -> None:
    in_features, out_features, group_size = 256, 128, 128
    pack_factor = 32 // bits
    num_groups = in_features // group_size
    generator = torch.Generator().manual_seed(7)
    save_file(
        {
            # GPTQ packs along in_features for qweight (rows) and along
            # out_features for qzeros (columns).
            "model.layers.0.mlp.down_proj.qweight": torch.randint(
                -(2 ** 31), 2 ** 31 - 1, (in_features // pack_factor, out_features),
                dtype=torch.int32, generator=generator,
            ),
            "model.layers.0.mlp.down_proj.qzeros": torch.randint(
                0, 2 ** 31 - 1, (num_groups, out_features // pack_factor),
                dtype=torch.int32, generator=generator,
            ),
            "model.layers.0.mlp.down_proj.scales": torch.rand(
                (num_groups, out_features), dtype=torch.float16, generator=generator,
            ),
            "model.layers.0.mlp.down_proj.g_idx": (
                torch.arange(in_features, dtype=torch.int32) // group_size
            ),
            "model.layers.0.mlp.down_proj.bias": torch.randn(
                out_features, dtype=torch.float16, generator=generator,
            ),
        },
        str(path),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("bits", [4, 8])
def test_gptq_dequant_cuda_shard_is_bitwise_equal_to_cpu(tmp_path, bits):
    """``device="cuda"`` must work and produce exactly the CPU bytes."""

    shard = tmp_path / "model.safetensors"
    _write_gptq_shard(shard, bits)
    config = {"bits": bits, "checkpoint_format": "gptq"}

    cpu_tensors = convert_gptq_file(shard, torch.float16, config, device="cpu")
    cuda_tensors = convert_gptq_file(shard, torch.float16, config, device="cuda")

    assert cpu_tensors.keys() == cuda_tensors.keys()
    for key, expected in cpu_tensors.items():
        actual = cuda_tensors[key]
        assert actual.device.type == "cpu", key
        assert actual.dtype == expected.dtype, key
        assert torch.equal(actual, expected), key
