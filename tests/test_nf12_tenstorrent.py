# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.utils.nf12_codec import encode
from gptqmodel.utils.nf12_tenstorrent import normalize_ids, prepare_nf12, read_nf12, verify_nf12


def test_canonical_slabs_and_descriptor(tmp_path):
    generator = torch.Generator().manual_seed(12)
    w = torch.randn((529, 67), generator=generator).bfloat16()
    w.view(torch.int16)[0, :8] = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32641, -63], dtype=torch.int16)
    descriptor, weight = prepare_nf12(w, "example.weight")
    packed, escapes, base = encode(torch.nn.functional.pad(w, (0, 61, 0, 15)))
    assert torch.equal(weight.packed, packed)
    assert torch.equal(weight.escapes, escapes)
    assert weight.exp_base == base
    path = tmp_path / "weight.safetensors"
    save_file(weight.state_dict(prefix="example.weight."), path)
    restored = read_nf12(path, descriptor)
    assert torch.equal(restored.packed, packed)
    assert torch.equal(restored.escapes, escapes)

    with pytest.raises(ValueError, match="descriptor mismatch"):
        verify_nf12(dataclasses.replace(descriptor, packed_sha256="0" * 64), restored)


def test_signed_ids_before_narrowing():
    ids = torch.tensor([-1, 0, 16, 31, 32, 2**32 + 16, -(2**32) + 16], dtype=torch.int64)
    actual = normalize_ids(ids, 32, padding_idx=-1)
    assert actual.dtype == torch.int32
    assert actual.tolist() == [-1, 0, 16, -1, -1, -1, -1]


def test_sfpu_transcode_offsets_and_verifies_source_bits():
    from gptqmodel.utils.nf12_ttnn_sfpu import _pack_four, _to_tiles, _transcode

    base = 60
    words = torch.randint(0, 256, (544, 128), generator=torch.Generator().manual_seed(56), dtype=torch.int32)
    words |= 0x3e00
    words[511, :2] = torch.tensor([0, 0x7f83])
    words[512, :2] = torch.tensor([0x8000, 0xff80])
    source = words.to(torch.int16).view(torch.bfloat16)
    descriptor, weight = prepare_nf12(source, "lm_head.weight", exp_base=base)

    packed, (headers, patches) = _transcode(descriptor, weight)
    direct, (direct_headers, direct_patches) = _pack_four(_to_tiles(words).transpose(-2, -1).contiguous(), base)
    assert torch.equal(packed, direct.to(torch.uint16))
    assert torch.equal(headers, direct_headers)
    assert torch.equal(patches, direct_patches)
    assert patches.flatten()[:4].bitwise_right_shift(12).tolist() == [0, 0x7f83, 0x8000, 0xff80]

    with pytest.raises(ValueError, match="source hash mismatch"):
        _transcode(dataclasses.replace(descriptor, source_sha256="0" * 64), weight)


@pytest.mark.parametrize("suffix", ["linear.16.64.1.57.1", "linear.16.64.1.57.1.0.0"])
def test_ttxla_artifact_rejects_wrong_number_of_fields(suffix):
    from gptqmodel.utils.nf12_ttxla import _PREFIX, _artifact

    with pytest.raises(ValueError, match="expected seven fields"):
        _artifact(_PREFIX + suffix, "", [], [], [], (1, 1), "")


def test_ttxla_packed_linear_buffers_two_k_pairs():
    from gptqmodel.utils.nf12_ttxla import _PREFIX, _artifact, _version_tag

    def artifact(dense):
        weight = (256, 32) if dense else (96, 32)
        shapes = [(160, 128), weight, (32, 32), (160, 64)]
        dtypes = ["bf16", "bf16" if dense else "si32", "si32", "bf16"]
        layouts = ["<interleaved><dram>!ttcore.tile<32x32"] * 4
        return json.loads(_artifact(
            _PREFIX + f"linear.48.128.129.0.4.{int(dense)}", _version_tag(),
            shapes, dtypes, layouts, (1, 1), "in,in,in,out",
        ))

    packed = artifact(False)
    dense = artifact(True)
    assert {cb["buffer_index"]: cb["total_size"] for cb in packed["cb_configs"]} == {
        0: 8192, 1: 32768, 16: 4096, 24: 23552,
    }
    assert {cb["buffer_index"]: cb["total_size"] for cb in dense["cb_configs"]} == {
        0: 4096, 1: 16384, 16: 4096,
    }
    assert packed["kernels"][2]["kernel_config"] == dense["kernels"][2]["kernel_config"]


def test_ttxla_sfpu_artifact_requires_base_tag_and_one_weight_slab():
    from gptqmodel.utils.nf12_ttxla import _PREFIX, _artifact, _version_tag

    layout = "!ttcore.tile<32x32 <interleaved> <dram>"
    shapes = [(32, 128), (192, 32), (184, 32), (32, 128)]
    args = (_version_tag(), shapes, ["bf16", "si32", "si32", "bf16"], [layout] * 4, (1, 1), "in,in,in,out")
    with pytest.raises(ValueError, match="exponent base"):
        _artifact(_PREFIX + "sfpu_linear.128.128.1.55.1.0", *args)

    resolved = json.loads(_artifact(_PREFIX + "sfpu_linear.128.128.1.56.1.0", *args))
    buffers = {cb["buffer_index"]: cb for cb in resolved["cb_configs"]}
    assert buffers[16]["total_size"] == 4 * 2048
    assert buffers[16]["data_format"] == "UInt16"
    assert buffers[2]["total_size"] == 2048
    assert buffers[2]["data_format"] == "BFloat16"

    multirow = [(96, 128), *shapes[1:3], (96, 128)]
    multiple_tiles = json.loads(_artifact(_PREFIX + "sfpu_linear.128.128.65.56.1.0", args[0], multirow, *args[2:]))
    assert multiple_tiles["cb_configs"] == resolved["cb_configs"]

    grouped = list(multirow)
    grouped[2] = (1, *grouped[2])
    group_id = _PREFIX + "sfpu_linear.128.128.65.56.1.2"
    with pytest.raises(ValueError, match="geometry"):
        _artifact(group_id, args[0], multirow, *args[2:])
    group_buffers = json.loads(_artifact(group_id, args[0], grouped, *args[2:]))["cb_configs"]
    expected = {cb["buffer_index"]: cb["total_size"] for cb in multiple_tiles["cb_configs"]}
    expected[1] *= 2
    assert {cb["buffer_index"]: cb["total_size"] for cb in group_buffers} == expected
