# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import dataclasses

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
