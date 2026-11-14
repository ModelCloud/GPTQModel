# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent BF16-bit checks for the experimental packed row-major layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel_nf12.api import NF12Weight
from gptqmodel_nf12.experimental_rowmajor import (
    NF12RowMajorWeight,
    embedding,
    embedding_repeat4,
    from_nf12,
)


def _codec():
    path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/nf12_codec.py"
    spec = importlib.util.spec_from_file_location("nf12_rowmajor_test_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collision_fixture() -> tuple[torch.Tensor, NF12Weight]:
    bits = torch.full((16, 128), 0x3f00, dtype=torch.int16)
    bits[8:, :64] = -16640
    bits[:8, 64:] = -16640
    bits[0, 64] = 0x7fc0  # NaN payload
    bits[8, 64] = -32768  # -0.0
    bits[1, 65] = 1  # subnormal
    bits[9, 65] = -1
    dense = bits.view(torch.bfloat16)
    packed, escapes, base = _codec().encode(dense, exp_base=63)
    return dense, NF12Weight(packed, escapes, base)


def _exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_cpu_transform_and_versioned_state(tmp_path: Path) -> None:
    _, source = _collision_fixture()
    row = from_nf12(source, backend="cpu")
    assert row.shape == (16, 128)
    assert row.original_escapes == 2
    assert row.selected_half_collisions == 254
    assert row.nbytes == source.nbytes + 8 * row.selected_half_collisions
    assert set(vars(row)) == {"packed", "escapes", "exp_base", "original_escapes",
                              "selected_half_collisions", "shape"}
    state = row.state_dict("weight.")
    assert "weight.nf12_packed" not in state
    assert "weight.nf12_row_packed" in state and "weight.nf12_row_escapes" in state
    restored = NF12RowMajorWeight.from_state_dict(state, "weight.")
    assert torch.equal(restored.packed, row.packed)
    assert torch.equal(restored.escapes, row.escapes)
    checkpoint = tmp_path / "row-major.safetensors"
    save_file(state, checkpoint)
    loaded = NF12RowMajorWeight.from_state_dict(load_file(checkpoint), "weight.")
    assert torch.equal(loaded.packed, row.packed)
    assert torch.equal(loaded.escapes, row.escapes)
    bad = dict(state)
    bad["weight.nf12_row_layout_version"] = torch.tensor([2], dtype=torch.uint8)
    with pytest.raises(ValueError, match="layout version"):
        NF12RowMajorWeight.from_state_dict(bad, "weight.")
    # Conversion is an explicit packed copy, not an inference-owned cache of source.
    saved = row.packed.clone()
    source.packed.zero_()
    assert torch.equal(row.packed, saved)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA CUDA")
def test_gpu_transform_exact_collision_and_invalid_ids() -> None:
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("requires SM80 or newer")
    dense, source = _collision_fixture()
    cpu = from_nf12(source, backend="cpu")
    gpu = from_nf12(source.to("cuda:0"), backend="cuda")
    assert gpu.nbytes == cpu.nbytes == source.nbytes + 8 * gpu.selected_half_collisions
    assert gpu.original_escapes == cpu.original_escapes == 2
    assert gpu.selected_half_collisions == cpu.selected_half_collisions == 254
    assert torch.equal(gpu.escapes[:4].cpu(), cpu.escapes[:4])
    for dtype in (torch.int32, torch.int64):
        values = [-1, 0, 1, 7, 8, 15, 16, 17]
        if dtype == torch.int64:
            values[-1] = 2**40
        ids = torch.tensor(values, dtype=dtype, device="cuda:0")
        expected = torch.zeros((len(values), 128), dtype=torch.bfloat16, device="cuda:0")
        for i, value in enumerate(values):
            if 0 <= value < 16:
                expected[i] = dense[value].cuda()
        for row in (gpu, cpu.to("cuda:0")):
            before = {name: (value.data_ptr(), value.dtype, value.nbytes)
                      for name, value in vars(row).items() if isinstance(value, torch.Tensor)}
            for _ in range(2):
                _exact(embedding(ids, row), expected)
            expected4 = expected.unsqueeze(1).repeat(1, 4, 1)
            if torch.cuda.get_device_capability(0) == (8, 0):
                for policy in ("default", "cg", "cs", "transpose_cs", "transpose_cg"):
                    for _ in range(2):
                        _exact(embedding_repeat4(ids, row, threads=128, ids_per_cta=2,
                                                 store_policy=policy), expected4)
            after = {name: (value.data_ptr(), value.dtype, value.nbytes)
                     for name, value in vars(row).items() if isinstance(value, torch.Tensor)}
            assert before == after
            assert set(after) == {"packed", "escapes"}
            assert after["packed"][1] == torch.uint8 and after["escapes"][1] == torch.int16
    restored = NF12RowMajorWeight.from_state_dict(gpu.state_dict())
    _exact(embedding(ids, restored), expected)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA CUDA")
def test_gpu_transform_zero_escape_table() -> None:
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("requires SM80 or newer")
    bits = torch.full((16, 64), -32768, dtype=torch.int16)
    dense = bits.view(torch.bfloat16)
    packed, escapes, base = _codec().encode(dense, exp_base=0)
    source = NF12Weight(packed, escapes, base)
    assert source.num_escapes == 0
    row = from_nf12(source.to("cuda:0"), backend="cuda")
    assert row.escapes.shape == (0, 4)
    ids = torch.tensor([0, 15, -1, 16], dtype=torch.int64, device="cuda:0")
    expected = torch.zeros((4, 64), dtype=torch.bfloat16, device="cuda:0")
    expected[:2] = dense[[0, 15]].cuda()
    _exact(embedding(ids, row), expected)
    if torch.cuda.get_device_capability(0) == (8, 0):
        _exact(embedding_repeat4(ids, row), expected.unsqueeze(1).repeat(1, 4, 1))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA CUDA")
def test_transform_rejects_bad_original_escape_index() -> None:
    if torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("requires SM80 or newer")
    _, source = _collision_fixture()
    bad = NF12Weight(source.packed.clone(), source.escapes.clone(), source.exp_base)
    words = bad.packed.numpy().view(np.uint32).reshape(1, 2, 3, 32, 4)
    kb, lane, atom = np.argwhere(words[0, :, 0] == 0)[0]
    plane = 1 if atom < 2 else 2
    words[0, kb, plane, lane, 2 * (atom & 1)] = bad.num_escapes + 100
    with pytest.raises(ValueError, match="escape index"):
        from_nf12(bad, backend="cpu")
    with pytest.raises(ValueError, match="escape index"):
        from_nf12(bad.to("cuda:0"), backend="cuda")
