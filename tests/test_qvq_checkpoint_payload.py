# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint tensor ownership must control mixed dense/QVQ reloads."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.utils.model import (
    prepare_qvq_checkpoint_rank8_buffers,
    select_qvq_checkpoint_modules,
    validate_loaded_qvq_checkpoint_modules,
)


def _checkpoint(tmp_path, tensors):
    shard = "model.safetensors"
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard)}), encoding="utf-8"
    )
    return tmp_path / "model.safetensors.index.json"


def test_qvq_mixed_payload_selects_dense_by_checkpoint_not_dynamic_rate(tmp_path):
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    checkpoint = _checkpoint(tmp_path, {"dense.weight": weight, "compressed.trellis": torch.ones(4, dtype=torch.int32)})
    modules = {"dense": nn.Linear(2, 2, bias=False), "compressed": nn.Linear(2, 2, bias=False)}
    selected, dense = select_qvq_checkpoint_modules(modules, checkpoint)
    assert tuple(selected) == ("compressed",)
    assert dense == ("dense",)


@pytest.mark.parametrize("tensors", [
    {},
    {"proj.weight": torch.ones(2, 2), "proj.trellis": torch.ones(4, dtype=torch.int32)},
])
def test_qvq_mixed_payload_rejects_missing_or_ambiguous_module(tmp_path, tensors):
    if tensors:
        checkpoint = _checkpoint(tmp_path, tensors)
    else:
        checkpoint = _checkpoint(tmp_path, {"unrelated.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="exactly one"):
        select_qvq_checkpoint_modules({"proj": nn.Linear(2, 2, bias=False)}, checkpoint)


def test_qvq_post_load_audit_rejects_nonzero_placeholder_and_zero_weight(tmp_path):
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    checkpoint = _checkpoint(tmp_path, {"dense.weight": original})
    model = nn.Module()
    model.dense = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.dense.weight.copy_(original)
    validate_loaded_qvq_checkpoint_modules(model, checkpoint, (), ("dense",))
    with torch.no_grad():
        model.dense.weight.add_(1.0)
    with pytest.raises(ValueError, match="differs from its serialized payload"):
        validate_loaded_qvq_checkpoint_modules(model, checkpoint, (), ("dense",))
    with torch.no_grad():
        model.dense.weight.zero_()
    with pytest.raises(ValueError):
        validate_loaded_qvq_checkpoint_modules(model, checkpoint, (), ("dense",))


def test_qvq_rank8_buffers_are_registered_for_checkpoint_load_and_audited(tmp_path):
    model = nn.Module()
    model.proj = QVQLinear(bits=3, in_features=16, out_features=16)
    tensors = {
        "proj.trellis": torch.ones_like(model.proj.trellis),
        "proj.SU": torch.ones_like(model.proj.SU),
        "proj.SV": torch.ones_like(model.proj.SV),
        "proj.rank8_A": torch.ones((16, 8), dtype=torch.float16),
        "proj.rank8_B": torch.ones((8, 16), dtype=torch.float16),
        "proj.rank8_metadata": torch.tensor([123, 125], dtype=torch.uint8),
    }
    checkpoint = _checkpoint(tmp_path, tensors)
    assert model.proj.rank8_A is None
    prepare_qvq_checkpoint_rank8_buffers(model, checkpoint, ("proj",))
    assert model.proj.rank8_A.shape == (16, 8)
    assert model.proj.rank8_B.shape == (8, 16)
    with torch.no_grad():
        for key, value in tensors.items():
            setattr(model.proj, key.removeprefix("proj."), value.clone())
    validate_loaded_qvq_checkpoint_modules(model, checkpoint, ("proj",), ())
    model.proj.rank8_A[0, 0] = 2
    with pytest.raises(ValueError, match="differs from its serialized payload"):
        validate_loaded_qvq_checkpoint_modules(model, checkpoint, ("proj",), ())
