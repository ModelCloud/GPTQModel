# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5502: a resumed rebuild must not keep the previous layout's shards.

``dequantize_model(..., resume=True)`` rebuilds when the resume manifest does
not match the source checkpoint, layout or dtype. The rebuild rewrote only the
new layout's shard names, so every file the previous export wrote under another
name stayed in the directory. A stale ``model.safetensors`` next to a new
sharded index is what ``from_pretrained`` prefers, so the resumed export loaded
the previous model's weights with the new config -- #5396 again, through a
layout change.
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import dequantize_model
from tests.test_dequant_sharded_gptq_awq_5398 import _split_source
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]


def _source(path, *, bits=4):
    path.mkdir()
    _tiny_llama_checkpoint(path, bits=bits, lm_head_bits=bits, legacy_config=False)
    return path


def _sharded(source, target, assign):
    return _split_source(source, target, assign)


def _packed_only(key):
    return "packed" if key.endswith(".qweight") else "other"


def _exported(out):
    """Read every exported tensor, detached from the shard's memory map."""

    tensors = {}
    for shard in sorted(out.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as reader:
            for key in reader.keys():
                tensors[key] = reader.get_tensor(key).clone()
    return tensors


def _assert_close(left, right):
    assert set(left) == set(right)
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], rtol=0.0, atol=0.0)


def test_rebuild_removes_a_stale_single_file_export(tmp_path):
    """Single-file -> sharded: the stale ``model.safetensors`` must go (#5502)."""

    single = _source(tmp_path / "single", bits=4)
    sharded = _sharded(_source(tmp_path / "sharded_src", bits=8), tmp_path / "sharded", _packed_only)

    out = tmp_path / "out"
    dequantize_model(single, out, target_dtype=torch.float32)
    assert (out / "model.safetensors").is_file()

    dequantize_model(sharded, out, target_dtype=torch.float32, resume=True)

    assert not (out / "model.safetensors").exists(), (
        "a stale single-file export wins over the new sharded index in from_pretrained"
    )
    assert sorted(p.name for p in out.glob("*.safetensors")) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]

    control = tmp_path / "control"
    dequantize_model(sharded, control, target_dtype=torch.float32)
    _assert_close(_exported(out), _exported(control))


def test_rebuild_removes_orphan_shards_from_the_previous_layout(tmp_path):
    """Sharded -> single-file: the previous layout's shards must go (#5502)."""

    sharded = _sharded(_source(tmp_path / "sharded_src", bits=8), tmp_path / "sharded", _packed_only)
    single = _source(tmp_path / "single", bits=4)

    out = tmp_path / "out"
    dequantize_model(sharded, out, target_dtype=torch.float32)
    stale_shards = sorted(p.name for p in out.glob("model-*.safetensors"))
    assert len(stale_shards) == 2

    dequantize_model(single, out, target_dtype=torch.float32, resume=True)

    assert sorted(p.name for p in out.glob("*.safetensors")) == ["model.safetensors"]

    control = tmp_path / "control"
    dequantize_model(single, control, target_dtype=torch.float32)
    _assert_close(_exported(out), _exported(control))


def test_rebuild_keeps_files_the_manifest_does_not_list(tmp_path):
    """Only this tool's own recorded output is removed; user files stay (#5502)."""

    single = _source(tmp_path / "single", bits=4)
    sharded = _sharded(_source(tmp_path / "sharded_src", bits=8), tmp_path / "sharded", _packed_only)

    out = tmp_path / "out"
    dequantize_model(single, out, target_dtype=torch.float32)

    unrelated = out / "user-extra.safetensors"
    save_file({"user": torch.zeros(1)}, str(unrelated))

    dequantize_model(sharded, out, target_dtype=torch.float32, resume=True)

    assert unrelated.is_file(), "a file the resume manifest never listed must not be deleted"
    assert not (out / "model.safetensors").exists()
