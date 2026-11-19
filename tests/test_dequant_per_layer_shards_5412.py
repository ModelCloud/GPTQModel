# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5412: dequantize a per-layer checkpoint into a nested shard layout.

``reshard(..., layout=ShardLayout.PER_LAYER)`` writes
``model.layers.N/layer.safetensors`` and indexes those relative names. The
dense export kept the names, so it tried to write into a subdirectory it had
never created. Creating that directory and resuming was worse: the recursive
auxiliary copy matched only top-level entry names, so it copied the source
layer shard over the freshly written dense file while the index still
advertised the dense tensor.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from gptqmodel import ShardConfig, ShardLayout, reshard
from gptqmodel.utils.model_dequant import dequantize_model
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]

_PACKED_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


def _per_layer_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=4, lm_head_bits=4, legacy_config=False)

    per_layer = tmp_path / "per_layer"
    reshard(source, per_layer, config=ShardConfig(layout=ShardLayout.PER_LAYER))

    index = json.loads(
        (per_layer / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    nested = {shard for shard in index["weight_map"].values() if "/" in shard}
    assert nested, "the fixture did not produce a nested shard layout"
    return source, per_layer


def _exported_tensors(out):
    """Every tensor the export wrote, keyed by name, with its owning file."""

    weights = {}
    owners = {}
    for path in sorted(out.rglob("*.safetensors")):
        with safe_open(str(path), framework="pt") as reader:
            for key in reader.keys():
                assert key not in weights, f"{key} written by {owners[key]} and {path}"
                weights[key] = reader.get_tensor(key)
                owners[key] = path
    return weights, owners


def _assert_dense(weights, owners):
    packed = [
        name for name in weights if name.endswith(_PACKED_SUFFIXES)
    ]
    assert not packed, f"packed buffers survived the export: {packed[:3]}"
    for name in weights:
        if name.endswith(".weight"):
            assert weights[name].dtype == torch.float32, (name, weights[name].dtype)


def test_dequantize_per_layer_checkpoint_writes_nested_shards(tmp_path):
    source, per_layer = _per_layer_checkpoint(tmp_path)

    flat_out = tmp_path / "flat_out"
    dequantize_model(source, flat_out, target_dtype=torch.float32)
    expected, _ = _exported_tensors(flat_out)

    out = tmp_path / "out"
    dequantize_model(per_layer, out, target_dtype=torch.float32)
    actual, owners = _exported_tensors(out)

    _assert_dense(actual, owners)
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0.0, atol=0.0)

    # The nested shard names are preserved, so the export stays per-layer.
    assert any(path.parent.name.startswith("model.layers.") for path in owners.values())


def test_resumed_per_layer_export_keeps_the_dense_weights(tmp_path):
    _source, per_layer = _per_layer_checkpoint(tmp_path)
    out = tmp_path / "out"

    dequantize_model(per_layer, out, target_dtype=torch.float32)
    first, _ = _exported_tensors(out)

    # A second run reuses the completed shards; the auxiliary copy must not
    # overwrite them with the packed source files.
    dequantize_model(per_layer, out, target_dtype=torch.float32, resume=True)
    second, owners = _exported_tensors(out)

    _assert_dense(second, owners)
    assert set(second) == set(first)
    for name, tensor in first.items():
        torch.testing.assert_close(second[name], tensor, rtol=0.0, atol=0.0)


def test_per_layer_aux_files_still_reach_the_export(tmp_path):
    _source, per_layer = _per_layer_checkpoint(tmp_path)
    (per_layer / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 64}), encoding="utf-8"
    )
    (per_layer / "nested_assets").mkdir()
    (per_layer / "nested_assets" / "chat_template.jinja").write_text(
        "{{ messages }}", encoding="utf-8"
    )

    out = tmp_path / "out"
    dequantize_model(per_layer, out, target_dtype=torch.float32)

    assert (out / "tokenizer_config.json").is_file()
    assert (out / "nested_assets" / "chat_template.jinja").is_file()
