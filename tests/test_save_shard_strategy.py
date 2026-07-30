# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import torch
from safetensors.torch import load_file

from gptqmodel import QuantizeConfig, ShardStrategy
from gptqmodel.models.writer import _stream_state_dict_per_layer_shards
from gptqmodel.utils.model import TensorSource


def _state_dict_from_tensors(tensors):
    state = {}
    for name, tensor in tensors.items():
        state[name] = TensorSource(
            name=name,
            torch_dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            source=tensor,
        )
    return state


def test_quantize_config_shard_strategy_roundtrip():
    cfg = QuantizeConfig(bits=4, group_size=128, shard_strategy=ShardStrategy.PER_LAYER)
    assert cfg.shard_strategy is ShardStrategy.PER_LAYER

    payload = cfg.to_dict()
    # shard_strategy is a save-time knob and is stored under meta, not as a top-level key
    assert "shard_strategy" not in payload
    assert payload["meta"]["shard_strategy"] == ShardStrategy.PER_LAYER.value

    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.shard_strategy is ShardStrategy.PER_LAYER
    assert restored.meta["shard_strategy"] == ShardStrategy.PER_LAYER.value


def test_quantize_config_shard_strategy_none_roundtrip():
    cfg = QuantizeConfig(bits=4, group_size=128, shard_strategy=None)
    assert cfg.shard_strategy is None

    payload = cfg.to_dict()
    assert payload["meta"]["shard_strategy"] is None

    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.shard_strategy is None


def test_stream_state_dict_per_layer_shards_groups_layers_and_non_layer():
    tensors = {
        "model.embed_tokens.weight": torch.randn(10, 4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.norm.weight": torch.randn(4),
        "lm_head.weight": torch.randn(10, 4),
    }
    state_dict = _state_dict_from_tensors(tensors)

    with tempfile.TemporaryDirectory() as tmp:
        expected_files, weight_map, total_size = _stream_state_dict_per_layer_shards(
            state_dict,
            save_dir=tmp,
            model_base_name="model",
            model_save_name="model.safetensors",
            metadata={},
            max_shard_size=None,
            layer_prefixes=["model.layers"],
        )

        assert len(expected_files) == 3, expected_files
        for i, name in enumerate(expected_files):
            assert name == f"model-{i + 1:05d}-of-00003.safetensors"
            assert os.path.isfile(os.path.join(tmp, name))

        # All layer 0 tensors are in the first shard, layer 1 in the second,
        # and all non-layer tensors in the final shard.
        layer0_file = weight_map["model.layers.0.self_attn.q_proj.weight"]
        layer1_file = weight_map["model.layers.1.self_attn.q_proj.weight"]
        non_layer_file = weight_map["model.embed_tokens.weight"]

        assert layer0_file == "model-00001-of-00003.safetensors"
        assert layer1_file == "model-00002-of-00003.safetensors"
        assert non_layer_file == "model-00003-of-00003.safetensors"

        assert weight_map["model.norm.weight"] == non_layer_file
        assert weight_map["lm_head.weight"] == non_layer_file

        # Spot-check tensor values round-trip correctly.
        saved = {}
        for fn in expected_files:
            saved.update(load_file(os.path.join(tmp, fn)))
        for name, original in tensors.items():
            assert (saved[name] - original).abs().max().item() == 0.0

        # Verify no stale `model.safetensors` or staging directory remains.
        assert not os.path.exists(os.path.join(tmp, "model.safetensors"))
        assert not os.path.exists(os.path.join(tmp, ".per_layer_staging"))


def test_stream_state_dict_per_layer_respects_max_shard_size():
    # Each tensor is roughly 4*4*4 = 64 bytes; with max_shard_size=80 bytes a
    # layer containing two tensors should split into two shards.
    large = torch.randn(4, 4)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": large,
        "model.layers.0.self_attn.k_proj.weight": large,
    }
    state_dict = _state_dict_from_tensors(tensors)

    with tempfile.TemporaryDirectory() as tmp:
        expected_files, weight_map, _ = _stream_state_dict_per_layer_shards(
            state_dict,
            save_dir=tmp,
            model_base_name="model",
            model_save_name="model.safetensors",
            metadata={},
            max_shard_size=80,
            layer_prefixes=["model.layers"],
        )

        assert len(expected_files) == 2
        assert all(f.startswith("model-") for f in expected_files)
        assert expected_files[0] != expected_files[1]
