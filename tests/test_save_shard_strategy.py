# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from gptqmodel import QuantizeConfig, ShardStrategy
from gptqmodel.models.writer import _stream_state_dict_per_layer_shards
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.utils.reshard import routed_module_templates_from_model_definition
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


def test_quantize_config_per_layer_moe_strategy_roundtrip():
    cfg = QuantizeConfig(bits=4, group_size=128, shard_strategy=ShardStrategy.PER_LAYER_MOE)
    payload = cfg.to_dict()
    assert payload["meta"]["shard_strategy"] == "per_layer_moe"
    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.shard_strategy is ShardStrategy.PER_LAYER_MOE


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


def test_stream_state_dict_per_layer_moe_groups_routed_modules_without_splitting_state():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(2, 2),
    }
    for expert in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            tensors[f"{prefix}.qweight"] = torch.randint(0, 16, (2, 2), dtype=torch.int32)
            tensors[f"{prefix}.scales"] = torch.randn(2, 1)
    state_dict = _state_dict_from_tensors(tensors)

    with tempfile.TemporaryDirectory() as tmp:
        expected_files, weight_map, _ = _stream_state_dict_per_layer_shards(
            state_dict,
            save_dir=tmp,
            model_base_name="model",
            model_save_name="model.safetensors",
            metadata={},
            max_shard_size=None,
            layer_prefixes=["model.layers"],
            shard_strategy=ShardStrategy.PER_LAYER_MOE,
            routed_module_templates=routed_module_templates_from_model_definition(DeepSeekV4QModel),
            moe_modules_per_shard=2,
        )

        # One dense/shared shard plus ceil(9 routed projection modules / 2).
        assert len(expected_files) == 6
        dense_file = weight_map["model.layers.0.self_attn.q_proj.weight"]
        assert weight_map["model.layers.0.mlp.shared_experts.gate_proj.weight"] == dense_file
        routed_files = set()
        for expert in range(3):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
                assert weight_map[f"{prefix}.qweight"] == weight_map[f"{prefix}.scales"]
                routed_files.add(weight_map[f"{prefix}.qweight"])
        assert dense_file not in routed_files
        assert len(routed_files) == 5

        saved = {}
        for filename in expected_files:
            saved.update(load_file(os.path.join(tmp, filename)))
        assert set(saved) == set(tensors)
        for name, original in tensors.items():
            assert torch.equal(saved[name], original)


def test_stream_state_dict_per_layer_moe_max_size_never_splits_module_state():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
    }
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}.gate_proj"
        tensors[f"{prefix}.qweight"] = torch.randint(0, 16, (4, 4), dtype=torch.int32)
        tensors[f"{prefix}.scales"] = torch.randn(4, 2)
        tensors[f"{prefix}.qzeros"] = torch.randint(0, 16, (4, 1), dtype=torch.int32)

    with tempfile.TemporaryDirectory() as tmp:
        expected_files, weight_map, _ = _stream_state_dict_per_layer_shards(
            _state_dict_from_tensors(tensors),
            save_dir=tmp,
            model_base_name="model",
            model_save_name="model.safetensors",
            metadata={},
            # One routed module is larger than this cap. The shard may exceed
            # the cap, but its logically inseparable state must stay together.
            max_shard_size=64,
            layer_prefixes=["model.layers"],
            shard_strategy=ShardStrategy.PER_LAYER_MOE,
            routed_module_templates=routed_module_templates_from_model_definition(DeepSeekV4QModel),
            moe_modules_per_shard=2,
        )

        assert len(expected_files) == 3  # dense plus one atomic shard per routed module
        for expert in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert}.gate_proj"
            filenames = {
                weight_map[f"{prefix}.qweight"],
                weight_map[f"{prefix}.scales"],
                weight_map[f"{prefix}.qzeros"],
            }
            assert len(filenames) == 1


def test_save_quantized_per_layer_moe_api_and_unsupported_kernel_fallback(tmp_path, monkeypatch):
    from test_out_of_model_tensors import _build_writer_with_out_of_model_file, _patch_basic_env

    original = tmp_path / "original"
    original.mkdir()
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.experts.0.gate_proj.qweight": torch.randint(0, 16, (4, 4), dtype=torch.int32),
        "model.layers.0.mlp.experts.0.gate_proj.scales": torch.randn(4, 2),
    }
    state_dict = _state_dict_from_tensors(tensors)
    writer = _build_writer_with_out_of_model_file(str(original))
    writer.quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        shard_strategy=ShardStrategy.PER_LAYER,
    )

    def save_pretrained_with_quantization_config(save_dir, state_dict=None, is_main_process=True):
        del state_dict, is_main_process
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dummy": True,
                    "quantization_config": writer.model.config.quantization_config,
                },
                handle,
            )
        with open(os.path.join(save_dir, "generation_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"do_sample": True}, handle)

    monkeypatch.setattr(writer.model, "save_pretrained", save_pretrained_with_quantization_config)
    writer.extract_layers_node = lambda: ["model.layers"]
    _patch_basic_env(monkeypatch, state_dict)
    monkeypatch.setattr(
        "gptqmodel.models.writer.routed_module_templates_from_model_definition",
        lambda _model_cls: ["mlp.experts.#.gate_proj"],
    )

    moe_dir = tmp_path / "moe-save"
    writer.save_quantized(
        save_dir=str(moe_dir),
        shard_strategy=ShardStrategy.PER_LAYER_MOE,
        moe_modules_per_shard=1,
        max_shard_size=None,
    )
    with open(moe_dir / "model.safetensors.index.json", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    assert set(weight_map) == set(tensors)
    assert weight_map["model.layers.0.mlp.experts.0.gate_proj.qweight"] == weight_map[
        "model.layers.0.mlp.experts.0.gate_proj.scales"
    ]
    with open(moe_dir / "quantize_config.json", encoding="utf-8") as handle:
        saved_quantize_config = json.load(handle)
    with open(moe_dir / "config.json", encoding="utf-8") as handle:
        saved_model_config = json.load(handle)
    assert saved_quantize_config["meta"]["shard_strategy"] == ShardStrategy.PER_LAYER_MOE.value
    assert (
        saved_model_config["quantization_config"]["meta"]["shard_strategy"]
        == ShardStrategy.PER_LAYER_MOE.value
    )
    assert writer.quantize_config.shard_strategy is ShardStrategy.PER_LAYER

    fallback_state = _state_dict_from_tensors({"model.layers.0.weight": torch.ones(2, 2)})
    _patch_basic_env(monkeypatch, fallback_state)
    writer.qlinear_kernel = SimpleNamespace(SUPPORTS_SHARDS=False, REQUIRES_FORMAT_V2=False)
    fallback_dir = tmp_path / "fallback-save"
    fallback_dir.mkdir()
    stale_index = fallback_dir / "model.safetensors.index.json"
    stale_index.write_text("stale", encoding="utf-8")
    writer.save_quantized(
        save_dir=str(fallback_dir),
        shard_strategy=ShardStrategy.PER_LAYER_MOE,
        max_shard_size=None,
    )

    assert (fallback_dir / "model.safetensors").is_file()
    assert not stale_index.exists()
    with open(fallback_dir / "quantize_config.json", encoding="utf-8") as handle:
        fallback_quantize_config = json.load(handle)
    with open(fallback_dir / "config.json", encoding="utf-8") as handle:
        fallback_model_config = json.load(handle)
    assert fallback_quantize_config["meta"]["shard_strategy"] is None
    assert fallback_model_config["quantization_config"]["meta"]["shard_strategy"] is None
    assert writer.quantize_config.shard_strategy is ShardStrategy.PER_LAYER
