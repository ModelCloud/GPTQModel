# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel import ShardConfig, ShardLayout, reshard


def _make_state():
    return {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
        "model.layers.0.self_attn.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.layers.0.self_attn.scales": torch.ones(4, 2, dtype=torch.float16),
        "model.layers.1.mlp.down_proj.qweight": torch.arange(16, dtype=torch.int32).reshape(4, 4),
        "model.layers.1.mlp.down_proj.qzeros": torch.zeros(4, 1, dtype=torch.int32),
        "model.norm.weight": torch.ones(4, dtype=torch.float16),
        "lm_head.weight": torch.arange(32, dtype=torch.float16).reshape(8, 4),
        "model.rotary_emb.inv_freq": torch.ones(2, dtype=torch.float32),
    }


def _build_source(root: Path, *, with_config: bool = True):
    state = _make_state()
    root.mkdir(parents=True)

    names = sorted(state)
    split = len(names) // 2
    shard_names = [names[:split], names[split:]]
    weight_map = {}
    total_size = 0

    for index, group in enumerate(shard_names, start=1):
        filename = f"model-{index:05d}-of-{len(shard_names):05d}.safetensors"
        save_file(
            {name: state[name].contiguous() for name in group},
            str(root / filename),
            metadata={"format": "pt", "gptq_bits": "4"},
        )
        total_size += (root / filename).stat().st_size
        for name in group:
            weight_map[name] = filename

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    if with_config:
        (root / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
            encoding="utf-8",
        )
    (root / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")
    return state


def _read_index(target: Path):
    return json.loads((target / "model.safetensors.index.json").read_text(encoding="utf-8"))


def _assert_tensors_equal(loaded, expected):
    assert set(loaded) == set(expected)
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, expected[name])


def test_reshard_per_layer_with_single_global(tmp_path):
    source = tmp_path / "source"
    state = _build_source(source)
    target = tmp_path / "target"

    result = reshard(source, target)

    assert result == str(target)
    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.layers.1" / "layer.safetensors").is_file()
    assert (target / "model.safetensors").is_file()

    # Every non-layer tensor lives in the single global shard, not in per-module files.
    assert not (target / "model.embed_tokens.safetensors").exists()
    assert not (target / "lm_head.safetensors").exists()
    assert not (target / "model.norm.safetensors").exists()
    assert not (target / "model.rotary_emb.safetensors").exists()
    assert not (target / "model.embed_tokens").exists()

    assert (target / "config.json").is_file()
    assert (target / "tokenizer_config.json").is_file()

    weight_map = _read_index(target)["weight_map"]
    assert weight_map["model.layers.0.self_attn.qweight"] == "model.layers.0/layer.safetensors"
    assert weight_map["model.layers.0.self_attn.scales"] == "model.layers.0/layer.safetensors"
    assert weight_map["model.layers.1.mlp.down_proj.qweight"] == "model.layers.1/layer.safetensors"
    assert weight_map["model.embed_tokens.weight"] == "model.safetensors"
    assert weight_map["model.norm.weight"] == "model.safetensors"
    assert weight_map["lm_head.weight"] == "model.safetensors"
    assert weight_map["model.rotary_emb.inv_freq"] == "model.safetensors"

    global_state = load_file(str(target / "model.safetensors"))
    assert set(global_state) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.rotary_emb.inv_freq",
    }
    _assert_tensors_equal(global_state, {name: state[name] for name in global_state})
    for layer in ("model.layers.0", "model.layers.1"):
        loaded = load_file(str(target / layer / "layer.safetensors"))
        _assert_tensors_equal(loaded, {name: state[name] for name in loaded})

    total_size = sum(path.stat().st_size for path in target.rglob("*.safetensors"))
    assert _read_index(target)["metadata"]["total_size"] == total_size


def test_reshard_preserves_safetensors_metadata(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"

    reshard(source, target)

    with safe_open(str(target / "model.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["gptq_bits"] == "4"
    assert metadata["format"] == "pt"

    with safe_open(str(target / "model.layers.0" / "layer.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["gptq_bits"] == "4"


def test_reshard_keeps_source_untouched(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    before = {path.name: path.stat().st_size for path in source.iterdir()}

    reshard(source, tmp_path / "target")

    after = {path.name: path.stat().st_size for path in source.iterdir()}
    assert after == before


def test_reshard_splits_large_layer_but_not_global_shard(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"

    reshard(source, target, ShardConfig(max_shard_size=64))

    layer_shards = sorted(path.name for path in (target / "model.layers.0").glob("*.safetensors"))
    assert len(layer_shards) > 1
    assert all(name.startswith("layer-") for name in layer_shards)

    root_shards = sorted(path.name for path in target.glob("*.safetensors"))
    assert root_shards == ["model.safetensors"]

    weight_map = _read_index(target)["weight_map"]
    assert weight_map["model.embed_tokens.weight"] == "model.safetensors"
    assert weight_map["model.norm.weight"] == "model.safetensors"
    assert weight_map["model.layers.0.self_attn.qweight"].startswith("model.layers.0/layer-")


def test_reshard_single_layout_writes_one_file(tmp_path):
    source = tmp_path / "source"
    state = _build_source(source)
    target = tmp_path / "target"

    reshard(source, target, ShardConfig(layout=ShardLayout.SINGLE))

    weight_files = sorted(target.glob("*.safetensors"))
    assert len(weight_files) == 1
    assert not (target / "model.layers.0").exists()
    assert not (target / "model.safetensors.index.json").exists()

    _assert_tensors_equal(load_file(str(weight_files[0])), state)


def test_reshard_requires_overwrite_for_non_empty_destination(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    target = tmp_path / "target"
    target.mkdir()
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        reshard(source, target)

    reshard(source, target, overwrite=True)
    assert not (target / "stale.txt").exists()
    assert (target / "model.safetensors").is_file()


def test_reshard_accepts_loaded_model_like_object(tmp_path):
    source = tmp_path / "source"
    _build_source(source)

    class _ModelLike:
        def __init__(self, path):
            self.model_local_path = str(path)

        @classmethod
        def extract_layers_node(cls):
            return ["model.layers"]

    target = tmp_path / "target"
    reshard(_ModelLike(source), target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()


def test_reshard_explicit_layer_prefixes(tmp_path):
    source = tmp_path / "source"
    _build_source(source)

    target = tmp_path / "target"
    reshard(source, target, ShardConfig(layer_prefixes=["model.layers"]))

    assert (target / "model.layers.1" / "layer.safetensors").is_file()


def test_reshard_detects_layers_without_config(tmp_path):
    source = tmp_path / "source"
    _build_source(source, with_config=False)

    target = tmp_path / "target"
    reshard(source, target)

    assert (target / "model.layers.0" / "layer.safetensors").is_file()
    assert (target / "model.safetensors").is_file()


def test_reshard_is_repeatable_on_nested_layout(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    reshard(source, first)
    reshard(first, second)

    assert (second / "model.layers.0" / "layer.safetensors").is_file()
    assert _read_index(second)["weight_map"] == _read_index(first)["weight_map"]
    assert (second / "model.safetensors").is_file()


def test_shard_config_rejects_unknown_layout():
    with pytest.raises(ValueError):
        ShardConfig(layout="not-a-layout")
