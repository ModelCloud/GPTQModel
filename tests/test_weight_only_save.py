# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.models import writer as writer_module
from gptqmodel.models.writer import ModelWriter, _save_embedding_replacement_safetensors
from gptqmodel.quantization.config import FORMAT, METHOD


class _DummyKernel:
    REQUIRES_FORMAT_V2 = False
    SUPPORTS_SHARDS = True


class _DummyQuantizeConfig:
    method = METHOD.GPTQ
    format = FORMAT.GPTQ
    checkpoint_format = FORMAT.GPTQ
    quant_method = METHOD.GPTQ
    damp_percent = 0.0
    damp_auto_increment = 0.0
    static_groups = False
    true_sequential = False
    mse = False
    gptaq = None
    foem = None
    act_group_aware = False
    adapter = None
    dynamic = False
    offload_to_disk = False
    offload_to_disk_path = None
    lm_head = False

    def __init__(self):
        self._meta = {}

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        clone._meta = copy.deepcopy(self._meta, memo)
        return clone

    def meta_set_versionable(self, key, value):
        self._meta[key] = value

    def meta_set(self, key, value):
        self._meta[key] = value

    def to_dict(self):
        return {"meta": dict(self._meta)}

    def save_pretrained(self, save_dir):
        with open(os.path.join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"meta": dict(self._meta)}, handle)

    def extract_adapter_rank_patterns(self):
        return {}


class _DummyConfig:
    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        clone.__dict__ = copy.deepcopy(self.__dict__, memo)
        return clone


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyConfig()
        self.generation_config = _DummyConfig()

    def save_pretrained(self, save_dir, state_dict=None, is_main_process=True):
        del state_dict, is_main_process
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"dummy": True}, handle)
        with open(os.path.join(save_dir, "generation_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"do_sample": False}, handle)


def _build_writer(tmp_path):
    class _Base:
        @classmethod
        def extract_layers_node(cls):
            return ["layers"]

    DummyWriter = ModelWriter(_Base)
    instance = DummyWriter()
    instance.quantized = True
    instance.quantize_config = _DummyQuantizeConfig()
    instance.quant_log = []
    instance.load_quantized_model = False
    instance.qlinear_kernel = _DummyKernel()
    instance.model_local_path = str(tmp_path / "original")
    instance.trust_remote_code = False
    instance.tokenizer = None
    instance.processor = None
    instance.model = _DummyModel()
    instance.turtle_model = SimpleNamespace(
        _weight_map={"embed_tokens.weight": "model.safetensors"},
        model_local_path=instance.model_local_path,
    )
    os.makedirs(instance.model_local_path, exist_ok=True)
    return instance


def _patch_save_io(monkeypatch, captured):
    monkeypatch.setattr(writer_module, "get_model_files_size", lambda _path: 1)
    monkeypatch.setattr(writer_module, "sanitize_model_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(writer_module, "sanitize_generation_config_file", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(writer_module, "_normalize_legacy_tied_weights_keys", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(writer_module, "_cleanup_saved_weight_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        writer_module,
        "alias_all_from_turtle_if_meta",
        lambda *args, **kwargs: captured.setdefault("alias_all_called", True),
    )

    def fake_stream_state_dict_to_shards(state_dict, **kwargs):
        captured["stream_state_dict"] = state_dict
        captured["stream_kwargs"] = kwargs
        return ["model.safetensors"], {}, 0

    monkeypatch.setattr(writer_module, "streaming_state_dict_to_shards", fake_stream_state_dict_to_shards)


def test_save_quantized_embeddings_uses_embedding_shard_replacement_helper(tmp_path, monkeypatch):
    captured = {}
    writer = _build_writer(tmp_path)
    writer._embedding_replacement_prefixes = {"embed_tokens", "lm_head"}
    writer.quantize_config.dynamic = {
        "embed_tokens": {"bits": 8, "group_size": 32},
    }
    _patch_save_io(monkeypatch, captured)

    def fake_embedding_helper(model, turtle_model, prefixes, *, save_dir, metadata):
        captured["embedding_shard_helper"] = {
            "model": model,
            "turtle_model": turtle_model,
            "prefixes": list(prefixes),
            "save_dir": save_dir,
            "metadata": metadata,
        }
        return (
            ["model.safetensors"],
            {"embed_tokens.qweight": "model.safetensors", "lm_head.qweight": "model.safetensors"},
            0,
            ["embed_tokens.weight", "lm_head.weight"],
        )

    def fail_streaming_shards(*_args, **_kwargs):
        raise AssertionError("embedding-only save should not rewrite all tensors through streaming_state_dict_to_shards")

    monkeypatch.setattr(writer_module, "_save_embedding_replacement_safetensors", fake_embedding_helper)
    monkeypatch.setattr(writer_module, "streaming_state_dict_to_shards", fail_streaming_shards)
    monkeypatch.setattr(
        writer_module,
        "_cleanup_saved_weight_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("embedding-only save should not clean up untouched shards")
        ),
    )

    save_dir = tmp_path / "save"
    save_dir.mkdir()
    with open(save_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"model_type": "dummy", "quantization_config": {"bits": 4}}, handle)
    with open(save_dir / "quantize_config.json", "w", encoding="utf-8") as handle:
        json.dump({"bits": 4}, handle)
    with open(save_dir / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "embed_tokens.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                    "layers.0.linear.weight": "model.safetensors",
                },
            },
            handle,
        )
    writer.save_quantized_embeddings(save_dir=str(save_dir), max_shard_size=None)

    assert captured["embedding_shard_helper"] == {
        "model": writer.model,
        "turtle_model": writer.turtle_model,
        "prefixes": ["embed_tokens", "lm_head"],
        "save_dir": str(save_dir),
        "metadata": {"format": "pt"},
    }
    assert "alias_all_called" not in captured
    with open(save_dir / "config.json", "r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    with open(save_dir / "quantize_config.json", "r", encoding="utf-8") as handle:
        quant_config_payload = json.load(handle)
    with open(save_dir / "model.safetensors.index.json", "r", encoding="utf-8") as handle:
        index_payload = json.load(handle)
    assert config_payload["quantization_config"]["dynamic"] == writer.quantize_config.dynamic
    assert quant_config_payload["dynamic"] == writer.quantize_config.dynamic
    assert "embed_tokens.weight" not in index_payload["weight_map"]
    assert "lm_head.weight" not in index_payload["weight_map"]
    assert index_payload["weight_map"]["embed_tokens.qweight"] == "model.safetensors"
    assert index_payload["weight_map"]["lm_head.qweight"] == "model.safetensors"
    assert index_payload["weight_map"]["layers.0.linear.weight"] == "model.safetensors"


def test_embedding_replacement_safetensors_only_rewrites_affected_shard(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {
            "embed_tokens.weight": torch.ones(4, 3, dtype=torch.float16),
        },
        str(source_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "layers.0.linear.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
        },
        str(source_dir / "model-00002-of-00002.safetensors"),
    )
    turtle_model = SimpleNamespace(
        _weight_map={
            "embed_tokens.weight": "model-00001-of-00002.safetensors",
            "layers.0.linear.weight": "model-00002-of-00002.safetensors",
        },
        model_local_path=str(source_dir),
    )
    model = nn.Module()
    model.embed_tokens = nn.Module()
    model.embed_tokens.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
    model.embed_tokens.register_buffer("scales", torch.ones(3, 1, dtype=torch.float16))
    saved_files = []

    original_save_file = writer_module.save_file

    def tracking_save_file(tensors, filename, metadata=None):
        saved_files.append(os.path.basename(filename))
        return original_save_file(tensors, filename, metadata=metadata)

    monkeypatch.setattr(writer_module, "save_file", tracking_save_file)

    save_dir = tmp_path / "save"
    save_dir.mkdir()
    save_file(
        {
            "embed_tokens.weight": torch.full((4, 3), 2, dtype=torch.float16),
        },
        str(save_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "layers.0.linear.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
        },
        str(save_dir / "model-00002-of-00002.safetensors"),
    )

    rewritten_files, tensor_to_filename, total_size, removed_tensor_names = _save_embedding_replacement_safetensors(
        model,
        turtle_model,
        ["embed_tokens"],
        save_dir=str(save_dir),
        metadata={"format": "pt"},
    )

    assert saved_files == ["model-00001-of-00002.safetensors"]
    assert rewritten_files == ["model-00001-of-00002.safetensors"]
    assert removed_tensor_names == ["embed_tokens.weight"]
    assert total_size > 0
    with safe_open(str(save_dir / "model-00001-of-00002.safetensors"), framework="pt", device="cpu") as handler:
        keys = set(handler.keys())
        assert "embed_tokens.weight" not in keys
        assert keys == {"embed_tokens.qweight", "embed_tokens.scales"}
    with safe_open(str(save_dir / "model-00002-of-00002.safetensors"), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"layers.0.linear.weight"}
        assert torch.equal(handler.get_tensor("layers.0.linear.weight"), torch.arange(6, dtype=torch.float16).reshape(2, 3))
    assert tensor_to_filename == {
        "embed_tokens.qweight": "model-00001-of-00002.safetensors",
        "embed_tokens.scales": "model-00001-of-00002.safetensors",
    }


def test_embedding_replacement_removes_runtime_dense_weight_from_existing_index(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {"embed_tokens.weight": torch.ones(4, 3, dtype=torch.float16)},
        str(source_dir / "model-00001-of-00002.safetensors"),
    )
    turtle_model = SimpleNamespace(
        _weight_map={"embed_tokens.weight": "model-00001-of-00002.safetensors"},
        model_local_path=str(source_dir),
    )

    def resolve_checkpoint_tensor_source(prefix, leaf):
        if prefix == "lm_head" and leaf == "weight":
            return "embed_tokens.weight", None, None, None
        return None, None, None, None

    turtle_model._resolve_checkpoint_tensor_source = resolve_checkpoint_tensor_source
    model = nn.Module()
    model.lm_head = nn.Module()
    model.lm_head.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
    model.lm_head.register_buffer("scales", torch.ones(3, 1, dtype=torch.float16))

    save_dir = tmp_path / "save"
    save_dir.mkdir()
    save_file(
        {"embed_tokens.weight": torch.full((4, 3), 2, dtype=torch.float16)},
        str(save_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {
            "lm_head.weight": torch.full((4, 3), 3, dtype=torch.float16),
            "layers.0.linear.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
        },
        str(save_dir / "model-00002-of-00002.safetensors"),
    )
    with open(save_dir / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors",
                    "layers.0.linear.weight": "model-00002-of-00002.safetensors",
                },
            },
            handle,
        )

    rewritten_files, tensor_to_filename, _total_size, removed_tensor_names = _save_embedding_replacement_safetensors(
        model,
        turtle_model,
        ["lm_head"],
        save_dir=str(save_dir),
        metadata={"format": "pt"},
    )

    assert rewritten_files == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert removed_tensor_names == ["embed_tokens.weight", "lm_head.weight"]
    assert tensor_to_filename["lm_head.qweight"] == "model-00001-of-00002.safetensors"
    assert tensor_to_filename["lm_head.scales"] == "model-00001-of-00002.safetensors"
    with safe_open(str(save_dir / "model-00001-of-00002.safetensors"), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"lm_head.qweight", "lm_head.scales"}
    with safe_open(str(save_dir / "model-00002-of-00002.safetensors"), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"layers.0.linear.weight"}
