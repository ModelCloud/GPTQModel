# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import save_file

from gptqmodel.models.base import BaseQModel
from gptqmodel.models import writer as writer_module
from gptqmodel.models.writer import ModelWriter, _model_free_embedding_replacement_state_dict
from gptqmodel.quantization.config import FORMAT, METHOD
from gptqmodel.utils.model import OffloadTensorRef


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


def test_base_save_routes_embedding_only_to_dedicated_save(tmp_path):
    captured = {}

    def save_quantized_embeddings(**kwargs):
        captured["embedding_save"] = kwargs

    def save_quantized(**_kwargs):
        raise AssertionError("regular save_quantized should not be used for embedding-only quantization")

    model = SimpleNamespace(
        quantized=True,
        _model_free_weight_only_embeddings_only=True,
        save_quantized_embeddings=save_quantized_embeddings,
        save_quantized=save_quantized,
        quant_override_files={},
    )

    BaseQModel.save(
        model,
        save_dir=str(tmp_path / "save"),
        safetensors_metadata={"source": "test"},
        max_shard_size=None,
        meta_quantizer="unit:test",
        eora_path=str(tmp_path / "eora"),
        split_by="layer",
    )

    assert captured["embedding_save"] == {
        "save_dir": str(tmp_path / "save"),
        "safetensors_metadata": {"source": "test"},
        "max_shard_size": None,
        "meta_quantizer": "unit:test",
    }


def test_save_quantized_embeddings_uses_embedding_replacement_helper(tmp_path, monkeypatch):
    captured = {}
    writer = _build_writer(tmp_path)
    writer._model_free_weight_only_replacement_prefixes = {"embed_tokens", "lm_head"}
    writer._model_free_weight_only_embedding_replacement_prefixes = {"embed_tokens", "lm_head"}
    _patch_save_io(monkeypatch, captured)

    def fake_embedding_helper(model, turtle_model, prefixes):
        captured["embedding_helper"] = {
            "model": model,
            "turtle_model": turtle_model,
            "prefixes": list(prefixes),
        }
        return collections.OrderedDict()

    def fail_regular_helper(*_args, **_kwargs):
        raise AssertionError("regular model-free replacement helper should not be used")

    monkeypatch.setattr(writer_module, "_model_free_embedding_replacement_state_dict", fake_embedding_helper)
    monkeypatch.setattr(writer_module, "_model_free_replacement_state_dict", fail_regular_helper)

    writer.save_quantized_embeddings(save_dir=str(tmp_path / "save"), max_shard_size=None)

    assert captured["embedding_helper"] == {
        "model": writer.model,
        "turtle_model": writer.turtle_model,
        "prefixes": ["embed_tokens", "lm_head"],
    }
    assert "alias_all_called" not in captured
    assert captured["stream_state_dict"] == collections.OrderedDict()


def test_save_quantized_uses_regular_replacement_helper_for_non_embedding_prefix(tmp_path, monkeypatch):
    captured = {}
    writer = _build_writer(tmp_path)
    writer._model_free_weight_only_replacement_prefixes = {"embed_tokens", "layers.0.linear"}
    writer._model_free_weight_only_embedding_replacement_prefixes = {"embed_tokens"}
    _patch_save_io(monkeypatch, captured)

    def fail_embedding_helper(*_args, **_kwargs):
        raise AssertionError("embedding-only replacement helper should not be used")

    def fake_regular_helper(model, turtle_model, prefixes):
        captured["regular_helper"] = {
            "model": model,
            "turtle_model": turtle_model,
            "prefixes": list(prefixes),
        }
        return collections.OrderedDict()

    monkeypatch.setattr(writer_module, "_model_free_embedding_replacement_state_dict", fail_embedding_helper)
    monkeypatch.setattr(writer_module, "_model_free_replacement_state_dict", fake_regular_helper)

    writer.save_quantized(save_dir=str(tmp_path / "save"), max_shard_size=None)

    assert captured["regular_helper"] == {
        "model": writer.model,
        "turtle_model": writer.turtle_model,
        "prefixes": ["embed_tokens", "layers.0.linear"],
    }
    assert "alias_all_called" not in captured
    assert captured["stream_state_dict"] == collections.OrderedDict()


def test_embedding_replacement_state_dict_replaces_embedding_tensor_and_streams_others(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(
        {
            "embed_tokens.weight": torch.ones(4, 3, dtype=torch.float16),
            "layers.0.linear.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
        },
        str(source_dir / "model.safetensors"),
    )
    turtle_model = SimpleNamespace(
        _weight_map={
            "embed_tokens.weight": "model.safetensors",
            "layers.0.linear.weight": "model.safetensors",
        },
        model_local_path=str(source_dir),
    )
    model = nn.Module()
    model.embed_tokens = nn.Module()
    model.embed_tokens.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
    model.embed_tokens.register_buffer("scales", torch.ones(3, 1, dtype=torch.float16))

    state_dict = _model_free_embedding_replacement_state_dict(model, turtle_model, ["embed_tokens"])

    assert "embed_tokens.weight" not in state_dict
    assert torch.equal(state_dict["embed_tokens.qweight"].source, model.embed_tokens.qweight)
    assert torch.equal(state_dict["embed_tokens.scales"].source, model.embed_tokens.scales)
    source = state_dict["layers.0.linear.weight"].source
    assert isinstance(source, OffloadTensorRef)
    assert source.format == "safetensors"
    assert source.weight_name == "layers.0.linear.weight"
    assert source.shape == (2, 3)
