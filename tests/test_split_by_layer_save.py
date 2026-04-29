import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import load_checkpoint_in_model

from gptqmodel.models.writer import ModelWriter
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


class _DummyGenerationConfig(_DummyConfig):
    pass


class _TinySplitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(6, 4)
        self.model.layers = nn.ModuleList(
            [
                nn.Linear(4, 4),
                nn.Linear(4, 4),
            ]
        )
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 6, bias=False)
        self.config = _DummyConfig()
        self.generation_config = _DummyGenerationConfig()

        with torch.no_grad():
            for idx, (_, tensor) in enumerate(self.state_dict().items(), start=1):
                tensor.copy_(torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape) + idx)

    def save_pretrained(self, save_dir, state_dict=None, is_main_process=True):
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"dummy": True}, handle)
        with open(os.path.join(save_dir, "generation_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"do_sample": False}, handle)


def _build_writer(tmp_path):
    class _Base:
        @classmethod
        def extract_layers_node(cls):
            return ["model.layers"]

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
    instance.turtle_model = SimpleNamespace()
    instance.model = _TinySplitModel()
    os.makedirs(instance.model_local_path, exist_ok=True)
    return instance


def _patch_writer_env(monkeypatch):
    monkeypatch.setattr("gptqmodel.models.writer.get_model_files_size", lambda _: 1)
    monkeypatch.setattr("gptqmodel.models.writer.alias_all_from_turtle_if_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr("gptqmodel.models.writer.sanitize_model_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gptqmodel.models.writer.sanitize_generation_config_file", lambda *_args, **_kwargs: False)


@pytest.mark.skip(reason="see gptqmodel/models/writer.py:SUPPORTED_SPLIT_BY")
def test_save_quantized_split_by_layer_writes_per_layer_dirs(tmp_path, monkeypatch):
    writer = _build_writer(tmp_path)
    _patch_writer_env(monkeypatch)

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir), split_by="layer", max_shard_size=None)

    assert (save_dir / "model.layers.0" / "layer.safetensors").exists()
    assert (save_dir / "model.layers.1" / "layer.safetensors").exists()
    assert (save_dir / "model.embed_tokens.safetensors").exists()
    assert (save_dir / "model.norm.safetensors").exists()
    assert (save_dir / "lm_head.safetensors").exists()
    assert not (save_dir / "model.embed_tokens").exists()
    assert not (save_dir / "model.norm").exists()
    assert not (save_dir / "lm_head").exists()
    assert not (save_dir / "model.safetensors").exists()

    index = json.loads((save_dir / "model.safetensors.index.json").read_text())

    assert index["weight_map"]["model.layers.0.weight"] == "model.layers.0/layer.safetensors"
    assert index["weight_map"]["model.layers.1.bias"] == "model.layers.1/layer.safetensors"
    assert index["weight_map"]["model.embed_tokens.weight"] == "model.embed_tokens.safetensors"
    assert index["weight_map"]["model.norm.weight"] == "model.norm.safetensors"
    assert index["weight_map"]["lm_head.weight"] == "lm_head.safetensors"


@pytest.mark.skip(reason="see gptqmodel/models/writer.py:SUPPORTED_SPLIT_BY")
def test_save_quantized_split_by_layer_still_shards_large_layer(tmp_path, monkeypatch):
    writer = _build_writer(tmp_path)
    _patch_writer_env(monkeypatch)

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir), split_by="layer", max_shard_size=64)

    layer0_dir = save_dir / "model.layers.0"
    layer0_shards = sorted(path.name for path in layer0_dir.glob("*.safetensors"))
    assert layer0_shards == [
        "layer-00001-of-00002.safetensors",
        "layer-00002-of-00002.safetensors",
    ]

    index = json.loads((save_dir / "model.safetensors.index.json").read_text())
    weight_file = index["weight_map"]["model.layers.0.weight"]
    bias_file = index["weight_map"]["model.layers.0.bias"]

    assert weight_file.startswith("model.layers.0/layer-")
    assert bias_file.startswith("model.layers.0/layer-")
    assert weight_file != bias_file


@pytest.mark.skip(reason="see gptqmodel/models/writer.py:SUPPORTED_SPLIT_BY")
def test_split_by_layer_index_loads_nested_layer_shards(tmp_path, monkeypatch):
    writer = _build_writer(tmp_path)
    _patch_writer_env(monkeypatch)

    save_dir = tmp_path / "save"
    expected_state = {name: tensor.clone() for name, tensor in writer.model.state_dict().items()}

    writer.save_quantized(save_dir=str(save_dir), split_by="layer", max_shard_size=64)

    reloaded = _TinySplitModel()
    with torch.no_grad():
        for tensor in reloaded.state_dict().values():
            tensor.zero_()

    load_checkpoint_in_model(reloaded, checkpoint=str(save_dir / "model.safetensors.index.json"))

    reloaded_state = reloaded.state_dict()
    for name, expected in expected_state.items():
        torch.testing.assert_close(reloaded_state[name], expected)
