# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import os
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from gptqmodel.models.writer import ModelWriter
from gptqmodel.quantization.config import FORMAT, METHOD


class _DummyKernel:
    REQUIRES_FORMAT_V2 = False
    SUPPORTS_SHARDS = True


class _DummyQuantizeConfig:
    format = FORMAT.GPTQ
    quant_method = METHOD.GPTQ
    damp_percent = 0.0
    damp_auto_increment = 0.0
    static_groups = False
    true_sequential = False
    mse = False
    v2 = False
    v2_alpha = 0.0
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
    def __init__(self):
        self.some_field = 1

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        clone.__dict__ = copy.deepcopy(self.__dict__, memo)
        return clone


class _DummyGenerationConfig(_DummyConfig):
    pass


class _DummyModel:
    def __init__(self):
        self.config = _DummyConfig()
        self.generation_config = _DummyGenerationConfig()

    def save_pretrained(self, save_dir, state_dict=None, is_main_process=True):
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"dummy": True}, handle)
        with open(os.path.join(save_dir, "generation_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"do_sample": True}, handle)


def _build_writer_with_out_of_model_file(model_local_path):
    class _Base:
        out_of_model_tensor_files = ["dangling.safetensors"]

    DummyWriter = ModelWriter(_Base)
    instance = DummyWriter()
    instance.quantized = True
    instance.quantize_config = _DummyQuantizeConfig()
    instance.quant_log = []
    instance.load_quantized_model = False
    instance.qlinear_kernel = _DummyKernel()
    instance.model_local_path = model_local_path
    instance.trust_remote_code = False
    instance.tokenizer = None
    instance.processor = None
    instance.turtle_model = SimpleNamespace()
    instance.model = _DummyModel()
    return instance


def test_out_of_model_tensor_files_are_copied_and_indexed(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()

    dangling_path = original_dir / "dangling.safetensors"
    save_file({"dangling.weight": torch.ones(1)}, str(dangling_path))

    writer = _build_writer_with_out_of_model_file(str(original_dir))

    monkeypatch.setattr("gptqmodel.models.writer.get_model_files_size", lambda _: 1)
    monkeypatch.setattr("gptqmodel.models.writer.alias_all_from_turtle_if_meta", lambda *args, **kwargs: None)

    base_state_dict = {"model.weight": torch.zeros(1)}

    def _fake_get_state_dict_for_save(*_args, **_kwargs):
        return dict(base_state_dict)

    monkeypatch.setattr("gptqmodel.models.writer.get_state_dict_for_save", _fake_get_state_dict_for_save)

    def _fake_streaming_state_dict_to_shards(state_dict, save_dir, model_base_name, single_file_name, metadata, *_args, **_kwargs):
        file_path = os.path.join(save_dir, single_file_name)
        save_file(state_dict, file_path, metadata=metadata)
        tensor_to_filename = dict.fromkeys(state_dict.keys(), single_file_name)
        total_size = os.path.getsize(file_path)
        return [single_file_name], tensor_to_filename, total_size

    monkeypatch.setattr(
        "gptqmodel.models.writer.streaming_state_dict_to_shards",
        _fake_streaming_state_dict_to_shards,
    )

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir))

    copied_path = save_dir / "dangling.safetensors"
    assert copied_path.exists()

    index_path = save_dir / "model.safetensors.index.json"
    assert index_path.exists()

    with open(index_path, "r", encoding="utf-8") as handle:
        index_data = json.load(handle)

    assert index_data["weight_map"]["model.weight"] == "model.safetensors"
    assert index_data["weight_map"]["dangling.weight"] == "dangling.safetensors"

    model_size = os.path.getsize(save_dir / "model.safetensors")
    dangling_size = os.path.getsize(copied_path)
    assert index_data["metadata"]["total_size"] == model_size + dangling_size
