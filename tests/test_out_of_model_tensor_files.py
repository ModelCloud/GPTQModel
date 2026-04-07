# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import json
import os
from types import SimpleNamespace

import torch
from gptqmodel.utils.model import TensorSource
from safetensors import safe_open
from safetensors.torch import save_file

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


def _tensor_source(name: str, tensor: torch.Tensor) -> TensorSource:
    return TensorSource(name=name, torch_dtype=tensor.dtype, shape=tuple(tensor.shape), source=tensor)


def _build_writer_with_out_of_model_file(model_local_path, out_of_model_tensor_files=None):
    class _Base:
        pass

    _Base.out_of_model_tensors = out_of_model_tensor_files or []

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


def _patch_streaming(monkeypatch, shard_count=1):
    def _fake_streaming_state_dict_to_shards(state_dict, save_dir, model_base_name, single_file_name, metadata, *_args, **_kwargs):
        expected_files = []
        tensor_to_filename = {}
        for idx in range(shard_count):
            if shard_count == 1:
                shard_name = "model.safetensors"
            else:
                shard_name = f"{model_base_name}-{idx+1:05d}-of-{shard_count:05d}.safetensors"
            file_path = os.path.join(save_dir, shard_name)
            tensor_data = {
                name: ts.source if isinstance(ts, TensorSource) else ts
                for name, ts in state_dict.items()
            }
            save_file(tensor_data, file_path, metadata=metadata)
            expected_files.append(shard_name)
            for name in state_dict:
                tensor_to_filename.setdefault(name, shard_name)
        total_size = sum(os.path.getsize(os.path.join(save_dir, fname)) for fname in expected_files)
        return expected_files, tensor_to_filename, total_size

    monkeypatch.setattr(
        "gptqmodel.models.writer.streaming_state_dict_to_shards",
        _fake_streaming_state_dict_to_shards,
    )


def _patch_basic_env(monkeypatch, state_dict_tensor):
    monkeypatch.setattr("gptqmodel.models.writer.get_model_files_size", lambda _: 1)
    monkeypatch.setattr("gptqmodel.models.writer.alias_all_from_turtle_if_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr("gptqmodel.models.writer.sanitize_model_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("gptqmodel.models.writer.sanitize_generation_config_file", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "gptqmodel.models.writer.get_state_dict_for_save",
        lambda *_args, **_kwargs: state_dict_tensor,
    )


def test_merge_prefixed_tensors(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()

    shard_a_name = "model-00001-of-00002.safetensors"
    shard_b_name = "model-00002-of-00002.safetensors"

    save_file(
        {
            "base.weight": torch.zeros(1),
            "mtp.fc.weight": torch.ones(2),
        },
        str(original_dir / shard_a_name),
    )
    save_file(
        {
            "model.layers.0.weight": torch.full((1,), 2.0),
            "mtp.model.layers.0.weight": torch.full((3,), 3.0),
        },
        str(original_dir / shard_b_name),
    )

    with open(original_dir / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "mtp.fc.weight": shard_a_name,
                    "mtp.model.layers.0.weight": shard_b_name,
                },
            },
            handle,
        )

    writer = _build_writer_with_out_of_model_file(
        str(original_dir), out_of_model_tensor_files=[{"prefixes": ["mtp"]}]
    )
    state_dict_data = {"model.weight": _tensor_source("model.weight", torch.zeros(1))}

    _patch_basic_env(monkeypatch, state_dict_data)
    _patch_streaming(monkeypatch)

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir))

    assert not (save_dir / "mtp.safetensors").exists()

    with safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    assert {"mtp.fc.weight", "mtp.model.layers.0.weight"} <= keys


def test_merge_prefixed_tensors_with_multiple_shards(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()

    for shard_idx in range(2):
        shard_name = f"model-{shard_idx+1:05d}-of-00002.safetensors"
        save_file(
            {
                "model.weight": torch.zeros(1),
                "mtp.fc.weight": torch.ones(2),
            },
            str(original_dir / shard_name),
        )

    with open(original_dir / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "mtp.fc.weight": "model-00001-of-00002.safetensors",
                },
            },
            handle,
        )

    writer = _build_writer_with_out_of_model_file(
        str(original_dir), out_of_model_tensor_files=[{"prefixes": ["mtp"]}]
    )
    state_dict_data = {"model.weight": _tensor_source("model.weight", torch.zeros(1))}

    _patch_basic_env(monkeypatch, state_dict_data)
    _patch_streaming(monkeypatch, shard_count=2)

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir))

    assert (save_dir / "model-00001-of-00002.safetensors").exists()
    assert (save_dir / "model-00002-of-00002.safetensors").exists()
    assert (save_dir / "model.safetensors.index.json").exists()

    keys = []
    with safe_open(save_dir / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as handle:
        keys += handle.keys()
    with safe_open(save_dir / "model-00002-of-00002.safetensors", framework="pt", device="cpu") as handle:
        keys += handle.keys()
    assert {"mtp.fc.weight"} <= set(keys)

    with open(save_dir / "model.safetensors.index.json", "r", encoding="utf-8") as handle:
        index_data = json.load(handle)
    assert index_data["weight_map"]["mtp.fc.weight"] == "model-00001-of-00002.safetensors"


def test_copy_existing_file(tmp_path, monkeypatch):
    original_dir = tmp_path / "original"
    original_dir.mkdir()

    mtp_file = original_dir / "mtp.safetensors"
    save_file({"mtp.linear.weight": torch.ones(1)}, str(mtp_file))

    writer = _build_writer_with_out_of_model_file(
        str(original_dir), out_of_model_tensor_files=[{"files": ["mtp.safetensors"]}]
    )
    state_dict_data = {"model.weight": _tensor_source("model.weight", torch.zeros(1))}

    _patch_basic_env(monkeypatch, state_dict_data)
    _patch_streaming(monkeypatch)

    save_dir = tmp_path / "save"
    writer.save_quantized(save_dir=str(save_dir))

    with safe_open(save_dir / "mtp.safetensors", framework="pt", device="cpu") as handle:
        mtp_keys = set(handle.keys())
    assert mtp_keys == {"mtp.linear.weight"}

    with safe_open(save_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        mtp_keys = set(handle.keys())
    assert mtp_keys == {"model.weight"}
