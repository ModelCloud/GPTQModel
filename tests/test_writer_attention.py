# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest

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

    def save_pretrained(self, _):  # pragma: no cover - not exercised in this test
        return None

    def extract_adapter_rank_patterns(self):  # pragma: no cover - not exercised here
        return {}


class _DummyConfig:
    def __init__(self):
        self.attn_implementation = "flash_attention_2"
        self._attn_implementation = "flash_attention_2"

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        clone.__dict__ = copy.deepcopy(self.__dict__, memo)
        return clone


class _DummyGenerationConfig(_DummyConfig):
    pass


class _DummyModel:
    def __init__(self, tracker):
        self.config = _DummyConfig()
        self.generation_config = _DummyGenerationConfig()
        self._tracker = tracker

    def save_pretrained(self, *_args, **_kwargs):
        self._tracker["config_snapshot"] = dict(self.config.__dict__)
        self._tracker["generation_snapshot"] = dict(self.generation_config.__dict__)
        raise RuntimeError("stop after checks")


def _build_dummy_model_writer():
    class _Base:
        pass

    DummyWriter = ModelWriter(_Base)
    instance = DummyWriter()
    instance.quantized = True
    instance.quantize_config = _DummyQuantizeConfig()
    instance.quant_log = []
    instance.load_quantized_model = False
    instance.qlinear_kernel = _DummyKernel()
    instance.model_local_path = "/tmp/nonexistent"
    instance.trust_remote_code = False
    instance.tokenizer = None
    instance.processor = None
    instance.turtle_model = SimpleNamespace()
    instance.lm_head = "lm_head"
    return instance


def test_save_quantized_strips_attention_before_serialization(tmp_path, monkeypatch):
    tracker = {}
    writer = _build_dummy_model_writer()
    writer.model = _DummyModel(tracker)

    monkeypatch.setattr("gptqmodel.models.writer.get_model_files_size", lambda _: 1)

    with pytest.raises(RuntimeError, match="stop after checks"):
        writer.save_quantized(save_dir=str(tmp_path))

    config_snapshot = tracker["config_snapshot"]
    generation_snapshot = tracker["generation_snapshot"]

    assert "attn_implementation" not in config_snapshot
    assert "_attn_implementation" not in config_snapshot
    assert "attn_implementation" not in generation_snapshot
    assert "_attn_implementation" not in generation_snapshot

    assert writer.model.config.attn_implementation == "flash_attention_2"
    assert writer.model.config._attn_implementation == "flash_attention_2"
