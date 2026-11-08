# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_TIME,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_LOSS_KIND,
    QUANT_LOG_NSAMPLES,
    ModelWriter,
    _quantization_config_for_model_config,
    _save_model_configs_without_weights,
)
from gptqmodel.quantization.config import EXL3Config, FORMAT, METHOD
from gptqmodel.utils.exl3_error_ledger import (
    LEDGER_FILENAME,
    LEDGER_MANIFEST_FILENAME,
    build_projection_record,
)


class _DummyKernel:
    REQUIRES_FORMAT_V2 = False
    SUPPORTS_SHARDS = True


class _DummyQuantizeConfig:
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

    def stop_after_checks(model, _save_dir, _source_model_dir):
        tracker["config_snapshot"] = dict(model.config.__dict__)
        tracker["generation_snapshot"] = dict(model.generation_config.__dict__)
        raise RuntimeError("stop after checks")

    monkeypatch.setattr(
        "gptqmodel.models.writer._save_model_configs_without_weights",
        stop_after_checks,
    )

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


def test_save_quantized_persists_exl3_ledger_and_unambiguous_csv(tmp_path, monkeypatch):
    tracker = {}
    writer = _build_dummy_model_writer()
    writer.model = _DummyModel(tracker)
    ledger_record = build_projection_record(
        module_full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
        bits=2,
        codebook="mcg",
        sample_count=1024,
        duration_seconds=1.0,
        encoded_bytes=128,
        device_names=["cuda:0"],
        quantizer_metrics={
            "reported_metric_kind": "hessian_weighted_relative_error",
            "reconstruction": {
                "error_sum_sq": 1.0,
                "reference_sum_sq": 10.0,
                "element_count": 256,
                "max_abs_error": 0.5,
            },
        },
        provenance={"family_join": {"source_revision": "abc"}},
    )
    writer.quant_log = [
        {
            PROCESS_LOG_LAYER: 0,
            PROCESS_LOG_MODULE: "gate_proj",
            QUANT_LOG_LOSS: "0.1000000000",
            QUANT_LOG_LOSS_KIND: "hessian_weighted_relative_error",
            QUANT_LOG_NSAMPLES: "1024",
            QUANT_LOG_DAMP: "0.02500",
            PROCESS_LOG_TIME: "1.000",
            "exl3_error_ledger_record": ledger_record,
        }
    ]
    monkeypatch.setattr("gptqmodel.models.writer.get_model_files_size", lambda _: 1)

    def stop_after_checks(model, _save_dir, _source_model_dir):
        tracker["config_snapshot"] = dict(model.config.__dict__)
        tracker["generation_snapshot"] = dict(model.generation_config.__dict__)
        raise RuntimeError("stop after checks")

    monkeypatch.setattr(
        "gptqmodel.models.writer._save_model_configs_without_weights",
        stop_after_checks,
    )

    with pytest.raises(RuntimeError, match="stop after checks"):
        writer.save_quantized(save_dir=str(tmp_path))

    manifest = json.loads((tmp_path / LEDGER_MANIFEST_FILENAME).read_text())
    assert manifest["projection_records"] == 1
    assert manifest["complete_family_records"] == 0
    assert (tmp_path / LEDGER_FILENAME).is_file()
    with (tmp_path / "quant_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            PROCESS_LOG_LAYER: "0",
            PROCESS_LOG_MODULE: "gate_proj",
            QUANT_LOG_LOSS: "0.1000000000",
            QUANT_LOG_LOSS_KIND: "hessian_weighted_relative_error",
            QUANT_LOG_NSAMPLES: "1024",
            QUANT_LOG_DAMP: "0.02500",
            PROCESS_LOG_TIME: "1.000",
        }
    ]


def test_exl3_config_payload_can_omit_external_tensor_storage():
    quantize_config = EXL3Config(
        bits=3,
        codebook="mcg",
        tensor_storage={
            "model.layers.0.proj": {
                "quant_format": "exl3",
                "bits_per_weight": 3,
                "stored_tensors": {},
            }
        },
    )

    embedded = _quantization_config_for_model_config(
        quantize_config,
        FORMAT.EXL3,
    )

    assert embedded["quant_method"] == METHOD.EXL3
    assert embedded["bits"] == 3
    assert embedded["codebook"] == "mcg"
    assert "tensor_storage" not in embedded
    assert "tensor_storage" in quantize_config.to_dict()


def test_config_only_save_preserves_existing_weight_shards(tmp_path):
    class Config:
        def save_pretrained(self, save_dir):
            payload = {
                "architectures": self.architectures,
                "dtype": self.dtype,
            }
            (Path(save_dir) / "config.json").write_text(json.dumps(payload))

    class GenerationConfig:
        def save_pretrained(self, save_dir):
            (Path(save_dir) / "generation_config.json").write_text(
                json.dumps({"do_sample": False})
            )

    class ConfigOnlyModel:
        config = Config()
        generation_config = GenerationConfig()
        dtype = torch.bfloat16

        @staticmethod
        def can_generate():
            return True

        def save_pretrained(self, *_args, **_kwargs):
            raise AssertionError("config-only save must not call model.save_pretrained")

    shard = tmp_path / "model-00011-of-00013.safetensors"
    shard.write_bytes(b"authenticated-existing-shard")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SourceModel"],
                "dtype": "float16",
                "num_hash_layers": 3,
                "attn_implementation": "flash_attention_2",
            }
        )
    )

    _save_model_configs_without_weights(
        ConfigOnlyModel(),
        str(tmp_path),
        str(source),
    )

    assert shard.read_bytes() == b"authenticated-existing-shard"
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config == {
        "architectures": ["ConfigOnlyModel"],
        "dtype": "bfloat16",
        "num_hash_layers": 3,
    }
    assert json.loads((tmp_path / "generation_config.json").read_text()) == {
        "do_sample": False
    }
