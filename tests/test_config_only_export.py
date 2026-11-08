# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import torch
from gptqmodel.models.writer import _save_model_configs_without_weights


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


def test_failed_shard_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import pytest
    from gptqmodel.utils import model as model_utils

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"previous checkpoint")
    entry = model_utils.TensorSource("weight", torch.float32, (1,), torch.ones(1))

    def fail(*args):
        raise OSError("injected write failure")

    monkeypatch.setattr(model_utils, "_write_tensor_bytes", fail)
    with pytest.raises(OSError, match="injected write failure"):
        model_utils._write_shard_file(str(path), [entry], {})
    assert path.read_bytes() == b"previous checkpoint"
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]
