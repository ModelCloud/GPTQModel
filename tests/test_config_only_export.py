# SPDX-License-Identifier: Apache-2.0
import json
import stat
from pathlib import Path

import pytest
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


def test_config_only_save_preserves_nested_source_fields(tmp_path):
    class Config:
        def save_pretrained(self, save_dir):
            payload = {
                "architectures": ["MergedModel"],
                "text_config": {"hidden_size": 1024, "num_layers": 4},
                "quantization_config": {"bits": 4},
            }
            (Path(save_dir) / "config.json").write_text(json.dumps(payload))

    class Model:
        config = Config()
        generation_config = None
        dtype = None

        @staticmethod
        def can_generate():
            return False

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SourceModel"],
                "text_config": {"hidden_size": 4096, "source_only_field": 7},
                "quantization_config": {"bits": 8, "source_only": True},
                "source_only_top": "keep",
            }
        )
    )

    _save_model_configs_without_weights(Model(), str(tmp_path), str(source))

    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["architectures"] == ["MergedModel"]
    assert saved_config["text_config"] == {
        "hidden_size": 1024,
        "num_layers": 4,
        "source_only_field": 7,
    }
    # Regenerated sections must never inherit stale source values.
    assert saved_config["quantization_config"] == {"bits": 4}
    assert saved_config["source_only_top"] == "keep"


def test_atomic_shard_write_preserves_existing_file_mode(tmp_path):
    from gptqmodel.utils import model as model_utils

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"previous checkpoint")
    path.chmod(0o640)
    entry = model_utils.TensorSource("weight", torch.float32, (1,), torch.ones(1))

    model_utils._write_shard_file(str(path), [entry], {})

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_new_atomic_shard_honors_umask_without_changing_it(tmp_path, monkeypatch):
    from gptqmodel.utils import model as model_utils

    reference = tmp_path / "ordinary-file"
    reference.write_bytes(b"ordinary file creation")
    path = tmp_path / "model.safetensors"
    entry = model_utils.TensorSource("weight", torch.float32, (1,), torch.ones(1))

    def reject_umask_change(*_args):
        raise AssertionError("shard export must not change the process umask")

    monkeypatch.setattr(model_utils.os, "umask", reject_umask_change)
    model_utils._write_shard_file(str(path), [entry], {})

    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


def test_atomic_shard_closes_descriptor_if_mode_restore_fails(tmp_path, monkeypatch):
    from gptqmodel.utils import model as model_utils

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"previous checkpoint")
    entry = model_utils.TensorSource("weight", torch.float32, (1,), torch.ones(1))
    descriptors = []

    original_open = model_utils.os.open

    def capture_open(*args):
        descriptor = original_open(*args)
        descriptors.append(descriptor)
        return descriptor

    def fail_mode_restore(_temporary, _mode):
        raise OSError("injected mode restore failure")

    monkeypatch.setattr(model_utils.os, "open", capture_open)
    monkeypatch.setattr(model_utils.os, "chmod", fail_mode_restore)
    with pytest.raises(OSError, match="injected mode restore failure"):
        model_utils._write_shard_file(str(path), [entry], {})

    assert path.read_bytes() == b"previous checkpoint"
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        model_utils.os.fstat(descriptors[0])


def test_config_only_save_preserves_source_fields_on_in_place_export(tmp_path):
    class Config:
        def save_pretrained(self, save_dir):
            (Path(save_dir) / "config.json").write_text(
                json.dumps({"architectures": ["LiveModel"], "dtype": "bfloat16"})
            )

    class Model:
        config = Config()
        generation_config = None
        dtype = None

        @staticmethod
        def can_generate():
            return False

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["SourceModel"], "num_hash_layers": 3})
    )

    _save_model_configs_without_weights(Model(), str(tmp_path), str(tmp_path))

    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["architectures"] == ["LiveModel"]
    assert saved_config["num_hash_layers"] == 3


PORTED_REMOTE_CODE_CASES = {
    "minicpmv": {
        "architectures": ["MiniCPMV"],
        "auto_map": {
            "AutoConfig": "configuration_minicpm.MiniCPMVConfig",
            "AutoModel": "modeling_minicpmv.MiniCPMV",
            "AutoModelForCausalLM": "modeling_minicpmv.MiniCPMV",
        },
        "runtime_class": "MiniCPMV",
    },
    "minicpmo": {
        "architectures": ["MiniCPMO"],
        "auto_map": {
            "AutoConfig": "configuration_minicpm.MiniCPMOConfig",
            "AutoModel": "modeling_minicpmo.MiniCPMO",
            "AutoModelForCausalLM": "modeling_minicpmo.MiniCPMO",
        },
        # The port dispatches version 2.6 to this package-internal subclass.
        "runtime_class": "MiniCPMO26",
    },
    "internvl_chat": {
        "architectures": ["InternVLChatModel"],
        "auto_map": {
            "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
            "AutoModel": "modeling_internvl_chat.InternVLChatModel",
            "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel",
        },
        "runtime_class": "InternVLChatModel",
    },
    "ovis2_5": {
        "architectures": ["Ovis2_5"],
        "auto_map": {
            "AutoConfig": "configuration_ovis2_5.Ovis2_5_Config",
            "AutoModelForCausalLM": "modeling_ovis2_5.Ovis2_5",
        },
        "runtime_class": "Ovis2_5",
    },
    "deepseek_vl_v2": {
        "architectures": None,
        "auto_map": {
            "AutoConfig": "configuration_deepseek_vl_v2.DeepseekVLV2Config",
            "AutoModelForCausalLM": "modeling_deepseek_vl_v2.DeepseekVLV2ForCausalLM",
        },
        "runtime_class": "DeepseekVLV2ForCausalLM",
    },
}


def _remote_config_module(config_class, model_type):
    return (
        "from transformers import PretrainedConfig\n\n"
        f"class {config_class}(PretrainedConfig):\n"
        f"    model_type = {model_type!r}\n"
    )


def _write_remote_code_source(source, model_type, architectures, auto_map):
    """Write the released metadata plus every checkpoint `.py` module as data."""

    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": architectures,
                "auto_map": auto_map,
            }
        )
    )
    config_module, _, config_class = auto_map["AutoConfig"].rpartition(".")
    for reference in auto_map.values():
        module_name = reference.rpartition(".")[0]
        path = source / f"{module_name}.py"
        if not path.exists():
            if module_name == config_module:
                path.write_text(_remote_config_module(config_class, model_type))
            else:
                # Modeling code is copied as data and never imported by the writer.
                path.write_text("# released checkpoint modeling code\n")


def test_ported_config_only_save_keeps_remote_code_metadata(tmp_path):
    """Only a natively registered model type may drop `auto_map` (issue #6134).

    MiniCPM-V/o, InternVL-Chat, Ovis2.5 and DeepSeek-VL2 are package-owned ports
    of released remote-code checkpoints. Their export must keep the source
    `auto_map` and checkpoint `.py` files so `AutoConfig` can still resolve the
    config class. MiniCPM-o's runtime class is the internal `MiniCPMO26`
    subclass; the saved architecture must stay the released `MiniCPMO`.
    """

    from transformers import AutoConfig

    for model_type, case in PORTED_REMOTE_CODE_CASES.items():
        source = tmp_path / f"{model_type}-source"
        out = tmp_path / f"{model_type}-out"
        _write_remote_code_source(
            source,
            model_type,
            case["architectures"],
            case["auto_map"],
        )

        class Config:
            def __init__(self):
                self.architectures = case["architectures"]
                self.dtype = None

            def save_pretrained(self, save_dir):
                (Path(save_dir) / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": model_type,
                            "architectures": self.architectures,
                            "dtype": self.dtype,
                        }
                    )
                )

        class Model:
            config = Config()
            dtype = None

        Model.__name__ = case["runtime_class"]
        # Ported runtime classes live in the package-owned ports tree; the
        # writer uses that to recognize a package-internal subclass name.
        Model.__module__ = "gptqmodel.models.ports.test_port"
        _save_model_configs_without_weights(Model(), str(out), str(source))

        saved = json.loads((out / "config.json").read_text())
        assert saved["auto_map"] == case["auto_map"], model_type
        if case["architectures"] is not None:
            assert saved["architectures"] == case["architectures"], model_type
        for reference in case["auto_map"].values():
            module_name = reference.rpartition(".")[0]
            assert (out / f"{module_name}.py").is_file(), (model_type, module_name)

        config_class = case["auto_map"]["AutoConfig"].rpartition(".")[2]
        loaded = AutoConfig.from_pretrained(str(out), trust_remote_code=True)
        assert type(loaded).__name__ == config_class, model_type
        assert loaded.model_type == model_type

    assert json.loads((tmp_path / "minicpmo-out" / "config.json").read_text())[
        "architectures"
    ] == ["MiniCPMO"]
