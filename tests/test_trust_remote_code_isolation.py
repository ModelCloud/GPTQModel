# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``trust_remote_code=False`` must never execute checkpoint Python.

Issue #3968: load, save and dequantize imported ``auto_map`` classes from the
snapshot even when the caller left trust off. Each checkpoint below carries a
native ``model_type`` (so it loads as a built-in config and a native model) plus
an ``auto_map`` whose modules write a marker file at import time.

The entry points are stopped by a sentinel just after the import boundary. That
keeps the test focused on the security contract: the marker is written only if
the checkpoint module actually ran.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from gptqmodel.models import loader as loader_module
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils import model_dequant


_MARKER_NAME = "EXECUTED"

_EVIL_CONFIG = """
from pathlib import Path

from transformers import PretrainedConfig

Path({marker!r}).write_text("auto_map AutoConfig executed", encoding="utf-8")


class EvilConfig(PretrainedConfig):
    model_type = "llama"
"""

_EVIL_MODEL = """
from pathlib import Path

Path({marker!r}).write_text("auto_map AutoModelForCausalLM executed", encoding="utf-8")


class EvilModel:
    pass
"""


class _Stop(Exception):
    """Abort an entry point once it has crossed the import boundary."""


def _write_checkpoint(tmp_path: Path, *, compressed_tensors: bool = False):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    marker = checkpoint / _MARKER_NAME

    (checkpoint / "evil_config.py").write_text(
        _EVIL_CONFIG.format(marker=str(marker)), encoding="utf-8"
    )
    (checkpoint / "evil_modeling.py").write_text(
        _EVIL_MODEL.format(marker=str(marker)), encoding="utf-8"
    )

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "auto_map": {
            "AutoConfig": "evil_config.EvilConfig",
            "AutoModelForCausalLM": "evil_modeling.EvilModel",
        },
    }
    if compressed_tensors:
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
        }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {"model.embed_tokens.weight": torch.zeros(8, 8, dtype=torch.float16)},
        str(checkpoint / "model.safetensors"),
    )
    return checkpoint, marker


def test_load_does_not_import_auto_map_without_trust(tmp_path, monkeypatch):
    checkpoint, marker = _write_checkpoint(tmp_path)

    def stop(*_args, **_kwargs):
        raise _Stop

    # auto_dtype() runs immediately after the remote-code compat stage.
    monkeypatch.setattr(loader_module, "auto_dtype", stop)

    with pytest.raises(_Stop):
        BaseQModel.from_quantized(str(checkpoint), trust_remote_code=False, device="cpu")

    assert not marker.exists(), "load imported checkpoint code with trust_remote_code=False"


def test_save_does_not_import_auto_map_config_without_trust(tmp_path):
    checkpoint, marker = _write_checkpoint(tmp_path)

    class _Loader:
        def from_config(self, *_args, **_kwargs):
            raise _Stop

    # get_model_with_quantize() only needs the trust flag before from_config().
    owner = SimpleNamespace(trust_remote_code=False, loader=_Loader())

    with pytest.raises(_Stop):
        BaseQModel.get_model_with_quantize(
            owner, qcfg=None, model_id_or_path=str(checkpoint)
        )

    assert not marker.exists(), "save imported checkpoint code with trust_remote_code=False"


def _fake_compressed_tensors_deps():
    class _QuantizationConfig:
        @classmethod
        def model_validate(cls, payload):
            return SimpleNamespace(format=payload.get("format"))

    class _BaseCompressor:
        @classmethod
        def load_from_registry(cls, *_args, **_kwargs):
            raise _Stop

    return {
        "QuantizationConfig": _QuantizationConfig,
        "BaseCompressor": _BaseCompressor,
        "apply_quantization_config": lambda *_args, **_kwargs: None,
        "map_module_to_scheme": lambda _model: {},
        "init_empty_weights": lambda **_kwargs: nullcontext(),
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    }


def test_dequantize_does_not_import_auto_map_config_without_trust(
    tmp_path, monkeypatch
):
    checkpoint, marker = _write_checkpoint(tmp_path, compressed_tensors=True)
    monkeypatch.setattr(
        model_dequant,
        "_get_compressed_tensors_dependencies",
        _fake_compressed_tensors_deps,
    )

    with pytest.raises(_Stop):
        model_dequant.dequantize_model(checkpoint, tmp_path / "out")

    assert not marker.exists(), "dequantize imported checkpoint code with trust_remote_code=False"


def test_remote_model_init_opt_in_still_imports(tmp_path):
    checkpoint, marker = _write_checkpoint(tmp_path)
    config = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=False)

    from gptqmodel.utils.hf import prepare_remote_model_init_compat

    prepare_remote_model_init_compat(
        str(checkpoint), config, trust_remote_code=True
    )

    assert marker.exists(), "trust_remote_code=True must keep importing auto_map classes"


def test_dequantize_opt_in_still_imports(tmp_path, monkeypatch):
    checkpoint, marker = _write_checkpoint(tmp_path, compressed_tensors=True)
    monkeypatch.setattr(
        model_dequant,
        "_get_compressed_tensors_dependencies",
        _fake_compressed_tensors_deps,
    )

    with pytest.raises((_Stop, RuntimeError, ValueError)):
        model_dequant.dequantize_model(
            checkpoint, tmp_path / "out", trust_remote_code=True
        )

    assert marker.exists(), "trust_remote_code=True must keep importing auto_map classes"
