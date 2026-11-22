# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #4385:
Prevent unneeded CPU offload disk staging and TMPDIR leaks during model loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

from gptqmodel.models import loader as loader_module
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization.config import FORMAT, METHOD


class _StopLoad(Exception):
    """Sentinel exception to abort loader flow after checkpoint load call."""


def _setup_mock_checkpoint(tmp_path: Path, format_code: str = FORMAT.GPTQ) -> Path:
    checkpoint_dir = tmp_path / "mock_ckpt"
    checkpoint_dir.mkdir(exist_ok=True)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 32,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    quant_config = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "format": format_code,
        "method": METHOD.GPTQ,
    }
    (checkpoint_dir / "quantize_config.json").write_text(json.dumps(quant_config), encoding="utf-8")
    return checkpoint_dir


def test_loader_does_not_pass_offload_state_dict_by_default(tmp_path, monkeypatch):
    """Issue #4385: verify that GPTQ v1 loading does not force offload_state_dict=True."""
    ckpt_dir = _setup_mock_checkpoint(tmp_path, FORMAT.GPTQ)
    captured_kwargs = {}

    def mock_load_checkpoint(model, *args, **kwargs):
        captured_kwargs.update(kwargs)
        raise _StopLoad()

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(16, 16)])
            self.lm_head = nn.Linear(16, 16)
            self.config = SimpleNamespace(to_dict=lambda: {"max_position_embeddings": 2048})

        def get_input_embeddings(self):
            return None

        def get_output_embeddings(self):
            return None

    monkeypatch.setattr(loader_module, "load_checkpoint_in_model_then_tie_weights", mock_load_checkpoint)
    monkeypatch.setattr(loader_module, "get_model_local_path", lambda path, **kw: str(path))
    monkeypatch.setattr(loader_module, "load_hf_tokenizer", lambda *args, **kw: None)
    monkeypatch.setattr(loader_module, "get_checkpoints", lambda *args, **kw: (False, str(ckpt_dir / "model.safetensors"), "model"))
    monkeypatch.setattr(loader_module, "make_quant", lambda *args, **kw: None)
    monkeypatch.setattr(loader_module, "validate_checkpoint_qweights", lambda *args, **kw: None)
    monkeypatch.setattr(LlamaQModel.loader, "from_config", lambda cfg, **kw: _MockModel())

    loader_cls = loader_module.ModelLoader(LlamaQModel)

    with pytest.raises(_StopLoad):
        loader_cls.from_quantized(
            str(ckpt_dir),
            device="cpu",
        )

    # offload_state_dict and offload_buffers must NOT be True by default
    assert captured_kwargs.get("offload_state_dict") is not True, (
        f"Expected offload_state_dict not to be True by default, got: {captured_kwargs.get('offload_state_dict')}"
    )
    assert captured_kwargs.get("offload_buffers") is not True, (
        f"Expected offload_buffers not to be True by default, got: {captured_kwargs.get('offload_buffers')}"
    )


def test_loader_forwards_explicit_offload_kwargs(tmp_path, monkeypatch):
    """Issue #4385: verify explicit offload_state_dict and offload_folder kwargs are forwarded."""
    ckpt_dir = _setup_mock_checkpoint(tmp_path, FORMAT.GPTQ)
    captured_kwargs = {}

    def mock_load_checkpoint(model, *args, **kwargs):
        captured_kwargs.update(kwargs)
        raise _StopLoad()

    class _MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(16, 16)])
            self.lm_head = nn.Linear(16, 16)
            self.config = SimpleNamespace(to_dict=lambda: {"max_position_embeddings": 2048})

        def get_input_embeddings(self):
            return None

        def get_output_embeddings(self):
            return None

    monkeypatch.setattr(loader_module, "load_checkpoint_in_model_then_tie_weights", mock_load_checkpoint)
    monkeypatch.setattr(loader_module, "get_model_local_path", lambda path, **kw: str(path))
    monkeypatch.setattr(loader_module, "load_hf_tokenizer", lambda *args, **kw: None)
    monkeypatch.setattr(loader_module, "get_checkpoints", lambda *args, **kw: (False, str(ckpt_dir / "model.safetensors"), "model"))
    monkeypatch.setattr(loader_module, "make_quant", lambda *args, **kw: None)
    monkeypatch.setattr(loader_module, "validate_checkpoint_qweights", lambda *args, **kw: None)
    monkeypatch.setattr(LlamaQModel.loader, "from_config", lambda cfg, **kw: _MockModel())

    loader_cls = loader_module.ModelLoader(LlamaQModel)

    custom_offload_dir = str(tmp_path / "custom_offload")
    with pytest.raises(_StopLoad):
        loader_cls.from_quantized(
            str(ckpt_dir),
            device="cpu",
            offload_state_dict=True,
            offload_buffers=True,
            offload_folder=custom_offload_dir,
        )

    assert captured_kwargs.get("offload_state_dict") is True
    assert captured_kwargs.get("offload_buffers") is True
    assert captured_kwargs.get("offload_folder") == custom_offload_dir
