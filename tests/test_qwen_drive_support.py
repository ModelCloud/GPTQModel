# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn
from transformers import AutoModel

from gptqmodel.models import auto
from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils import hf


def test_qwen_drive_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="qwen_drive")
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/qwen-drive") is QwenDriveQModel


def test_qwen_drive_registration_imports_official_package(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen-drive"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen_drive"}', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "qwen_drive", ModuleType("qwen_drive"))

    assert hf.ensure_qwen_drive_registered(str(model_dir)) is True


def test_qwen_drive_registration_error_points_to_official_source(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen-drive"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen_drive"}', encoding="utf-8")
    monkeypatch.delitem(sys.modules, "qwen_drive", raising=False)
    original_import = builtins.__import__

    def fail_qwen_drive_import(name, *args, **kwargs):
        if name == "qwen_drive":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qwen_drive_import)

    with pytest.raises(ImportError, match="QwenLM/Qwen-Drive-1.0"):
        hf.ensure_qwen_drive_registered(str(model_dir))


def test_qwen_drive_registration_ignores_other_model_types(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")

    def unexpected_import(name, *args, **kwargs):
        if name == "qwen_drive":
            raise AssertionError("qwen_drive should not be imported for another model type")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", unexpected_import)

    assert hf.ensure_qwen_drive_registered(str(model_dir)) is False


def test_qwen_drive_definition_contract_and_module_tree():
    assert QwenDriveQModel.loader is AutoModel
    assert QwenDriveQModel.require_trust_remote_code is False
    assert QwenDriveQModel.require_load_processor is False
    assert QwenDriveQModel.lm_head == "vlm.lm_head"
    assert QwenDriveQModel.pre_lm_head_norm_module == "vlm.model.language_model.norm"
    assert QwenDriveQModel.rotary_embedding == "vlm.model.language_model.rotary_emb"
    assert QwenDriveQModel.extract_layers_node() == ["vlm.model.language_model.layers"]

    blocks = QwenDriveQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=QuantizeConfig(),
    )
    flat = {name for block in blocks for name in block}
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= flat
    assert {
        "self_attn.q_norm",
        "self_attn.k_norm",
        "linear_attn.norm",
        "linear_attn.conv1d",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
    }.isdisjoint(flat)


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.forward_kwargs = None
        self.generate_kwargs = None

    def forward(self, *args, **kwargs):
        self.forward_kwargs = (args, kwargs)
        return "forwarded"

    def generate(self, *args, **kwargs):
        self.generate_kwargs = (args, kwargs)
        return "generated"


class _FakeQwenDrive(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _FakeVLM()
        self.planning_expert = nn.Linear(2, 2, bias=False)


def _definition_for(model):
    definition = QwenDriveQModel.__new__(QwenDriveQModel)
    nn.Module.__init__(definition)
    definition.model = model
    return definition


def test_qwen_drive_text_calls_delegate_to_vlm():
    wrapped = _FakeQwenDrive()
    definition = _definition_for(wrapped)

    assert definition.forward(input_ids="ids") == "forwarded"
    assert wrapped.vlm.forward_kwargs == ((), {"input_ids": "ids"})
    assert definition.generate(input_ids="ids") == "generated"
    assert wrapped.vlm.generate_kwargs == ((), {"input_ids": "ids"})


def test_qwen_drive_discards_unreleased_random_planner_state():
    wrapped = _FakeQwenDrive()
    definition = _definition_for(wrapped)

    definition.after_model_load(wrapped)

    assert "planning_expert" not in wrapped._modules
    assert wrapped(input_ids="ids") == "forwarded"
    assert wrapped.vlm.forward_kwargs == ((), {"input_ids": "ids"})
    assert all(not name.startswith("planning_expert.") for name in wrapped.state_dict())
    assert any(name.startswith("vlm.") for name in wrapped.state_dict())
