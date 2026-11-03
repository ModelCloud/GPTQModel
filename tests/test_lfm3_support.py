# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig

from gptqmodel.models import auto
from gptqmodel.models.definitions.lfm3 import LFM3QModel
from gptqmodel.models.definitions.lfm3_omni import LFM3OmniQModel
from gptqmodel.models.definitions.lfm3_vl import LFM3VLQModel


MODEL_PATHS = {
    "lfm3": Path("/monster/data/models/LFM3-1.7B"),
    "lfm3_vl": Path("/monster/data/models/LFM3-VL-2B"),
    "lfm3_omni": Path("/monster/data/models/LFM3-Omni-2B"),
}


@pytest.mark.parametrize(
    ("model_type", "definition"),
    [
        ("lfm3", LFM3QModel),
        ("lfm3_vl", LFM3VLQModel),
        ("lfm3_omni", LFM3OmniQModel),
    ],
)
def test_lfm3_model_type_selects_definition(monkeypatch, model_type, definition):
    fake_config = SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition(f"/tmp/{model_type}") is definition


@pytest.mark.parametrize(("model_type", "definition"), [("lfm3", LFM3QModel), ("lfm3_vl", LFM3VLQModel), ("lfm3_omni", LFM3OmniQModel)])
def test_lfm3_local_config_selects_definition(model_type, definition):
    path = MODEL_PATHS[model_type]
    if not path.exists():
        pytest.skip(f"{path} not found")

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    assert config.model_type == model_type
    assert auto.check_and_get_model_definition(path, trust_remote_code=True) is definition


def _flat_layer_modules(definition):
    layer_modules = definition.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    return {name for block in layer_modules for name in block}


@pytest.mark.parametrize("definition", [LFM3QModel, LFM3VLQModel, LFM3OmniQModel])
def test_lfm3_module_tree_covers_conv_attention_and_mlp_paths(definition):
    flat_modules = _flat_layer_modules(definition)

    assert definition.layer_modules_strict is False
    assert definition.require_trust_remote_code is True
    assert definition.rotary_embedding.endswith("rotary_emb")
    # `:!` marks a module that keeps its source dtype, so it is not quantized.
    assert flat_modules == {
        "conv.in_proj",
        "conv.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    }


def test_lfm3_text_layers_node_and_norms():
    assert LFM3QModel.extract_layers_node() == ["model.layers"]
    assert LFM3QModel.pre_lm_head_norm_module == "model.embedding_norm"


def test_lfm3_vl_layers_node_and_multimodal_modules():
    assert LFM3VLQModel.extract_layers_node() == ["model.language_model.layers"]
    assert LFM3VLQModel.pre_lm_head_norm_module == "model.language_model.embedding_norm"
    assert LFM3VLQModel.require_load_processor is True


def test_lfm3_omni_layers_node_and_multimodal_modules():
    assert LFM3OmniQModel.extract_layers_node() == ["model.language_model.layers"]
    assert LFM3OmniQModel.pre_lm_head_norm_module == "model.language_model.embedding_norm"
    assert LFM3OmniQModel.require_load_processor is True


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])
        self.embed_tokens = nn.Embedding(4, 4)
        self.rotary_emb = nn.Identity()
        self.embedding_norm = nn.LayerNorm(4)


class _VLCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision_tower = nn.Identity()
        self.multi_modal_projector = nn.Identity()


class _VLWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _VLCore()
        self.lm_head = nn.Linear(4, 4, bias=False)


class _OmniCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        for name in LFM3OmniQModel._NON_TEXT_MODULES:
            setattr(self, name, nn.Identity())


class _OmniWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _OmniCore()
        self.lm_head = nn.Linear(4, 4, bias=False)


def test_lfm3_vl_base_modules_include_vision_children():
    base_modules = set(LFM3VLQModel.get_base_modules(_VLWrapper()))

    assert "model.vision_tower" in base_modules
    assert "model.multi_modal_projector" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.rotary_emb" in base_modules
    assert "model.language_model.embedding_norm" in base_modules


def test_lfm3_omni_base_modules_include_audio_and_vision_children():
    base_modules = set(LFM3OmniQModel.get_base_modules(_OmniWrapper()))

    for name in LFM3OmniQModel._NON_TEXT_MODULES:
        assert f"model.{name}" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.embedding_norm" in base_modules


class _FakeRotary(nn.Module):
    def forward(self, x, position_ids, layer_type):
        cos = torch.zeros(*position_ids.shape, 4, dtype=x.dtype, device=x.device)
        return cos, cos


class _TextCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rotary_emb = _FakeRotary()


class _NestedTextCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = _TextCore(config)


class _TextWrapper(nn.Module):
    def __init__(self, config, nested: bool = False):
        super().__init__()
        self.model = _NestedTextCore(config) if nested else _TextCore(config)


def _shell_text_model(definition, text_config):
    wrapper = _TextWrapper(text_config, nested=definition.extract_layers_node() != ["model.layers"])
    # Bypass nn.Module.__setattr__: the definition instance is built with
    # `object.__new__` because it needs no real quantizer state here.
    shell = object.__new__(definition)
    object.__setattr__(shell, "model", wrapper)
    return shell


@pytest.mark.parametrize("model_type", ["lfm3", "lfm3_vl", "lfm3_omni"])
def test_lfm3_replay_kwargs_rebuild_mask_and_rotary_per_layer_type(model_type):
    path = MODEL_PATHS[model_type]
    if not path.exists():
        pytest.skip(f"{path} not found")

    root_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_config = root_config.get_text_config()
    text_config._attn_implementation = "eager"
    assert set(text_config.layer_types) >= {"conv", "full_attention", "sliding_attention"}

    definition = {"lfm3": LFM3QModel, "lfm3_vl": LFM3VLQModel, "lfm3_omni": LFM3OmniQModel}[model_type]
    model_def = _shell_text_model(definition, text_config)

    hidden_states = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    position_ids = torch.arange(4).unsqueeze(0)
    raw_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    inputs = {
        "__gptqmodel_lfm3_attention_mask": raw_mask,
        "__gptqmodel_lfm3_position_ids": position_ids,
    }

    conv_index = text_config.layer_types.index("conv")
    conv_layer = SimpleNamespace(self_attn=None, conv=SimpleNamespace(layer_idx=conv_index))
    conv_kwargs = model_def.prepare_layer_replay_kwargs(conv_layer, [hidden_states], dict(inputs), torch.device("cpu"))
    # Conv layers take no rotary state, only the 2D padding mask.
    assert "position_embeddings" not in conv_kwargs
    assert "position_ids" not in conv_kwargs
    assert torch.equal(conv_kwargs["attention_mask"], raw_mask)

    attention_layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=text_config.layer_types.index("full_attention")))
    attention_kwargs = model_def.prepare_layer_replay_kwargs(
        attention_layer, [hidden_states], dict(inputs), torch.device("cpu")
    )
    cos, sin = attention_kwargs["position_embeddings"]
    assert cos.shape == (*position_ids.shape, 4)
    assert attention_kwargs["position_ids"] is not None
    assert attention_kwargs["attention_mask"] is not None
