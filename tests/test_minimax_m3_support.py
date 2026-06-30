# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText

from gptqmodel.models import auto
from gptqmodel.models.definitions.minimax_m3_vl import MiniMaxM3VLGPTQ


MODEL_PATH = Path("/monster/data/model/MiniMax-M3")


def test_minimax_m3_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="minimax_m3_vl")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/minimax-m3") is MiniMaxM3VLGPTQ


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MiniMax-M3 model not found")
def test_minimax_m3_local_config_selects_definition():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)

    assert config.model_type == "minimax_m3_vl"
    assert auto.check_and_get_model_definition(MODEL_PATH) is MiniMaxM3VLGPTQ


def test_minimax_m3_module_tree_covers_dense_sparse_and_indexer_paths():
    layer_modules = MiniMaxM3VLGPTQ.simple_layer_modules(
        model_config=SimpleNamespace(text_config=SimpleNamespace(num_local_experts=2)),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert MiniMaxM3VLGPTQ.layer_modules_strict is False
    assert MiniMaxM3VLGPTQ.require_load_processor is False
    assert MiniMaxM3VLGPTQ.pre_lm_head_norm_module == "model.language_model.norm"
    assert MiniMaxM3VLGPTQ.rotary_embedding == "model.language_model.rotary_emb"

    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.indexer.q_proj" in flat_modules
    assert "self_attn.indexer.k_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules
    assert not any("block_sparse_moe" in name for name in flat_modules)


def test_minimax_m3_multimodal_base_modules_include_non_language_children():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])
            self.embed_tokens = nn.Embedding(4, 4)
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _MiniMaxM3Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _LanguageModel()
            self.vision_tower = nn.Identity()
            self.multi_modal_projector = nn.Identity()

    class _MiniMaxM3Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MiniMaxM3Core()
            self.lm_head = nn.Linear(4, 4, bias=False)

    base_modules = set(MiniMaxM3VLGPTQ.get_base_modules(_MiniMaxM3Wrapper()))

    assert MiniMaxM3VLGPTQ.extract_layers_node() == ["model.language_model.layers"]
    assert "model.vision_tower" in base_modules
    assert "model.multi_modal_projector" in base_modules
    assert "model.language_model.embed_tokens" in base_modules
    assert "model.language_model.norm" in base_modules
    assert "model.language_model.rotary_emb" in base_modules


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="MiniMax-M3 model not found")
def test_minimax_m3_defuser_splits_native_packed_experts():
    from defuser import convert_model
    from defuser.model_registry import MODEL_CONFIG

    config = _tiny_minimax_m3_config()
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

    experts = model.model.language_model.layers[3].mlp.experts
    assert "minimax_m3_vl" in MODEL_CONFIG
    assert hasattr(experts, "gate_up_proj")
    assert hasattr(experts, "down_proj")

    assert convert_model(model, cleanup_original=False) is True

    experts = model.model.language_model.layers[3].mlp.experts
    expert0 = getattr(experts, "0")
    assert not hasattr(experts, "gate_up_proj")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")


def _tiny_minimax_m3_config():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
    text_config = config.text_config
    text_config.hidden_size = 16
    text_config.intermediate_size = 8
    text_config.dense_intermediate_size = 32
    text_config.shared_intermediate_size = 8
    text_config.num_hidden_layers = 4
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 8
    text_config.num_local_experts = 2
    text_config.num_experts_per_tok = 1
    text_config.n_shared_experts = 1
    text_config.moe_layer_freq = [0, 0, 0, 1]
    text_config.vocab_size = 128
    config.vocab_size = 128

    vision_config = config.vision_config
    vision_config.hidden_size = 8
    vision_config.intermediate_size = 16
    vision_config.num_hidden_layers = 1
    vision_config.num_attention_heads = 2
    vision_config.image_size = 16
    vision_config.patch_size = 8
    if hasattr(vision_config, "spatial_merge_size"):
        vision_config.spatial_merge_size = 1

    return config
