# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import defuser
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe import HunYuanMoEV1Config

from gptqmodel.models import auto
from gptqmodel.models.definitions.hunyuan_v1_dense import HunYuanDenseV1QModel
from gptqmodel.models.definitions.hunyuan_v1_moe import HunYuanMoEV1QModel


def test_hunyuan_v1_dense_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="hunyuan_v1_dense")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/hunyuan_v1_dense") is HunYuanDenseV1QModel


def test_hunyuan_v1_dense_module_tree_skips_qk_norms():
    attn_modules = HunYuanDenseV1QModel.module_tree[-1]["self_attn"]

    assert "q_proj:0" in attn_modules
    assert "k_proj:0" in attn_modules
    assert "v_proj:0" in attn_modules
    assert "o_proj:1" in attn_modules
    assert "query_layernorm:!" in attn_modules
    assert "key_layernorm:!" in attn_modules


def test_hunyuan_v1_moe_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="hunyuan_v1_moe")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/hunyuan_v1_moe") is HunYuanMoEV1QModel


def test_hunyuan_v1_moe_module_tree_matches_defused_experts():
    cfg = HunYuanMoEV1Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        moe_topk=2,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(cfg)

    assert defuser.convert_model(model, cleanup_original=False) is True

    layer = model.model.layers[0]
    expert = layer.mlp.experts[0]

    assert hasattr(layer.self_attn, "query_layernorm")
    assert hasattr(layer.self_attn, "key_layernorm")
    assert hasattr(layer.mlp, "shared_mlp")
    assert hasattr(expert, "gate_proj")
    assert hasattr(expert, "up_proj")
    assert hasattr(expert, "down_proj")

    attn_modules = HunYuanMoEV1QModel.module_tree[-1]["self_attn"]
    mlp_tree = HunYuanMoEV1QModel.module_tree[-1]["mlp:moe:?"]
    layer_modules = HunYuanMoEV1QModel.simple_layer_modules(
        model_config=cfg,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert "query_layernorm:!" in attn_modules
    assert "key_layernorm:!" in attn_modules
    assert "shared_mlp" in mlp_tree
    assert "experts:0" in mlp_tree
    assert ["mlp.shared_mlp.gate_proj", "mlp.shared_mlp.up_proj"] in layer_modules
    assert ["mlp.shared_mlp.down_proj"] in layer_modules
    assert any("mlp.experts.0.gate_proj" in block for block in layer_modules)
    assert any("mlp.experts.0.down_proj" in block for block in layer_modules)
