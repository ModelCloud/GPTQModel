# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.ouro import OuroQModel
from gptqmodel.models.ports.ouro.configuration_ouro import OuroConfig
from gptqmodel.utils.hf import normalize_hf_config_compat


def test_ouro_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="ouro")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto,
        "load_model_config",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition(
        "/monster/data/model/Ouro-1.4B", trust_remote_code=True
    ) is OuroQModel


def test_ouro_module_tree_matches_decoder_forward_boundaries():
    layer_modules = OuroQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert OuroQModel.require_trust_remote_code is False
    assert OuroQModel.extract_layers_node() == ["model.layers"]
    assert OuroQModel.pre_lm_head_norm_module == "model.norm"
    assert OuroQModel.rotary_embedding == "model.rotary_emb"
    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    full_modules = {module for block in OuroQModel.full_layer_modules() for module in block}
    assert {
        "input_layernorm:!",
        "input_layernorm_2:!",
        "post_attention_layernorm:!",
        "post_attention_layernorm_2:!",
    } <= full_modules
    assert "early_exit_gate" not in full_modules


def test_ouro_owned_config_does_not_receive_remote_compat_rewrites():
    config = SimpleNamespace(
        model_type="ouro",
        auto_map={"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"},
        eos_token_id=0,
        pad_token_id=None,
    )

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.pad_token_id is None


def test_ouro_declared_loader_constructs_owned_config_without_remote_class_lookup(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Ouro must not request checkpoint Python")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", denied)
    OuroConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
               num_attention_heads=4, num_key_value_heads=2, vocab_size=64).save_pretrained(tmp_path)
    config = OuroQModel.loader.load_config(str(tmp_path), trust_remote_code=True)
    assert type(config) is OuroConfig
    assert config.total_ut_steps == 4
    assert "before_model_load" not in OuroQModel.__dict__
    assert OuroQModel.awq_get_modules_for_scaling(None, None, None, None) == []
    assert OuroQModel.fusable is None
