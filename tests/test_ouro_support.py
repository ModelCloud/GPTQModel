# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
import types
from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.ouro import OuroQModel
from gptqmodel.utils.hf import normalize_hf_config_compat


def test_ouro_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="ouro")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
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

    assert OuroQModel.require_trust_remote_code is True
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


def test_ouro_remote_config_compat_restores_missing_pad_token_id():
    config = SimpleNamespace(
        model_type="ouro",
        auto_map={"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"},
        eos_token_id=0,
    )
    assert not hasattr(config, "pad_token_id")

    normalize_hf_config_compat(config, trust_remote_code=True)

    assert config.pad_token_id == config.eos_token_id


def test_ouro_before_model_load_patches_legacy_rotary_class(monkeypatch):
    class FakeOuroRotaryEmbedding:
        pass

    sentinel = ("inv_freq", 1.0)
    remote_module = types.ModuleType("fake_ouro_remote")

    def fake_causal_mask(**kwargs):
        return kwargs

    remote_module.create_causal_mask = fake_causal_mask
    remote_module.create_sliding_window_causal_mask = fake_causal_mask
    FakeOuroRotaryEmbedding.__module__ = remote_module.__name__
    monkeypatch.setitem(sys.modules, remote_module.__name__, remote_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_local_path: FakeOuroRotaryEmbedding,
    )
    monkeypatch.setitem(
        pytest.importorskip("transformers.modeling_rope_utils").ROPE_INIT_FUNCTIONS,
        "default",
        lambda config: sentinel,
    )

    OuroQModel.before_model_load(OuroQModel, "/monster/data/model/Ouro-1.4B", False)

    assert FakeOuroRotaryEmbedding().compute_default_rope_parameters(object()) == sentinel


def test_ouro_before_model_load_adapts_legacy_mask_keyword_locally(monkeypatch):
    class FakeOuroRotaryEmbedding:
        pass

    remote_module = types.ModuleType("fake_ouro_mask_remote")

    def fake_causal_mask(**kwargs):
        return kwargs

    remote_module.create_causal_mask = fake_causal_mask
    remote_module.create_sliding_window_causal_mask = fake_causal_mask
    FakeOuroRotaryEmbedding.__module__ = remote_module.__name__
    monkeypatch.setitem(sys.modules, remote_module.__name__, remote_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_local_path: FakeOuroRotaryEmbedding,
    )

    OuroQModel.before_model_load(OuroQModel, "/monster/data/model/Ouro-1.4B", False)

    for mask_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        mask = getattr(remote_module, mask_name)
        assert mask._gptqmodel_ouro_mask_compat is True
        assert mask(input_embeds="hidden", cache_position="positions") == {"inputs_embeds": "hidden"}
