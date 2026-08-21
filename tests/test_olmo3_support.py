# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.olmo3 import Olmo3QModel, _prepare_olmo3_replay_kwargs


def _tiny_olmo3_model():
    config = Olmo3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "linear",
                "factor": 2.0,
                "rope_theta": 10_000.0,
            },
        },
    )
    return Olmo3ForCausalLM(config)


def test_olmo3_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="olmo3")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert auto.check_and_get_model_definition("/tmp/olmo3") is Olmo3QModel


def test_olmo3_quantization_groups_match_forward_boundaries():
    layer_modules = Olmo3QModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    role_flags = {}
    for module_spec in Olmo3QModel.module_tree[-1]["self_attn"]:
        name, flags = Olmo3QModel._parse_module_flags(module_spec)
        role_flags[name] = frozenset(
            flag for flag in flags if not flag.isdigit() and flag not in {"!", "?"}
        )

    assert role_flags["q_proj"] == frozenset({"q"})
    assert role_flags["k_proj"] == frozenset({"k"})
    assert role_flags["v_proj"] == frozenset({"v"})


def test_olmo3_transformers_runtime_matches_explicit_module_tree():
    model = _tiny_olmo3_model()
    layer_modules = set(dict(model.model.layers[0].named_modules()))

    assert Olmo3QModel.extract_layers_node() == ["model.layers"]
    assert Olmo3QModel.pre_lm_head_norm_module == "model.norm"
    assert Olmo3QModel.rotary_embedding == "model.rotary_emb"
    assert "input_layernorm" not in layer_modules
    assert {
        "self_attn.q_proj",
        "self_attn.q_norm",
        "self_attn.k_proj",
        "self_attn.k_norm",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "post_feedforward_layernorm",
    } <= layer_modules


def test_olmo3_replay_refreshes_rope_for_each_attention_type():
    model = _tiny_olmo3_model()
    model_def = SimpleNamespace(
        model=model, rotary_embedding=Olmo3QModel.rotary_embedding
    )
    hidden_states = torch.zeros(1, 8, model.config.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)

    sliding_kwargs = _prepare_olmo3_replay_kwargs(
        model_def,
        model.model.layers[0],
        [hidden_states],
        {"position_ids": position_ids},
        torch.device("cpu"),
    )
    full_kwargs = _prepare_olmo3_replay_kwargs(
        model_def,
        model.model.layers[3],
        [hidden_states],
        {"position_ids": position_ids},
        torch.device("cpu"),
    )

    expected_full = model.model.rotary_emb(
        hidden_states, position_ids, "full_attention"
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(full_kwargs["position_embeddings"], expected_full)
    )
    assert not torch.equal(
        sliding_kwargs["position_embeddings"][0], full_kwargs["position_embeddings"][0]
    )
