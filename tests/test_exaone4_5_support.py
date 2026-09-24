# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.models import auto
from gptqmodel.models.definitions.exaone4 import Exaone4QModel, Exaone4_5QModel


def test_exaone4_5_model_types_select_expected_definitions(monkeypatch):
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )

    for model_type, expected in (
        ("exaone4_5", Exaone4_5QModel),
        ("exaone4_5_text", Exaone4QModel),
    ):
        monkeypatch.setattr(
            auto.AutoConfig,
            "from_pretrained",
            lambda *args, model_type=model_type, **kwargs: SimpleNamespace(model_type=model_type),
        )
        assert auto.check_and_get_model_definition("exaone4-5-fixture") is expected


def test_exaone4_5_module_tree_targets_language_model_only():
    modules = Exaone4_5QModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in modules for name in block}

    assert Exaone4_5QModel.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]
    assert Exaone4_5QModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert Exaone4_5QModel.out_of_model_tensors == {"prefixes": ["mtp"]}
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= flat_modules
    assert "self_attn.q_norm" not in flat_modules
    assert "self_attn.k_norm" not in flat_modules
