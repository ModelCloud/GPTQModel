# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from transformers import Exaone4_5_Config, Exaone4Config

from gptqmodel.models import auto
from gptqmodel.models.definitions.exaone4 import Exaone4_5QModel, Exaone4QModel


@pytest.mark.parametrize("model_type,expected", [
    ("exaone4_5", Exaone4_5QModel),
    ("exaone4_5_text", Exaone4QModel),
])
def test_exaone4_5_model_types_select_expected_definitions(tmp_path, model_type, expected):
    if model_type == "exaone4_5":
        config = Exaone4_5_Config(text_config=Exaone4Config().to_dict())
    else:
        config = Exaone4Config()
    config.save_pretrained(tmp_path)
    if model_type == "exaone4_5_text":
        config_path = tmp_path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["model_type"] = model_type
        config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert auto.check_and_get_model_definition(str(tmp_path)) is expected


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
