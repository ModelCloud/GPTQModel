# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every released Ripple decoder owner resolves through the real module plan."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.ricursive.configuration_ricursive import RicursiveConfig
from gptqmodel.models.ports.ricursive.modeling_ricursive import RicursiveForCausalLM
from tests.port_assets import DATA, read_port_json
from tests.test_ricursive_local_port import FIXTURES, LINEARS, _config


@pytest.mark.parametrize("release", ["preview", "base"])
def test_released_1203_keys_and_all_shapes_equal_live_meta_owners(release):
    config = RicursiveConfig.from_pretrained(DATA / "ricursive/fixtures" / release)
    with torch.device("meta"):
        model = RicursiveForCausalLM(config)
    # Both releases publish the same tensor headers.
    expected = read_port_json(FIXTURES / "header_inventory.json")
    assert len(expected) == 1203
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == {k: v["shape"] for k, v in expected.items()}
    assert len(model.model.layers) == 80
    assert model.model.layers[4].self_attn.layer_type == "full_attention"
    assert not hasattr(model.model.layers[4].self_attn, "rotary_emb")
    assert model.config.embed_scale == 71.55417527999327


@pytest.mark.parametrize("gated,valres", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_all_real_decoder_owners_are_quantized_or_explicitly_dense(gated, valres, kv_heads):
    config = _config(use_gated_attention=gated, use_valres=valres, num_key_value_heads=kv_heads)
    with torch.device("meta"):
        model = RicursiveForCausalLM(config)
    _convert_model_with_defuser(RicursiveQModel, model, cleanup_original=False)
    wrapper = RicursiveQModel.__new__(RicursiveQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    tree = wrapper._resolve_effective_module_tree(model, SimpleNamespace(method="gptq"))
    plan = ModulePlan.from_tree(tree)
    quantized = LINEARS if gated else LINEARS - {"self_attn.g_proj"}
    expected_keys = {name + ".weight" for name in quantized} | {
        n + ".weight" for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                 "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm")}
    if valres:
        expected_keys.add("self_attn.value_res_lambda")
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected_keys
        for key in expected_keys:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in quantized), key
            if not spec.quantize:
                assert "!" in spec.flags
        for name in quantized:
            assert isinstance(layer.get_submodule(name), nn.Linear)
            assert plan.find(name + ".undeclared") is None
        assert "in=x" in plan.find("self_attn.q_proj").flags
        assert not any(flag.startswith("in=") for flag in plan.find("self_attn.v_proj").flags)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert RicursiveQModel.fusable is None
    assert not RicursiveQModel.require_trust_remote_code


def test_default_registered_module_plan_is_unchanged():
    from pathlib import Path
    inventory = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    expected = inventory["cases"]["RicursiveQModel/gptq"]
    assert RicursiveQModel.extract_layers_node() == expected["layers"]
    assert RicursiveQModel.build_layer_modules(RicursiveQModel.module_tree, include_capture_only=True) == expected["blocks"]
