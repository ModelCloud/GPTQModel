# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned release inventories and real dense/routed/shared XVERSE owners."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.xverse import XverseQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.xverse import XverseConfig, XverseForCausalLM
from tests.port_assets import read_port_json
from tests.test_xverse_local_port import PORT, tiny_config


INVENTORIES = read_port_json(PORT / "checkpoint_inventories.json")


@pytest.mark.parametrize("release", sorted(INVENTORIES))
def test_exact_pinned_release_keys_and_available_header_shapes(release):
    inventory = INVENTORIES[release]
    raw = dict(inventory["config"])
    raw.pop("_attn_implementation", None)
    config = XverseConfig(
        **raw,
        xverse_rope_persistent=any(k.endswith("inv_freq") for k in inventory["keys"]),
        attn_implementation="eager",
    )
    with torch.device("meta"):
        model = XverseForCausalLM(config)
    actual = {name: list(t.shape) for name, t in model.state_dict().items()}
    assert set(actual) == set(inventory["keys"])
    if "state" in inventory:
        assert actual == {name: item["shape"] for name, item in inventory["state"].items()}
    assert all(t.device.type == "meta" for t in model.state_dict().values())
    assert len(inventory["revision"]) == 40
    assert config.xverse_rope_type == ("dynamic_256k" if release == "XVERSE-13B-256K" else "default")


@pytest.mark.parametrize("moe,shared,kv,bias", [(0, None, 1, True), (4, None, 2, False), (4, 2, 4, True)])
def test_real_every_tensor_deepest_owner_roles_exclusions_and_plan(moe, shared, kv, bias):
    config = tiny_config(
        num_experts=moe, moe_top_k=2, num_shared_experts=shared, num_key_value_heads=kv, attention_bias=bias
    )
    with torch.device("meta"):
        model = XverseForCausalLM(config)
    _convert_model_with_defuser(XverseQModel, model, cleanup_original=False)
    tree = XverseQModel._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    expected = {"self_attn." + name for name in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    if moe:
        expected |= {f"mlp.experts.{i}.{name}" for i in range(moe) for name in ["gate_proj", "up_proj", "down_proj"]}
        if shared:
            expected |= {f"mlp.shared_experts.{name}" for name in ["gate_proj", "up_proj", "down_proj"]}
    else:
        expected |= {"mlp." + name for name in ["gate_proj", "up_proj", "down_proj"]}
    with XverseQModel._module_tree_context(tree):
        declared = {
            name
            for block in XverseQModel.simple_layer_modules(config, SimpleNamespace(dynamic=None))
            for name in block
        }
    assert declared == expected
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key if key.endswith("inv_freq") else key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in expected)
            if not spec.quantize:
                assert "!" in spec.flags
        for owner in expected:
            assert isinstance(layer.get_submodule(owner), nn.Linear)
            assert plan.find(owner + ".undeclared") is None
        if moe:
            assert not plan.find("mlp.router").quantize
            assert plan.is_expert("mlp.experts.3.gate_proj")
            assert plan.find("mlp.experts.3.gate_proj").roles == ("gate",)
            assert plan.find("mlp.experts.3.up_proj").roles == ("up",)
            assert plan.find("mlp.experts.3.down_proj").roles == ("down",)
    assert MODEL_MAP["xverse"] is XverseQModel
    assert XverseQModel.fusable is None
    assert XverseQModel.awq_get_modules_for_scaling(SimpleNamespace(), None, {}, {}) == []


def test_selection_is_request_local_and_central_proposal_is_exact():
    with torch.device("meta"):
        moe = XverseForCausalLM(tiny_config(num_experts=4, moe_top_k=2, num_shared_experts=1))
        dense = XverseForCausalLM(tiny_config())
    a = XverseQModel._resolve_effective_module_tree(moe, None)
    b = XverseQModel._resolve_effective_module_tree(dense, None)
    assert XverseQModel.build_layer_modules(b, include_capture_only=True) == [
        ["input_layernorm:!"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["self_attn.rotary_emb.inv_freq:!"],
        ["post_attention_layernorm:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    a[-1].clear()
    assert XverseQModel._resolve_effective_module_tree(moe, None)[-1]
    assert XverseQModel._resolve_effective_module_tree(dense, None) == b
