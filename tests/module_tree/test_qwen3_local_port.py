# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inventories are independent of the definition's projection tree."""

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.module_plan import ModulePlan
from tests.test_qwen3_local_port import asset, tiny_config


QUANTIZED = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
DENSE = {"self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm"}


@pytest.mark.parametrize("family,count", [("marin", 707), ("qwen3", 311)])
def test_all_released_header_keys_and_shapes_equal_exact_native_meta(family, count):
    config = Qwen3Config.from_dict(json.loads(asset(family, "config.json")))
    inventory = json.loads(asset(family, "headers.json"))
    assert len(inventory) == count
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == {name: info["shape"] for name, info in inventory.items()}
    plan = ModulePlan.from_tree(Qwen3QModel.module_tree)
    for name in inventory:
        if not name.startswith("model.layers."):
            assert name in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
            continue
        owner = name.split(".", 3)[3].rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, name
        assert spec.quantize == (owner in QUANTIZED)
        if owner in DENSE:
            assert "!" in spec.flags
    del model


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("head_dim", [8, 10])
def test_native_layout_live_owners_and_alias_contract(kv_heads, bias, device, head_dim):
    config = tiny_config(num_key_value_heads=kv_heads, attention_bias=bias, head_dim=head_dim)
    with torch.device(device):
        model = Qwen3QModel.loader.from_config(config)
    plan = ModulePlan.from_tree(Qwen3QModel.module_tree)
    for layer in model.model.layers:
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            assert owner in QUANTIZED | DENSE
            spec = plan.find(owner)
            assert spec is not None
            assert spec.quantize == (owner in QUANTIZED)
            assert layer.get_submodule(owner).weight.device.type == device
        assert layer.self_attn.q_norm.weight.shape == (head_dim,)
        assert layer.self_attn.k_proj.weight.shape == (kv_heads * head_dim, 32)
    assert MODEL_MAP["marin"] is MODEL_MAP["qwen3"] is Qwen3QModel
    assert Qwen3QModel.simple_layer_modules(config, None) == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    assert plan.find("self_attn.q_projection") is None
    assert plan.find("self_attn.q_norm.undeclared") is None


def test_explicit_plain_awq_policy_preserves_quantizable_inventory():
    wrapper = Qwen3QModel.__new__(Qwen3QModel)
    model = Qwen3ForCausalLM(tiny_config())
    assert wrapper.awq_get_modules_for_scaling(model.model.layers[0], {}, {}) == []
    assert len(QUANTIZED) == 7
