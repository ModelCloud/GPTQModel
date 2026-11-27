# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact real MiniCPM checkpoint inventories and live dense/MoE owners."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.minicpm import MiniCPMGPTQ
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.minicpm import MiniCPMConfig, MiniCPMForCausalLM
from gptqmodel.utils.hf import load_model_config
from tests.port_assets import read_port_json
from tests.test_minicpm_local_port import _config, _model


FIXTURE = Path(__file__).parents[2] / "gptqmodel/models/ports/minicpm/checkpoint_inventories.json"
INVENTORIES = read_port_json(FIXTURE)
RELEASED_CONFIGS = Path(__file__).parents[1] / "fixtures/ports/minicpm/released_configs"


@pytest.mark.parametrize("checkpoint", sorted(path.name for path in RELEASED_CONFIGS.iterdir()))
def test_released_minicpm4_longrope_configs_load(checkpoint):
    path = RELEASED_CONFIGS / checkpoint
    released = json.loads((path / "config.json").read_text())
    # This is the same first call GPTQModel.load()/from_quantized() make.
    config = load_model_config(path)
    assert isinstance(config, MiniCPMConfig)
    rope = released["rope_scaling"]
    assert config.rope_parameters["rope_type"] == "longrope"
    assert config.rope_parameters["short_factor"] == rope["short_factor"]
    assert config.rope_parameters["long_factor"] == rope["long_factor"]
    assert config.rope_parameters["original_max_position_embeddings"] == rope["original_max_position_embeddings"]
    # Released 4/4.1 configs keep the pretraining context, so the source
    # attention scale is one; the GPTQ config also serializes head_dim.
    assert config.rope_parameters["attention_factor"] == 1.0
    derived = released["hidden_size"] // released["num_attention_heads"]
    assert config.head_dim == released.get("head_dim", derived)
    with torch.device("meta"):
        model = MiniCPMForCausalLM(config)
    assert model.model.layers[0].self_attn.head_dim == config.head_dim


@pytest.mark.parametrize("checkpoint", sorted(INVENTORIES))
def test_pinned_real_checkpoint_keys_and_all_shapes_equal_meta_model(checkpoint):
    inventory = INVENTORIES[checkpoint]
    config = MiniCPMConfig(**inventory["config"])
    with torch.device("meta"):
        model = MiniCPMForCausalLM(config)
    expected = {name: entry["shape"] for name, entry in inventory["state"].items()}
    if config.tie_word_embeddings:
        assert "lm_head.weight" not in expected
        expected["lm_head.weight"] = expected["model.embed_tokens.weight"]
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == expected
    assert all(tensor.device.type == "meta" for tensor in model.state_dict().values())
    assert len(inventory["revision"]) == 40


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_every_checkpoint_tensor_has_deepest_live_plan_owner(moe, kv, bias):
    config = _config(num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0,
                      num_key_value_heads=kv, attention_bias=bias)
    with torch.device("meta"):
        model = MiniCPMForCausalLM(config)
    _convert_model_with_defuser(MiniCPMGPTQ, model, cleanup_original=False)
    tree = MiniCPMGPTQ._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    expected = {"self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    if moe:
        expected |= {f"mlp.experts.{i}.{name}" for i in range(3) for name in ("w1", "w3", "w2")}
        assert plan.find("mlp.gate").quantize is False
        assert [spec.template for spec in plan.moe_roots()] == ["mlp"]
    else:
        expected |= {"mlp." + name for name in ("gate_proj", "up_proj", "down_proj")}
    with MiniCPMGPTQ._module_tree_context(tree):
        declared = {name for block in MiniCPMGPTQ.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == expected
    for layer in model.model.layers:
        owners = {name.rsplit(".", 1)[0] for name in layer.state_dict()}
        assert owners == expected | {"input_layernorm", "post_attention_layernorm"} | ({"mlp.gate"} if moe else set())
        for owner in owners:
            spec = plan.find(owner)
            assert spec is not None
            assert spec.quantize == (owner in declared)
            if not spec.quantize:
                assert "!" in spec.flags
        for owner in declared:
            assert isinstance(layer.get_submodule(owner), nn.Linear)
            assert plan.find(owner + ".undeclared") is None
        if moe:
            assert plan.is_expert("mlp.experts.2.w1")
            assert plan.find("mlp.experts.2.w1").roles == ("gate",)
            assert plan.find("mlp.experts.2.w3").roles == ("up",)
            assert plan.find("mlp.experts.2.w2").roles == ("down",)
            assert not plan.is_expert("mlp.gate")
    assert MODEL_MAP["minicpm"] is MiniCPMGPTQ
    assert MiniCPMGPTQ.fusable is None


def test_dense_registry_plan_is_unchanged_and_moe_selection_is_request_local():
    expected = [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]
    assert MiniCPMGPTQ.build_layer_modules(MiniCPMGPTQ.module_tree, include_capture_only=True) == expected
    with torch.device("meta"):
        moe = MiniCPMForCausalLM(_config(num_experts=3, num_experts_per_tok=2))
        dense = MiniCPMForCausalLM(_config())
    a = MiniCPMGPTQ._resolve_effective_module_tree(moe, None)
    b = MiniCPMGPTQ._resolve_effective_module_tree(dense, None)
    assert MiniCPMGPTQ.build_layer_modules(b, include_capture_only=True) == expected
    blocks = MiniCPMGPTQ.build_layer_modules(a, include_capture_only=True)
    assert blocks[4:] == [["mlp.gate:!"], ["mlp.experts.{expert_index}.w1", "mlp.experts.{expert_index}.w3"],
                          ["mlp.experts.{expert_index}.w2"]]
    a[-1].clear()
    assert MiniCPMGPTQ._resolve_effective_module_tree(moe, None)[-1]
    assert MiniCPMGPTQ.build_layer_modules(MiniCPMGPTQ.module_tree, include_capture_only=True) == expected
    tree = MiniCPMGPTQ._resolve_effective_module_tree(moe, None)
    moe.gptqmodel_effective_module_tree = tree
    assert MiniCPMGPTQ._resolve_effective_module_tree(moe, None) == tree


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("kv", [1, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_awq_scale_folds_preserve_dense_and_moe_layer_output(moe, kv, bias):
    # MiniCPM scales each branch output by scale_depth/sqrt(L) before the
    # residual add; every fold below is internal to a branch, so it is exact.
    model = _model(num_key_value_heads=kv, attention_bias=bias,
                   num_experts=3 if moe else 0, num_experts_per_tok=2 if moe else 0)
    _convert_model_with_defuser(MiniCPMGPTQ, model, cleanup_original=False)
    tree = MiniCPMGPTQ._resolve_effective_module_tree(model, None)
    ids = torch.tensor([[4, 5, 6, 7, 4, 7, 6, 5]])

    def forward():
        return model(ids).logits

    expected = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], [])]
    if kv == 4:
        # GQA leaves v_proj narrower than o_proj, so only MHA folds v -> o.
        expected.append(("self_attn.v_proj", ["self_attn.o_proj"], []))
    if moe:
        # The unquantized router reads the same normed tensor and is compensated.
        expected.append(("post_attention_layernorm",
                         [f"mlp.experts.{i}.{name}" for i in range(3) for name in ("w1", "w3")], ["mlp.gate"]))
        expected += [(f"mlp.experts.{i}.w3", [f"mlp.experts.{i}.w2"], []) for i in range(3)]
    else:
        expected += [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], []),
                     ("mlp.up_proj", ["mlp.down_proj"], [])]
    # The processor plans against the compiled effective (MoE) tree.
    with MiniCPMGPTQ._module_tree_context(tree):
        for layer in model.model.layers:
            assert assert_awq_folds_neutral(MiniCPMGPTQ, model, layer, forward, tree=tree) == expected
