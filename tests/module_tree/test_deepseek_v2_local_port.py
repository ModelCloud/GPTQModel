# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V2's native/defused weights must reach the module-plan consumers."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config as LocalConfig
from gptqmodel.models.ports.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM as LocalModel
from gptqmodel.quantization.config import QuantizeConfig


def tiny_config(**overrides):
    values = {
        "vocab_size": 43, "hidden_size": 24, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2, "kv_lora_rank": 8,
        "q_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 6,
        "n_routed_experts": 4, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
        "moe_intermediate_size": 12, "n_shared_experts": 2, "first_k_dense_replace": 1,
        "max_position_embeddings": 64, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
        "attention_dropout": 0.0, "experts_implementation": "eager",
    }
    values.update(overrides)
    return DeepseekV2Config(**values)


@pytest.mark.parametrize("q_rank", [None, 8], ids=["lite", "full"])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("implementation", ["native", "local"])
def test_real_checkpoint_owners_and_module_plan(q_rank, device, implementation):
    config = tiny_config(q_lora_rank=q_rank, attention_bias=True, mlp_bias=True)
    model_cls = DeepseekV2ForCausalLM
    if implementation == "local":
        config = LocalConfig(**config.to_dict())
        model_cls = LocalModel
    with torch.device(device):
        model = model_cls(config).eval()
    assert _convert_model_with_defuser(DeepSeekV2QModel, model, cleanup_original=True)
    wrapper = DeepSeekV2QModel(model, quantized=False, quantize_config=QuantizeConfig())
    assert MODEL_MAP["deepseek_v2"] is DeepSeekV2QModel
    plan = wrapper.module_plan()
    quantized = {
        name.split(":", 1)[0]
        for block in wrapper.simple_layer_modules(config, wrapper.quantize_config)
        for name in block
    }
    expected = {"input_layernorm", "post_attention_layernorm", "self_attn.kv_a_layernorm", "mlp.gate"}
    if q_rank is not None:
        expected.add("self_attn.q_a_layernorm")
    owners = set()
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            owners.add(owner)
            spec = plan.find(owner)
            assert spec is not None, f"undeclared checkpoint owner: {owner}"
            assert spec.quantize == (owner in quantized), owner
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert name in quantized, name
    assert expected <= owners
    assert all(not plan.find(name).quantize for name in expected)
    assert plan.roles("self_attn.q_b_proj") == ("q",)
    assert plan.roles("self_attn.kv_b_proj") == ("k", "v")
    assert plan.roles("self_attn.o_proj") == ("o",)
    assert plan.expert_root("mlp.experts.3.gate_proj") == "mlp.experts.3"
    assert plan.shared_container_leaf("mlp") == "shared_experts"
    assert plan.find("self_attn.kv_a_layernorm_extra") is None
    assert plan.find("mlp.gate.undeclared") is None
    hooks = copy.copy(wrapper.moe_lifecycle_hooks)
    hooks.module_plan = plan
    block = hooks.get_moe_block(model.model.layers[1], wrapper)
    assert hooks.get_experts_module(block, wrapper) is block.experts
    assert hooks.get_shared_experts_module(block, wrapper) is block.shared_experts
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(module_plan=lambda: plan)
    assert looper._is_attention_module_name("self_attn.kv_b_proj")
    assert looper._extract_moe_group_key("mlp.experts.3.up_proj") == "mlp.experts.3"


def test_awq_is_explicitly_unqualified_even_with_complete_features():
    model = DeepseekV2ForCausalLM(tiny_config()).eval()
    _convert_model_with_defuser(DeepSeekV2QModel, model, cleanup_original=True)
    wrapper = DeepSeekV2QModel(model, quantized=False, quantize_config=QuantizeConfig())
    for layer in model.model.layers:
        features = {
            name: torch.randn(2, 5, module.in_features)
            for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)
        }
        features["mlp"] = torch.randn(2, 5, model.config.hidden_size)
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []


def test_independent_registry_case_and_fusion_exclusion():
    assert DeepSeekV2QModel.extract_layers_node() == ["model.layers"]
    assert DeepSeekV2QModel.build_layer_modules(DeepSeekV2QModel.module_tree, include_capture_only=True) == [
        ["input_layernorm:!"],
        ["self_attn.q_a_proj", "self_attn.q_a_layernorm:!", "self_attn.q_proj",
         "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm:!"],
        ["self_attn.q_b_proj", "self_attn.kv_b_proj"],
        ["self_attn.o_proj"],
        ["post_attention_layernorm:!"],
        ["mlp.gate:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
        ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        ["mlp.experts.{expert_index}.down_proj"],
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
    ]
    # Distinct low-rank attention stages are not a qualified QKV fusion set.
    assert DeepSeekV2QModel.fusable is None
