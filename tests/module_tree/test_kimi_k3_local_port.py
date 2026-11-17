# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real decoder checkpoint ownership for K3 and the AttnRes text alias."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_kimi_k3_local_port import tiny_config, tiny_k3


def test_port_registry_cases_are_exact_for_both_aliases():
    from gptqmodel.models.ports import kimi_k3

    cases = json.loads((Path(kimi_k3.__file__).parent / "module_plan_cases.json").read_text())
    for definition in (KimiK3QModel, KimiLinearQModel):
        expected = cases[f"{definition.__name__}/gptq"]
        assert expected["layers"] == definition.extract_layers_node()
        assert expected["blocks"] == definition.build_layer_modules(definition.module_tree, include_capture_only=True)


from gptqmodel.models.definitions.kimi_k3 import KimiK3QModel, KimiLinearQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.kimi_k3.modeling_kimi_k3 import KimiK3ForConditionalGeneration
from gptqmodel.models.ports.kimi_k3.modeling_kimi_linear import KimiLinearForCausalLM
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AWQConfig


@pytest.mark.parametrize("mode", ["mla", "hybrid", "kda"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_live_checkpoint_owners_and_routed_expert_tags(mode, nested, device):
    config = tiny_k3(mode=mode) if nested else tiny_config(mode)
    definition = KimiK3QModel if nested else KimiLinearQModel
    with torch.device(device):
        model = (KimiK3ForConditionalGeneration if nested else KimiLinearForCausalLM)(config).eval()
    decoder = model.language_model.model if nested else model.model
    plan = ModulePlan.from_tree(definition.module_tree)
    routed = {f"block_sparse_moe.experts.{i}.{projection}" for i in range(4) for projection in ("w1", "w3", "w2")}
    qnames = {name for group in definition.simple_layer_modules(config, QuantizeConfig()) for name in group}
    assert qnames == routed
    assert plan.find("block_sparse_moe.experts.1.w1.undeclared") is None
    for index, layer in enumerate(decoder.layers):
        expected = {
            f"{name}.weight"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attention_res_norm",
                "self_attention_res_proj",
                "mlp_res_norm",
                "mlp_res_proj",
            )
        }
        if layer.is_linear_attn:
            names = (
                "q_proj",
                "k_proj",
                "v_proj",
                "b_proj",
                "f_a_proj",
                "f_b_proj",
                "g_proj",
                "o_proj",
                "q_conv1d",
                "k_conv1d",
                "v_conv1d",
                "o_norm",
            )
            expected |= {"self_attn.A_log", "self_attn.dt_bias"}
        else:
            names = (
                "q_a_proj",
                "q_a_layernorm",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_a_layernorm",
                "kv_b_proj",
                "o_proj",
                "g_proj",
            )
        expected |= {f"self_attn.{name}.weight" for name in names}
        if index == 0:
            expected |= {f"mlp.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")}
        else:
            expected |= {name + ".weight" for name in routed}
            expected |= {
                f"block_sparse_moe.{name}.weight"
                for name in (
                    "gate",
                    "routed_expert_down_proj",
                    "routed_expert_up_proj",
                    "routed_expert_norm",
                    "shared_experts.gate_proj",
                    "shared_experts.up_proj",
                    "shared_experts.down_proj",
                )
            }
            expected.add("block_sparse_moe.gate.e_score_correction_bias")
            expert = layer.block_sparse_moe.experts[0]
            assert expert.w1.weight.shape == (12, 8) and expert.w2.weight.shape == (8, 12)
        assert set(layer.state_dict()) == expected
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            assert layer.get_submodule(owner) is not None
            spec = plan.find(owner)
            assert spec is not None, (index, key)
            assert spec.quantize == (owner in routed)
            if not spec.quantize:
                assert "!" in spec.flags
            else:
                assert spec.roles == (
                    ("gate",) if owner.endswith("w1") else ("up",) if owner.endswith("w3") else ("down",)
                )
    prefix = "language_model." if nested else ""
    allowed = {
        prefix + name
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.output_attn_res_norm.weight",
            "model.output_attn_res_proj.weight",
            "lm_head.weight",
        )
    }
    external = {key for key in model.state_dict() if not key.startswith(prefix + "model.layers.")}
    if nested:
        assert any(key.startswith("vision_tower.") for key in external)
        assert any(key.startswith("mm_projector.") for key in external)
        external = {key for key in external if not key.startswith(("vision_tower.", "mm_projector."))}
    assert external == allowed


@pytest.mark.parametrize("nested", [False, True])
def test_awq_rejected_before_capture_or_weight_mutation(nested):
    definition = KimiK3QModel if nested else KimiLinearQModel
    model = KimiLinearForCausalLM(tiny_config()).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for quantize_config in (QuantizeConfig(), AWQConfig()):
        wrapper = SimpleNamespace(model=model, quantize_config=quantize_config)
        with pytest.raises(NotImplementedError, match="prefix_sum and block_residual"):
            definition.quantize(wrapper, object())
    with pytest.raises(NotImplementedError, match="AWQ is unqualified"):
        definition.awq_get_modules_for_scaling(SimpleNamespace(), model.model.layers[1], {}, {})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
