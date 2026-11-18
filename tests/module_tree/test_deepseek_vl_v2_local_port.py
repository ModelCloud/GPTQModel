import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.deepseek_vl_v2 import DeepSeekVLV2QModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.deepseek_vl_v2.configuration_deepseek_vl_v2 import DeepseekVLV2Config
from gptqmodel.models.ports.deepseek_vl_v2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
from tests.test_deepseek_vl_v2_local_port import tiny_config


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ports" / "deepseek_vl_v2"

EXPECTED_BLOCKS = [
    ["input_layernorm:!"],
    [
        "self_attn.q_a_proj", "self_attn.q_a_layernorm:!", "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm:!",
    ],
    ["self_attn.q_b_proj", "self_attn.kv_b_proj"],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
    ["mlp.gate:!"],
    ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
    ["mlp.experts.{expert_index}.down_proj"],
    ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
    ["mlp.shared_experts.down_proj"],
]


def test_explicit_registry_case_for_integrator():
    assert DeepSeekVLV2QModel.extract_layers_node() == ["language.model.layers"]
    assert (
        DeepSeekVLV2QModel.build_layer_modules(DeepSeekVLV2QModel.module_tree, include_capture_only=True)
        == EXPECTED_BLOCKS
    )
    assert DeepSeekVLV2QModel.fusable is None


def assert_owned(model):
    assert MODEL_MAP["deepseek_vl_v2"] is DeepSeekVLV2QModel
    plan = ModulePlan.from_tree(DeepSeekVLV2QModel.module_tree)
    quantized = {
        name
        for block in DeepSeekVLV2QModel.simple_layer_modules(model.config, SimpleNamespace(dynamic=None))
        for name in block
    }
    seen = set()
    for key in model.state_dict():
        if not key.startswith("language.model.layers."):
            continue
        owner = key.rsplit(".", 1)[0]
        relative = owner.split(".", 4)[4]
        spec = plan.find(relative)
        assert spec is not None, owner
        if spec.quantize:
            assert relative in quantized, owner
            assert isinstance(model.get_submodule(owner), nn.Linear), owner
        else:
            assert "!" in spec.flags or "?" in spec.flags, owner
        seen.add(owner)
    assert seen
    assert plan.find("self_attn.q_proj.undeclared") is None
    return plan


@pytest.mark.parametrize(
    "mla,q_lora,method",
    [(False, None, "greedy"), (True, None, "greedy"), (True, 8, "group_limited_greedy"), (True, 8, "noaux_tc")],
)
def test_real_layout_weight_owners(mla, q_lora, method):
    config = tiny_config(mla, language={"q_lora_rank": q_lora, "topk_method": method})
    with torch.device("meta"):
        model = DeepseekVLV2ForCausalLM(config)
    plan = assert_owned(model)
    names = {key.rsplit(".", 1)[0] for key in model.state_dict()}
    if mla:
        assert "language.model.layers.0.self_attn.kv_a_layernorm" in names
        assert not plan.find("self_attn.kv_a_layernorm").quantize
    assert not plan.find("mlp.gate").quantize


@pytest.mark.parametrize("variant", ["tiny", "small", "base"])
def test_pinned_checkpoint_key_inventory_matches_actual_meta_model(variant):
    # Key count + sha256 of the sorted safetensors index keys of the pinned hub revision.
    fixture = json.loads((FIXTURES / "checkpoints.json").read_text())[variant]
    with torch.device("meta"):
        model = DeepseekVLV2ForCausalLM(DeepseekVLV2Config(**fixture["config"]))
    keys = sorted(model.state_dict())
    assert len(keys) == fixture["key_count"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == fixture["keys_sha256"]
    assert_owned(model)
