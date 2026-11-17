import itertools
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


def test_explicit_registry_case_for_integrator():
    import gptqmodel.models.ports.deepseek_vl_v2 as port

    expected = json.loads((Path(port.__file__).parent / "registry_plan.json").read_text())
    assert DeepSeekVLV2QModel.extract_layers_node() == expected["layers"]
    assert (
        DeepSeekVLV2QModel.build_layer_modules(DeepSeekVLV2QModel.module_tree, include_capture_only=True)
        == expected["blocks"]
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
    import gptqmodel.models.ports.deepseek_vl_v2 as port

    fixture = json.loads((Path(port.__file__).parent / "checkpoint_metadata.json").read_text())[variant]
    expected = set()
    for group in fixture["patterns"]:
        for layer, expert in itertools.product(group.get("layers", [0]), group.get("experts", [0])):
            expected.add(group["template"].format(layer=layer, expert=expert))
    assert len(expected) == fixture["key_count"]
    with torch.device("meta"):
        model = DeepseekVLV2ForCausalLM(DeepseekVLV2Config(**fixture["config"]))
    assert set(model.state_dict()) == expected
    assert_owned(model)
