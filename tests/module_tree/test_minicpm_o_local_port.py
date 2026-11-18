# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.minicpm_o import MiniCPMOQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.minicpm_o.configuration_minicpmo import MiniCPMOConfig
from gptqmodel.models.ports.minicpm_o.modeling_minicpmo import MiniCPMO
from tests.test_minicpm_o_local_port import LEGACY_TTS_AVAILABLE, PORT, tiny_config


EXPECTED_PLAN = {
    "layers": ["llm.model.layers"],
    "blocks": [
        ["input_layernorm:!"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm:!", "self_attn.k_norm:!"],
        ["self_attn.o_proj"],
        ["post_attention_layernorm:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ],
}


def test_exact_registry_contract():
    registry = Path(__file__).with_name("fixtures") / "registered_module_plans.json"
    assert json.loads(registry.read_text())["cases"]["MiniCPMOQModel/gptq"] == EXPECTED_PLAN
    tree = MiniCPMOQModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert MiniCPMOQModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert MiniCPMOQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]


@pytest.mark.parametrize("version", ["4.5", "2.6"])
@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("bias", [False, True])
def test_actual_cpu_meta_owners_and_plain_policy(version, kv_heads, bias):
    with torch.device("meta"):
        model = MiniCPMO(tiny_config(version=version, kv_heads=kv_heads, bias=bias))
    wrapper = MiniCPMOQModel.__new__(MiniCPMOQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    declared = {
        name for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block
    }
    assert declared == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    count = 0
    for layer in model.llm.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert owner in declared if spec.quantize else "!" in spec.flags
            if spec.quantize:
                assert isinstance(layer.get_submodule(owner), nn.Linear)
                count += key.endswith("weight")
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert count == 14
    assert plan.find("self_attn.q_norm.weight.undeclared") is None


@pytest.mark.parametrize("version,keys", [("4_5", 1414), ("2_6", 1599)])
def test_every_real_pinned_checkpoint_key_and_header_shape(version, keys):
    if version == "2_6" and not LEGACY_TTS_AVAILABLE:
        pytest.skip("MiniCPM-o 2.6 TTS requires gptqmodel[minicpm-o]")
    config = MiniCPMOConfig(**json.loads((PORT / f"checkpoint-{version}-config.json").read_text()))
    with torch.device("meta"):
        model = MiniCPMO(config)
    expected = json.loads((PORT / f"checkpoint-{version}-checkpoint-layout.json").read_text())
    assert len(expected) == keys
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert actual == expected
    assert all(
        key.startswith(("llm.", "vpm.", "resampler.", "apm.", "audio_projection_layer.", "tts.")) for key in actual
    )
    assert expected["llm.model.layers.0.self_attn.k_proj.weight"][0] == config.num_key_value_heads * config.head_dim
    assert model.tts.model.config.hidden_size == config.tts_config.hidden_size
