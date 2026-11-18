# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The actual legacy phi4mm tensor owners, not the similarly named native layout."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.phi4 import Phi4MMGPTQ
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.phi4.configuration_phi4 import Phi4MMConfig
from gptqmodel.models.ports.phi4.modeling_phi4 import Phi4MMForCausalLM
from tests.port_assets import read_port_json
from tests.test_phi4_local_port import _config


PROJECTIONS = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm")
ADAPTERS = tuple(f"{p}.lora_{ab}.{mode}" for p in PROJECTIONS for ab in ("A", "B") for mode in ("vision", "speech"))
EXPECTED_PLAN = {
    "prefix": "model.layers",
    "blocks": [
        ["input_layernorm:!"],
        ["self_attn.qkv_proj.base_layer"],
        ["self_attn.qkv_proj.lora_A.vision:!", "self_attn.qkv_proj.lora_A.speech:!"],
        ["self_attn.qkv_proj.lora_B.vision:!", "self_attn.qkv_proj.lora_B.speech:!"],
        ["self_attn.o_proj.base_layer"],
        ["self_attn.o_proj.lora_A.vision:!", "self_attn.o_proj.lora_A.speech:!"],
        ["self_attn.o_proj.lora_B.vision:!", "self_attn.o_proj.lora_B.speech:!"],
        ["post_attention_layernorm:!"],
        ["mlp.gate_up_proj.base_layer"],
        ["mlp.gate_up_proj.lora_A.vision:!", "mlp.gate_up_proj.lora_A.speech:!"],
        ["mlp.gate_up_proj.lora_B.vision:!", "mlp.gate_up_proj.lora_B.speech:!"],
        ["mlp.down_proj.base_layer"],
        ["mlp.down_proj.lora_A.vision:!", "mlp.down_proj.lora_A.speech:!"],
        ["mlp.down_proj.lora_B.vision:!", "mlp.down_proj.lora_B.speech:!"],
    ],
}


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_real_legacy_decoder_plan_and_excluded_adapters(kv, device):
    config = _config(num_key_value_heads=kv)
    with torch.device(device):
        model = Phi4MMForCausalLM(config)
    _convert_model_with_defuser(Phi4MMGPTQ, model, cleanup_original=False)
    wrapper = Phi4MMGPTQ.__new__(Phi4MMGPTQ)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    assert wrapper.build_layer_modules(wrapper.module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    declared = {
        name for group in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in group
    }
    assert declared == {p + ".base_layer" for p in PROJECTIONS}
    owners = {p + ".base_layer" for p in PROJECTIONS} | set(ADAPTERS) | set(NORMS)
    for layer in model.model.layers:
        assert set(layer.state_dict()) == {name + ".weight" for name in owners}
        for name in owners:
            spec = plan.find(name)
            assert spec is not None, name
            assert spec.quantize == (name in declared)
            if name not in declared:
                assert "!" in spec.flags
            assert plan.find(name + ".undeclared") is None
            if name in declared:
                assert isinstance(layer.get_submodule(name), nn.Linear)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {"input_mode": 3}) == []


def test_complete_pinned_released_tensor_header_inventory():
    root = Path(__file__).parents[2] / "gptqmodel/models/ports/phi4"
    schema = read_port_json(root / "released_schema.json")
    config = Phi4MMConfig(**schema["config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Phi4MMForCausalLM(config)
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert len(schema["shapes"]) == 2047
    assert set(actual) - set(schema["shapes"]) == {"lm_head.weight"}
    for key, shape in schema["shapes"].items():
        assert actual[key] == shape, key
    assert model.lm_head.weight is model.model.embed_tokens.weight
