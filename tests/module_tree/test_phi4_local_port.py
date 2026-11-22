# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The actual legacy phi4mm tensor owners, not the similarly named native layout."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.definitions.phi4 import Phi4MMGPTQ
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.phi4.configuration_phi4 import Phi4MMConfig
from gptqmodel.models.ports.phi4.modeling_phi4 import Phi4MMForCausalLM
from tests.test_phi4_local_port import _config, _model


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


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_awq_norm_folds_compensate_modality_loras(mode):
    # `_model` gives every lora_B a nonzero weight, so vision/speech modes
    # really add the adapter branch that reads the folded norm output.
    model = _model()
    _convert_model_with_defuser(Phi4MMGPTQ, model, cleanup_original=False)
    ids = torch.randint(3, 50, (1, 8), generator=torch.Generator().manual_seed(5))

    def forward():
        return model(input_ids=ids, input_mode=mode).logits

    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(Phi4MMGPTQ, model, layer, forward)
        # Each norm folds into the base projection and both input-side adapters
        # of that projection; the previous projection's adapters are not
        # consumers. Fused qkv/gate_up outputs cannot feed o_proj/down_proj.
        assert checked == [
            ("input_layernorm", ["self_attn.qkv_proj.base_layer"],
             ["self_attn.qkv_proj.lora_A.vision", "self_attn.qkv_proj.lora_A.speech"]),
            ("post_attention_layernorm", ["mlp.gate_up_proj.base_layer"],
             ["mlp.gate_up_proj.lora_A.vision", "mlp.gate_up_proj.lora_A.speech"]),
        ]


# sha256 of the sorted (name, shape) inventory of all three microsoft/Phi-4-multimodal-instruct
# safetensors headers at the pinned revision. The tied lm_head alias is not serialized.
RELEASED_HEADER_SHA256 = "a614e91b7668889a00fc95ba7969b5dfc28fd8d87ee855ec8543e63d1680d4f1"


def test_complete_pinned_released_tensor_header_inventory():
    released = json.loads((Path(__file__).parents[1] / "fixtures/ports/phi4/released_config.json").read_text())
    config = Phi4MMConfig(**released["config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Phi4MMForCausalLM(config)
    inventory = sorted((key, list(value.shape)) for key, value in model.state_dict().items() if key != "lm_head.weight")
    assert len(inventory) == 2047
    assert hashlib.sha256(json.dumps(inventory).encode()).hexdigest() == RELEASED_HEADER_SHA256
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_device_map_no_split_covers_wrapped_legacy_layers():
    # accelerate matches _no_split_modules by class name; the legacy wrappers replace native layers.
    model = Phi4MMForCausalLM(_config())
    classes = {type(module).__name__ for module in model.modules()}
    for name in ("Phi4MMDecoderLayer", "Phi4MMAudioLayer", "Phi4MultimodalVisionEncoderLayer"):
        assert name in classes and name in model._no_split_modules
    assert "Phi4MultimodalDecoderLayer" not in classes
