# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Inspect actual Omni checkpoint tensor ownership, not similarly shaped stubs."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.lfm3_omni import LFM3OmniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.lfm3_omni.configuration_lfm3_omni import Lfm3OmniConfig
from gptqmodel.models.ports.lfm3_omni.configuration_liquid_codec import LiquidCodecConfig
from gptqmodel.models.ports.lfm3_omni.modeling_lfm3_omni import Lfm3OmniForConditionalGeneration
from gptqmodel.models.ports.lfm3_omni.modeling_liquid_codec import LiquidCodecModel
from tests.port_assets import DATA
from tests.test_lfm3_omni_local_port import tiny_config


FIXTURES = DATA / "lfm3_omni"


EXPECTED_PLAN = {
    "layers": ["model.language_model.layers"],
    "blocks": [
        ["operator_norm:!"],
        ["post_operator_norm:!"],
        ["conv.in_proj"],
        ["conv.conv:!"],
        ["conv.out_proj"],
        ["self_attn.q_proj", "self_attn.q_norm:!", "self_attn.k_proj", "self_attn.k_norm:!",
         "self_attn.v_proj", "self_attn.gate_proj:!"],
        ["self_attn.o_proj"],
        ["ffn_norm:!"],
        ["post_ffn_norm:!"],
        ["feed_forward.w1", "feed_forward.w3"],
        ["feed_forward.w2"],
    ],
}


def test_omni_exact_registry_contract():
    tree = LFM3OmniQModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert LFM3OmniQModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert LFM3OmniQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]


def wrapper_for(model):
    wrapper = LFM3OmniQModel.__new__(LFM3OmniQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    return wrapper


@pytest.mark.parametrize("kv_heads,bias,sandwich", [(4, False, True), (2, True, False), (1, True, True)])
def test_real_decoder_weight_owners_quantization_and_plan_tags(kv_heads, bias, sandwich):
    with torch.device("meta"):
        model = Lfm3OmniForConditionalGeneration(tiny_config(kv_heads=kv_heads, bias=bias, sandwich=sandwich))
    _convert_model_with_defuser(LFM3OmniQModel, model, cleanup_original=False)
    wrapper = wrapper_for(model)
    plan = ModulePlan.from_tree(wrapper.module_tree)
    declared = {name for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == {"conv.in_proj", "conv.out_proj", "self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj", "self_attn.o_proj", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"}
    quantized = 0
    for layer in model.model.language_model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), nn.Linear)
                quantized += key.endswith("weight")
            else:
                assert set(spec.flags) & {"!", "?"}
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert quantized == 19
    assert not plan.find("conv.conv").quantize
    assert not plan.find("self_attn.gate_proj").quantize
    assert plan.find("conv.conv.undeclared") is None


def _layout_sha256(model):
    layout = {key: list(value.shape) for key, value in model.state_dict().items()}
    return len(layout), hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_exact_pinned_checkpoint_shapes_and_every_decoder_owner():
    config = Lfm3OmniConfig(**json.loads((FIXTURES / "checkpoint-config.json").read_text()))
    with torch.device("meta"):
        model = Lfm3OmniForConditionalGeneration(config)
    # Canonical sorted {key: shape} of the pinned safetensors headers (665 tensors) plus the tied lm_head alias.
    assert _layout_sha256(model) == (666, "daa5713ca2ad5297b84bd8d1b36e4c396db012e129461c2ffff13a89348dc2ed")
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    plan = ModulePlan.from_tree(LFM3OmniQModel.module_tree)
    wrapper = wrapper_for(model)
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    count = 0
    for key in model.state_dict():
        if not key.startswith("model.language_model.layers."):
            continue
        relative = key.split(".", 4)[-1]
        owner = relative.rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, key
        assert owner in declared if spec.quantize else set(spec.flags) & {"!", "?"}
        count += 1
    assert count > 200


def test_exact_pinned_codec_shapes():
    config = LiquidCodecConfig(**json.loads((FIXTURES / "codec-config.json").read_text()))
    with torch.device("meta"):
        model = LiquidCodecModel(config)
    # Canonical sorted {key: shape} of the pinned codec safetensors header.
    assert _layout_sha256(model) == (203, "8cf8486423a504c5cc516117179def353093d6398079f3744ce0e604b06b153a")
    assert list(model.state_dict()["decoder.istft.window"].shape) == [1280]
