# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real decoder weight owners, source layouts and released OpenVLA schema."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.openvla import OpenVLAQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.openvla.configuration_prismatic import OpenVLAConfig
from gptqmodel.models.ports.openvla.modeling_prismatic import OpenVLAForActionPrediction
from tests.test_openvla_local_port import _config, _write_assets


@pytest.fixture(scope="module")
def processor_assets(tmp_path_factory):
    path = tmp_path_factory.mktemp("openvla-plan-processor")
    _write_assets(path, _config())
    return str(path)


@pytest.mark.parametrize("llm,kv_heads,bias,device", [
    ("llama2-7b-pure", 1, False, "cpu"), ("vicuna-v15-13b", 4, True, "meta"),
    ("mistral-v0.1-7b-pure", 2, True, "cpu"), ("phi-2-3b", 4, True, "meta"), ("phi-2-3b", 2, False, "cpu"),
])
def test_openvla_real_weight_owners_and_module_plans(llm, kv_heads, bias, device, processor_assets):
    with torch.device(device):
        model = OpenVLAForActionPrediction(_config(
            llm=llm, text_config={"num_key_value_heads": kv_heads, "attention_bias": bias, "mlp_bias": bias,
                                    "qk_layernorm": bias},
        ))
    _convert_model_with_defuser(OpenVLAQModel, model, cleanup_original=False)
    wrapper = OpenVLAQModel(model, quantized=False, quantize_config=None, model_local_path=processor_assets)
    plan = wrapper.module_plan()
    phi = llm == "phi-2-3b"
    declared = {path for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None))
                for path in block}
    expected = {"self_attn." + name + "_proj" for name in ("q", "k", "v")}
    expected |= {"self_attn.dense", "mlp.fc1", "mlp.fc2"} if phi else {
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    assert declared == expected
    assert wrapper.lm_head == "language_model.lm_head"
    norm_name = "final_layernorm" if phi else "norm"
    assert wrapper.pre_lm_head_norm_module == "language_model.model." + norm_name
    assert wrapper.extract_layers_node() == ["language_model.model.layers"]
    base_modules = wrapper.get_base_modules(model)
    assert {"vision_backbone", "projector", "language_model.model.embed_tokens",
            "language_model.model." + norm_name, "language_model.lm_head"} <= set(base_modules)
    for name, value in model.state_dict().items():
        assert value.device.type == device
        if name.startswith("language_model.model.layers."):
            leaf = name.split(".", 4)[-1].rsplit(".", 1)[0]
            spec = plan.find(leaf)
            assert spec is not None, name
            assert spec.quantize == (leaf in expected), name
            assert plan.find(leaf + ".foreign_child") is None
        else:
            assert any(name.startswith(root + ".") for root in base_modules), name
    for layer in model.language_model.model.layers:
        for path in expected:
            assert isinstance(layer.get_submodule(path), nn.Linear)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert wrapper.fusable is None
    assert MODEL_MAP["openvla"] is OpenVLAQModel


# openvla/openvla-7b @ 47a0ec7fc4ec123775a391911046cf33cf9ed83f: config geometry and the
# sha256 of "name [shape]" lines over the sorted union of its three safetensors headers.
RELEASED_CONFIG = {
    "vision_backbone_id": "dinosiglip-vit-so-224px", "llm_backbone_id": "llama2-7b-pure",
    "text_config": {"vocab_size": 32064, "hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32,
                    "num_attention_heads": 32, "num_key_value_heads": 32, "pad_token_id": 32000},
}
RELEASED_INVENTORY_SHA256 = "99212ba7b2a20bff01a94924f778724810deca58ecd0cfc4beddd9b81bd1398f"


def test_openvla_full_pinned_release_header_inventory():
    with torch.device("meta"):
        model = OpenVLAForActionPrediction(OpenVLAConfig(**RELEASED_CONFIG))
    _convert_model_with_defuser(OpenVLAQModel, model, cleanup_original=False)
    state = model.state_dict()
    assert len(state) == 982
    canonical = "\n".join(f"{name} {list(value.shape)}" for name, value in sorted(state.items()))
    assert hashlib.sha256(canonical.encode()).hexdigest() == RELEASED_INVENTORY_SHA256
    # Last vision blocks/norms are owned even though the reference extracts the penultimate block.
    assert "vision_backbone.featurizer.blocks.23.ls1.scale_factor" in state
    assert "vision_backbone.fused_featurizer.blocks.26.attn.qkv.weight" in state
    assert "vision_backbone.featurizer.norm.weight" in state
