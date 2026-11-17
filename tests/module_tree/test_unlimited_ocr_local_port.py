# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent released names/shapes and real decoder ownership."""

import json
import re

import pytest
import torch
from defuser.modeling.update_module import update_module
from transformers import UnlimitedOcrConfig, UnlimitedOcrForConditionalGeneration

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.unlimited_ocr import UnlimitedOCRQModel
from gptqmodel.models.module_plan import ModulePlan
from tests.test_unlimited_ocr_local_port import asset, legacy_key, tiny_config


def test_complete_real_header_native_meta_inventory():
    config = UnlimitedOcrConfig.from_dict(json.loads(asset("config.json")))
    with torch.device("meta"):
        model = UnlimitedOcrForConditionalGeneration(config)
    update_module(model, cleanup_original=False)
    header = json.loads(asset("model-00001-of-000001.safetensors.header.json"))
    header.pop("__metadata__")
    assert len(header) == 2710
    expected = {}
    for name, tensor in model.state_dict().items():
        key = legacy_key(name)
        shape = list(tensor.shape)
        if key.startswith("model.vision_model.") and re.search(r"[qkv]_proj\.(weight|bias)$", key):
            if ".q_proj." not in key:
                continue
            key = key.replace(".q_proj.", ".qkv_proj.")
            shape[0] *= 3
        expected[key] = shape
    unused = header.pop("model.vision_model.embeddings.patch_embedding.weight")
    assert unused["shape"] == [1024, 3, 14, 14]
    assert expected == {name: info["shape"] for name, info in header.items()}


@pytest.mark.parametrize("layout", ["dense", "sparse", "mixed"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_live_module_plan_all_layout_owners(layout, kv_heads, bias, device):
    config = tiny_config(kv_heads, bias, layout=layout)
    with torch.device(device):
        model = UnlimitedOCRQModel.loader.from_config(config, attn_implementation="sdpa")
    update_module(model, cleanup_original=False)
    plan = ModulePlan.from_tree(UnlimitedOCRQModel.module_tree)
    for layer in model.model.language_model.layers:
        for name, value in layer.state_dict().items():
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, owner
            excluded = owner in {"input_layernorm", "post_attention_layernorm", "mlp.gate"}
            assert spec.quantize is not excluded
            assert value.device.type == device
    assert plan.find("self_attn.q_proj.unknown") is None
    assert plan.find("mlp.experts.0.unknown") is None
    assert MODEL_MAP["unlimited_ocr"] is MODEL_MAP["unlimited-ocr"] is UnlimitedOCRQModel
    assert UnlimitedOCRQModel.extract_layers_node() == ["model.language_model.layers"]
    assert UnlimitedOCRQModel.fusable is None


def test_explicit_plain_awq_policy():
    wrapper = object.__new__(UnlimitedOCRQModel)
    model = UnlimitedOcrForConditionalGeneration(tiny_config())
    for layer in model.model.language_model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
