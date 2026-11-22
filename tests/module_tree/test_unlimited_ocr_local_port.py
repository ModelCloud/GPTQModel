# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent released names/shapes and real decoder ownership."""

import json
import re

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
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


@pytest.mark.parametrize("layout", ["mixed", "sparse"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_awq_scale_folds_preserve_dense_and_moe_text_layers(layout, kv_heads, bias):
    torch.manual_seed(7)
    # Native eager attention only supports MHA/MQA; GQA runs through SDPA.
    implementation = "sdpa" if kv_heads == 2 else "eager"
    model = UnlimitedOCRQModel.loader.from_config(tiny_config(kv_heads, bias, layout=layout),
                                                  attn_implementation=implementation).eval()
    update_module(model, cleanup_original=False)
    text = model.model.language_model
    with torch.no_grad():
        # Trained-like gains and a non-degenerate router (native init is ones
        # and zeros, which would hide a missing router compensation).
        for layer in text.layers:
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.weight.normal_()
    ids = torch.randint(4, 60, (1, 16), generator=torch.Generator().manual_seed(1))

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    for layer in text.layers:
        checked = assert_awq_folds_neutral(UnlimitedOCRQModel, model, layer, forward)
        expected = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])]
        if kv_heads == 4:
            # Grouped K/V make v_proj narrower than o_proj's input.
            expected.append(("self_attn.v_proj", ["self_attn.o_proj"]))
        if hasattr(layer.mlp, "gate"):
            owners = ["mlp.shared_experts"] + [f"mlp.experts.{expert}" for expert in range(4)]
            expected.append(("post_attention_layernorm", [f"{owner}.{name}" for owner in owners
                                                          for name in ("gate_proj", "up_proj")]))
            expected += [(f"{owner}.up_proj", [f"{owner}.down_proj"]) for owner in owners]
        else:
            expected += [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
                         ("mlp.up_proj", ["mlp.down_proj"])]
        assert [(source, targets) for source, targets, _ in checked] == expected
        consumers = {source: consumers for source, _, consumers in checked}
        assert consumers["post_attention_layernorm"] == (["mlp.gate"] if hasattr(layer.mlp, "gate") else [])
