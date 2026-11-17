# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.qwen_drive import QwenDriveForConditionalGeneration
from gptqmodel.quantization.config import AWQConfig
from tests.test_qwen_drive_local_port import tiny_config


def test_explicit_released_plan_proposal_and_input_tags():
    fixture = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/qwen_drive/central_fixture_proposals.json"
    expected = json.loads(fixture.read_text())["registered_module_plans"]["cases"]["QwenDriveQModel/gptq"]
    tree = QwenDriveQModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert QwenDriveQModel.extract_layers_node() == expected["layers"]
    assert QwenDriveQModel.build_layer_modules(tree, include_capture_only=True) == expected["blocks"]
    model = QwenDriveForConditionalGeneration(tiny_config())
    wrapper = QwenDriveQModel(model, quantized=False, quantize_config=AWQConfig(device="cpu", group_size=16))
    plan = wrapper.module_plan()
    for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv",
                 "linear_attn.in_proj_z", "mlp.gate_proj", "mlp.up_proj"):
        assert "in=x" in plan.find(path).flags
    assert not plan.find("linear_attn").quantize


@pytest.mark.parametrize("layout", ["hybrid", "linear", "full"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_live_decoder_owners_plan_tags_aliases_and_plain_awq(layout, kv_heads, bias, device):
    with torch.device(device):
        model = QwenDriveForConditionalGeneration(tiny_config(layout, kv_heads, bias))
    _convert_model_with_defuser(QwenDriveQModel, model, cleanup_original=False)
    wrapper = QwenDriveQModel(model, quantized=False, quantize_config=AWQConfig(device="cpu", group_size=16))
    plan = wrapper.module_plan()
    assert wrapper.extract_layers_node() == ["vlm.model.language_model.layers"]
    assert wrapper.awq_get_modules_for_scaling(None, None, None) == []
    assert wrapper.fusable is None
    resolved = set()
    for layer in model.vlm.model.language_model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if "." in key else ""
            spec = plan.find(owner)
            assert spec is not None, key
            resolved.add(owner)
            if owner.startswith("linear_attn") and owner not in {
                "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            }:
                assert not spec.quantize, key
        for prefix in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert plan.find(prefix).quantize
    assert "mlp.gate_proj" in resolved
    assert plan.find("self_attn.q_proj_extra") is None
    assert plan.find("mlp.down_proj.extra") is None
    assert plan.find("planning_expert") is None
