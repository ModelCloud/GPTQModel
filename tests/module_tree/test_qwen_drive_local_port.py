# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from eager_ssm_kernels import eager_torch_ssm_kernels

from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.qwen_drive import QwenDriveForConditionalGeneration
from gptqmodel.quantization.config import AWQConfig
from tests.test_qwen_drive_local_port import tiny_config


def test_explicit_released_plan_proposal_and_input_tags():
    fixtures = Path(__file__).resolve().parent / "fixtures"
    registry = json.loads((fixtures / "registered_module_plans.json").read_text())
    assert registry["models"]["qwen_drive"] == "QwenDriveQModel"
    assert json.loads((fixtures / "fusable_models.json").read_text())["models"]["qwen_drive"]["definition"] == "QwenDriveQModel"
    expected = registry["cases"]["QwenDriveQModel/gptq"]
    assert expected["layers"] == ["vlm.model.language_model.layers"]
    assert ["linear_attn:!"] in expected["blocks"]
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


@pytest.mark.parametrize("layout,kv_heads,bias,device", [
    ("hybrid", 2, False, "cpu"), ("hybrid", 4, True, "meta"), ("linear", 1, True, "cpu"), ("full", 1, False, "meta"),
])
def test_live_decoder_owners_plan_tags_and_aliases(layout, kv_heads, bias, device):
    with torch.device(device):
        model = QwenDriveForConditionalGeneration(tiny_config(layout, kv_heads, bias))
    _convert_model_with_defuser(QwenDriveQModel, model, cleanup_original=False)
    wrapper = QwenDriveQModel(model, quantized=False, quantize_config=AWQConfig(device="cpu", group_size=16))
    plan = wrapper.module_plan()
    assert wrapper.extract_layers_node() == ["vlm.model.language_model.layers"]
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


@pytest.mark.parametrize("layout,kv_heads", [("hybrid", 2), ("hybrid", 4), ("full", 4), ("linear", 2)])
def test_awq_scale_folds_preserve_real_media_decoder_output(layout, kv_heads):
    from awq_neutrality import assert_awq_folds_neutral

    from tests.test_qwen_drive_local_port import media_inputs

    torch.manual_seed(5)
    model = QwenDriveForConditionalGeneration(tiny_config(layout, kv_heads)).eval()
    inputs = media_inputs()

    def forward():
        with eager_torch_ssm_kernels():
            return model(**inputs, use_cache=False).logits

    mlp = {("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")), ("mlp.up_proj", ("mlp.down_proj",))}
    for layer, kind in zip(model.vlm.model.language_model.layers, model.config.vlm_config.text_config.layer_types):
        checked = assert_awq_folds_neutral(QwenDriveQModel, model, layer, forward)
        groups = {(source, tuple(targets)) for source, targets, _ in checked}
        if kind == "full_attention":
            expected = mlp | {("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))}
            if kv_heads == 4:
                expected.add(("self_attn.v_proj", ("self_attn.o_proj",)))
        else:
            # Gated DeltaNet keeps its projections unscaled: conv1d, recurrence
            # and the gated norm separate them from any foldable source.
            expected = mlp
        assert groups == expected
