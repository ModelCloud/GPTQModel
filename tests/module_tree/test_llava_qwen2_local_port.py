# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.llava_qwen2 import LlavaQwen2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.llava_qwen2.configuration_llava_qwen2 import LlavaConfig
from gptqmodel.models.ports.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM
from tests.test_llava_qwen2_local_port import _config


PORT = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/llava_qwen2"
LINEARS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"],
               ["mlp.down_proj"]],
}


def test_exact_native_mapping_is_absent_and_all_registered_aliases_select_owned_definition():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "llava_qwen2" not in CONFIG_MAPPING_NAMES
    assert "llava_qwen2" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert [name for name, definition in MODEL_MAP.items() if definition is LlavaQwen2QModel] == ["llava_qwen2"]


@pytest.mark.parametrize("size", ["0.5B", "1.5B", "7B"])
def test_pinned_real_checkpoint_config_has_exact_flat_vision_and_decoder_owners_on_meta(size):
    payload = json.loads((PORT / f"FastVLM-{size}.json").read_text())
    config = LlavaConfig(**payload)
    with torch.device("meta"):
        model = LlavaQwen2ForCausalLM(config)
    _convert_model_with_defuser(LlavaQwen2QModel, model, cleanup_original=False)
    state = model.state_dict()
    original = json.loads((PORT / "checkpoint_shapes.json").read_text())["tensors"]
    vision = {name: tuple(shape) for name, shape in original.items() if name.startswith("model.vision_tower.")}
    assert len(vision) == 629
    assert {name: tuple(value.shape) for name, value in state.items() if name.startswith("model.vision_tower.")} == vision
    if size == "0.5B":
        assert {name: list(value.shape) for name, value in state.items()} == original
        assert len(state) == 924
    plan = ModulePlan.from_tree(LlavaQwen2QModel.module_tree)
    declared = {name for block in LlavaQwen2QModel.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == LINEARS
    expected = {name + ".weight" for name in LINEARS} | {"input_layernorm.weight", "post_attention_layernorm.weight"}
    expected |= {f"self_attn.{name}_proj.bias" for name in ("q", "k", "v")}
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, name
            if spec.quantize:
                assert owner in LINEARS
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert "!" in spec.flags
    assert {"model.embed_tokens", "model.norm", "model.rotary_emb", "model.mm_projector", "model.vision_tower"} <= set(
        LlavaQwen2QModel.get_base_modules(model))
    assert not any(name.startswith("language_model.") for name in state)


@pytest.mark.parametrize("tower", ["mobileclip", "clip"])
@pytest.mark.parametrize("merge", ["flat", "spatial", "spatial_unpad"])
def test_live_tiny_plan_excludes_vision_projector_norms_and_preserves_all_seven_decoder_linears(tower, merge):
    config = _config(tower, mm_patch_merge_type=merge)
    with torch.device("meta"):
        model = LlavaQwen2ForCausalLM(config)
    wrapper = LlavaQwen2QModel.__new__(LlavaQwen2QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    assert wrapper.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) == [
        [name for name in group if not name.endswith(":!")] for group in EXPECTED_PLAN["blocks"]
        if not all(name.endswith(":!") for name in group)]
    assert wrapper.awq_get_modules_for_scaling(model.model.layers[0], {}, {}) == []
    assert wrapper.pre_lm_head_norm_module == "model.norm"
    assert wrapper.rotary_embedding == "model.rotary_emb"
    assert ("model.image_newline" in model.state_dict()) == (merge == "spatial_unpad")


def test_central_registered_plan_is_unchanged():
    fixtures = Path(__file__).parent / "fixtures/registered_module_plans.json"
    assert json.loads(fixtures.read_text())["cases"]["LlavaQwen2QModel/gptq"] == EXPECTED_PLAN
