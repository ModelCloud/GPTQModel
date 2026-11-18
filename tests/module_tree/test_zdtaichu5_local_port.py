# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual native hybrid decoder owners and full released C-RADIO inventory."""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.zdtaichu5 import ZDTaichu5QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.zdtaichu5.modeling import ZDTaichu5_0_ForConditionalGeneration
from tests.test_zdtaichu5_local_port import tiny_config, visual_inputs


FULL = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
LINEAR = {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"}
MLP = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
EXCLUDED = {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm",
            "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.norm", "linear_attn.conv1d"}


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv_heads,bias", [(1, False), (2, True), (4, False)])
@pytest.mark.parametrize("layout", ["full", "linear", "hybrid"])
def test_weight_owners_and_defuser_aliases(device, kv_heads, bias, layout):
    config = tiny_config(kv_heads, bias, layout)
    with torch.device(device):
        model = ZDTaichu5_0_ForConditionalGeneration(config)
    _convert_model_with_defuser(ZDTaichu5QModel, model, cleanup_original=False)
    plan = ModulePlan.from_tree(ZDTaichu5QModel.module_tree)
    quantizable = {name for block in ZDTaichu5QModel.simple_layer_modules(config, SimpleNamespace(dynamic=None))
                   for name in block}
    assert quantizable == FULL | LINEAR | MLP
    for index, layer in enumerate(model.language_model.model.layers):
        projection = (FULL if config.llm_config.layer_types[index] == "full_attention" else LINEAR) | MLP
        actual = {name for name, item in layer.named_modules() if isinstance(item, nn.Linear)}
        dense = actual & EXCLUDED
        assert actual - dense == projection
        for key, value in layer.state_dict().items():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if owner in projection:
                assert spec.quantize and "!" not in spec.flags
            elif owner != "linear_attn":
                assert not spec.quantize and "!" in spec.flags
            else:
                assert key in {"linear_attn.A_log", "linear_attn.dt_bias"}
            assert value.device.type == device
        for name in projection:
            assert plan.find(name + ".foreign") is None
    base = ZDTaichu5QModel.get_base_modules(model)
    assert {"vision_model", "mlp1"}.issubset(base)
    assert model.get_submodule(ZDTaichu5QModel.lm_head) is model.language_model.lm_head
    assert model.get_submodule(ZDTaichu5QModel.pre_lm_head_norm_module) is model.language_model.model.norm
    assert model.get_submodule(ZDTaichu5QModel.rotary_embedding) is model.language_model.model.rotary_emb
    assert ZDTaichu5QModel.extract_layers_node() == ["language_model.model.layers"]
    assert {name for name, definition in MODEL_MAP.items() if definition is ZDTaichu5QModel} == {"zdtaichu5_0"}
    assert ZDTaichu5QModel.fusable is None


@pytest.mark.parametrize("kv_heads,bias", [(4, False), (2, True), (1, False)])
@pytest.mark.parametrize("layout", ["full", "linear", "hybrid"])
def test_awq_scale_folds_preserve_hybrid_layer_output(kv_heads, bias, layout):
    torch.manual_seed(71)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config(kv_heads, bias, layout)).eval()
    _convert_model_with_defuser(ZDTaichu5QModel, model, cleanup_original=False)
    tree = ZDTaichu5QModel._resolve_effective_module_tree(model, None)
    # Run the real wrapper with an injected image so the decoder layer sees
    # the projected visual tokens it is calibrated on.
    inputs = visual_inputs()

    def forward():
        return model(**inputs, use_cache=False).logits

    mlp = [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], []),
           ("mlp.up_proj", ["mlp.down_proj"], [])]
    for index, layer in enumerate(model.language_model.model.layers):
        checked = assert_awq_folds_neutral(ZDTaichu5QModel, model, layer, forward, tree=tree)
        if config_layer_type(model, index) == "linear_attention":
            # DeltaNet's b/a gates read the same normed input and are
            # compensated; out_proj reads a gated norm and has no source.
            attention = [("input_layernorm", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
                          ["linear_attn.in_proj_b", "linear_attn.in_proj_a"])]
        else:
            attention = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], [])]
            if kv_heads == 4:
                # The sigmoid output gate multiplies per channel, so v -> o
                # stays exact; GQA leaves v_proj narrower than o_proj.
                attention.append(("self_attn.v_proj", ["self_attn.o_proj"], []))
        assert checked == attention + mlp


def config_layer_type(model, index):
    return model.config.llm_config.layer_types[index]
