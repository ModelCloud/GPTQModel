# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual native hybrid decoder owners and full released C-RADIO inventory."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.zdtaichu5 import ZDTaichu5QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.zdtaichu5.modeling import ZDTaichu5_0_ForConditionalGeneration
from tests.test_zdtaichu5_local_port import tiny_config


FULL = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
LINEAR = {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"}
MLP = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
EXCLUDED = {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm",
            "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.norm", "linear_attn.conv1d"}


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layout", ["full", "linear", "hybrid"])
def test_weight_owners_defuser_aliases_and_conservative_policy(device, kv_heads, bias, layout):
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
    assert ZDTaichu5QModel.awq_get_modules_for_scaling(None, model.language_model.model.layers[0], {}, {}) == []
