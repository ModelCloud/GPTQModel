# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-owned InternLM 1 weight owners resolve in the live module plan."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
from tests.test_internlm_local_port import _config


@pytest.mark.parametrize("bias", [False, True])
def test_internlm_real_meta_model_owns_every_declared_decoder_weight(bias):
    config = _config(bias=bias)
    with torch.device("meta"):
        model = InternLMForCausalLM(config)
    # Exercise the same conversion boundary as normal model loading.
    _convert_model_with_defuser(InternLMQModel, model, cleanup_original=False)
    wrapper = InternLMQModel.__new__(InternLMQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    expected_linears = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == expected_linears
    # Names independently inspected in internlm/internlm-7b's original
    # pytorch_model.bin.index.json at revision 96e127d08d851a88cac736a9b091dd953ae1b873.
    expected_keys = {name + ".weight" for name in expected_linears} | {
        "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.rotary_emb.inv_freq"}
    if bias:
        expected_keys |= {"self_attn." + name + ".bias" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected_keys
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
            else:
                assert set(spec.flags) & {"!", "?"}
        for name in declared:
            assert isinstance(layer.get_submodule(name), nn.Linear)
            assert plan.find(name + ".undeclared") is None
    assert wrapper.awq_get_modules_for_scaling(model.model.layers[0], {}, {}) == []
