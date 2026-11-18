# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real ChatGLM state_dict checkpoint ownership, aliases and layout variants."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.chatglm import ChatGLMQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from tests.test_chatglm_local_port import _config


@pytest.mark.parametrize("checkpoint", ["THUDM/chatglm2-6b", "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat"])
def test_chatglm_pinned_checkpoint_exact_weight_contract(checkpoint):
    from gptqmodel.models.ports.chatglm.configuration_chatglm import ChatGLMConfig
    contracts = json.loads((Path(__file__).parents[1] / "fixtures/ports/chatglm/checkpoint_contracts.json").read_text())
    reference = contracts[checkpoint]
    config = ChatGLMConfig.from_dict(reference["config"])
    with torch.device("meta"):
        model = ChatGLMQModel.loader.from_config(config)
    # Digest of the sorted released safetensors key inventory at the pinned revision.
    keys = sorted(model.state_dict())
    assert len(keys) == reference["key_count"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == reference["keys_sha256"]
    plan = ModulePlan.from_tree(ChatGLMQModel.module_tree)
    declared = {name for group in ChatGLMQModel.simple_layer_modules(config, SimpleNamespace(dynamic=None))
                for name in group}
    for name in keys:
        if not name.startswith("transformer.encoder.layers."):
            continue
        relative = ".".join(name.split(".")[4:-1])
        owner = plan.find(relative)
        assert owner is not None, (checkpoint, name)
        if isinstance(model.get_submodule(".".join(name.split(".")[:-1])), torch.nn.Linear):
            assert relative in declared
        else:
            assert set(owner.flags) & {"!", "?"}


@pytest.mark.parametrize("rope_ratio", [None, 500])
@pytest.mark.parametrize("mqa", [False, True])
@pytest.mark.parametrize("rmsnorm", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_chatglm_real_state_dict_and_live_plan(rope_ratio, mqa, rmsnorm, bias):
    config = _config(rope_ratio=rope_ratio, multi_query_attention=mqa, rmsnorm=rmsnorm,
                     add_bias_linear=bias, add_qkv_bias=bias)
    with torch.device("meta"):
        model = ChatGLMForConditionalGeneration(config)
    _convert_model_with_defuser(ChatGLMQModel, model, cleanup_original=False)
    wrapper = ChatGLMQModel.__new__(ChatGLMQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    linears = {"self_attention.query_key_value", "self_attention.dense",
               "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"}
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == linears
    # Independently audited against THUDM/chatglm3-6b's original
    # pytorch_model.bin.index.json @ e9e0406d062cdb887444fe5bd546833920abd4ac.
    expected = {name + ".weight" for name in linears} | {
        "input_layernorm.weight", "post_attention_layernorm.weight",
    }
    if bias:
        expected |= {name + ".bias" for name in linears}
    if not rmsnorm:
        expected |= {"input_layernorm.bias", "post_attention_layernorm.bias"}
    global_keys = {"transformer.embedding.word_embeddings.weight", "transformer.output_layer.weight",
                   "transformer.rotary_pos_emb.inv_freq", "transformer.encoder.final_layernorm.weight"}
    if not rmsnorm:
        global_keys.add("transformer.encoder.final_layernorm.bias")
    all_keys = set(global_keys)
    for index, layer in enumerate(model.transformer.encoder.layers):
        assert set(layer.state_dict()) == expected
        all_keys |= {f"transformer.encoder.layers.{index}." + key for key in expected}
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in linears)
            if not spec.quantize:
                assert set(spec.flags) & {"!", "?"}
        for name in declared:
            assert isinstance(layer.get_submodule(name), nn.Linear)
            assert plan.find(name + ".undeclared") is None
        qkv = layer.self_attention.query_key_value
        assert qkv.out_features == config.kv_channels * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert set(model.state_dict()) == all_keys
    assert model.max_sequence_length == config.seq_length
    assert isinstance(model.all_tied_weights_keys, dict)


def test_chatglm_registry_alias_inventory_and_unchanged_plan():
    assert {name for name, definition in MODEL_MAP.items() if definition is ChatGLMQModel} == {"chatglm"}
    inventory = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    assert inventory["models"]["chatglm"] == "ChatGLMQModel"
    assert inventory["cases"]["ChatGLMQModel/gptq"] == {
        "layers": ["transformer.encoder.layers"],
        "blocks": [["input_layernorm:!"], ["self_attention.query_key_value"],
                   ["self_attention.dense"], ["post_attention_layernorm:!"],
                   ["mlp.dense_h_to_4h"], ["mlp.dense_4h_to_h"]],
    }
