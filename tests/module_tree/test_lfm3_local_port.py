# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native decoder ownership, including nonquantized convolution/gate weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Lfm3Config, Lfm3ForCausalLM

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.lfm3 import LFM3QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from tests.test_lfm3_local_port import shell, tiny_config


@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
@pytest.mark.parametrize("sandwich", [False, True])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("conv_bias", [False, True])
@pytest.mark.parametrize("attention_bias", [False, True])
def test_native_state_dict_owners_and_quantization_plan(kv_heads, sandwich, gate, conv_bias, attention_bias):
    config = tiny_config(num_key_value_heads=kv_heads, use_sandwich_norm=sandwich,
                         use_attention_gate=gate, conv_bias=conv_bias, attention_bias=attention_bias)
    with torch.device("meta"):
        model = Lfm3ForCausalLM(config)
    _convert_model_with_defuser(LFM3QModel, model, cleanup_original=False)
    wrapper = shell(model)
    plan = wrapper.module_plan()
    declared = {name for group in wrapper.simple_layer_modules(
        model_config=config, quantize_config=SimpleNamespace(dynamic=None)) for name in group}
    assert {name for name, definition in MODEL_MAP.items() if definition is LFM3QModel} == {"lfm3"}
    for layer_index, layer in enumerate(model.model.layers):
        expected = {"feed_forward.w1", "feed_forward.w3", "feed_forward.w2"}
        expected |= ({"conv.in_proj", "conv.out_proj"} if layer_index == 0 else
                     {f"self_attn.{projection}_proj" for projection in ("q", "k", "v", "o")})
        for owner in expected:
            projection = layer.get_submodule(owner)
            expected_bias = (conv_bias if owner.startswith("conv.") else
                             attention_bias if owner.startswith("self_attn.") else False)
            assert (projection.bias is not None) is expected_bias
        keys = layer.state_dict()
        assert ("post_operator_norm.weight" in keys) is sandwich
        assert ("post_ffn_norm.weight" in keys) is sandwich
        if layer_index:
            assert ("self_attn.g_proj.weight" in keys) is gate
        actual = set()
        for key in keys:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, f"unowned checkpoint tensor: {layer_index}.{key}"
            if spec.quantize:
                actual.add(owner)
                assert owner in declared
            else:
                assert "!" in spec.flags or "?" in spec.flags
        assert actual == expected
    assert plan.find("conv.conv") is not None
    assert not plan.find("conv.conv").quantize
    assert not plan.find("self_attn.g_proj").quantize
    assert not plan.find("self_attn.gate_proj").quantize
    assert plan.find("conv.conv.undeclared") is None
    assert plan.find("self_attn.g_proj_suffix") is None


def test_pinned_checkpoint_shapes_and_conversion_owners():
    manifest = json.loads((Path(__file__).parents[2] /
                           "gptqmodel/models/ports/lfm3/checkpoint_manifest.json").read_text())
    config = Lfm3Config.from_dict(manifest["config"])
    with torch.device("meta"):
        model = Lfm3ForCausalLM(config)
    wrapper = shell(model)
    expected = {key.replace(".gate_proj.", ".g_proj."): tuple(shape)
                for key, shape in manifest["checkpoint_shapes"].items()}
    assert len(expected) == 286
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == {
        **expected, "lm_head.weight": expected["model.embed_tokens.weight"]}
    for key in expected:
        if not key.startswith("model.layers."):
            continue
        owner = key.split(".", 3)[3].rsplit(".", 1)[0]
        spec = wrapper.module_plan().find(owner)
        assert spec is not None, f"unowned checkpoint tensor: {key}"
        if not spec.quantize:
            assert "!" in spec.flags
