# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import ApertusConfig, ApertusForCausalLM

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.apertus import ApertusQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from tests.test_apertus_local_port import KINDS, tiny_model


LINEARS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.up_proj", "mlp.down_proj"}
DENSE = {"attention_layernorm", "feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm", "mlp.act_fn"}
BLOCKS = [
    ["attention_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm:!", "self_attn.k_norm:!"],
    ["self_attn.o_proj"], ["feedforward_layernorm:!"], ["mlp.up_proj", "mlp.act_fn:!"], ["mlp.down_proj"],
]


def test_native_apertus_checkpoint_owners_are_declared():
    config = ApertusConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    model = ApertusForCausalLM(config)
    plan = ModulePlan.from_tree(ApertusQModel.module_tree)
    for key in model.model.layers[0].state_dict():
        owner = key.rsplit(".", 1)[0]
        assert plan.find(owner) is not None, key


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hidden_act", ["xielu", "silu"])
@pytest.mark.parametrize("attention_bias", [False, True])
def test_exact_real_meta_checkpoint_ownership_and_consumer_plan(kind, hidden_act, attention_bias):
    definition = MODEL_MAP[kind.removesuffix("_pruned")]
    with torch.device("meta"):
        model = tiny_model(kind, hidden_act=hidden_act, attention_bias=attention_bias)
    before = set(model.state_dict())
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    assert set(model.state_dict()) == before
    prefix = "model.language_model.layers" if kind.removesuffix("_pruned") == "apertus1p5" else "model.layers"
    assert definition.extract_layers_node() == [prefix]
    assert definition.build_layer_modules(definition.module_tree, include_capture_only=True) == BLOCKS
    for method in ("gptq", "awq"):
        qc = SimpleNamespace(dynamic=None, method=method)
        quantized = {name for block in definition.simple_layer_modules(model.config, qc, is_awq_quantize=method == "awq") for name in block}
        assert quantized == LINEARS
    plan = ModulePlan.from_tree(definition.module_tree)
    for layer in model.get_submodule(prefix):
        expected_keys = {f"{name}.weight" for name in LINEARS | (DENSE - {"mlp.act_fn"})}
        if attention_bias:
            expected_keys |= {f"self_attn.{p}_proj.bias" for p in ("q", "k", "v", "o")}
        if hidden_act == "xielu":
            expected_keys |= {f"mlp.act_fn.{k}" for k in ("alpha_p", "alpha_n", "beta", "eps")}
        assert set(layer.state_dict()) == expected_keys
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in LINEARS)
            if owner not in LINEARS:
                assert "!" in spec.flags or "?" in spec.flags
        assert layer.self_attn.q_proj.weight.shape == (16, 16)
        assert layer.self_attn.k_proj.weight.shape == (8, 16)
        assert layer.self_attn.v_proj.weight.shape == (8, 16)
        for name in LINEARS:
            assert isinstance(layer.get_submodule(name), torch.nn.Linear)
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            assert "in=x" in plan.find(name).flags
        assert "nosrc" in plan.find("mlp.up_proj").flags
        for bad in ("self_attn.q_norm_extra", "self_attn.q_proj.fake", "mlp.gate_proj"):
            assert plan.find(bad) is None
    assert model.lm_head.weight.shape[0] == (48 if kind.endswith("_pruned") else 96)
    assert model.get_input_embeddings().weight.shape == (96, 16)
    if kind.removesuffix("_pruned") == "apertus1p5":
        assert {key.split(".")[1] for key in before if key.startswith("model.")} == {
            "language_model", "vision_tokenizer", "audio_tokenizer",
        }
        assert any("parametrizations.weight.original" in key for key in before)
        assert "model.audio_tokenizer.quantizer.codebook.embed" in before
