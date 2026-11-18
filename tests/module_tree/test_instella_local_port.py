# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.instella import InstellaQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.instella import InstellaConfig, InstellaForCausalLM
from tests.test_instella_local_port import tiny_model


LINEARS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}
NORMS = {"pre_attention_layernorm", "pre_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"}


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_instella_real_checkpoint_weight_owners(heads, bias, device):
    config = InstellaConfig(
        vocab_size=47, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=heads, attention_bias=bias,
    )
    with torch.device(device):
        model = InstellaForCausalLM(config)
    assert [key for key, value in MODEL_MAP.items() if value is InstellaQModel] == ["instella"]
    wrapper = InstellaQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert isinstance(plan, ModulePlan)
    quantizable = {path for block in wrapper.simple_layer_modules(config, None) for path in block}
    assert quantizable == LINEARS
    expected = {path + ".weight" for path in LINEARS | NORMS}
    if bias:
        expected |= {path + ".bias" for path in LINEARS if path.startswith("self_attn.")}
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in LINEARS)
            if owner in NORMS:
                assert "!" in spec.flags
        for role in ("q", "k", "v", "o", "gate", "up", "down"):
            path = ("self_attn." if role in ("q", "k", "v", "o") else "mlp.") + role + "_proj"
            assert role in plan.find(path).flags
    assert set(model.state_dict()) == {
        *(f"model.layers.{idx}.{key}" for idx in range(2) for key in expected),
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    }
    for path in ("self_attn.q_norm.fake", "self_attn.q_proj_extra", "fake.self_attn.k_proj"):
        assert plan.find(path) is None


def test_instella_production_checkpoint_index_and_registry_contract():
    # The pinned AMD safetensors index has these 11 owners per layer and three
    # root tensors (399 total), no persistent rotary tensors or biases.
    config = InstellaConfig(
        vocab_size=50304, hidden_size=2560, intermediate_size=6912,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=4096, bos_token_id=0, eos_token_id=0,
    )
    with torch.device("meta"):
        model = InstellaForCausalLM(config)
    expected = {
        *(f"model.layers.{idx}.{owner}.weight" for idx in range(36) for owner in LINEARS | NORMS),
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
    }
    assert len(expected) == 399
    assert set(model.state_dict()) == expected
    # Independently extracted from the pinned upstream safetensors index:
    # sha256("\n".join(sorted(index["weight_map"])).encode()).
    digest = hashlib.sha256("\n".join(sorted(model.state_dict())).encode()).hexdigest()
    assert digest == "f21aaded620da6bf22f2a85f3328c144644250e0975533cef7d1a56ea5185a1a"
    assert model.model.layers[0].self_attn.q_norm.weight.shape == (2560,)
    assert InstellaQModel.build_layer_modules(InstellaQModel.module_tree, include_capture_only=True) == [
        ["pre_attention_layernorm:!"],
        ["self_attn.q_norm:!", "self_attn.k_norm:!", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["pre_feedforward_layernorm:!"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


# The full-width Q/K RMSNorms sit after q_proj/k_proj, so scaling their input
# columns leaves the normalized queries/keys unchanged. Equal KV heads also
# qualify v_proj -> o_proj, which the released GQA checkpoint cannot emit.
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_awq_scale_folds_preserve_layer_output(kv_heads, bias):
    model = tiny_model(num_key_value_heads=kv_heads, attention_bias=bias)
    ids = torch.tensor([[2, 4, 8, 6, 9, 11, 5, 7]])

    def forward():
        return model(ids, use_cache=False).logits

    expected = [("pre_attention_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])]
    if kv_heads == 4:
        expected.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    expected += [("pre_feedforward_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
                 ("mlp.up_proj", ["mlp.down_proj"])]
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(InstellaQModel, model, layer, forward)
        assert [(source, targets) for source, targets, _ in checked] == expected
