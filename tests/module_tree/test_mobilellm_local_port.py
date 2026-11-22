# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import itertools
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.mobilellm import MobileLLMQModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.mobilellm import MobileLLMConfig, MobileLLMForCausalLM


EXPECTED_BLOCKS = [
    ["input_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


@pytest.mark.parametrize("device,kv,bias,sharing,shared_embedding", list(itertools.product(
    ["cpu", "meta"], [1, 2, 4], [False, True], [False, True], [False, True])))
def test_live_weight_owners_and_plan(device, kv, bias, sharing, shared_embedding):
    config = MobileLLMConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv,
                            attention_bias=bias, mlp_bias=bias,
                            layer_sharing=sharing, share_embedding=shared_embedding)
    with torch.device(device):
        model = MobileLLMForCausalLM(config)
    _convert_model_with_defuser(MobileLLMQModel, model, cleanup_original=False)
    wrapper = MobileLLMQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    if not shared_embedding:
        expected.add("lm_head.weight")
    linears = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v",
               "self_attn.o_proj": "o", "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "down"}
    for index, layer in enumerate(model.model.layers):
        root = f"model.layers.{index}."
        for path, role in linears.items():
            module = layer.get_submodule(path)
            assert isinstance(module, torch.nn.Linear)
            spec = plan.find(path)
            assert spec is not None and spec.quantize
            assert role in spec.flags
            assert plan.find(path + ".undeclared") is None
            expected.add(root + path + ".weight")
            if bias:
                expected.add(root + path + ".bias")
        for path in ["input_layernorm", "post_attention_layernorm"]:
            spec = plan.find(path)
            assert spec is not None and not spec.quantize and "norm" in spec.flags
            expected.add(root + path + ".weight")
        assert layer.self_attn.k_proj.out_features == kv * config.head_dim
        assert layer.self_attn.v_proj.out_features == kv * config.head_dim
    assert set(model.state_dict()) == expected
    assert len(model.model.layers) == 2  # Sharing repeats execution, not checkpoint owners.
    assert MODEL_MAP["mobilellm"] is MobileLLMQModel
    with pytest.raises(ValueError):
        require_fusable_config(MobileLLMQModel, "mobilellm")
    tree = MobileLLMQModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert MobileLLMQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_BLOCKS


@pytest.mark.parametrize("kv,bias,sharing", list(itertools.product([1, 2, 4], [False, True], [False, True])))
def test_awq_scale_folds_preserve_every_decoder_layer_output(kv, bias, sharing):
    # A fold reparametrizes one physical layer, so it is exact for every shared use.
    torch.manual_seed(3)
    config = MobileLLMConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv,
                            attention_bias=bias, mlp_bias=bias, layer_sharing=sharing,
                            max_position_embeddings=32, pad_token_id=0)
    model = MobileLLMForCausalLM(config).eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm" in name:
                value.uniform_(0.5, 1.5)
    _convert_model_with_defuser(MobileLLMQModel, model, cleanup_original=False)
    ids = torch.tensor([[1, 4, 9, 13, 7, 5, 19, 8]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    # v_proj -> o_proj is only exact without GQA head repetition.
    expected = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], [])]
    if kv == 4:
        expected.append(("self_attn.v_proj", ["self_attn.o_proj"], []))
    expected += [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], []),
                 ("mlp.up_proj", ["mlp.down_proj"], [])]
    for layer in model.model.layers:
        assert assert_awq_folds_neutral(MobileLLMQModel, model, layer, forward) == expected


@pytest.mark.parametrize("hidden,intermediate,layers,heads,kv", [
    (576, 1536, 30, 9, 3), (960, 2560, 32, 15, 5), (1152, 3072, 40, 18, 6),
    (1280, 3584, 54, 20, 5), (1600, 4352, 54, 25, 5),
])
@pytest.mark.parametrize("sharing", [False, True])
def test_official_training_config_geometry_meta(hidden, intermediate, layers, heads, kv, sharing):
    # Dimensions pinned from the authoritative GitHub configs, not HF tensor headers.
    config = MobileLLMConfig(hidden_size=hidden, intermediate_size=intermediate,
                            num_hidden_layers=layers, num_attention_heads=heads,
                            num_key_value_heads=kv, layer_sharing=sharing)
    with torch.device("meta"):
        model = MobileLLMForCausalLM(config)
    assert model.model.embed_tokens.weight.shape == (32000, hidden)
    assert len(model.state_dict()) == 9 * layers + 2
    for layer in model.model.layers:
        assert layer.self_attn.q_proj.weight.shape == (hidden, hidden)
        assert layer.self_attn.k_proj.weight.shape == (kv * 64, hidden)
        assert layer.mlp.gate_proj.weight.shape == (intermediate, hidden)
