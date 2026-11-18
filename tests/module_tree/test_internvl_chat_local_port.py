# SPDX-License-Identifier: Apache-2.0
"""Inventory real decoder checkpoint owners after normal Defuser conversion."""

import re

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from test_internvl_chat_local_port import ARCHITECTURES, tiny_config

from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.internvl_chat.modeling_internvl_chat import InternVLChatModel
from gptqmodel.quantization import QuantizeConfig


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_checkpoint_owners_and_quantization_consumers(architecture):
    model = InternVLChatModel(tiny_config(architecture)).eval()
    ids = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        before = model(input_ids=ids).logits
    _convert_model_with_defuser(InternVLChatQModel, model, cleanup_original=False)
    tree = InternVLChatQModel._resolve_effective_module_tree(model, None)
    model.gptqmodel_effective_module_tree = tree
    qmodel = InternVLChatQModel(model, quantized=False, quantize_config=QuantizeConfig(bits=4, group_size=8))
    plan = qmodel.module_plan()
    quantizable = {name for block in qmodel.simple_layer_modules(model.config, qmodel.quantize_config) for name in block}
    assert quantizable
    resolved = set()
    for layer in model.language_model.model.layers:
        # state_dict includes persistent buffers as well as parameters.
        for key in layer.state_dict():
            owner = key.rpartition(".")[0]
            spec = plan.find(owner)
            assert spec is not None, (architecture, owner)
            if spec.quantize:
                assert owner in quantizable, (architecture, owner)
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                resolved.add(owner)
            else:
                assert "!" in spec.flags or spec.capture_only
            assert plan.find(owner + ".undeclared") is None
    assert resolved == quantizable
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, before, atol=1e-6, rtol=1e-5)
    if architecture == "Qwen3MoeForCausalLM":
        assert qmodel.get_num_experts(model.config) == 2
        assert plan.moe_roots()
        assert plan.is_expert("mlp.experts.1.gate_proj")
        assert not plan.find("mlp.gate").quantize
        # The routed gate/up AWQ group replays the MoE root from its captured input.
        assert qmodel.awq_input_feature_aggregation("mlp") == {"mode": "token_rows", "capture_root": True}
    else:
        assert not plan.moe_roots()
        assert qmodel.awq_input_feature_aggregation("mlp") == {"mode": "token_rows"}
    assert qmodel.awq_input_feature_aggregation("mlp.experts.0.gate_proj") == {"mode": "token_rows"}
    assert re.fullmatch(r"language_model.model.layers", qmodel.extract_layers_node()[0])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_prequant_hooks_preserve_embedding_and_rotary_owners(architecture, monkeypatch):
    model = InternVLChatModel(tiny_config(architecture)).eval()
    qmodel = InternVLChatQModel(model, quantized=False, quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False))
    visited = []
    def prequant(module):
        visited.append(module)
        return module
    monkeypatch.setattr(qmodel, "pre_quantize", prequant)
    qmodel.pre_quantize_generate_hook_start()
    qmodel.pre_quantize_generate_hook_end()
    assert model.get_input_embeddings() is visited[0]
    rotary_paths = qmodel._decoder_rotary_paths()
    assert len(rotary_paths) == (2 if architecture == "InternLM2ForCausalLM" else 1)
    assert all(model.get_submodule(path) in visited for path in rotary_paths)


ATTENTION = ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
VALUE = ("self_attn.v_proj", ["self_attn.o_proj"])
MLP = ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"])
DOWN = ("mlp.up_proj", ["mlp.down_proj"])
MOE = ("post_attention_layernorm",
       ["mlp.experts.0.gate_proj", "mlp.experts.0.up_proj", "mlp.experts.1.gate_proj", "mlp.experts.1.up_proj"])
# Each routed expert also folds its own up_proj -> down_proj.
EXPERT_DOWN = [(f"mlp.experts.{index}.up_proj", [f"mlp.experts.{index}.down_proj"]) for index in range(2)]
EXPECTED_AWQ_GROUPS = {
    # kv=1 is GQA (v_proj narrower than o_proj); kv=2 is MHA, where v_proj -> o_proj is exact.
    ("LlamaForCausalLM", 1): [ATTENTION, MLP, DOWN],
    ("LlamaForCausalLM", 2): [ATTENTION, VALUE, MLP, DOWN],
    ("Qwen2ForCausalLM", 1): [ATTENTION, MLP, DOWN],
    ("Qwen2ForCausalLM", 2): [ATTENTION, VALUE, MLP, DOWN],
    # Qwen3 q_norm/k_norm act per head after the projections, so input scaling is exact.
    ("Qwen3ForCausalLM", 1): [ATTENTION, MLP, DOWN],
    ("Qwen3ForCausalLM", 2): [ATTENTION, VALUE, MLP, DOWN],
    # The router reads the same normed input and is compensated as a scale consumer.
    ("Qwen3MoeForCausalLM", 1): [ATTENTION, MOE, *EXPERT_DOWN],
    ("Qwen3MoeForCausalLM", 2): [ATTENTION, VALUE, MOE, *EXPERT_DOWN],
    # Fused/interleaved QKV never sources o_proj/wo: its output is not o_proj's input.
    ("Phi3ForCausalLM", 1): [("input_layernorm", ["self_attn.qkv_proj"]), MLP, DOWN],
    ("Phi3ForCausalLM", 2): [("input_layernorm", ["self_attn.qkv_proj"]), MLP, DOWN],
    ("InternLM2ForCausalLM", 1): [("attention_norm", ["attention.wqkv"]),
                                  ("ffn_norm", ["feed_forward.w1", "feed_forward.w3"]),
                                  ("feed_forward.w3", ["feed_forward.w2"])],
    ("InternLM2ForCausalLM", 2): [("attention_norm", ["attention.wqkv"]),
                                  ("ffn_norm", ["feed_forward.w1", "feed_forward.w3"]),
                                  ("feed_forward.w3", ["feed_forward.w2"])],
}


@pytest.mark.parametrize("kv", [1, 2])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_awq_scale_folds_preserve_decoder_logits(architecture, kv):
    torch.manual_seed(3)
    model = InternVLChatModel(tiny_config(architecture, llm={"num_key_value_heads": kv})).eval()
    _convert_model_with_defuser(InternVLChatQModel, model, cleanup_original=False)
    with torch.no_grad():
        # Non-unit norm gains so a fold into the wrong norm cannot cancel out.
        for name, parameter in model.language_model.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.linspace(0.6, 1.4, parameter.numel()).reshape_as(parameter))
    tree = InternVLChatQModel._resolve_effective_module_tree(model, None)
    ids = torch.tensor([[7, 8, 9, 10, 11, 12]])

    def forward():
        return model(input_ids=ids).logits

    for layer in model.language_model.model.layers:
        checked = assert_awq_folds_neutral(InternVLChatQModel, model, layer, forward, tree=tree)
        assert [(source, targets) for source, targets, _ in checked] == EXPECTED_AWQ_GROUPS[architecture, kv]
        consumers = {source: names for source, _, names in checked}
        if architecture == "Qwen3MoeForCausalLM":
            assert consumers["post_attention_layernorm"] == ["mlp.gate"]
