# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Instella must declare its own decoder norm names.

Issue #4093 item 1: Instella inherits ``LlamaQModel`` but its decoder layer
names the norms ``pre_attention_layernorm`` / ``pre_feedforward_layernorm``
(``modeling_instella.py:663-664``). The inherited Llama tree therefore found
no scaling source for q/k/v and folded ``mlp.gate/up`` into
``self_attn.o_proj`` across the residual add (PPL 14616 vs RTN 27.71).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gptqmodel.models.definitions import LlamaQModel
from gptqmodel.models.definitions.instella import InstellaQModel


INSTELLA_PATH = Path("/monster/data/model/Instella-3B-Instruct")


def test_instella_tree_uses_real_norm_names():
    layer_specs = InstellaQModel.module_tree[3]
    assert "pre_attention_layernorm" in layer_specs
    assert "pre_feedforward_layernorm" in layer_specs
    # The inherited Llama names do not exist on an Instella layer.
    assert "input_layernorm" not in layer_specs
    assert "post_attention_layernorm" not in layer_specs

    blocks = InstellaQModel.simple_layer_modules(
        model_config=SimpleNamespace(model_type="instella"),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat = {name for block in blocks for name in block}

    assert "self_attn.q_proj" in flat
    assert "input_layernorm" not in flat
    assert "post_attention_layernorm" not in flat


def test_instella_attention_and_mlp_read_their_own_norm():
    blocks = InstellaQModel.simple_layer_modules(
        model_config=SimpleNamespace(model_type="instella"),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    block_sets = [set(block) for block in blocks]

    assert {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"} in block_sets
    assert {"mlp.gate_proj", "mlp.up_proj"} in block_sets
    assert ["self_attn.o_proj"] in blocks
    assert ["mlp.down_proj"] in blocks
    # A definition that just reuses the Llama tree is the bug being pinned.
    assert InstellaQModel.module_tree != LlamaQModel.module_tree


def test_instella_module_plan_resolves_on_the_real_decoder_layer():
    from gptqmodel.models.module_plan import ModulePlan
    from gptqmodel.models.ports.instella import InstellaConfig, InstellaForCausalLM

    config = InstellaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
    )
    hf = InstellaForCausalLM(config).eval()
    layer = hf.model.layers[0]

    assert hasattr(layer, "pre_attention_layernorm")
    assert hasattr(layer, "pre_feedforward_layernorm")
    assert not hasattr(layer, "input_layernorm")
    assert not hasattr(layer, "post_attention_layernorm")

    tree = InstellaQModel._resolve_effective_module_tree(hf, None)
    plan = ModulePlan.from_tree(tree)
    assert plan.find("pre_attention_layernorm") is not None
    assert plan.find("pre_feedforward_layernorm") is not None
    assert plan.find("input_layernorm") is None
    assert plan.find("post_attention_layernorm") is None

    # Every declared quantizable path must resolve on the real layer.
    for spec in plan.specs:
        if spec.quantize:
            layer.get_submodule(spec.template)
