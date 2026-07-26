# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoConfig

from gptqmodel.models.definitions.olmoe import OlmoeQModel


def _make_olmoe_config(num_experts: int = 8, num_hidden_layers: int = 2):
    """Build a minimal OLMoE config for unit testing without loading a checkpoint."""
    return AutoConfig.from_pretrained(
        "allenai/OLMoE-1B-7B-0924",
        num_experts=num_experts,
        num_hidden_layers=num_hidden_layers,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )


@pytest.mark.parametrize("num_experts", [1, 4, 64])
def test_olmoe_module_tree_expands_all_experts(num_experts):
    config = _make_olmoe_config(num_experts=num_experts, num_hidden_layers=1)
    modules = OlmoeQModel.simple_layer_modules(
        model_config=config,
        quantize_config=None,
        is_awq_quantize=False,
        include_capture_only=False,
    )

    # Each layer should expose the dense attention path plus one block per expert.
    assert len(modules) == 4, f"expected 4 subset blocks, got {len(modules)}"
    assert set(modules[0]) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert modules[1] == ["self_attn.o_proj"]
    assert modules[2] == [f"mlp.experts.{i}.gate_proj" for i in range(num_experts)] + [
        f"mlp.experts.{i}.up_proj" for i in range(num_experts)
    ]
    assert modules[3] == [f"mlp.experts.{i}.down_proj" for i in range(num_experts)]


def test_olmoe_dynamic_pattern_matches_first_half_of_experts():
    """Verify the exact dynamic regex used in the OLMoE quant script."""
    import re

    pattern = r"+:^.*\.mlp\.experts\.(3[0-1]|[12][0-9]|[0-9])\..*$"
    compiled = re.compile(pattern[2:])  # strip leading '+:'

    matched = []
    for i in range(64):
        name = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
        if compiled.match(name):
            matched.append(i)

    assert matched == list(range(32)), f"expected experts 0-31, got {matched[:5]}..{matched[-5:]}"
