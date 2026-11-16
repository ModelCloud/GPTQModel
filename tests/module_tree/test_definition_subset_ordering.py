# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Regression tests for module-tree subset ordering and linear coverage.

Issue #4041: a subset is one quantization/forward-replay block. A consumer
must never share a subset with the producer whose output it consumes, because
the producer's quantized weights are installed after the whole block is
processed. These tests pin the producer/consumer order and the explicit
coverage of every called ``nn.Linear`` for the audited definitions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import transformers

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)


def _blocks(model_type: str, **config_overrides):
    definition = MODEL_MAP[model_type]
    config = SimpleNamespace(model_type=model_type, **config_overrides)
    if isinstance(definition.dynamic_expert_index, str):
        setattr(config, definition.dynamic_expert_index, 2)
    return definition.simple_layer_modules(model_config=config, quantize_config=QC)


def _subset_indices(blocks) -> dict[str, int]:
    indices: dict[str, int] = {}
    for subset_index, block in enumerate(blocks):
        for raw_name in block:
            name = raw_name.split(":!", 1)[0].split(":?", 1)[0]
            indices.setdefault(name, subset_index)
    return indices


@pytest.mark.parametrize(
    "model_type, expected",
    [
        # `q_b_proj` consumes `q_a_proj`; `o_b_proj` consumes `kv_proj`.
        (
            "deepseek_v4",
            {
                "self_attn.q_a_proj": 0,
                "self_attn.q_b_proj": 1,
                "self_attn.kv_proj": 0,
                "self_attn.o_b_proj": 2,
            },
        ),
        # MLA split mirrors DeepSeek-V3: low-rank producers, then consumers,
        # then the attention output projection.
        (
            "deepseek_v2",
            {
                "self_attn.q_a_proj": 0,
                "self_attn.kv_a_proj_with_mqa": 0,
                "self_attn.q_b_proj": 1,
                "self_attn.kv_b_proj": 1,
                "self_attn.o_proj": 2,
            },
        ),
        # The convolution output projection consumes `in_proj`'s convolution.
        ("lfm2", {"conv.in_proj": 0, "conv.out_proj": 1}),
        # `x_proj` consumes the convolved `in_proj`; `out_proj` consumes the SSM.
        (
            "falcon_mamba",
            {"mixer.in_proj": 0, "mixer.x_proj": 1, "mixer.out_proj": 2},
        ),
        # The gate and QKV read the same input; `out_proj` consumes the gated result.
        (
            "minimax",
            {"self_attn.qkv_proj": 0, "self_attn.output_gate": 0, "self_attn.out_proj": 2},
        ),
    ],
)
def test_consumer_is_quantized_after_its_producer(model_type, expected):
    indices = _subset_indices(_blocks(model_type))

    for name, subset_index in expected.items():
        assert indices.get(name) == subset_index, (model_type, name, indices)


def test_jamba_declares_mamba_x_proj():
    indices = _subset_indices(_blocks("jamba"))

    assert indices["mamba.in_proj"] < indices["mamba.x_proj"] < indices["mamba.out_proj"]


def _tiny_axk2_config():
    return transformers.AXK2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        index_head_dim=8,
        index_n_heads=2,
        gated_norm_rank=4,
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=32,
    )


def test_axk2_declares_every_called_linear():
    """Every called ``nn.Linear`` is either quantized or explicitly excluded."""

    definition = MODEL_MAP["axk2"]
    config = _tiny_axk2_config()
    model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers

    plan = ModulePlan.from_tree(definition.module_tree)
    declared = {name.split(":", 1)[0] for block in _blocks("axk2") for name in block}

    for layer in layers:
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            spec = plan.find(name)
            assert spec is not None, f"axk2: undeclared nn.Linear {name}"
            if spec.quantize:
                assert name in declared, f"axk2: quantizable linear missing from subsets: {name}"
            else:
                assert "!" in spec.flags or "?" in spec.flags, (name, spec.flags)

    for excluded in (
        "input_layernorm.mlp.fc1",
        "input_layernorm.mlp.fc2",
        "post_attention_layernorm.mlp.fc1",
        "post_attention_layernorm.mlp.fc2",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
    ):
        spec = plan.find(excluded)
        assert spec is not None, excluded
        assert not spec.quantize, excluded

    # Sanity check the real layers expose the paths the plan declares.
    resolved = set(find_modules(layers[0]))
    assert "self_attn.indexer.wq_b" in resolved
    assert "input_layernorm.mlp.fc1" in resolved
