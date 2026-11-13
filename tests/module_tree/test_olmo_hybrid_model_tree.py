# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-architecture module-plan coverage for OLMo Hybrid.

OLMo Hybrid interleaves gated-DeltaNet ``linear_attn`` layers with full
``self_attn`` layers. The two decoder classes expose different checkpoint
weights (the DeltaNet branch owns ``q/k/v/g/o_proj`` plus a depthwise ``conv1d``
and a gated output norm, and only the full-attention branch owns
``post_feedforward_layernorm``). These tests build the actual Transformers
model on the ``meta`` device and check the compiled plan against every
checkpoint weight owner in both block shapes.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)

MODEL_TYPE = "olmo_hybrid"
LAYERS_NODE = "model.layers"

QUANTIZED_LEAVES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.q_proj",
    "linear_attn.k_proj",
    "linear_attn.v_proj",
    "linear_attn.g_proj",
    "linear_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

DENSE_LEAVES = (
    "input_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "linear_attn",
    "linear_attn.a_proj",
    "linear_attn.b_proj",
    "linear_attn.conv1d",
    "linear_attn.o_norm",
    "post_attention_layernorm",
    "post_feedforward_layernorm",
)

# Owners that must resolve for every checkpoint tensor in each block shape.
LINEAR_LAYER_OWNERS = {
    "input_layernorm",
    "linear_attn",
    "linear_attn.a_proj",
    "linear_attn.b_proj",
    "linear_attn.conv1d",
    "linear_attn.g_proj",
    "linear_attn.k_proj",
    "linear_attn.o_norm",
    "linear_attn.o_proj",
    "linear_attn.q_proj",
    "linear_attn.v_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "post_attention_layernorm",
}

FULL_LAYER_OWNERS = {
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "post_attention_layernorm",
    "post_feedforward_layernorm",
    "self_attn.k_norm",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.q_proj",
    "self_attn.v_proj",
}


def _tiny_config():
    config = transformers.AutoConfig.for_model(MODEL_TYPE)
    config.vocab_size = 64
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.max_position_embeddings = 128
    config.pad_token_id = 0
    config.layer_types = ["linear_attention", "full_attention"]
    config.linear_num_key_heads = 4
    config.linear_num_value_heads = 4
    config.linear_key_head_dim = 32
    config.linear_value_head_dim = 64
    config.linear_conv_kernel_dim = 4
    config._attn_implementation = "eager"
    return config


def _weight_owners(layer) -> set[str]:
    """Return modules that own a checkpoint ``state_dict()`` entry."""

    modules = {name for name, _ in layer.named_modules() if name}
    owners = set()
    for weight_path in layer.state_dict():
        owner = max(
            (
                name
                for name in modules
                if weight_path.startswith(f"{name}.")
                and "." not in weight_path[len(name) + 1:]
            ),
            key=len,
            default="",
        )
        assert owner, f"state entry {weight_path!r} is not owned by a module"
        owners.add(owner)
    return owners


def _missing_specs(specs: tuple[str, ...], resolved: set[str]) -> list[str]:
    missing = []
    for spec in specs:
        pattern = re.compile(
            "^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
        )
        if not any(pattern.match(name) for name in resolved):
            missing.append(spec)
    return missing


@pytest.fixture(scope="module")
def hybrid_case():
    if MODEL_TYPE not in CONFIG_MAPPING:
        pytest.skip("Installed Transformers has no 'olmo_hybrid' implementation")

    config = _tiny_config()
    definition = MODEL_MAP[MODEL_TYPE]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == config.num_hidden_layers

    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))

    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        layers=layers,
        layer_names=layer_names,
        specs=tuple(
            sorted(
                {
                    name.split(":", 1)[0]
                    for block in definition.simple_layer_modules(model_config=config, quantize_config=QC)
                    for name in block
                }
            )
        ),
        resolved=resolved,
        plan=ModulePlan.from_tree(definition.module_tree),
    )


def test_hybrid_registration_and_block_shapes(hybrid_case):
    definition = hybrid_case.definition
    assert MODEL_MAP[MODEL_TYPE] is definition
    assert definition.extract_layers_node() == [LAYERS_NODE]
    assert hybrid_case.layer_names[0] == f"{LAYERS_NODE}.0"

    linear_layer, full_layer = hybrid_case.layers
    assert type(linear_layer).__name__ == "OlmoHybridLinearAttentionDecoderLayer"
    assert type(full_layer).__name__ == "OlmoHybridAttentionDecoderLayer"

    linear_children = {name for name, _ in linear_layer.named_children()}
    full_children = {name for name, _ in full_layer.named_children()}
    assert {"input_layernorm", "linear_attn", "post_attention_layernorm", "mlp"} <= linear_children
    assert "post_feedforward_layernorm" not in linear_children
    assert {"self_attn", "post_attention_layernorm", "mlp", "post_feedforward_layernorm"} <= full_children
    assert "input_layernorm" not in full_children
    assert "linear_attn" not in full_children


def test_every_quantizable_spec_resolves_on_the_real_model(hybrid_case):
    missing = _missing_specs(hybrid_case.specs, hybrid_case.resolved)
    assert not missing, f"quantizable entries matched no decoder layer: {missing}"
    assert set(QUANTIZED_LEAVES) <= hybrid_case.resolved
    assert set(QUANTIZED_LEAVES) == set(hybrid_case.specs)


def test_every_checkpoint_weight_owner_is_declared_with_the_right_tag(hybrid_case):
    plan = hybrid_case.plan
    quantized_patterns = [
        re.compile("^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$")
        for spec in hybrid_case.specs
    ]

    linear_layer, full_layer = hybrid_case.layers
    assert _weight_owners(linear_layer) == LINEAR_LAYER_OWNERS
    assert _weight_owners(full_layer) == FULL_LAYER_OWNERS

    for layer in hybrid_case.layers:
        for owner in _weight_owners(layer):
            spec = plan.find(owner)
            assert spec is not None, f"checkpoint weights owned by undeclared module {owner!r}"
            is_quantized = any(pattern.match(owner) for pattern in quantized_patterns)
            if not is_quantized:
                assert not spec.quantize, (
                    f"non-quantized weight owner {owner!r} must carry an explicit `:!`/`:?` tag"
                )


def test_dense_leaves_are_declared_and_rejected_when_undeclared(hybrid_case):
    definition = hybrid_case.definition
    plan = hybrid_case.plan

    simple = {
        name.split(":", 1)[0]
        for block in definition.simple_layer_modules(model_config=hybrid_case.config, quantize_config=QC)
        for name in block
    }
    full = {
        name.split(":", 1)[0]
        for block in definition.full_layer_modules(
            model_config=hybrid_case.config, include_capture_only=True
        )
        for name in block
    }

    assert set(DENSE_LEAVES) <= full
    assert set(DENSE_LEAVES).isdisjoint(simple)
    for name in DENSE_LEAVES:
        spec = plan.find(name)
        assert spec is not None, f"dense owner {name!r} is undeclared"
        assert not spec.quantize, f"dense owner {name!r} must stay dense"
        assert plan.find(f"{name}.undeclared") is None
        assert plan.find(f"other.{name}") is None

    # ``linear_attn`` owns the ``A_log``/``dt_bias`` parameters directly.
    assert plan.find("linear_attn").quantize is False


def test_hybrid_roles_resolve_for_both_token_mixers(hybrid_case):
    plan = hybrid_case.plan
    expected = {
        "self_attn.q_proj": ("q",),
        "self_attn.k_proj": ("k",),
        "self_attn.v_proj": ("v",),
        "self_attn.o_proj": ("o",),
        "linear_attn.q_proj": ("q",),
        "linear_attn.k_proj": ("k",),
        "linear_attn.v_proj": ("v",),
        "linear_attn.o_proj": ("o",),
        "mlp.gate_proj": ("gate",),
        "mlp.up_proj": ("up",),
        "mlp.down_proj": ("down",),
    }
    for path, roles in expected.items():
        assert plan.roles(path) == roles, path

    # Undeclared siblings and similar prefixes must not resolve.
    for path in (
        "self_attn.undeclared",
        "linear_attn.undeclared",
        "mlp.undeclared",
        "linear_attn.conv1d.undeclared",
    ):
        assert plan.find(path) is None, path


def test_checkpoint_conversions_resolve_to_declared_dense_owners(hybrid_case):
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    # The source checkpoint stores three convolutions; Transformers combines
    # them into conv1d. Audit ownership after the declared conversions rather
    # than treating these source names as additional runtime modules.
    conversions = get_checkpoint_conversion_mapping(MODEL_TYPE)
    assert conversions
    expected = {
        "linear_attn.q_conv1d.weight": "linear_attn.conv1d.weight",
        "linear_attn.k_conv1d.weight": "linear_attn.conv1d.weight",
        "linear_attn.v_conv1d.weight": "linear_attn.conv1d.weight",
        "attention_layer_norm.weight": "input_layernorm.weight",
        "feedforward_layer_norm.weight": "post_attention_layernorm.weight",
    }
    for source, target in expected.items():
        converted = f"{LAYERS_NODE}.0.{source}"
        for conversion in conversions:
            converted, _ = conversion.rename_source_key(converted)
        assert converted == f"{LAYERS_NODE}.0.{target}"
        owner = target.rsplit(".", 1)[0]
        spec = hybrid_case.plan.find(owner)
        assert spec is not None and not spec.quantize, (source, owner)
        assert owner not in hybrid_case.specs
