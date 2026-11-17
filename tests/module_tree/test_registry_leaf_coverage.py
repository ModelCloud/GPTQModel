# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Registry-wide coverage of checkpoint-owned decoder leaves.

The existing plan contracts compare a definition with itself: they prove that
every declared spec resolves, but a Linear/Conv/expert leaf that the tree never
mentions is invisible to them. Such a leaf silently stays in the source dtype
(issue #4174). This test builds each locally constructible ``MODEL_MAP`` entry
on the ``meta`` device, runs the same Defuser conversion as the loader, and
fails when a decoder leaf is neither declared by ``module_tree`` nor covered by
an explicit ``:!``/``:?`` exclusion (on the leaf or one of its ancestors).

Three-dimensional expert parameters are part of the leaf inventory. The
pre-existing runtime warning only inspects ``nn.Linear``/``nn.Conv1d``/
``nn.Conv2d``, which is exactly how Gemma-4 MoE's ``Gemma4TextExperts``
parameter stayed undeclared and unquantized without a warning.

Types whose config is only available through remote code (and therefore needs
network access) are skipped with a precise reason. The focused model tests in
``test_issue_4174_model_trees.py`` cover those layouts with local fixtures.
"""

from __future__ import annotations

import pytest
import torch
import transformers
from torch import nn
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models._const import SUPPORTS_MODULE_TYPES
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.kimi_linear import (
    KimiLinearNativeQModel,
    is_kimi_k3_config,
)
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import get_layers_with_prefixes


# Models whose default config does not exercise every declared layout branch.
# The override stays local to the config object; it never changes the tree.
def _llama4_mixed_layers(config):
    text_config = getattr(config, "text_config", config)
    text_config.num_hidden_layers = 4
    text_config.interleave_moe_layer_step = 2
    text_config.moe_layers = [0, 2]


def _gemma4_moe(config):
    text_config = getattr(config, "text_config", config)
    text_config.num_hidden_layers = 4
    text_config.enable_moe_block = True
    text_config.num_experts = 8
    text_config.top_k_experts = 2
    text_config.moe_intermediate_size = 256


def _nemotron_h_latent(config):
    # `moe_latent_size` adds `fc1_latent_proj`/`fc2_latent_proj` to every MoE
    # mixer; the default config omits that layout branch.
    config.moe_latent_size = 64


def _qwen4_exp_text_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "indexed_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        hc_count=2,
        hc_lowrank=4,
        # A non-empty PLE layer set exercises `ple.*`, which is optional.
        ple_layer_ids=[1],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


def _qwen4_exp_text(config):
    # The auto config's derived `ple_embed_dim`/cache fields are stale after
    # shrinking the model, so return a fresh tiny config instead of mutating.
    return _qwen4_exp_text_config()


def _qwen4_exp(config):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpConfig,
        Qwen4ExpVisionConfig,
    )

    # The outer auto config has no default `text_config`/`vision_config`.
    return Qwen4ExpConfig(
        text_config=_qwen4_exp_text_config(),
        vision_config=Qwen4ExpVisionConfig(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=16,
        ),
    )


CONFIG_OVERRIDES = {
    "gemma4": _gemma4_moe,
    "gemma4_text": _gemma4_moe,
    "llama4": _llama4_mixed_layers,
    "llama4_text": _llama4_mixed_layers,
    "nemotron_h": _nemotron_h_latent,
    "qwen4_exp": _qwen4_exp,
    "qwen4_exp_text": _qwen4_exp_text,
}

# Local-config models this issue fixes. A construction failure here is a
# regression, not an environment limitation.
REQUIRED_MODELS = frozenset(
    {
        "falcon_h1",
        "gemma4",
        "gemma4_text",
        "hrm_text",
        "jamba",
        "kimi_linear",
        "llama4",
        "llama4_text",
        "nemotron_h",
        "phi3",
        "phimoe",
        "qwen4_exp",
        "qwen4_exp_text",
        "voxtral",
        "zamba",
        "zamba2",
    }
)

# Explicit, reviewed exemptions. Each entry must name the reason and the
# tracking location; never add a model merely to make the sweep green.
KNOWN_UNDECLARED = {
    "mllama": "cross-attention intentionally stays in the source dtype (issue #4174 known TODO)",
    "mllama_text_model": "cross-attention intentionally stays in the source dtype (issue #4174 known TODO)",
    "unlimited_ocr": "no decoder layers resolve; definition is owned outside issue #4174",
    "granitemoehybrid": "mamba.conv1d is checkpoint-owned but the definition is owned outside issue #4174",
}

def _definition_for(model_type, config):
    definition = MODEL_MAP[model_type]
    if model_type == "kimi_linear" and not is_kimi_k3_config(config):
        return KimiLinearNativeQModel
    return definition


def _build_meta_model(model_type):
    config = transformers.AutoConfig.for_model(model_type)
    override = CONFIG_OVERRIDES.get(model_type)
    if override is not None:
        replacement = override(config)
        if replacement is not None:
            config = replacement

    definition = _definition_for(model_type, config)
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    return definition, config, layers or []


def _is_expert_owner(path: str, module: nn.Module) -> bool:
    """Return whether ``module`` directly owns a routed/shared expert tensor."""

    if "expert" not in path.lower() and "expert" not in type(module).__name__.lower():
        return False
    return any(param.dim() >= 2 for param in module.parameters(recurse=False))


def _is_fused_expert_parameter(path: str, module: nn.Module) -> bool:
    """Catch a 3-D fused expert tensor even when the module name is generic."""

    if "expert" in path.lower() or "expert" in type(module).__name__.lower():
        return False
    expert_names = {"gate_up_proj", "down_proj", "up_proj", "gate_proj", "w1", "w2", "w3"}
    return any(
        param.dim() >= 3 and name.rsplit(".", 1)[-1] in expert_names
        for name, param in module.named_parameters(recurse=False)
    )


def _leaf_paths(layer: nn.Module):
    for name, module in layer.named_modules():
        if not name:
            continue
        if isinstance(module, tuple(SUPPORTS_MODULE_TYPES)):
            yield name
            continue
        if _is_expert_owner(name, module) or _is_fused_expert_parameter(name, module):
            yield name


def _declared(plan: ModulePlan, path: str) -> bool:
    """Return whether ``path`` is declared, including a tagged ancestor subtree."""

    if plan.find(path) is not None:
        return True
    parts = path.split(".")
    for end in range(len(parts) - 1, 0, -1):
        spec = plan.find(".".join(parts[:end]))
        if spec is not None and any(flag in {"!", "?"} for flag in spec.flags):
            return True
    return False


def _undeclared_leaves(definition, layers):
    plan = ModulePlan.from_tree(definition.module_tree)
    undeclared = {}
    for layer in layers:
        for path in _leaf_paths(layer):
            if not _declared(plan, path):
                undeclared[path] = undeclared.get(path, 0) + 1
    return undeclared


@pytest.mark.parametrize("model_type", sorted(set(MODEL_MAP) & set(CONFIG_MAPPING)))
def test_every_decoder_leaf_is_declared_or_explicitly_excluded(model_type):
    try:
        definition, _, layers = _build_meta_model(model_type)
    except Exception as exc:  # noqa: BLE001 - report offline/unbuildable configs
        if model_type in REQUIRED_MODELS:
            raise
        pytest.skip(
            f"{model_type}: config/model is not constructible offline "
            f"({type(exc).__name__}: {exc})"
        )

    if not layers:
        reason = KNOWN_UNDECLARED.get(model_type)
        if reason is not None:
            pytest.xfail(f"{model_type}: {reason}")
        pytest.fail(
            f"{model_type}: {definition.__name__} declares "
            f"{definition.extract_layers_node()} but no decoder layers resolved"
        )

    undeclared = _undeclared_leaves(definition, layers)
    if not undeclared:
        return

    summary = ", ".join(
        f"{path} ({count} layer(s))" for path, count in sorted(undeclared.items())[:10]
    )
    reason = KNOWN_UNDECLARED.get(model_type)
    if reason is not None:
        pytest.xfail(f"{model_type}: {reason}: {summary}")
    pytest.fail(
        f"{model_type}: {len(undeclared)} decoder leaf(s) are neither declared "
        f"nor tagged :!/:? and stay in the source dtype: {summary}"
    )
