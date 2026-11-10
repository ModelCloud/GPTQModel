# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Meta-model contract test for ``module_tree`` coverage.

A definition that drifts from the installed Transformers layout used to fail
silently: with ``layer_modules_strict = False`` the looper skipped every spec it
could not resolve, so a model could "quantize successfully" while leaving all of
its routed experts in the source dtype.

This test rebuilds each model on the ``meta`` device, runs Defuser's expert
expansion, and checks that every quantizable spec emitted by
``simple_layer_modules`` actually resolves on at least one decoder layer.

The focused six-model check always runs. The exhaustive ``MODEL_MAP`` sweep is
opt-in because it builds every model in the registry:

    GPTQMODEL_META_COVERAGE=1 pytest tests/module_tree/test_moe_expert_coverage.py
"""

import os
import re
from importlib import metadata
from types import SimpleNamespace

import pytest
import torch
import transformers
from packaging.version import InvalidVersion, Version
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


# `afmoe` and `bailing_hybrid` only defuse their routed experts from 0.0.29, so
# on an older wheel their specs legitimately match no layer. Skip rather than
# fail: without this the suite looks like a MoE regression on a stale install.
_MIN_DEFUSER = Version("0.0.29")


def _defuser_version():
    try:
        return Version(metadata.version("defuser"))
    except (metadata.PackageNotFoundError, InvalidVersion):
        return None


def _requires_defuser(name: str) -> None:
    if name not in _NEEDS_DEFUSER_029:
        return
    installed = _defuser_version()
    if installed is None or installed < _MIN_DEFUSER:
        pytest.skip(
            f"{name} needs defuser>={_MIN_DEFUSER} to defuse routed experts; "
            f"installed: {installed or 'unknown'}"
        )


QC = SimpleNamespace(dynamic=None)
_EXPERT_PLACEHOLDER = re.compile(r"\{expert_index\}")

# Models fixed by the MoE expert-coverage audit. Every quantizable spec in
# these definitions must resolve on the installed Transformers layout.
FIXED_MODELS = (
    "minimax_m2",
    "minimax",
    "granitemoe",
    "granitemoehybrid",
    "bailing_hybrid",
    "afmoe",
    "ernie4_5_vl_moe",
)

# Configs whose default layer mix does not exercise every branch of the tree.
# ``GraniteMoeHybridConfig`` defaults to 100% Mamba layers, so the ``self_attn``
# branch would look "dead" even though the definition is correct for real
# checkpoints.
def _mix_granitemoehybrid_layers(cfg):
    cfg.layer_types = [
        "full_attention" if index % 2 else "linear_attention"
        for index in range(cfg.num_hidden_layers)
    ]


CONFIG_OVERRIDES = {
    "granitemoehybrid": _mix_granitemoehybrid_layers,
}

# Model types whose routed experts the *installed* Defuser still leaves fused as
# 3-D ``gate_up_proj``/``down_proj`` parameters. Their ``.experts.`` specs cannot
# resolve until Defuser gains a registry entry, so the exhaustive sweep marks
# them xfail. Delete an entry once the matching Defuser release is required.
DEFUSER_FUSED_EXPERTS = frozenset({
    "axk2",
    "cohere2_moe",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
    "deepseek_v4",
    "falcon_mamba",
    "gemma3n",
    "glm4_moe_lite",
    "glm_moe_dsa",
    "gpt2",
    "inkling_mm_model",
    "jamba",
    "laguna",
    "lfm2",
    "lfm2_vl",
    "minicpmv4_6",
    "minimax_m3_vl",
})

# Defuser gained the `afmoe` / `bailing_hybrid` expert registry entries in
# 0.0.29. Older wheels leave those routed experts fused in 3-D parameters.
_NEEDS_DEFUSER_029 = frozenset({"afmoe", "bailing_hybrid"})


def _quantizable_specs(definition, model_config):
    blocks = definition.simple_layer_modules(
        model_config=model_config,
        quantize_config=QC,
    )
    # ``simple_layer_modules`` already drops ``:!``/``:?`` entries; strip any
    # remaining group/flag suffixes defensively.
    return sorted({name.split(":")[0] for block in blocks for name in block})


def _build_defused_model(model_type):
    """Return ``(config, definition, layers)`` for a meta-device model."""
    config = transformers.AutoConfig.for_model(model_type)
    override = CONFIG_OVERRIDES.get(model_type)
    if override is not None:
        override(config)

    definition = MODEL_MAP[model_type]
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    # Imported lazily so the fast test does not pay for Defuser at collection.
    from defuser import convert_model

    convert_model(model, cleanup_original=False)

    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    return config, definition, layers


def _resolved_module_names(layers):
    """Union of quantizable leaf module names across every decoder layer."""
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))
    return resolved


def _missing_specs(specs, resolved):
    missing = []
    for spec in specs:
        if "{expert_index}" in spec:
            pattern = re.compile(
                "^" + re.escape(spec).replace(r"\{expert_index\}", r"\d+") + "$"
            )
            if not any(pattern.match(name) for name in resolved):
                missing.append(spec)
        elif spec not in resolved:
            missing.append(spec)
    return missing


@pytest.mark.parametrize("model_type", FIXED_MODELS)
def test_fixed_moe_definitions_expand_every_expert_placeholder(model_type):
    """A definition must not leak the literal placeholder into quantization.

    ``create_named_modules`` matches concrete module names, so a surviving
    ``{expert_index}`` spec can never hit; with ``layer_modules_strict = False``
    it is recorded as a silent miss and the experts stay in the source dtype.
    """
    _requires_defuser(model_type)
    config, definition, _ = _build_defused_model(model_type)

    specs = _quantizable_specs(definition, config)
    assert not [spec for spec in specs if "{expert_index}" in spec]

    blocks = definition.simple_layer_modules(model_config=config, quantize_config=QC)
    for block in blocks:
        for entry in block:
            if ":!" in entry or ":?" in entry:
                continue
            assert "{expert_index}" not in entry, (model_type, entry)


@pytest.mark.parametrize("model_type", FIXED_MODELS)
def test_fixed_moe_definitions_resolve_every_quantizable_spec(model_type):
    _requires_defuser(model_type)
    config, definition, layers = _build_defused_model(model_type)

    assert layers, f"{model_type}: no decoder layers resolved"
    specs = _quantizable_specs(definition, config)
    resolved = _resolved_module_names(layers)
    missing = _missing_specs(specs, resolved)

    assert not missing, (
        f"{model_type}: {len(missing)} layer-module spec(s) matched no layer and "
        f"would be silently left unquantized: {missing[:10]}"
    )


@pytest.mark.parametrize("model_type", FIXED_MODELS)
def test_fixed_moe_definitions_resolve_routed_experts(model_type):
    _requires_defuser(model_type)
    config, definition, layers = _build_defused_model(model_type)

    specs = _quantizable_specs(definition, config)
    expert_specs = [spec for spec in specs if ".experts." in spec]
    assert expert_specs, f"{model_type}: definition emits no routed-expert specs"

    missing = _missing_specs(expert_specs, _resolved_module_names(layers))
    assert not missing, (
        f"{model_type}: {len(missing)} routed-expert spec(s) never resolved "
        f"(experts left in the source dtype): {missing[:10]}"
    )


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_META_COVERAGE") != "1",
    reason="set GPTQMODEL_META_COVERAGE=1 to build every MODEL_MAP entry",
)
@pytest.mark.parametrize(
    "model_type", sorted(set(MODEL_MAP) & set(CONFIG_MAPPING))
)
def test_every_defined_model_type_resolves_routed_experts(model_type):
    _requires_defuser(model_type)
    try:
        config, definition, layers = _build_defused_model(model_type)
    except Exception as exc:  # noqa: BLE001 - unsupported configs are skipped
        pytest.skip(
            f"{model_type}: cannot build meta model "
            f"({type(exc).__name__}: {exc})"
        )

    if not layers:
        pytest.skip(f"{model_type}: no decoder layers resolved")

    specs = _quantizable_specs(definition, config)
    expert_specs = [spec for spec in specs if ".experts." in spec]
    if not expert_specs:
        return

    if model_type in DEFUSER_FUSED_EXPERTS:
        pytest.xfail(
            f"{model_type}: installed Defuser leaves routed experts fused as 3-D "
            "parameters"
        )

    missing = _missing_specs(expert_specs, _resolved_module_names(layers))
    assert not missing, (
        f"{model_type}: {len(missing)} routed-expert spec(s) never resolved "
        f"(experts left in the source dtype): {missing[:10]}"
    )
