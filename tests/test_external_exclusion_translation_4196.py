# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4196.

Producer exclusions (`modules_to_not_convert`, llm-compressor `ignore`) used to
be translated into `dynamic` rules that either kept llm-compressor's `re:`
prefix (so they never matched) or only matched a module ending with the entry
(so parent names did not exclude their children).  The AWQ export also wrote
`pcre.escape`d runtime exclusions, which substring-matching consumers such as
vLLM never skip.
"""

import pcre
import pytest

from gptqmodel.quantization import AWQConfig
from gptqmodel.quantization.config import (
    QuantizeConfig,
    _dynamic_exclusion_patterns,
    _external_exclusion_dynamic_pattern,
)


RUNTIME_RULE = r"-:model\.layers\.3\.mlp\.experts\.17\.down_proj"


def _matches(pattern, module_name):
    return pcre.compile(pattern).match(module_name) is not None


def _vllm_skips(module_name, patterns):
    """Mirror vLLM's `is_layer_skipped_awq` substring check."""

    return any(pattern in module_name for pattern in patterns)


def test_re_prefix_is_stripped_and_regex_kept():
    pattern = _external_exclusion_dynamic_pattern(r"re:.*mlp\.gate$")

    assert pattern == r".*mlp\.gate$"
    assert _matches(pattern, "model.layers.0.mlp.gate")
    assert not _matches(pattern, "model.layers.0.mlp.gate_proj")


@pytest.mark.parametrize(
    "entry,matches,not_matches",
    [
        (
            "down_proj",
            ("down_proj", "model.layers.0.mlp.down_proj"),
            ("down_proj_extra", "model.layers.0.mlp.down_proj_extra"),
        ),
        (
            "visual",
            ("visual", "model.visual", "model.visual.blocks.0.attn.qkv"),
            ("model.visualize.qkv",),
        ),
        (
            "model.layers.0.",
            ("model.layers.0", "model.layers.0.mlp.down_proj"),
            ("model.layers.01.mlp.down_proj", "model.layers.0x.mlp.down_proj"),
        ),
    ],
)
def test_plain_names_match_module_boundaries_and_children(entry, matches, not_matches):
    pattern = _external_exclusion_dynamic_pattern(entry)

    assert pattern is not None
    for module_name in matches:
        assert _matches(pattern, module_name), module_name
    for module_name in not_matches:
        assert not _matches(pattern, module_name), module_name


def test_regex_entries_are_kept_verbatim():
    assert _external_exclusion_dynamic_pattern(r"model\.layers\.1\..*") == r"model\.layers\.1\..*"


def test_glob_entries_fall_back_to_a_module_boundary():
    pattern = _external_exclusion_dynamic_pattern("*.mlp.down_proj")

    assert pattern is not None
    assert _matches(pattern, "model.layers.0.mlp.down_proj")


def test_llm_compressor_ignore_regex_excludes_module():
    config = QuantizeConfig.from_quant_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "sym": True,
            "ignore": ["lm_head", r"re:.*mlp\.gate$"],
        }
    )

    assert config.dynamic_get("lm_head") is False
    assert config.dynamic_get("model.layers.0.mlp.gate") is False
    assert config.dynamic_get("model.layers.0.mlp.up_proj") is not False


def test_parent_module_name_excludes_children():
    config = QuantizeConfig.from_quant_config(
        {
            "quant_method": "awq",
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "modules_to_not_convert": ["visual"],
        }
    )

    assert config.dynamic_get("model.visual.blocks.0.attn.qkv") is False
    assert config.dynamic_get("visual") is False
    assert config.dynamic_get("model.visualize.qkv") is not False


def test_awq_export_uses_plain_names_for_runtime_exclusions():
    config = AWQConfig(bits=4, group_size=128, dynamic={RUNTIME_RULE: {}})
    config._runtime_dynamic_exclusions = {RUNTIME_RULE}

    exported = config.to_dict()["modules_to_not_convert"]

    assert exported == ["model.layers.3.mlp.experts.17.down_proj"]
    assert _vllm_skips("model.layers.3.mlp.experts.17.down_proj", exported)
    assert _dynamic_exclusion_patterns(config.dynamic) == exported


def test_awq_export_unescapes_literal_re_module_names():
    config = QuantizeConfig.from_quant_config(
        {
            "quant_method": "awq",
            "bits": 4,
            "zero_point": True,
            "version": "gemm",
            "ignore": [r"re:model\.layers\.0\.down_proj"],
        }
    )

    exported = config.to_dict()["modules_to_not_convert"]

    assert exported == ["model.layers.0.down_proj"]
    assert config.dynamic_get("model.layers.0.down_proj") is False


def test_awq_export_drops_re_prefix_and_round_trips():
    payload = {
        "quant_method": "awq",
        "bits": 4,
        "zero_point": True,
        "version": "gemm",
        "modules_to_not_convert": ["visual", r"re:.*mlp\.gate$"],
    }
    config = QuantizeConfig.from_quant_config(payload)

    exported = config.to_dict()["modules_to_not_convert"]

    assert exported == ["visual", r".*mlp\.gate$"]
    assert all(not pattern.startswith("re:") for pattern in exported)

    reloaded = QuantizeConfig.from_quant_config(config.to_dict())

    assert reloaded.dynamic_get("model.visual.blocks.0.attn.qkv") is False
    assert reloaded.dynamic_get("model.layers.0.mlp.gate") is False
