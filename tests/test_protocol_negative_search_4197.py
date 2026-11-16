# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4197.

Protocol `-:` selectors are evaluated with search semantics (`_pattern_matches`
uses `pcre.search`), but the compiled `dynamic` rules are evaluated with a
start-anchored `pcre.match`.  A selector that does not start with `.*` used to
exclude the module in the protocol and quantize it in the compiled config.
"""

import pytest

from gptqmodel.quantization import AWQConfig
from gptqmodel.quantization.protocol import (
    compile_protocol,
    compile_protocol_to_quantize_config,
)


MODULES = (
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.gate",
    "model.layers.0.mlp.up_proj",
    "model.layers.1.self_attn.q_proj",
    "lm_head",
)


def _gptq_source(match):
    return {
        "version": 2,
        "stages": [
            {
                "name": "ptq",
                "rules": [
                    {
                        "match": match,
                        "weight": {
                            "quantize": {"method": "gptq", "bits": 4, "group_size": 128}
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "match,excluded",
    [
        (["*", "-:down_proj"], {"model.layers.0.mlp.down_proj"}),
        (["*", r"-:mlp\.gate$"], {"model.layers.0.mlp.gate"}),
        (["*", r"-:\.self_attn\.q_proj$"], {"model.layers.1.self_attn.q_proj"}),
        (["*", r"-:.*mlp\.gate$"], {"model.layers.0.mlp.gate"}),
    ],
)
def test_compiled_exclusions_follow_protocol_search_semantics(match, excluded):
    source = _gptq_source(match)
    rule = compile_protocol(source).stages[0].rules[0]
    config = compile_protocol_to_quantize_config(source)

    for name in MODULES:
        protocol_excludes = not rule.matches(name)
        assert protocol_excludes == (name in excluded), name
        assert (config.dynamic_get(name) is False) == protocol_excludes, name


def test_exclude_everything_selector_excludes_every_module():
    config = compile_protocol_to_quantize_config(_gptq_source(["*", "-:*"]))

    for name in MODULES:
        assert config.dynamic_get(name) is False, name


def test_awq_compilation_keeps_search_semantics():
    source = {
        "version": 2,
        "stages": [
            {
                "name": "ptq",
                "rules": [
                    {
                        "match": ["*", "-:down_proj"],
                        "weight": {
                            "quantize": {"method": "awq", "bits": 4, "group_size": 128}
                        },
                    }
                ],
            }
        ],
    }
    config = compile_protocol_to_quantize_config(source)

    assert isinstance(config, AWQConfig)
    assert config.dynamic_get("model.layers.0.mlp.down_proj") is False
    assert config.dynamic_get("model.layers.0.mlp.up_proj") is not False
