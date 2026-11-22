# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #4198.

`dynamic` resolves first-match in insertion order. Only ``__post_init__`` used
to move ``-:`` exclusions to the front, so a rule appended after construction
(``looper/stage_subset.py``, ``models/definitions/gemma3.py``,
``looper/gptq_checkpoint.py``, or a plain ``cfg.dynamic = {...}`` assignment)
lost to an earlier positive rule in-process while the same rule set excluded
the module after a save/reload. Every mutation now keeps exclusions first.
"""

import json

import pcre
import pytest

from gptqmodel.quantization.config import BaseQuantizeConfig, GPTQConfig


MODULE = "model.layers.3.mlp.experts.17.down_proj"
POSITIVE_RULE = r".*\.experts\..*"


def _config() -> GPTQConfig:
    return GPTQConfig(bits=4, dynamic={POSITIVE_RULE: {"bits": 8}})


def _exclusion(module: str = MODULE) -> str:
    return f"-:{pcre.escape(module)}"


def test_appended_exclusion_wins_in_process_and_after_reload():
    cfg = _config()
    cfg.dynamic[_exclusion()] = {}

    assert cfg.dynamic_get(MODULE) is False
    assert cfg.resolve_module_config(MODULE).excluded is True
    assert list(cfg.dynamic) == [_exclusion(), POSITIVE_RULE]

    payload = json.loads(json.dumps(cfg.to_dict()))
    assert list(payload["dynamic"]) == [_exclusion(), POSITIVE_RULE]
    assert BaseQuantizeConfig.from_quant_config(payload).dynamic_get(MODULE) is False


def test_exclusion_inserted_first_still_wins_over_earlier_positive_rule():
    cfg = _config()
    cfg.dynamic.update({_exclusion(): {}})

    assert cfg.dynamic_get(MODULE) is False


def test_setdefault_and_inplace_update_keep_exclusions_first():
    cfg = _config()
    cfg.dynamic.setdefault(_exclusion(), {})
    assert list(cfg.dynamic) == [_exclusion(), POSITIVE_RULE]

    cfg.dynamic |= {_exclusion("model.layers.4.mlp.experts.2.down_proj"): {}}
    assert list(cfg.dynamic) == [
        _exclusion(),
        _exclusion("model.layers.4.mlp.experts.2.down_proj"),
        POSITIVE_RULE,
    ]


def test_dynamic_assignment_keeps_exclusions_first():
    cfg = _config()
    cfg.dynamic = {POSITIVE_RULE: {"bits": 8}, _exclusion(): {}}

    assert list(cfg.dynamic) == [_exclusion(), POSITIVE_RULE]
    assert cfg.dynamic_get(MODULE) is False


def test_reordering_preserves_relative_order_and_nested_overrides():
    second_positive = r".*\.mlp\.down_proj$"
    cfg = GPTQConfig(
        bits=4,
        dynamic={
            POSITIVE_RULE: {"bits": 8},
            _exclusion(): {"adapter": {"rank": 4}},
            second_positive: {"bits": 4},
        },
    )
    cfg.dynamic[_exclusion("model.layers.5.mlp.experts.1.down_proj")] = {}

    assert list(cfg.dynamic) == [
        _exclusion(),
        _exclusion("model.layers.5.mlp.experts.1.down_proj"),
        POSITIVE_RULE,
        second_positive,
    ]
    # Nested override payloads are not touched by the reorder.
    assert cfg.dynamic[_exclusion()] == {"adapter": {"rank": 4}}


def test_exclusion_only_dynamic_is_unchanged():
    o_proj_rule = r".*self_attn\.o_proj$"
    cfg = GPTQConfig(bits=4, dynamic={_exclusion(): {}})
    cfg.dynamic[o_proj_rule] = {"bits": 8}

    assert list(cfg.dynamic) == [_exclusion(), o_proj_rule]
    assert cfg.dynamic_get(MODULE) is False
    assert cfg.dynamic_get("model.layers.3.self_attn.o_proj") == {"bits": 8}


@pytest.mark.parametrize("mutation", ["setitem", "update", "setdefault", "assign"])
def test_every_mutation_path_agrees_with_reload(mutation):
    cfg = _config()
    rule = _exclusion()
    if mutation == "setitem":
        cfg.dynamic[rule] = {}
    elif mutation == "update":
        cfg.dynamic.update({rule: {}})
    elif mutation == "setdefault":
        cfg.dynamic.setdefault(rule, {})
    else:
        cfg.dynamic = {POSITIVE_RULE: {"bits": 8}, rule: {}}

    reloaded = BaseQuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
    assert cfg.dynamic_get(MODULE) is False
    assert reloaded.dynamic_get(MODULE) is False
