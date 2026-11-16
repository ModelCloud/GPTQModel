# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #4195.

``AWQConfig(modules_to_not_convert=[...])`` was only folded into ``-:``
dynamic rules by ``from_quant_config``. A freshly constructed config therefore
quantized the listed module (saving qweight/qzeros/scales) while the saved
config made the loader build a dense ``Linear`` with random-init weights.
``AWQConfig.__post_init__`` now performs the same fold, so quantize time and
load time resolve the module identically.
"""

import json

from gptqmodel.quantization.config import AWQConfig, BaseQuantizeConfig


MODULE = "model.layers.0.mlp.down_proj"


def _round_trip(config: AWQConfig) -> BaseQuantizeConfig:
    return BaseQuantizeConfig.from_quant_config(json.loads(json.dumps(config.to_dict())))


def test_modules_to_not_convert_is_excluded_at_quantize_time():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])

    assert cfg.has_dynamic_exclusions() is True
    assert cfg.resolve_module_config(MODULE).excluded is True
    assert cfg.dynamic_get(MODULE) is False


def test_quantize_and_reload_agree():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])

    reloaded = _round_trip(cfg)

    assert cfg.resolve_module_config(MODULE).excluded is True
    assert reloaded.resolve_module_config(MODULE).excluded is True
    assert dict(reloaded.dynamic) == dict(cfg.dynamic)


def test_saved_payload_keeps_exclusion_first_and_original_spelling():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])

    payload = cfg.to_dict()

    assert payload["modules_to_not_convert"] == ["down_proj"]
    assert list(payload["dynamic"])[0].startswith("-:")
    # One derived rule, no duplicate from a second fold during reload.
    assert len(payload["dynamic"]) == 1
    assert list(_round_trip(cfg).to_dict()["dynamic"]) == list(payload["dynamic"])


def test_unlisted_modules_stay_quantized():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])

    assert cfg.resolve_module_config("model.layers.0.mlp.up_proj").excluded is False


def test_exclusion_wins_over_a_matching_positive_dynamic_rule():
    cfg = AWQConfig(
        bits=4,
        group_size=128,
        dynamic={r".*\.down_proj$": {"bits": 8}},
        modules_to_not_convert=["down_proj"],
    )

    assert cfg.resolve_module_config(MODULE).excluded is True
    assert list(cfg.dynamic)[0].startswith("-:")


def test_no_exclusion_list_leaves_dynamic_untouched():
    cfg = AWQConfig(bits=4, group_size=128)

    assert cfg.dynamic is None
    assert cfg.resolve_module_config(MODULE).excluded is False
    assert "modules_to_not_convert" not in cfg.to_dict()


def test_empty_exclusion_list_is_a_noop():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=[])

    assert cfg.dynamic is None
    assert "modules_to_not_convert" not in cfg.to_dict()
