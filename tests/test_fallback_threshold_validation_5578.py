# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fallback.threshold must fail closed at construction (#5578).

``Fallback.__post_init__`` validated only ``smooth``. ``NaN`` and negative
thresholds were accepted and silently disabled the low-sample fallback
(``observed < NaN`` is always false), and ``NaN`` was written to
``quantize_config.json`` as invalid JSON. Bad strings failed only once
``quantize()`` started.
"""

from __future__ import annotations

import pytest

from gptqmodel.quantization import config as config_module


CONFIG_CLASSES = [
    config_module.GPTQConfig,
    config_module.AWQConfig,
    config_module.RTNConfig,
    config_module.ParoConfig,
    config_module.QQQConfig,
]

BAD_THRESHOLDS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param(-1, id="negative-int"),
    pytest.param(-0.5, id="negative-float"),
    pytest.param("-5%", id="negative-percent"),
    pytest.param("abc", id="bad-string"),
    pytest.param([], id="list"),
    pytest.param(object(), id="object"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("threshold", BAD_THRESHOLDS)
def test_fallback_rejects_an_unusable_threshold_at_construction(threshold):
    with pytest.raises((ValueError, TypeError)):
        config_module.Fallback(threshold=threshold)


@pytest.mark.cpu
@pytest.mark.parametrize("config_cls", CONFIG_CLASSES, ids=lambda cls: cls.__name__)
def test_every_config_rejects_a_nan_fallback_threshold(config_cls):
    with pytest.raises(ValueError, match="finite"):
        config_cls(fallback=config_module.Fallback(threshold=float("nan")))


@pytest.mark.cpu
def test_dict_and_dynamic_fallback_thresholds_are_validated():
    with pytest.raises(ValueError):
        config_module.GPTQConfig(fallback={"threshold": "abc"})
    with pytest.raises(ValueError):
        config_module.GPTQConfig(dynamic={"+:.*q_proj": {"fallback": {"threshold": -1}}})


@pytest.mark.cpu
@pytest.mark.parametrize("threshold", [None, 0, 0.0, "0", "0%", "", False])
def test_documented_off_values_are_still_accepted(threshold):
    assert config_module.Fallback(threshold=threshold).threshold == threshold


@pytest.mark.cpu
@pytest.mark.parametrize("threshold", [1, 5.0, "0.5%", "5%", True])
def test_positive_thresholds_are_still_accepted(threshold):
    assert config_module.Fallback(threshold=threshold).threshold == threshold
