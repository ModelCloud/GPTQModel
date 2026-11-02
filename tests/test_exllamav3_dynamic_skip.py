# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression: a `dynamic=None` config must not skip every EXL3 module.

``QuantizeConfig.dynamic_get`` returns ``None`` when no override rule matches, and
``False`` only for an explicit exclusion.  The processor previously tested the
result for truthiness, so a config without `dynamic` (the default, and the common
case on a freshly created EXL3 config) returned ``None`` for every layer and
skipped the whole model -- quantization silently became a no-op.
"""

from __future__ import annotations

from gptqmodel.looper.exllamav3_processor import clone_exllamav3_config_for_module
from gptqmodel.quantization.config import METHOD, QuantizeConfig


def _qcfg(**kwargs) -> QuantizeConfig:
    return QuantizeConfig(bits=4, group_size=128, method=METHOD.EXL3, **kwargs)


def test_dynamic_none_does_not_skip_module():
    qcfg = _qcfg()
    assert qcfg.dynamic is None
    assert qcfg.dynamic_get(layer_name="model.layers.0.mlp.down_proj") is None

    clone = clone_exllamav3_config_for_module(qcfg, "model.layers.0.mlp.down_proj")
    assert clone is not None
    assert clone.bits == qcfg.bits


def test_explicit_false_still_skips_module(monkeypatch):
    # EXL3Config rejects a literal ``False`` entry while normalizing, so drive the
    # skip branch the way a non-EXL3 weight-only config would reach it.
    name = "model.layers.0.mlp.down_proj"
    qcfg = _qcfg()
    monkeypatch.setattr(qcfg, "dynamic_get", lambda *a, **k: False)

    assert clone_exllamav3_config_for_module(qcfg, name) is None


def test_matching_override_is_applied_not_skipped():
    name = "model.layers.1.mlp.down_proj"
    qcfg = _qcfg(dynamic={"model.layers.1": {"bits": 3}})
    assert qcfg.dynamic_get(layer_name=name) == {"bits": 3}

    clone = clone_exllamav3_config_for_module(qcfg, name)
    assert clone is not None
    assert clone.bits == 3
