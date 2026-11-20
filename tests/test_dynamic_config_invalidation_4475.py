# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for issue #4475.

A fresh dynamic resolution must observe the value currently stored in a live
typed override. Previously issued immutable snapshots may keep their old value.
"""

import pytest

from gptqmodel.quantization.config import (
    _DYNAMIC_ALL_EXACT_CACHE,
    _DYNAMIC_EXACT_LOOKUP_CACHE,
    _DYNAMIC_IDENTITY_CACHE,
    _DYNAMIC_OVERRIDE_CACHE,
    _DYNAMIC_PATTERN_CACHE,
    _DYNAMIC_REGEX_PATTERN_CACHE,
    AWQConfig,
    GPTQConfig,
    MSEConfig,
)


def _clear_dynamic_caches() -> None:
    _DYNAMIC_PATTERN_CACHE.clear()
    _DYNAMIC_EXACT_LOOKUP_CACHE.clear()
    _DYNAMIC_REGEX_PATTERN_CACHE.clear()
    _DYNAMIC_OVERRIDE_CACHE.clear()
    _DYNAMIC_ALL_EXACT_CACHE.clear()
    _DYNAMIC_IDENTITY_CACHE.clear()


@pytest.fixture(autouse=True)
def clear_caches():
    _clear_dynamic_caches()
    yield
    _clear_dynamic_caches()


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("norm", 4.0),
        ("grid_budget", 2 << 20),
        ("clip_grid_budget", 3 << 20),
    ],
)
def test_mse_config_leaf_edit_invalidates_fresh_resolution(field_name, replacement):
    if field_name == "clip_grid_budget":
        # Only AWQ reads the clip budget; GPTQ refuses it (#5583).
        mse = MSEConfig(clip_grid_budget=1 << 20)
        cfg = AWQConfig(bits=4, group_size=128, dynamic={"^layer$": {"mse": mse}})
    else:
        mse = MSEConfig(norm=2.0, grid_budget=1 << 20)
        cfg = GPTQConfig(dynamic={"^layer$": {"mse": mse}})

    # Warm every cache layer before the edit.
    first = cfg.dynamic_get("layer", "mse")
    assert getattr(first, field_name) == getattr(mse, field_name)
    snapshot = cfg.resolve_module_config("layer")
    snapshot_mse = snapshot.get("mse")
    original = getattr(snapshot_mse, field_name)
    assert original == getattr(mse, field_name)
    assert snapshot_mse is not mse

    getattr(cfg.dynamic["^layer$"]["mse"], "__setattr__")(field_name, replacement)

    assert getattr(cfg.dynamic["^layer$"]["mse"], field_name) == replacement

    fresh_get = cfg.dynamic_get("layer", "mse")
    assert getattr(fresh_get, field_name) == replacement

    fresh_resolved = cfg.resolve_module_config("layer")
    assert getattr(fresh_resolved.get("mse"), field_name) == replacement

    # The already-issued immutable snapshot is allowed to remain old.
    assert getattr(snapshot.get("mse"), field_name) == original
    assert getattr(snapshot_mse, field_name) == original


def test_plain_dict_control_invalidates_fresh_resolution():
    cfg = GPTQConfig(dynamic={"^layer$": {"mse": {"norm": 2.0}}})

    assert cfg.dynamic_get("layer", "mse")["norm"] == 2.0
    snapshot = cfg.resolve_module_config("layer")

    cfg.dynamic["^layer$"]["mse"]["norm"] = 4.0

    assert cfg.dynamic["^layer$"]["mse"]["norm"] == 4.0
    assert cfg.dynamic_get("layer", "mse")["norm"] == 4.0
    assert cfg.resolve_module_config("layer").get("mse")["norm"] == 4.0
    assert snapshot.get("mse")["norm"] == 2.0
