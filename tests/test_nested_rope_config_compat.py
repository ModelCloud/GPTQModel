# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace

import pytest

from gptqmodel.utils.hf import normalize_hf_config_compat


PER_LAYER = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 50_000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 80_000_000.0},
}


def _config(**kwargs):
    return SimpleNamespace(model_type="dots3_note", rope_theta=80_000_000.0, **kwargs)


def test_nested_attention_rope_parameters_are_preserved():
    config = _config(
        layer_types=list(PER_LAYER) * 2,
        rope_parameters=copy.deepcopy(PER_LAYER),
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters == PER_LAYER


def test_nested_rope_parameters_gain_no_flat_scalar_keys():
    """Backfill must not append scalar siblings onto a per-layer dictionary.

    Transformers tolerates a dangling flat key, but adding ``rope_type`` next to
    the per-layer entries is exactly the pollution this guard exists to prevent.
    """
    before = copy.deepcopy(PER_LAYER)
    config = _config(layer_types=list(PER_LAYER), rope_parameters=copy.deepcopy(before))
    normalize_hf_config_compat(config)
    assert config.rope_parameters == before
    assert not {"rope_type", "rope_theta"} & set(config.rope_parameters)


def test_nested_rope_parameters_are_idempotent():
    config = _config(layer_types=list(PER_LAYER), rope_parameters=copy.deepcopy(PER_LAYER))
    normalize_hf_config_compat(config)
    once = copy.deepcopy(config.rope_parameters)
    normalize_hf_config_compat(config)
    normalize_hf_config_compat(config)
    assert config.rope_parameters == once == PER_LAYER


def test_rope_type_labels_take_priority_over_layer_types():
    """Configs keyed by ``_rope_type_labels`` must not be flattened.

    DeepSeek V4 declares ``_rope_type_labels = ("main", "compress")`` while its
    ``layer_types`` name attention blocks, and Transformers resolves nested keys
    through ``_rope_type_labels`` first.
    """
    parameters = {
        "main": {"rope_type": "default", "rope_theta": 10_000.0},
        "compress": {"rope_type": "default", "rope_theta": 160_000.0},
    }
    config = _config(
        layer_types=["heavily_compressed_attention"] * 4,
        _rope_type_labels=("main", "compress"),
        rope_parameters=copy.deepcopy(parameters),
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters == parameters


def test_nope_layers_are_preserved_as_none():
    config = _config(
        layer_types=["no_rope", "sliding_attention"],
        rope_parameters={"no_rope": None, "sliding_attention": copy.deepcopy(PER_LAYER["sliding_attention"])},
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters["no_rope"] is None
    assert isinstance(config.rope_parameters["sliding_attention"], dict)


def test_flat_legacy_rope_parameters_still_receive_defaults():
    config = SimpleNamespace(model_type="legacy", rope_parameters=None, rope_theta=50_000.0)
    normalize_hf_config_compat(config)
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 50_000.0}


def test_flat_parameters_with_layer_types_still_receive_defaults():
    """Declaring ``layer_types`` must not be enough to skip the backfill."""
    config = SimpleNamespace(
        model_type="legacy",
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 50_000.0},
        rope_theta=50_000.0,
    )
    normalize_hf_config_compat(config)
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 50_000.0}


@pytest.mark.parametrize("parameters", [{}, None])
def test_missing_rope_parameters_are_still_backfilled(parameters):
    config = SimpleNamespace(model_type="legacy", rope_parameters=parameters, rope_theta=50_000.0)
    normalize_hf_config_compat(config)
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 50_000.0}
