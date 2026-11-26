# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""JointQ is nested inside GPTQConfig like FOEM/GPTAQ.

The option must accept the same shapes (instance, dict, bool), serialize into
``meta.jointq``, reload through both config factories, and reject every
combination the refinement cannot honor.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.quantization.config import (
    ADMMQConfig,
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    GSQConfig,
    JointQConfig,
    QuantizeConfig,
    RTNConfig,
)


def _json_round_trip(config):
    return QuantizeConfig(**json.loads(json.dumps(config.to_dict())))


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (False, None),
        (True, JointQConfig()),
        ({"regularization": 0.01, "dtype": "float32"}, JointQConfig(regularization=0.01, dtype="float32")),
    ],
    ids=["none", "false", "true", "dict"],
)
def test_jointq_accepts_the_nested_shapes_foem_accepts(value, expected):
    config = GPTQConfig(bits=4, group_size=128, jointq=value)

    if expected is None:
        assert config.jointq is None
    else:
        assert isinstance(config.jointq, JointQConfig)
        assert config.jointq.regularization == expected.regularization
        assert config.jointq.dtype == expected.dtype


def test_jointq_instance_is_preserved():
    nested = JointQConfig(regularization=0.15, adaptive_retry=False)

    config = GPTQConfig(bits=3, group_size=-1, jointq=nested)

    assert config.jointq is nested


def test_jointq_rejects_unknown_shape():
    with pytest.raises(ValueError, match="`jointq` must be a JointQConfig, dict, boolean, or None"):
        GPTQConfig(bits=4, jointq="paper")


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"gptaq": GPTAQConfig()}, "cannot be combined with `gptaq` or `foem`"),
        ({"foem": FOEMConfig()}, "cannot be combined with `gptaq` or `foem`"),
        ({"gsq": GSQConfig(enabled=True)}, "plain GPTQ only"),
        ({"mock_quantization": True}, "incompatible with mock quantization"),
        ({"sym": False}, "symmetric quantization only"),
        ({"static_groups": True}, "does not support `static_groups=True`"),
        ({"bits": 1}, "not supported for method"),
        ({"bits": 9}, "not supported for method"),
    ],
    ids=[
        "gptaq",
        "foem",
        "gsq",
        "mock",
        "asymmetric",
        "static-groups",
        "bits-too-low",
        "bits-too-high",
    ],
)
def test_jointq_rejects_unsupported_combinations(kwargs, fragment):
    options = {"bits": 4, "group_size": 128, "sym": True, "jointq": JointQConfig()}
    options.update(kwargs)

    with pytest.raises(ValueError, match=fragment):
        GPTQConfig(**options)


def test_jointq_serializes_every_option_into_meta_and_reloads():
    nested = JointQConfig(
        regularization=0.01,
        eta=0.25,
        adaptive_retry=False,
        retry_ratio=0.3,
        retry_regularization=0.05,
        tolerance=1e-10,
        inner_stall_fraction=0.2,
        outer_active_fraction=0.3,
        re_solve_low=0.1,
        re_solve_high=0.8,
        max_iterations_per_group=7,
        max_passes=5,
        device="cpu",
        dtype="float32",
    )
    config = GPTQConfig(bits=3, group_size=-1, jointq=nested)

    payload = config.to_dict()["meta"]["jointq"]
    assert payload == nested.meta_payload()

    restored = _json_round_trip(config)
    assert isinstance(restored, GPTQConfig)
    assert restored.jointq == nested


def test_jointq_absent_from_meta_when_disabled():
    payload = GPTQConfig(bits=4, group_size=128, jointq=None).to_dict()

    assert "jointq" not in payload["meta"]


def test_jointq_meta_payload_renders_device_as_text():
    nested = JointQConfig(device=torch.device("cpu"))

    assert nested.meta_payload()["device"] == "cpu"


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"regularization": -1.0}, "non-negative"),
        ({"regularization": "table-6"}, "`\"paper\"` or a nonnegative float"),
        ({"eta": float("nan")}, "finite"),
        ({"adaptive_retry": 1}, "boolean or None"),
        ({"tolerance": 0.0}, "must be positive"),
        ({"inner_stall_fraction": 0.0}, "must be in \\(0, 1\\]"),
        ({"outer_active_fraction": 1.5}, "must be in \\(0, 1\\]"),
        ({"re_solve_low": 0.7, "re_solve_high": 0.6}, "must be smaller than"),
        ({"max_iterations_per_group": 0}, "integer >= 1 or None"),
        ({"max_passes": 0}, "integer >= 1"),
        ({"dtype": "float16"}, "`float64` or `float32`"),
    ],
    ids=[
        "negative-regularization",
        "unknown-regularization",
        "eta-nan",
        "retry-not-bool",
        "zero-tolerance",
        "zero-stall",
        "active-over-one",
        "re-solve-order",
        "iteration-cap",
        "pass-cap",
        "dtype",
    ],
)
def test_jointq_option_validation(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        JointQConfig(**kwargs)


LAYER = "model.layers.0.self_attn.v_proj"


def test_jointq_dynamic_override_is_normalized_by_the_processor():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        bits=4,
        group_size=128,
        jointq=JointQConfig(),
        dynamic={LAYER: {"jointq": {"regularization": 0.15, "dtype": "float32"}}},
    )

    clone = clone_gptq_config_for_module(config, LAYER)

    assert isinstance(clone.jointq, JointQConfig)
    assert clone.jointq.regularization == pytest.approx(0.15)
    assert clone.jointq.dtype == "float32"


@pytest.mark.parametrize("value", [True, False, None])
def test_jointq_dynamic_override_accepts_switches(value):
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        bits=4,
        group_size=128,
        jointq=JointQConfig(),
        dynamic={LAYER: {"jointq": value}},
    )

    clone = clone_gptq_config_for_module(config, LAYER)

    if value:
        assert isinstance(clone.jointq, JointQConfig)
    else:
        assert clone.jointq is None


def test_jointq_dynamic_override_enables_one_module_only():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={LAYER: {"jointq": True}},
    )

    assert config.jointq is None
    assert isinstance(clone_gptq_config_for_module(config, LAYER).jointq, JointQConfig)
    assert clone_gptq_config_for_module(config, "model.layers.0.mlp.down_proj").jointq is None


def test_jointq_dynamic_override_is_revalidated():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        bits=4,
        group_size=128,
        jointq=JointQConfig(),
        dynamic={LAYER: {"jointq": {}, "sym": False}},
    )

    with pytest.raises(ValueError, match="symmetric quantization only"):
        clone_gptq_config_for_module(config, LAYER)


def test_jointq_dynamic_override_serializes():
    config = GPTQConfig(
        bits=4,
        group_size=128,
        jointq=JointQConfig(),
        dynamic={LAYER: {"jointq": {"regularization": 0.15}}},
    )

    payload = config.to_dict()

    assert payload["dynamic"] == {LAYER: {"jointq": {"regularization": 0.15}}}
    assert _json_round_trip(config).dynamic == payload["dynamic"]


@pytest.mark.parametrize("value", [True, {}, False, None])
def test_sequential_checkpoint_rejects_dynamic_jointq(value):
    from gptqmodel.looper.gptq_checkpoint import validate_checkpoint_support

    config = GPTQConfig(dynamic={LAYER: {"jointq": value}})
    model = SimpleNamespace(
        quantize_config=config,
        model=SimpleNamespace(config=SimpleNamespace(model_type="llama")),
    )

    if value is False or value is None:
        validate_checkpoint_support(model)
    else:
        with pytest.raises(NotImplementedError, match="without GPTAQ/FOEM/JointQ"):
            validate_checkpoint_support(model)


def test_admmq_rejects_enabled_jointq_in_a_dynamic_rule():
    with pytest.raises(ValueError, match=r"dynamic.*jointq.*requires method"):
        ADMMQConfig(dynamic={LAYER: {"jointq": True}})


@pytest.mark.parametrize("as_dict", [False, True])
def test_dynamic_jointq_device_round_trips(as_dict):
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    device = torch.device("cpu")
    option = {"device": device} if as_dict else JointQConfig(device=device)
    restored = _json_round_trip(GPTQConfig(dynamic={LAYER: {"jointq": option}}))

    assert clone_gptq_config_for_module(restored, LAYER).jointq.device == "cpu"


@pytest.mark.parametrize("value", [True, {}, False, None])
def test_weight_only_dynamic_jointq(value):
    if value is False or value is None:
        assert RTNConfig(dynamic={LAYER: {"jointq": value}}).uses_weight_only_lifecycle()
    else:
        with pytest.raises(ValueError, match="jointq.*calibrat"):
            RTNConfig(dynamic={LAYER: {"jointq": value}})
