# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every numeric quantize option must reject NaN/inf (#5437).

#5316 added ``_validate_finite_float`` for ``MSEConfig``, ``FOEMConfig``,
``GPTAQConfig`` and ``HessianConfig`` only. Every other config still accepted
NaN/inf, and ``NaN`` compares false against every bound, so the option was
either silently ignored or silently replaced:

* ``MacaqConfig(ridge=nan)`` turned the flip refinement off (it applies only
  when ``ridge > 0``) and the run reported success;
* ``ParoConfig(opt_rotation_lr=nan)`` quantized at +13.8% wikitext-2 PPL with
  no warning;
* ``MacaqConfig(bits=nan)`` was replaced by the default 4.

The sweep below walks every numeric field of every ``QuantizeConfig``
subclass, so a new field cannot reintroduce the hole.
"""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import pytest

from gptqmodel.quantization import config as config_module


NON_FINITE = (float("nan"), float("inf"), float("-inf"))

# A rejected value may surface as ValueError/TypeError from an explicit check,
# or as OverflowError from `int(inf)` where an integer field is coerced. All
# three are loud; only a silent accept is a bug.
REJECTED = (ValueError, TypeError, OverflowError)


def _config_classes():
    classes = []
    for name in dir(config_module):
        obj = getattr(config_module, name)
        if (
            isinstance(obj, type)
            and dataclasses.is_dataclass(obj)
            and issubclass(obj, config_module.QuantizeConfig)
            and obj is not config_module.QuantizeConfig
        ):
            classes.append(pytest.param(obj, id=name))
    return sorted(classes, key=lambda param: param.id)


def _numeric_fields(config_cls):
    try:
        hints = get_type_hints(config_cls)
    except Exception:  # pragma: no cover - defensive for exotic hints
        hints = {}
    for field in dataclasses.fields(config_cls):
        hint = str(hints.get(field.name, field.type))
        if "bool" in hint:
            continue
        if "float" in hint or "int" in hint:
            yield field.name


@pytest.mark.cpu
@pytest.mark.parametrize("config_cls", _config_classes())
def test_every_numeric_option_rejects_non_finite_values(config_cls):
    """No config may accept NaN/inf in a numeric field."""

    checked = 0
    for name in _numeric_fields(config_cls):
        for bad in NON_FINITE:
            with pytest.raises(REJECTED):
                config_cls(offload_to_disk=False, **{name: bad})
            checked += 1
    assert checked, f"{config_cls.__name__} has no numeric field to validate"


@pytest.mark.cpu
@pytest.mark.parametrize(
    "config_cls,kwargs",
    [
        (config_module.ParoConfig, {"opt_rotation_lr": float("nan")}),
        (config_module.ParoConfig, {"opt_weight_lr": float("inf")}),
        (config_module.ParoConfig, {"opt_eps": float("nan")}),
        (config_module.MacaqConfig, {"ridge": float("nan")}),
        (config_module.MacaqConfig, {"ridge": float("inf")}),
        (config_module.VPTQConfig, {"npercent": float("nan")}),
        (config_module.VPTQConfig, {"kmeans_iters": float("nan")}),
        (config_module.VPTQConfig, {"kmeans_seed": float("inf")}),
        (config_module.VPTQConfig, {"block_size": float("nan")}),
        (config_module.RTNConfig, {"group_size": float("nan")}),
        (config_module.AWQConfig, {"group_size": float("inf")}),
        (config_module.QQQConfig, {"group_size": float("nan")}),
    ],
)
def test_reported_options_reject_non_finite_values(config_cls, kwargs):
    """The concrete options from the issue report."""

    with pytest.raises((ValueError, TypeError)):
        config_cls(offload_to_disk=False, **kwargs)


@pytest.mark.cpu
def test_macaq_rejects_a_non_mxfp4_bits_or_group_size():
    """MXFP4 constants are enforced, but a wrong value is an error, not a rewrite."""

    with pytest.raises(ValueError, match="bits"):
        config_module.MacaqConfig(bits=8, offload_to_disk=False)
    with pytest.raises(ValueError, match="group_size"):
        config_module.MacaqConfig(group_size=128, offload_to_disk=False)


@pytest.mark.cpu
def test_macaq_accepts_the_mxfp4_settings():
    """The supported values still construct, including the "unset" sentinel."""

    for group_size in (32, -1):
        config = config_module.MacaqConfig(bits=4, group_size=group_size, offload_to_disk=False)
        assert (config.bits, config.group_size) == (4, 32)


@pytest.mark.cpu
@pytest.mark.parametrize("config_cls", _config_classes())
def test_every_numeric_option_rejects_non_finite_values_through_dynamic(config_cls):
    """The same numeric options must be checked inside a per-module rule (#5503).

    ``dynamic`` values were merged verbatim: ``damp_percent=nan`` quantized with
    every damp-recovery attempt failing and was then rescued by the Hessian
    floor, ``group_size=nan`` only failed later inside ``quantize()``, and a
    bare ``NaN`` was written into ``quantize_config.json``. A config that
    rejects every ``dynamic`` override (VPTQ) raises for that reason and is
    still loud.
    """

    checked = 0
    # Match on the annotation members: the string-based helper above also
    # catches `calibration: Union[CalibrationConfig, ...]` ("int" in the class
    # name), which is not a numeric option.
    for name in sorted(config_module._numeric_annotated_fields(config_cls)):
        for bad in NON_FINITE:
            with pytest.raises(REJECTED):
                config_cls(offload_to_disk=False, dynamic={"+:.*": {name: bad}})
            checked += 1
    assert checked, f"{config_cls.__name__} has no numeric field to validate"


@pytest.mark.cpu
@pytest.mark.parametrize(
    "config_cls,override",
    [
        (config_module.GPTQConfig, {"damp_percent": float("nan")}),
        (config_module.GPTQConfig, {"damp_percent": 5.0}),
        (config_module.GPTQConfig, {"damp_percent": -1.0}),
        (config_module.GPTQConfig, {"damp_auto_increment": -0.5}),
        (config_module.GPTQConfig, {"group_size": float("nan")}),
        (config_module.GPTQConfig, {"group_size": -7}),
        (config_module.ParoConfig, {"opt_pair_ratio": 0.75}),
        (config_module.ParoConfig, {"opt_betas": [float("nan"), 0.9]}),
        (config_module.MacaqConfig, {"ridge": -1.0}),
        (config_module.MacaqConfig, {"pre_alpha": 2.0}),
    ],
)
def test_reported_dynamic_overrides_are_rejected(config_cls, override):
    """The concrete per-module values from the issue report."""

    with pytest.raises((ValueError, TypeError)):
        config_cls(offload_to_disk=False, dynamic={"+:.*": override})


@pytest.mark.cpu
def test_dynamic_numeric_overrides_keep_valid_values():
    """An in-range rule still constructs and is not rewritten."""

    override = {"damp_percent": 0.01, "group_size": 32, "damp_auto_increment": 0.0}
    config = config_module.GPTQConfig(
        bits=4,
        group_size=128,
        offload_to_disk=False,
        dynamic={"+:.*": dict(override)},
    )

    assert config.dynamic["+:.*"] == override
