# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Config validation must reject bad input instead of silently changing it (#4668).

Two paths accepted input that changed, or failed to change, what got quantized:

* every bool field outside a hand-maintained list of eight was coerced with
  ``bool(...)`` (so ``"False"`` became ``True``) or stored as a truthy string;
* an unknown key inside a ``dynamic`` per-module rule was accepted, had no
  effect, and was written into the checkpoint.

Both are silent, so the tests below assert the *rejection*, not just the value.
"""

import dataclasses
import typing

import pytest

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.quantization.config import (
    AWQConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    MacaqConfig,
    ParoConfig,
    QQQConfig,
    VPTQConfig,
)


#: ``(config class, kwargs)`` that construct a valid instance with defaults.
_CONFIG_FACTORIES = (
    (GPTQConfig, {"bits": 4, "group_size": 128}),
    (QQQConfig, {"bits": 4, "group_size": 128}),
    (AWQConfig, {"bits": 4, "group_size": 128}),
    (ParoConfig, {"bits": 4, "group_size": 128}),
    (BitsAndBytesConfig, {"bits": 4}),
    (VPTQConfig, {}),
    (MacaqConfig, {"bits": 4, "group_size": 32}),
)


def _bool_fields(cls):
    """Every ``bool`` / ``Optional[bool]`` dataclass field of ``cls``."""

    hints = typing.get_type_hints(cls)
    out = []
    for field_info in dataclasses.fields(cls):
        hint = hints.get(field_info.name)
        optional = hint is not bool and set(typing.get_args(hint)) == {bool, type(None)}
        if hint is bool or optional:
            out.append((field_info.name, optional))
    return out


def test_every_concrete_config_has_bool_fields_to_check():
    """Guard the guard: the enumeration below must not silently become empty."""

    for cls, _ in _CONFIG_FACTORIES:
        assert _bool_fields(cls), f"{cls.__name__} has no bool fields to validate"


@pytest.mark.parametrize("cls,kwargs", _CONFIG_FACTORIES)
def test_bool_fields_reject_truthy_strings(cls, kwargs):
    """``"False"`` must raise for every bool field, not become ``True``."""

    checked = 0
    for field_name, optional in _bool_fields(cls):
        for bad in ("False", "True"):
            with pytest.raises(TypeError, match=rf"`{field_name}` must be a bool"):
                cls(**{**kwargs, field_name: bad})
            checked += 1
        if not optional:
            # An int is not a bool even though ``bool`` subclasses ``int``.
            with pytest.raises(TypeError, match=rf"`{field_name}` must be a bool"):
                cls(**{**kwargs, field_name: 1})
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("cls,kwargs", _CONFIG_FACTORIES)
def test_bool_fields_still_accept_real_booleans(cls, kwargs):
    """The stricter validator must not reject legitimate ``True``/``False``.

    Each field is re-passed its own already-normalised value, so per-backend
    domain constraints (QQQ rejects ``desc_act=True``, MACaQ rejects ``lm_head``)
    cannot make this a false failure.
    """

    baseline = cls(**kwargs)
    for field_name, optional in _bool_fields(cls):
        current = getattr(baseline, field_name)
        if current is not None:
            cls(**{**kwargs, field_name: current})
        if optional:
            cls(**{**kwargs, field_name: None})


def test_macaq_validates_before_forcing_sym():
    """MACaQ forces ``sym=True``; a bad caller value must still be rejected."""

    with pytest.raises(TypeError, match="`sym` must be a bool"):
        MacaqConfig(bits=4, group_size=32, sym="False")

    # The forced value is still applied for a valid (or omitted) input.
    assert MacaqConfig(bits=4, group_size=32).sym is True


def test_dynamic_rejects_unknown_keys_with_a_close_match_hint():
    with pytest.raises(TypeError, match=r"unknown `dynamic\['\.\*down_proj'\]`") as excinfo:
        GPTQConfig(bits=4, group_size=128, dynamic={r".*down_proj": {"bit": 8}})

    message = str(excinfo.value)
    assert "`bit`" in message
    assert "did you mean `bits`" in message


def test_dynamic_accepts_known_and_legacy_keys():
    """A real field, and the documented ``fmt`` alias, must still work."""

    GPTQConfig(bits=4, group_size=128, dynamic={r".*down_proj": {"bits": 8, "group_size": 32}})
    GPTQConfig(bits=4, group_size=128, dynamic={r".*proj": {"bits": 8, "fmt": "gemm"}})


def test_dynamic_rejects_an_uncompilable_pattern_at_construction():
    """A bad regex used to fail only when a module was first matched."""

    with pytest.raises(ValueError, match="not a valid regular expression"):
        GPTQConfig(bits=4, group_size=128, dynamic={"+:[": {"bits": 8}})


#: Every bool field a ``dynamic`` rule can override, with a valid config for it.
_DYNAMIC_BOOL_OVERRIDES = (
    (GPTQConfig, {"bits": 4, "group_size": 128}, "static_groups"),
    (GPTQConfig, {"bits": 4, "group_size": 128}, "act_group_aware"),
    (GPTQConfig, {"bits": 4, "group_size": 128}, "mock_quantization"),
    (GPTQConfig, {"bits": 4, "group_size": 128}, "true_sequential"),
    (AWQConfig, {"bits": 4, "group_size": 128}, "scale_search_chunked_activations"),
    (BitsAndBytesConfig, {"bits": 4}, "compress_statistics"),
)


@pytest.mark.parametrize("cls,kwargs,field_name", _DYNAMIC_BOOL_OVERRIDES)
@pytest.mark.parametrize("bad", ("False", "True"))
def test_dynamic_bool_overrides_reject_truthy_strings(cls, kwargs, field_name, bad):
    """``"False"`` must not silently turn a per-module bool override on (#4728)."""

    with pytest.raises(TypeError, match=rf"dynamic\['\.\*down_proj'\]\.{field_name}` must be a bool"):
        cls(**kwargs, dynamic={r".*down_proj": {field_name: bad}})


def test_dynamic_bool_overrides_still_accept_real_booleans():
    """The stricter check must keep ``False`` and apply it to the module clone."""

    cfg = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={
            r".*down_proj": {
                "static_groups": False,
                "act_group_aware": False,
                "mock_quantization": False,
                "true_sequential": False,
            }
        },
    )
    clone = clone_gptq_config_for_module(cfg, "model.layers.0.mlp.down_proj")
    assert clone.static_groups is False
    assert clone.act_group_aware is False
    assert cfg.dynamic[r".*down_proj"]["mock_quantization"] is False
    assert cfg.dynamic[r".*down_proj"]["true_sequential"] is False


def test_bitsandbytes_dynamic_accepts_and_normalizes_legacy_aliases():
    """The documented BnB aliases must survive the ``dynamic`` key check (#4728)."""

    cfg = BitsAndBytesConfig(
        bits=4,
        dynamic={
            r".*down_proj": {
                "bnb_quant_type": "fp4",
                "bnb_block_size": 128,
                "bnb_compress_statistics": False,
            }
        },
    )
    assert cfg.dynamic[r".*down_proj"] == {
        "format": "fp4",
        "block_size": 128,
        "compress_statistics": False,
    }


def test_bitsandbytes_dynamic_rejects_string_for_alias_bool():
    """The alias must reach the bool validator, not just the key check."""

    with pytest.raises(TypeError, match=r"bnb_compress_statistics` must be a bool"):
        BitsAndBytesConfig(
            bits=4,
            dynamic={r".*down_proj": {"bnb_compress_statistics": "False"}},
        )
