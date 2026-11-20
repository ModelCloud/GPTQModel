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
    FallbackStrategy,
    GGUFConfig,
    GPTQConfig,
    MacaqConfig,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    SmootherConfig,
    SmoothMAD,
    SmoothPercentile,
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


# --- nested smoother/fallback payload keys (#5122) -------------------------
#
# A smoother or fallback dict was parsed with ``.get(...)`` only. A misspelled
# key was accepted and had no effect, and a smoother dict without ``type``
# returned ``None`` -- silently turning smoothing off. The tests below assert
# rejection; the control cases prove the documented spellings still compile.


@pytest.mark.parametrize(
    ("payload", "bad_key", "hint"),
    [
        ({"type": "mad", "kk": 1.0}, "kk", "k"),
        ({"smooth": {"type": "percentile", "percentil": 95}}, "percentil", "percentile"),
        ({"smoth": {"type": "mad"}}, "smoth", "smooth"),
    ],
)
def test_misspelled_smoother_payload_keys_are_rejected(payload, bad_key, hint):
    with pytest.raises((TypeError, ValueError)) as excinfo:
        GPTQConfig(smoother=payload)

    message = str(excinfo.value)
    assert f"`{bad_key}`" in message, message
    assert f"did you mean `{hint}`?" in message, message


@pytest.mark.parametrize("payload", [{"typ": "mad", "k": 1.0}, {}])
def test_smoother_dict_without_type_is_rejected_not_silently_disabled(payload):
    """A dict payload must never normalize to ``SmootherConfig(smooth=None)``."""

    with pytest.raises((TypeError, ValueError)):
        GPTQConfig(smoother=payload)


def test_nested_smoother_dicts_validate_in_every_owner():
    with pytest.raises((TypeError, ValueError)):
        SmootherConfig(smooth={"type": "mad", "kk": 1.0})
    with pytest.raises((TypeError, ValueError)):
        RTNConfig(bits=4, smooth={"type": "mad", "kk": 1.0})
    with pytest.raises((TypeError, ValueError)):
        GGUFConfig(bits=4, smoother={"type": "mad", "kk": 1.0})


@pytest.mark.parametrize(
    ("payload", "bad_key", "hint"),
    [
        ({"strategy": "rtn", "threshhold": "5%"}, "threshhold", "threshold"),
        ({"stratgy": "median"}, "stratgy", "strategy"),
    ],
)
def test_misspelled_fallback_keys_are_rejected(payload, bad_key, hint):
    for build in (
        lambda: GPTQConfig(fallback=payload),
        lambda: QuantizeConfig.from_quant_config(
            {"bits": 4, "group_size": 128, "fallback": payload}
        ),
    ):
        with pytest.raises((TypeError, ValueError)) as excinfo:
            build()

        message = str(excinfo.value)
        assert f"`{bad_key}`" in message, message
        assert f"did you mean `{hint}`?" in message, message


def test_smoother_dict_defaults_match_the_dataclasses():
    """``smoother="mad"`` must not use a second, drifting ``k`` literal.

    The weight-only RTN config owns `smoother`: the calibrated GPTQ configs
    refuse it because their solve never applies it (#5302).
    """

    from_dict = RTNConfig(bits=4, smoother="mad").smoother.smooth
    from_type = RTNConfig(bits=4, smoother={"type": "mad"}).smoother.smooth
    direct = SmoothMAD()

    assert isinstance(from_dict, SmoothMAD)
    assert from_dict.k == pytest.approx(direct.k)
    assert from_type.k == pytest.approx(direct.k)


def test_valid_smoother_and_fallback_dicts_still_compile():
    assert RTNConfig(bits=4, smoother={"type": "mad", "k": 1.0}).smoother.smooth.k == pytest.approx(1.0)

    wrapped = RTNConfig(
        bits=4, smoother={"smooth": {"type": "percentile", "percentile": 95}}
    ).smoother.smooth
    assert isinstance(wrapped, SmoothPercentile)
    assert wrapped.percentile == pytest.approx(95.0)

    thresholded = RTNConfig(
        bits=4, smoother={"type": "mad", "group_size_threshold": 64}
    ).smoother.smooth
    assert thresholded.group_size_threshold == 64

    fallback = GPTQConfig(fallback={"strategy": "rtn", "threshold": "5%"}).fallback
    assert fallback.strategy is FallbackStrategy.RTN
    assert fallback.threshold == "5%"

    # The documented legacy aliases must keep working.
    for legacy, expected in (
        ({"clip_method": "mad"}, SmoothMAD),
        ({"smooth_method": "mad"}, SmoothMAD),
        ({"smooth_mad_k": 2.5}, SmoothMAD),
        ({"smooth_percentile": 95}, SmoothPercentile),
        ({"smooth": {"type": "percentile", "percentile": 90}}, SmoothPercentile),
    ):
        resolved = GPTQConfig(fallback={"strategy": "rtn", **legacy}).fallback.smooth
        assert isinstance(resolved, expected), legacy


def test_rtn_smoother_round_trip_still_works():
    """RTN applies only its top-level smoother; `fallback.smooth` is refused (#5401)."""

    config = RTNConfig(
        bits=4,
        group_size=128,
        smoother={"type": "mad", "k": 2.5},
        fallback={"strategy": "rtn", "threshold": "5%"},
    )
    restored = QuantizeConfig.from_quant_config(config.to_dict())

    assert isinstance(restored.smoother.smooth, SmoothMAD)
    assert restored.smoother.smooth.k == pytest.approx(2.5)
    assert restored.fallback.smooth is None


@pytest.mark.parametrize("config_cls", [GPTQConfig, QQQConfig], ids=["gptq", "qqq"])
def test_calibrated_still_round_trips_the_fallback_smoother_it_applies(config_cls):
    """`fallback.smooth` is applied by the calibrated fallback path (#5302 control)."""

    config = config_cls(
        bits=4,
        group_size=128,
        fallback={
            "strategy": "rtn",
            "threshold": "5%",
            "smooth": {"type": "percentile", "percentile": 95},
        },
    )
    restored = QuantizeConfig.from_quant_config(config.to_dict())

    assert restored.smoother is None
    assert isinstance(restored.fallback.smooth, SmoothPercentile)
    assert restored.fallback.smooth.percentile == pytest.approx(95.0)
