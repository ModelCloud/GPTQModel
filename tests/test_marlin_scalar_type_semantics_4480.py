# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``ScalarType`` NaN/IEEE predicates must compare ``NanRepr`` members (#4480).

``NanRepr`` is an ordinary ``Enum``, so its members never compare equal to their
integer values. ``has_nans()``/``is_ieee_754()`` compared the member against
``NanRepr.NONE.value``/``NanRepr.IEEE_754.value``, which is always False. A
no-NaN type therefore reported ``has_nans() == True``, standard IEEE FP16
reported ``is_ieee_754() == False``, and ``__str__`` appended a spurious ``n``
suffix.
"""

import pytest

from gptqmodel.utils.marlin_scalar_type import NanRepr, ScalarType, scalar_types


def test_no_nan_type_reports_no_nans():
    assert scalar_types.float4_e2m1f.nan_repr is NanRepr.NONE
    assert scalar_types.float4_e2m1f.has_nans() is False


def test_ieee_float_type_reports_ieee_and_plain_name():
    fp16 = scalar_types.float16

    assert fp16.nan_repr is NanRepr.IEEE_754
    assert fp16.has_nans() is True
    assert fp16.is_ieee_754() is True
    # No `f`/`n` flags: the type follows IEEE 754 conventions.
    assert str(fp16) == "float16_e5m10"
    assert repr(fp16) == "ScalarType.float16_e5m10"

    bf16 = scalar_types.bfloat16
    assert bf16.is_ieee_754() is True
    assert str(bf16) == "float16_e8m7"


def test_non_ieee_types_keep_their_suffix_flags():
    # NaN via an extended-range max/min encoding, finite values allowed.
    assert scalar_types.float8_e4m3fn.is_ieee_754() is False
    assert str(scalar_types.float8_e4m3fn) == "float8_e4m3fn"
    # Finite-only, no NaNs at all.
    assert scalar_types.float4_e2m1f.is_ieee_754() is False
    assert str(scalar_types.float4_e2m1f) == "float4_e2m1f"


@pytest.mark.parametrize("raw", [NanRepr.NONE, NanRepr.IEEE_754, NanRepr.EXTD_RANGE_MAX_MIN])
def test_raw_int_nan_repr_matches_the_enum_member(raw):
    """A caller may mirror the C++ encoding with a plain ``int``."""

    from_enum = ScalarType(5, 10, True, 0, False, raw)
    from_int = ScalarType(5, 10, True, 0, False, raw.value)

    assert from_int.has_nans() is from_enum.has_nans()
    assert from_int.is_ieee_754() is from_enum.is_ieee_754()
    assert str(from_int) == str(from_enum)
    # The C++ interop id packs the normalized member value.
    assert from_int.id == from_enum.id


def test_non_ieee_constructor_rejects_a_raw_ieee_nan_encoding():
    """``float_`` must reject the IEEE encoding whether it is an int or a member."""

    with pytest.raises(AssertionError, match="float_IEEE754"):
        ScalarType.float_(5, 10, False, NanRepr.IEEE_754)
    with pytest.raises(AssertionError, match="float_IEEE754"):
        ScalarType.float_(5, 10, False, NanRepr.IEEE_754.value)


def test_max_min_predicate_matches_the_nan_encoding():
    """``_floating_point_max_int`` must branch on the member, not on ``.value``."""

    ieee = ScalarType.float_IEEE754(5, 10)
    no_nan = ScalarType.float_(5, 10, True, NanRepr.NONE)

    # A no-NaN type spends the reserved max exponent on a finite value, so its
    # representable maximum is larger than the IEEE one.
    assert no_nan._floating_point_max_int() > ieee._floating_point_max_int()


def test_float8_e4m3fn_max_is_finite_and_positive():
    # Regression guard for the assert branch that formats via ``__str__``.
    assert scalar_types.float8_e4m3fn._floating_point_max_int() > 0
