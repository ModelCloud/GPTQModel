# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoother bounds and per-module ``dynamic`` smoothers must fail early (#5005).

``SmoothPercentileAsymmetric`` reads the same 0-100 scale as
``SmoothPercentile`` through ``fallback_smooth._quantile``. ``#4970`` rejected a
fraction ``percentile`` on ``SmoothPercentile``/``SmoothLog`` but left the
asymmetric sibling unguarded: ``low=0.005, high=0.995`` (meant 0.5% / 99.5%)
constructed cleanly, and with ``high <= 1`` the upper bound lands next to the row
minimum, so ``_clamp_block`` pins almost every weight to the bottom tail. The
model is destroyed with no error or warning.

The second half is the same class of bug: a smoother inside a per-module
``dynamic`` override is only merged when the module is first matched, so an
invalid one raised from the middle of ``quantize()`` -- after the model had
already been mutated -- with no module name in the message.
"""

import pytest
import torch

from gptqmodel.quantization import config as C
from gptqmodel.quantization.fallback_smooth import _clamp_block, _quantile


# The three cases from the issue. The first two put the *upper* bound at or
# below 1, which is the destructive form; the third keeps a sane upper bound and
# only passes a fractional lower bound.
FRACTION_BOUNDS = [
    ("both-fractions", 0.005, 0.995, "high"),
    ("fraction-high", 0.5, 0.99, "high"),
    ("fraction-low", 0.005, 99.5, "low"),
]


@pytest.mark.parametrize(
    "low,high,field",
    [(low, high, field) for _, low, high, field in FRACTION_BOUNDS],
    ids=[case_id for case_id, *_ in FRACTION_BOUNDS],
)
def test_asymmetric_fraction_bounds_are_rejected(low, high, field):
    with pytest.raises(ValueError) as excinfo:
        C.SmoothPercentileAsymmetric(low=low, high=high)

    message = str(excinfo.value)
    assert field in message, message
    # The remedy must name the scale the field is read on.
    assert "0-100" in message, message


LEGAL_BOUNDS = [
    C.SmoothPercentileAsymmetric,
    lambda: C.SmoothPercentileAsymmetric(low=0.5, high=99.5),
    lambda: C.SmoothPercentileAsymmetric(low=0.0, high=100.0),
    lambda: C.SmoothPercentileAsymmetric(low=1.0, high=99.9),
    lambda: C.SmoothPercentileAsymmetric(low=0.1, high=99.9),
    lambda: C.SmoothPercentileAsymmetric(low=2.5, high=97.5),
]


@pytest.mark.parametrize("factory", LEGAL_BOUNDS)
def test_legitimate_asymmetric_bounds_still_construct(factory):
    smoother = factory()
    assert isinstance(smoother, C.SmoothPercentileAsymmetric)
    assert 0.0 <= smoother.low < smoother.high <= 100.0


def test_percentile_and_log_rules_from_4970_are_unchanged():
    """The symmetric smoothers keep rejecting a fraction ``percentile``."""

    with pytest.raises(ValueError, match="`percentile`"):
        C.SmoothPercentile(percentile=0.99)
    with pytest.raises(ValueError, match="`percentile`"):
        C.SmoothLog(percentile=0.99)
    assert C.SmoothPercentile(percentile=99).percentile == 99.0
    assert C.SmoothLog(percentile=99).percentile == 99.0


def test_dict_and_wrapper_paths_reject_the_same_asymmetric_bounds():
    """A checkpoint's JSON and the ``SmootherConfig`` wrapper must fail too."""

    with pytest.raises(ValueError, match="`high`"):
        C.SmootherConfig(smooth={"type": "percentile_asym", "low": 0.5, "high": 0.99})
    with pytest.raises(ValueError, match="`high`"):
        C.Fallback(smooth={"type": "percentile_asym", "low": 0.005, "high": 0.995})
    with pytest.raises(ValueError, match="`low`"):
        C.SmootherConfig(smooth={"type": "percentile_asymmetric", "low": 0.005})

    parsed = C.SmootherConfig(smooth={"type": "percentile_asym", "low": 0.5, "high": 99.5})
    assert isinstance(parsed.smooth, C.SmoothPercentileAsymmetric)
    assert (parsed.smooth.low, parsed.smooth.high) == (0.5, 99.5)


def _assert_dynamic_rejects(dynamic, field, pattern):
    with pytest.raises((ValueError, TypeError)) as excinfo:
        C.GPTQConfig(bits=4, group_size=128, dynamic=dynamic)

    message = str(excinfo.value)
    # The offending rule has to be named: the runtime error used to point at the
    # smoother only, far from the config that caused it.
    assert pattern in message, message
    assert field in message, message


def test_dynamic_fallback_smoother_is_validated_at_construction():
    _assert_dynamic_rejects(
        {"+:.*q_proj.*": {"fallback": {"smooth": {"type": "mad", "k": 0}}}},
        "fallback",
        "+:.*q_proj.*",
    )
    _assert_dynamic_rejects(
        {"model.layers.0.mlp.down_proj": {"fallback": {"smooth": {"type": "mad", "k": -1.0}}}},
        "fallback",
        "model.layers.0.mlp.down_proj",
    )


def test_dynamic_preprocessor_smoother_is_validated_at_construction():
    _assert_dynamic_rejects(
        {
            "+:.*q_proj.*": {
                "preprocessors": [{"code": "smoother", "smooth": {"type": "mad", "k": 0}}]
            }
        },
        "preprocessors",
        "+:.*q_proj.*",
    )
    _assert_dynamic_rejects(
        {
            "+:.*k_proj.*": {
                "preprocessors": [
                    {"code": "smoother", "smooth": {"type": "percentile_asym", "high": 0.99}}
                ]
            }
        },
        "preprocessors",
        "+:.*k_proj.*",
    )


def test_dynamic_smoother_field_is_validated_at_construction():
    _assert_dynamic_rejects(
        {"+:.*q_proj.*": {"smoother": {"type": "percentile_asym", "low": 0.5, "high": 0.99}}},
        "smoother",
        "+:.*q_proj.*",
    )


def test_valid_dynamic_smoothers_still_construct_and_are_not_rewritten():
    """Validation must not mutate the rule; serialization stays raw-dict based."""

    rule = {
        "fallback": {"smooth": {"type": "mad", "k": 2.75}},
        "preprocessors": [{"code": "smoother", "smooth": {"type": "percentile", "percentile": 99.0}}],
    }
    config = C.GPTQConfig(bits=4, group_size=128, dynamic={"+:.*q_proj.*": rule})

    stored = config.dynamic["+:.*q_proj.*"]
    assert stored["fallback"] == {"smooth": {"type": "mad", "k": 2.75}}
    assert stored["preprocessors"] == [
        {"code": "smoother", "smooth": {"type": "percentile", "percentile": 99.0}}
    ]
    serialized = config.to_dict()["dynamic"]["+:.*q_proj.*"]
    assert serialized["fallback"]["smooth"]["type"] == "mad"


def test_rejected_upper_bound_would_clamp_almost_every_weight():
    """Document the mechanism: ``high <= 1`` pins the block to the bottom tail."""

    generator = torch.Generator().manual_seed(0)
    block = torch.randn(8, 128, generator=generator)

    def clamped_fraction(high: float) -> float:
        hi = _quantile(block, high)
        clamped = _clamp_block(block, _quantile(block, 0.5), hi)
        return float((clamped != block).float().mean())

    # A 0.99 "percentile" is the 0.99th percentile: nearly the row minimum, so
    # almost every weight is clipped. The documented 99.5 clips only the top
    # half of one percent.
    assert clamped_fraction(0.99) > 0.9
    assert clamped_fraction(99.5) < 0.02
