# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``ExpertsRoutingBypass.batch_size`` has to be a real, positive chunk size.

The value is consumed behind a ``batch_size > 0`` guard in the staged executor,
so zero or a negative value degrades to "one chunk" without a word - the opposite
of what the option is for. It exists to keep expert modules from all being
resident at once on large MoE checkpoints.

Numeric strings are coerced, matching ``ExpertsRoutingOverride`` which does the
same for ``num_experts_per_tok``, so a value that came through JSON still works.
"""

import pytest

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    MoEConfig,
    normalize_bypass_batch_size,
)


def test_none_keeps_the_single_chunk_default():
    assert ExpertsRoutingBypass().batch_size is None


@pytest.mark.parametrize("value", [1, 2, 4, 1024])
def test_positive_int_is_accepted(value):
    assert ExpertsRoutingBypass(batch_size=value).batch_size == value


@pytest.mark.parametrize("value,expected", [("2", 2), (" 16 ", 16)])
def test_numeric_string_is_coerced_like_the_other_routing_mode(value, expected):
    assert ExpertsRoutingBypass(batch_size=value).batch_size == expected


@pytest.mark.parametrize("value", [0, -1, -3, 2.0, True, "0", "-3", "abc", [2], {"size": 2}])
def test_values_that_cannot_batch_are_rejected(value):
    with pytest.raises(ValueError, match="positive int or None"):
        ExpertsRoutingBypass(batch_size=value)


def test_moe_config_propagates_the_rejection():
    with pytest.raises(ValueError, match="positive int or None"):
        MoEConfig(routing=ExpertsRoutingBypass(batch_size=0))


def test_dict_payload_round_trip_accepts_an_int_and_coerces_a_string():
    """The deserialization path the fix is about: routing arrives as a dict."""
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        moe={"routing": {"class": "ExpertsRoutingBypass", "batch_size": "2"}},
    )
    assert config.moe.routing.batch_size == 2


def test_dict_payload_rejects_zero():
    with pytest.raises(ValueError, match="positive int or None"):
        QuantizeConfig(
            bits=4,
            group_size=128,
            moe={"routing": {"class": "ExpertsRoutingBypass", "batch_size": 0}},
        )


def test_mutated_value_still_cannot_silently_disable_batching():
    """The attribute is writable after construction; the consumer re-validates."""
    routing = ExpertsRoutingBypass(batch_size=4)
    routing.batch_size = 0
    with pytest.raises(ValueError, match="positive int or None"):
        normalize_bypass_batch_size(routing.batch_size)
