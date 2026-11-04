# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``ExpertsRoutingBypass.batch_size`` has to be a real, positive chunk size.

The value is consumed in one place, behind a ``batch_size > 0`` guard:

    if is_moe_subset and batch_size is not None and batch_size > 0 and execute_forward:

so anything else degrades to "one chunk" without a word. That is the opposite of
what the option is for - it exists to keep expert modules from all being resident
at once on large MoE checkpoints.
"""

import pytest

from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig


def test_none_keeps_the_single_chunk_default():
    assert ExpertsRoutingBypass().batch_size is None
    assert ExpertsRoutingBypass(batch_size=None).batch_size is None


@pytest.mark.parametrize("value", [1, 2, 4, 1024])
def test_positive_int_is_accepted(value):
    assert ExpertsRoutingBypass(batch_size=value).batch_size == value


@pytest.mark.parametrize("value", [0, -1, -3])
def test_non_positive_is_rejected(value):
    with pytest.raises(ValueError, match="positive int or None"):
        ExpertsRoutingBypass(batch_size=value)


@pytest.mark.parametrize("value", ["2", 2.0, True, [2], {"size": 2}])
def test_non_int_is_rejected(value):
    with pytest.raises(ValueError, match="positive int or None"):
        ExpertsRoutingBypass(batch_size=value)


def test_moe_config_wraps_the_validation():
    with pytest.raises(ValueError, match="positive int or None"):
        MoEConfig(routing=ExpertsRoutingBypass(batch_size=0))
