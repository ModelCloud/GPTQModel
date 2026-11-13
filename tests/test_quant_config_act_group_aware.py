# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""`act_group_aware` must be a no-op when one group spans the whole input."""

from gptqmodel.quantization.config import GPTQConfig


def test_group_size_minus_one_disables_act_group_aware():
    config = GPTQConfig(bits=8, group_size=-1)
    assert config.act_group_aware is False


def test_explicit_act_group_aware_is_disabled_for_group_size_minus_one():
    config = GPTQConfig(bits=8, group_size=-1, act_group_aware=True)
    assert config.act_group_aware is False


def test_grouped_quantization_keeps_act_group_aware():
    config = GPTQConfig(bits=8, group_size=128)
    assert config.act_group_aware is True


def test_explicit_act_group_aware_survives_grouped_quantization():
    config = GPTQConfig(bits=8, group_size=128, act_group_aware=True)
    assert config.act_group_aware is True
