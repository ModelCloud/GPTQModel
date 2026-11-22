# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.adapter.peft import LoraConfig


def test_layers_pattern_accepts_the_zero_index():
    config = LoraConfig(
        r=2,
        target_modules=["q_proj"],
        layers_to_transform=0,
        layers_pattern="layers",
    )

    assert config.layers_to_transform == 0


def test_layers_pattern_accepts_another_integer_index():
    config = LoraConfig(
        r=2,
        target_modules=["q_proj"],
        layers_to_transform=3,
        layers_pattern="layers",
    )

    assert config.layers_to_transform == 3


def test_layers_pattern_accepts_a_list_containing_zero():
    config = LoraConfig(
        r=2,
        target_modules=["q_proj"],
        layers_to_transform=[0],
        layers_pattern="layers",
    )

    assert config.layers_to_transform == [0]


def test_layers_pattern_still_requires_an_index():
    with pytest.raises(ValueError, match="layers_to_transform"):
        LoraConfig(r=2, target_modules=["q_proj"], layers_pattern="layers")
