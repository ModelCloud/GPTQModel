# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.models import auto
from gptqmodel.models.definitions.flint import FlintQModel


def test_flint_model_definition_is_registered():
    assert auto.check_and_get_model_definition("flint-fixture") is FlintQModel


def test_flint_model_tree_covers_experts_and_attention_sinks():
    layer = FlintQModel.module_tree[3]
    assert "softmax_offset:!" in layer["self_attn"]
    assert "experts" in layer["mlp:moe"]
    assert "shared_expert" in layer["mlp:moe"]
