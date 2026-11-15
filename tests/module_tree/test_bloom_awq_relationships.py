# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bloom replay policy must not hide quantizable or excluded weight owners."""

import pytest
from torch import nn
from transformers import BloomConfig, BloomForCausalLM

from gptqmodel.models.definitions.bloom import BloomQModel
from gptqmodel.models.module_plan import ModulePlan


@pytest.mark.parametrize("post_layernorm", [False, True])
def test_bloom_live_weight_owners_remain_in_module_plan(post_layernorm):
    config = BloomConfig(
        hidden_size=32, n_layer=1, n_head=4, vocab_size=32, apply_residual_connection_post_layernorm=post_layernorm
    )
    model = BloomForCausalLM(config).eval()
    wrapper = BloomQModel.__new__(BloomQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    quantizable = {name for block in wrapper.simple_layer_modules(config, None) for name in block}
    assert quantizable == {
        "self_attention.query_key_value",
        "self_attention.dense",
        "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h",
    }
    for name in model.transformer.h[0].state_dict():
        owner = name.rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, name
        if spec.quantize:
            assert owner in quantizable
        else:
            assert set(spec.flags) & {"!", "?"}
