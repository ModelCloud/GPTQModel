# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Spark2.5 checkpoint geometry and executable module-plan ownership."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.spark2_5 import Spark2_5QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.spark2_5.configuration_spark import Spark2_5Config
from gptqmodel.models.ports.spark2_5.modeling_spark import Spark2_5ForCausalLM
from tests.port_assets import DATA
from tests.test_spark2_5_local_port import tiny_config


# sha256 of the sorted [key, shape] pairs from the pinned Spark-X2.5-4B safetensors headers
# (290 tensors; the tied output head is not serialized).
CHECKPOINT_SHAPES_SHA256 = "27ce703dd46fc370956fe6f965ac05884377bf413ae1cd6ef509475ec8fcdb1a"


def test_all_released_checkpoint_keys_and_shapes_match_exact_meta_model():
    config = Spark2_5Config.from_pretrained(DATA / "spark2_5/fixtures")
    with torch.device("meta"):
        model = Spark2_5ForCausalLM(config)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    # The published safetensors omit the tied output head, as native save does.
    lm_head_shape = actual.pop("lm_head.weight")
    assert lm_head_shape == actual["model.embedding.weight"]
    assert len(actual) == 290
    canonical = json.dumps(sorted(actual.items()), separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == CHECKPOINT_SHAPES_SHA256
    assert model.lm_head.weight is model.model.embedding.weight


@pytest.mark.parametrize(
    "bias,gate,kv_heads,device", [(False, False, 1, "meta"), (True, True, 2, "cpu"), (False, True, 4, "meta")]
)
def test_real_weight_owners_resolve_to_exact_module_plan(bias, gate, kv_heads, device):
    config = tiny_config(
        attention_bias=bias, mlp_bias=bias, headwise_attn_output_gate=gate, num_key_value_heads=kv_heads
    )
    with torch.device(device):
        model = Spark2_5ForCausalLM(config)
    _convert_model_with_defuser(Spark2_5QModel, model, cleanup_original=False)
    wrapper = Spark2_5QModel.__new__(Spark2_5QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(Spark2_5QModel.module_tree)
    declared = {
        name for group in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in group
    }
    expected = {"self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert declared == expected
    for layer in model.model.layers:
        owners = expected | {"input_layernorm", "post_attention_layernorm"} | ({"self_attn.g_proj"} if gate else set())
        keys = {name + ".weight" for name in owners}
        if bias:
            keys |= {name + ".bias" for name in owners - {"input_layernorm", "post_attention_layernorm"}}
        assert set(layer.state_dict()) == keys
        for key in keys:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize is (owner in expected)
            assert plan.find(owner + ".undeclared") is None
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert Spark2_5QModel.fusable is None
