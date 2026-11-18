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
from tests.port_assets import read_port_asset
from tests.test_spark2_5_local_port import FIXTURES, tiny_config


def test_all_released_checkpoint_keys_and_shapes_match_exact_meta_model():
    config = Spark2_5Config.from_pretrained(FIXTURES)
    with torch.device("meta"):
        model = Spark2_5ForCausalLM(config)
    expected = json.loads((FIXTURES / "checkpoint_shapes.json").read_text())
    manifest = json.loads((FIXTURES / "source_sha256.json").read_text())["sha256"]
    index_bytes = read_port_asset(FIXTURES / "model.safetensors.index.json")
    assert hashlib.sha256(index_bytes).hexdigest() == manifest["model.safetensors.index.json"]
    weight_map = json.loads(index_bytes)["weight_map"]
    published = {}
    for shard in sorted(set(weight_map.values())):
        name = shard + ".header.json"
        payload = read_port_asset(FIXTURES / name)
        assert hashlib.sha256(payload).hexdigest() == manifest[name]
        for key, metadata in json.loads(payload).items():
            if key == "__metadata__":
                continue
            assert key not in published
            assert weight_map[key] == shard
            published[key] = metadata["shape"]
    assert published == expected
    assert set(weight_map) == set(published)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    # The published safetensors omit the tied output head, as native save does.
    lm_head_shape = actual.pop("lm_head.weight")
    assert lm_head_shape == expected["model.embedding.weight"]
    assert actual == expected
    assert len(expected) == 290
    assert model.lm_head.weight is model.model.embedding.weight


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
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
