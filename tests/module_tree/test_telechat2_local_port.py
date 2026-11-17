# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real TeleChat checkpoint owners, not tree-generated stand-ins."""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.telechat2 import TeleChat2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.telechat2.configuration_telechat2 import Telechat2Config
from gptqmodel.models.ports.telechat2.modeling_telechat2 import Telechat2ForCausalLM, TelechatModel
from tests.test_telechat2_local_port import FIXTURES, _config


@pytest.mark.parametrize("release", ["3B", "7B-32K", "35B-32K", "115B", "TeleChat1-7B"])
def test_full_pinned_release_inventory_matches_meta(release):
    fixture = json.loads(gzip.decompress((FIXTURES / "checkpoint_inventories.json.gz").read_bytes()))[release]
    with torch.device("meta"):
        model_class = TelechatModel if release == "TeleChat1-7B" else Telechat2ForCausalLM
        model = model_class(Telechat2Config(**fixture["config"]))
    assert set(model.state_dict()) == set(fixture["weight_map"])
    assert sum(value.is_meta for value in model.state_dict().values()) == len(fixture["weight_map"])
    if release in {"35B-32K", "115B"}:
        headers = json.loads(gzip.decompress((FIXTURES / "safetensors_headers.json.gz").read_bytes()))["TeleChat2-" + release]
        independent_shapes = {name: tensor["shape"] for shard in headers.values() for name, tensor in shard["tensors"].items()}
        assert {name: list(value.shape) for name, value in model.state_dict().items()} == independent_shapes


@pytest.mark.parametrize("kvheads", [1, 2, 4])
@pytest.mark.parametrize("postnorm", [False, True])
@pytest.mark.parametrize("embednorm", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_all_decoder_owners_resolve_and_plain_policy(kvheads, postnorm, embednorm, device):
    config = _config(num_key_value_heads=kvheads, embed_layernorm=embednorm,
                     apply_residual_connection_post_layernorm=postnorm)
    with torch.device(device):
        model = Telechat2ForCausalLM(config)
    _convert_model_with_defuser(TeleChat2QModel, model, cleanup_original=False)
    wrapper = TeleChat2QModel.__new__(TeleChat2QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    plan = ModulePlan.from_tree(wrapper.module_tree)
    expected = {"self_attention.query", "self_attention.key_value", "self_attention.dense",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    assert declared == expected
    assert MODEL_MAP["telechat"] is TeleChat2QModel
    for layer in model.transformer.h:
        assert len(layer.state_dict()) == 10
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in expected
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert "!" in spec.flags
            assert plan.find(owner + ".undeclared") is None
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert wrapper.fusable is None
    assert not wrapper.require_trust_remote_code
    assert wrapper.require_dtype is None
