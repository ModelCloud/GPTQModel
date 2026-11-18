# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real TeleChat checkpoint owners, not tree-generated stand-ins."""

import hashlib
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
from tests.test_telechat2_local_port import _config


def _digest(lines):
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


# Geometry from each pinned release config.json, with the sha256 of the sorted
# tensor names of its weight index and, for 35B/115B, of "name:shape" read from
# the real safetensors headers (computed once from the upstream files).
RELEASES = {
    "3B": ((3072, 24, 24, 24, 6144, 131072, False), 243,
           "a234ba91b58ca3ccde2aa556afcfebd5f7524ac20baf686be63fd2f4c861615c", None),
    "7B-32K": ((4096, 32, 32, 30, 12288, 131072, False), 303,
               "53283293cabbdecd775818619c7969c5e3b8b29e30c04602302f9d93955bde9f", None),
    "35B-32K": ((6144, 48, 48, 64, 20480, 131072, False), 643,
                "ef2e2bae9c6c4e9ae6603a8eb1321d1a0d2b0b48fb4cf66bc91bc9901301c39f",
                "7b825c9b5c0ba1bafcfb729f0a053af7d2b7cf476626bc0863eb9cf8a7e0326a"),
    "115B": ((8192, 64, 8, 96, 40960, 131072, False), 963,
             "6979954d2ba1fe22786c7af5889459ad69d83575d9c7976b2255824b8ac73739",
             "96f7d78a1fdd9b41157f04eddc8598da1f5912459aec722cdd8983a2e0f6b954"),
    "TeleChat1-7B": ((4096, 32, 32, 30, 12288, 160256, False), 302,
                     "c43507c317160a5a482b44db1fa030fd81f8d000f0810da752bdb1385d9425a1", None),
}


@pytest.mark.parametrize("release", list(RELEASES))
def test_full_pinned_release_inventory_matches_meta(release):
    (hidden, heads, kv, layers, ffn, vocab, embednorm), count, keys_sha, shapes_sha = RELEASES[release]
    config = Telechat2Config(hidden_size=hidden, n_head=heads, num_key_value_heads=kv, n_layer=layers,
                             ffn_hidden_size=ffn, vocab_size=vocab, embed_layernorm=embednorm)
    with torch.device("meta"):
        model_class = TelechatModel if release == "TeleChat1-7B" else Telechat2ForCausalLM
        model = model_class(config)
    state = model.state_dict()
    assert len(state) == count and all(value.is_meta for value in state.values())
    assert _digest(state) == keys_sha
    if shapes_sha is not None:
        assert _digest(f"{name}:{list(value.shape)}" for name, value in state.items()) == shapes_sha


@pytest.mark.parametrize("kvheads", [1, 4])
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
