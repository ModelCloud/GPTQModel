# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint inventories and real decoder plan owners, without trained payloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.definitions.minicpmv import MiniCPMVQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.minicpmv.configuration_minicpm import MiniCPMVConfig
from gptqmodel.models.ports.minicpmv.modeling_minicpmv import MiniCPMV
from tests.test_minicpmv_local_port import VERSIONS, tiny_config


RELEASES = json.loads((Path(__file__).resolve().parents[1] / "fixtures/ports/minicpmv/release_inventory.json").read_text())


class PlanOnlyMiniCPMV(MiniCPMVQModel):
    # Media loading is qualified through GPTQModel.load in the sibling suite.
    # This constructor receives an in-memory CPU/meta model with no checkpoint.
    require_load_processor = False


EXPECTED_PLAN = {
    "layers": ["llm.model.layers"],
    "blocks": [["input_layernorm:!"],
               ["self_attn.q_norm:!", "self_attn.k_norm:!", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"],
               ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}


@pytest.mark.parametrize("name", sorted(RELEASES))
def test_every_released_checkpoint_key_and_shape(name):
    # Digest of the sorted (key, shape) inventory read from each pinned release's safetensors headers.
    entry = RELEASES[name]
    with torch.device("meta"):
        model = MiniCPMV(MiniCPMVConfig(**entry["config"]))
    rows = sorted([key, list(value.shape)] for key, value in model.state_dict().items())
    assert len(rows) == entry["tensors"]
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv,bias", [(1, False), (2, True), (4, False)])
def test_live_owners_all_attention_layouts(version, device, kv, bias):
    config = tiny_config(version, num_key_value_heads=kv, attention_bias=bias)
    with torch.device(device):
        model = MiniCPMV(config)
    _convert_model_with_defuser(MiniCPMVQModel, model, cleanup_original=False)
    wrapper = PlanOnlyMiniCPMV(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    for layer in model.llm.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                quantized.add(owner)
            else:
                assert "!" in spec.flags, key
        assert len(quantized) == 7
    assert MiniCPMVQModel.fusable is None
    for key in ["self_attn.q_norm.child", "self_attn.q_projection", "mlp.down_proj_extra"]:
        assert plan.find(key) is None


def test_exact_declared_subset_order_and_native_46_separation():
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.definitions.minicpmv_4_6 import MiniCPMV4_6QModel

    assert MODEL_MAP["minicpmv"] is MiniCPMVQModel
    assert MODEL_MAP["minicpmv4_6"] is MiniCPMV4_6QModel
    assert MiniCPMVQModel.simple_layer_modules(tiny_config(), SimpleNamespace(dynamic=None)) == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"],
    ]
    assert MiniCPMVQModel.build_layer_modules(MiniCPMVQModel.module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    assert not MiniCPMVQModel.require_trust_remote_code
