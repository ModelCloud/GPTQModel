# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.solar_open2 import SolarOpen2QModel
from gptqmodel.models.ports.solar_open2 import SolarOpen2Config, SolarOpen2ForCausalLM
from tests.port_assets import read_port_asset, read_port_json
from tests.test_solar_open2_local_port import tiny_values


PORT = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/solar_open2"


def test_pinned_release_fixture_and_license_hashes():
    manifest = json.loads((PORT / "source_manifest.json").read_text())
    for local, source in [
        ("LICENSE", "TRANSFORMERS_LICENSE"),
        ("SOLAR_LICENSE", "LICENSE"),
        ("released_config.json", "config.json"),
        ("released_generation_config.json", "generation_config.json"),
    ]:
        assert hashlib.sha256((PORT / local).read_bytes()).hexdigest() == manifest["sha256"][source]
    assert (
        hashlib.sha256(read_port_asset(PORT / "released_shapes.json")).hexdigest()
        == manifest["header_inventory"]["sha256"]
    )
    assert len(json.loads((PORT / "header_manifest.json").read_text())) == manifest["header_inventory"]["shards"] == 94


def test_entire_released_checkpoint_inventory_matches_real_meta_model():
    config = SolarOpen2Config.from_pretrained(PORT / "released_config.json")
    with torch.device("meta"):
        model = SolarOpen2ForCausalLM(config)
    released = read_port_json(PORT / "released_shapes.json")
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert len(actual) == 47019
    assert actual == {key: value["shape"] for key, value in released.items()}
    assert sum(kind == "full_attention" for kind in config.layer_types) == 12
    assert sum(kind == "linear_attention" for kind in config.layer_types) == 36
    del model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("full_proj", [False, True])
@pytest.mark.parametrize("dense_first", [False, True])
def test_all_live_checkpoint_weight_owners_have_exact_module_plan_tags(device, kv_heads, full_proj, dense_first):
    config = SolarOpen2Config(
        **tiny_values(
            num_key_value_heads=kv_heads,
            kda_use_full_proj=full_proj,
            first_k_dense_replace=int(dense_first),
            use_qk_norm=True,
            attention_bias=True,
            use_gqa_gate_bias=True,
            linear_attn_config={"head_dim": 8, "num_heads": 4, "num_kv_heads": kv_heads, "short_conv_kernel_size": 2},
        )
    )
    with torch.device(device):
        model = SolarOpen2ForCausalLM(config)
    wrapper = SolarOpen2QModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    for name, layer in enumerate(model.model.layers):
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if key.endswith((".weight", ".bias")) else key
            spec = plan.find(owner)
            assert spec is not None, (name, key)
            assert not spec.capture_only
            if owner.startswith("self_attn."):
                assert spec.quantize == (owner.split(".")[1] in {"q_proj", "k_proj", "v_proj", "g_proj", "o_proj"})
            elif owner.startswith("mlp."):
                assert spec.quantize == ("gate" not in owner.split(".") and "e_score_correction_bias" not in owner)
            else:
                assert not spec.quantize
            assert plan.find(owner + ".foreign") is None
        assert layer.layer_type == ("full_attention" if name == 0 else "linear_attention")
    assert wrapper.awq_get_modules_for_scaling(None, None, None) == []
    assert SolarOpen2QModel.fusable is None
    assert MODEL_MAP["solar_open2"] is SolarOpen2QModel


def test_plan_proposal_is_the_exact_model_owned_contract():
    proposal = json.loads((PORT / "central_fixture_proposals.json").read_text())
    assert (
        SolarOpen2QModel.build_layer_modules(SolarOpen2QModel.module_tree, include_capture_only=True)
        == proposal["registered_module_plans.cases.SolarOpen2QModel/gptq"]["blocks"]
    )
    assert (
        SolarOpen2QModel.extract_layers_node()
        == proposal["registered_module_plans.cases.SolarOpen2QModel/gptq"]["layers"]
    )
