# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Raon decoder owners, including byte-compressed experts and mHC controls."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.raon1_mini import Raon1MiniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.raon1_mini.configuration_raon1_mini import Raon1MiniConfig
from gptqmodel.models.ports.raon1_mini.modeling_raon1_mini import Raon1MiniForCausalLM
from tests.test_raon1_mini_local_port import _config


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/raon1_mini"


def test_reviewed_independent_registry_proposal_pins_rda_mla_and_moe_controls():
    proposal = json.loads((PORT / "central_fixture_proposals.json").read_text())
    expected = proposal["registered_module_plans.cases"]["Raon1MiniQModel/gptq"]
    assert Raon1MiniQModel.extract_layers_node() == expected["layers"]
    assert (
        Raon1MiniQModel.build_layer_modules(Raon1MiniQModel.module_tree, include_capture_only=True)
        == expected["blocks"]
    )


@pytest.mark.parametrize("layout", ["hybrid", "reverse", "rda", "mla"])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"num_experts": None},
        {"first_k_dense_replace": 0, "num_shared_experts": 0},
        {"first_k_dense_replace": 0, "moe_layer_freq": 2},
        {"raon_mhc_streams": 3, "num_attention_heads": 2},
    ],
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_live_raon_module_plan_owns_every_decoder_tensor(layout, settings, device):
    config = _config(layout, **settings)
    with torch.device(device):
        model = Raon1MiniForCausalLM(config)
    _convert_model_with_defuser(Raon1MiniQModel, model, cleanup_original=False)
    plan = ModulePlan.from_tree(Raon1MiniQModel.module_tree)
    declared = {
        name for group in Raon1MiniQModel.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in group
    }
    for layer in model.model.layers:
        linears = {name for name, module in layer.named_modules() if isinstance(module, nn.Linear)}
        assert linears <= declared
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if "." in key else key
            if key.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
                owner = key
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in linears), key
            assert plan.find(owner + ".undeclared") is None
            if ".experts." in owner:
                assert plan.is_expert(owner)
        for name in linears:
            assert plan.find(name).quantize
        assert Raon1MiniQModel.awq_get_modules_for_scaling(None, layer, {}, {}) == []


def test_all_19434_released_tensor_shapes_and_storage_dtypes_match_exact_meta_owners():
    config = Raon1MiniConfig(**json.loads((PORT / "released_config.json").read_text()))
    expected = json.loads(gzip.decompress((PORT / "released_tensors.json.gz").read_bytes()))
    assert len(expected) == 19434
    with torch.device("meta"):
        model = Raon1MiniForCausalLM._from_config(config, dtype=torch.bfloat16)
    dtype_names = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}
    actual = {
        name: {"shape": list(value.shape), "dtype": dtype_names[value.dtype]}
        for name, value in model.state_dict().items()
    }
    assert actual == expected
    assert sum(not layer.is_rda for layer in model.model.layers) == 7
    assert sum(layer.is_rda for layer in model.model.layers) == 18
    plan = ModulePlan.from_tree(Raon1MiniQModel.module_tree)
    for key in expected:
        if not key.startswith("model.layers."):
            assert key in {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}
            continue
        relative = ".".join(key.split(".")[3:])
        owner = (
            relative
            if relative.endswith(("A_log", "dt_bias", "e_score_correction_bias"))
            else relative.rsplit(".", 1)[0]
        )
        spec = plan.find(owner)
        assert spec is not None, key


def test_raon_stays_explicitly_unqualified_for_fusion():
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.fusable import require_fusable_config

    assert MODEL_MAP["raon1_mini"] is Raon1MiniQModel
    assert Raon1MiniQModel.fusable is None
    with pytest.raises(ValueError):
        require_fusable_config(Raon1MiniQModel, "raon1_mini")
