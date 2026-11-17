# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from test_minimax_m2_local_port import ASSETS, tiny_model

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan


@pytest.mark.parametrize(
    "model_type,layout", [("minimax", "mixed"), ("minimax", "full"), ("minimax", "linear"), ("minimax_m2", "mixed")]
)
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_real_weight_owners_and_subset_order(model_type, layout, kv, device):
    definition = MODEL_MAP[model_type]
    with torch.device(device):
        model = tiny_model(model_type, layout=layout, kv=kv)
        _convert_model_with_defuser(definition, model, cleanup_original=False)
    plan = ModulePlan.from_tree(definition.module_tree)
    blocks = definition.simple_layer_modules(model_config=model.config, quantize_config=SimpleNamespace(dynamic=None))
    declared = {name for block in blocks for name in block}
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if key.endswith((".weight", ".bias")) else key
            spec = plan.find(owner)
            assert spec is not None, (model_type, owner)
            assert spec.quantize == (owner in declared)
            if not spec.quantize:
                assert "!" in spec.flags or "?" in spec.flags
            assert plan.find(owner + ".undeclared") is None
        live_linears = {name for name, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        assert live_linears - {"mlp.gate", "coefficient"} <= declared
        for expert in range(model.config.num_local_experts):
            up = next(i for i, block in enumerate(blocks) if f"mlp.experts.{expert}.up_proj" in block)
            down = next(i for i, block in enumerate(blocks) if f"mlp.experts.{expert}.down_proj" in block)
            assert up < down


@pytest.mark.parametrize("postnorm", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_original_optional_shared_expert_is_a_real_bound_owner(postnorm, device):
    definition = MODEL_MAP["minimax"]
    with torch.device(device):
        model = tiny_model("minimax", shared=True, postnorm=postnorm)
        _convert_model_with_defuser(definition, model, cleanup_original=False)
    plan = ModulePlan.from_tree(definition.module_tree)
    for layer in model.model.layers:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            owner = f"shared_mlp.{projection}"
            assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
            assert plan.find(owner).quantize
        assert not plan.find("coefficient").quantize
        assert layer.postnorm is postnorm


@pytest.mark.parametrize("model_type", ["minimax", "minimax_m2"])
def test_owned_registry_proposal_pins_reviewed_blocks(model_type):
    definition = MODEL_MAP[model_type]
    proposal = json.loads((ASSETS / "module_plan_cases.json").read_text())[f"{definition.__name__}/gptq"]
    assert definition.extract_layers_node() == proposal["layers"]
    assert definition.build_layer_modules(definition.module_tree, include_capture_only=True) == proposal["blocks"]
    with pytest.raises(ValueError):
        require_fusable_config(definition, model_type)
