# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch

from gptqmodel.models.definitions.ovis2_5 import Ovis2_5QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from tests.port_assets import read_port_json
from tests.test_ovis2_6_moe_local_port import CASES, FIXTURES, build


PORT_FIXTURES = Path(__file__).parents[2] / "gptqmodel/models/ports/ovis2_6_moe/fixtures"


@pytest.mark.parametrize("definition,config_class", CASES)
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("kv,bias", [(4, False), (2, True), (1, False)])
def test_real_decoder_every_checkpoint_owner_has_explicit_plan(definition, config_class, device, kv, bias):
    with torch.device(device):
        model = build(definition, config_class, kv=kv, bias=bias)
    plan = ModulePlan.from_tree(definition.module_tree)
    for layer in model.llm.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (config_class.model_type, key)
            module = layer.get_submodule(owner)
            if spec.quantize:
                assert isinstance(module, torch.nn.Linear)
            if "experts." in owner:
                assert plan.is_expert(owner)
            else:
                assert not plan.is_expert(owner)
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert plan.find(name) is not None
                assert not plan.find(name + ".undeclared")
    assert definition.get_num_experts(model.config) == 3
    assert definition.extract_layers_node() == ["llm.model.layers"]
    # The decoders use the generic AWQ planner, not Ovis2.5's plain policy.
    assert definition.awq_get_modules_for_scaling is not Ovis2_5QModel.awq_get_modules_for_scaling
    assert definition.fusable is None


@pytest.mark.parametrize("size,case", list(zip(["30B", "80B"], CASES)))
def test_every_released_safetensors_header_shape_matches_meta(size, case):
    definition, config_class = case
    config = config_class(**json.loads((FIXTURES / f"{size}_config.json").read_text()))
    with torch.device("meta"):
        model = definition.loader.from_config(config)
        _convert_model_with_defuser(definition, model, cleanup_original=True)
    expected = read_port_json(PORT_FIXTURES / f"{size}_shapes.json")
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert len(expected) == (19308 if size == "30B" else 74832)
    assert actual == expected
    assert all(value.device.type == "meta" for value in model.state_dict().values())
