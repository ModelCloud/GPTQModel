# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real CPU/meta Pangu owners, including persistent causal buffers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.pangu_alpha import PanguAlphaQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.pangu_alpha import GPTPanguConfig, GPTPanguForCausalLM
from tests.awq_neutrality import _wrapper
from tests.port_assets import read_port_json


FIXTURES = Path(__file__).parents[2] / "gptqmodel/models/ports/pangu_alpha/fixtures"
EXPECTED_BLOCKS = [
    ["ln_1:!"],
    ["attn:!", "attn.q_proj", "attn.k_proj", "attn.v_proj"],
    ["attn.c_proj"],
    ["ln_2:!"],
    ["mlp.c_fc"],
    ["mlp.c_proj"],
]


def test_exact_real_582_tensor_checkpoint_keys_shapes_on_meta():
    config = GPTPanguConfig(**read_port_json(FIXTURES / "config.json"))
    shapes = read_port_json(FIXTURES / "released_key_shapes.json")
    assert len(shapes) == 582
    with torch.device("meta"):
        model = GPTPanguForCausalLM(config)
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == shapes


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("layers,heads,inner,tied", [(1, 1, None, True), (2, 2, 48, False), (3, 4, 64, True)])
def test_plan_owns_all_state_and_plain_awq_excludes_no_linear(device, layers, heads, inner, tied):
    config = GPTPanguConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=layers,
        num_heads=heads,
        intermediate_size=inner,
        max_position_embeddings=16,
        tie_word_embeddings=tied,
    )
    with torch.device(device):
        model = GPTPanguForCausalLM(config)
    assert MODEL_MAP["gpt_pangu"] is PanguAlphaQModel
    assert PanguAlphaQModel.fusable is None
    tree = PanguAlphaQModel._resolve_effective_module_tree(model, SimpleNamespace(method="gptq"))
    assert PanguAlphaQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_BLOCKS
    plan = ModulePlan.from_tree(tree)
    for layer in model.transformer.h:
        owners = {name.rsplit(".", 1)[0] for name in layer.state_dict()}
        assert owners == {
            "ln_1",
            "attn",
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.c_proj",
            "ln_2",
            "mlp.c_fc",
            "mlp.c_proj",
        }
        for owner in owners:
            spec = plan.find(owner)
            assert spec is not None, owner
            assert spec.quantize == isinstance(layer.get_submodule(owner), torch.nn.Linear)
        wrapper = _wrapper(PanguAlphaQModel, model, tree)
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
