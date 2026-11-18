# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint inventory and effective plan coverage for physical Ouro layers."""

import json
from pathlib import Path

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ouro import OuroQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.ouro import OuroConfig, OuroForCausalLM
from tests.port_assets import read_port_json


PORT = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/ouro"
EXPECTED_BLOCKS = [
    ["input_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["input_layernorm_2:!"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
    ["post_attention_layernorm_2:!"],
]


@pytest.mark.parametrize("release", ["Ouro-1.4B", "Ouro-2.6B", "Ouro-1.4B-Thinking", "Ouro-2.6B-Thinking"])
def test_every_pinned_released_key_and_shape_matches_meta(release):
    fixture = read_port_json(PORT / "checkpoint_inventory.json")[release]
    config = OuroConfig(**fixture["config"])
    with torch.device("meta"):
        model = OuroForCausalLM(config)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert actual == fixture["shapes"]
    assert len(actual) == (533 if "2.6B" in release else 269)
    assert model.model.early_exit_gate.weight.shape == (1, 2048)
    assert len(model.model.layers) == config.num_hidden_layers
    assert len(model.model.layers) in (24, 48)
    del model


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [8, 12])
@pytest.mark.parametrize("sliding", [False, True])
def test_live_checkpoint_weight_owners_alias_and_plain_policy(kv_heads, head_dim, sliding):
    config = OuroConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        use_sliding_window=sliding,
        sliding_window=3,
        max_window_layers=1,
    )
    model = OuroForCausalLM(config).eval()
    wrapper = OuroQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert MODEL_MAP["ouro"] is OuroQModel
    assert wrapper.extract_layers_node() == ["model.layers"]
    quantized = {name for block in wrapper.simple_layer_modules(config, wrapper.quantize_config) for name in block}
    assert quantized == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    for layer in model.model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in quantized
            else:
                assert set(spec.flags) & {"!", "?"}
    assert plan.find("early_exit_gate") is None
    assert plan.find("self_attn.q_proj.extra") is None
    assert not wrapper.fusable
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[4, 5, 6]]), use_cache=False).logits).all()


def test_registered_plan_baseline_remains_exact():
    fixture = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    case = fixture["cases"]["OuroQModel/gptq"]
    assert case["blocks"] == EXPECTED_BLOCKS
    plan = ModulePlan.from_tree(OuroQModel.module_tree)
    assert plan.find("input_layernorm_2").quantize is False
    assert plan.find("post_attention_layernorm_2").quantize is False
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
        assert set(plan.find(name).flags) == {"0", "in=x"}
    for name in ("self_attn.o_proj", "mlp.down_proj"):
        assert plan.find(name).flags == ("1",)
