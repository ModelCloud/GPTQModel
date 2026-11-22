# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.solar_open2 import SolarOpen2QModel
from gptqmodel.models.ports.solar_open2 import SolarOpen2Config, SolarOpen2ForCausalLM
from tests.port_assets import read_port_json
from tests.test_solar_open2_local_port import tiny_model, tiny_values


PORT = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/solar_open2"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/ports/solar_open2"


def test_entire_released_checkpoint_inventory_matches_real_meta_model():
    config = SolarOpen2Config.from_pretrained(FIXTURES / "released_config.json")
    with torch.device("meta"):
        model = SolarOpen2ForCausalLM(config)
    released = read_port_json(PORT / "released_shapes.json")
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert len(actual) == 47019
    assert actual == {key: value["shape"] for key, value in released.items()}
    assert sum(kind == "full_attention" for kind in config.layer_types) == 12
    assert sum(kind == "linear_attention" for kind in config.layer_types) == 36
    del model


@pytest.mark.parametrize(
    "device,kv_heads,full_proj,dense_first",
    [("cpu", 1, False, False), ("meta", 2, True, False), ("cpu", 4, True, True), ("meta", 1, False, True)],
)
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
    assert SolarOpen2QModel.fusable is None
    assert MODEL_MAP["solar_open2"] is SolarOpen2QModel


@pytest.mark.parametrize(
    "kv_heads,full_proj,dense_first,extras",
    [
        (2, False, False, {}),
        (4, True, True, {}),
        (4, False, False, {"use_qk_norm": True, "attention_bias": True, "use_gqa_gate_bias": True}),
        (1, True, False, {"use_gqa_gate": False}),
    ],
)
def test_awq_folds_are_neutral_on_full_linear_dense_and_moe_layers(kv_heads, full_proj, dense_first, extras):
    model = tiny_model(
        num_key_value_heads=kv_heads,
        kda_use_full_proj=full_proj,
        first_k_dense_replace=int(dense_first),
        linear_attn_config={"head_dim": 8, "num_heads": 4, "num_kv_heads": kv_heads, "short_conv_kernel_size": 2},
        **extras,
    )
    ids = torch.tensor([[1, 7, 9, 5, 12, 17, 21, 32, 15, 24, 36, 42, 51, 61, 11, 8]])

    def forward():
        return model(ids, use_cache=False).logits

    config = model.config
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(SolarOpen2QModel, model, layer, forward)
        groups = {(source, tuple(targets)): consumers for source, targets, consumers in checked}
        # The attention output (sigmoid-gated or KDA gated-norm) never reads the
        # input norm, so o_proj may not join its group on either layer type.
        assert all("self_attn.o_proj" not in targets for _source, targets in groups)
        attention = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        if layer.layer_type == "full_attention":
            if config.use_gqa_gate:
                attention.append("self_attn.g_proj")
            assert groups[("input_layernorm", tuple(attention))] == []
        else:
            # Dense KDA decay/beta/gate projections read the same normed input
            # and absorb the inverse scale; the conv/recurrence path never folds.
            if full_proj:
                attention.append("self_attn.g_proj")
                consumers = ["self_attn.b_proj", "self_attn.f_proj"]
            else:
                consumers = ["self_attn.b_proj", "self_attn.f_a_proj", "self_attn.g_a_proj"]
            assert groups[("input_layernorm", tuple(attention))] == consumers
        assert all(not source.startswith("self_attn.") for source, _targets in groups)
        if hasattr(layer.mlp, "experts"):
            gate_up = [f"mlp.shared_experts.{p}" for p in ("gate_proj", "up_proj")]
            norm_groups = [key for key in groups if key[0] == "post_attention_layernorm"]
            assert len(norm_groups) == 1 and list(norm_groups[0][1][:2]) == gate_up
            assert groups[norm_groups[0]] == ["mlp.gate"]
            assert ("mlp.shared_experts.up_proj", ("mlp.shared_experts.down_proj",)) in groups
            assert any(source.startswith("mlp.experts.") for source, _targets in groups)
        else:
            assert groups[("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"))] == []
            assert ("mlp.up_proj", ("mlp.down_proj",)) in groups
