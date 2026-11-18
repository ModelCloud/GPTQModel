# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every released Ripple decoder owner resolves through the real module plan."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.ricursive import RicursiveQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.ricursive.configuration_ricursive import RicursiveConfig
from gptqmodel.models.ports.ricursive.modeling_ricursive import RicursiveForCausalLM
from tests.port_assets import DATA, read_port_json
from tests.test_ricursive_local_port import FIXTURES, LINEARS, _config


@pytest.mark.parametrize("release", ["preview", "base"])
def test_released_1203_keys_and_all_shapes_equal_live_meta_owners(release):
    config = RicursiveConfig.from_pretrained(DATA / "ricursive/fixtures" / release)
    with torch.device("meta"):
        model = RicursiveForCausalLM(config)
    # Both releases publish the same tensor headers.
    expected = read_port_json(FIXTURES / "header_inventory.json")
    assert len(expected) == 1203
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == {k: v["shape"] for k, v in expected.items()}
    assert len(model.model.layers) == 80
    assert model.model.layers[4].self_attn.layer_type == "full_attention"
    assert not hasattr(model.model.layers[4].self_attn, "rotary_emb")
    assert model.config.embed_scale == 71.55417527999327


@pytest.mark.parametrize("gated,valres", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_all_real_decoder_owners_are_quantized_or_explicitly_dense(gated, valres, kv_heads):
    config = _config(use_gated_attention=gated, use_valres=valres, num_key_value_heads=kv_heads)
    with torch.device("meta"):
        model = RicursiveForCausalLM(config)
    _convert_model_with_defuser(RicursiveQModel, model, cleanup_original=False)
    wrapper = RicursiveQModel.__new__(RicursiveQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    tree = wrapper._resolve_effective_module_tree(model, SimpleNamespace(method="gptq"))
    plan = ModulePlan.from_tree(tree)
    quantized = LINEARS if gated else LINEARS - {"self_attn.g_proj"}
    expected_keys = {name + ".weight" for name in quantized} | {
        n + ".weight" for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                                 "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm")}
    if valres:
        expected_keys.add("self_attn.value_res_lambda")
    for layer in model.model.layers:
        assert set(layer.state_dict()) == expected_keys
        for key in expected_keys:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in quantized), key
            if not spec.quantize:
                assert "!" in spec.flags
        for name in quantized:
            assert isinstance(layer.get_submodule(name), nn.Linear)
            assert plan.find(name + ".undeclared") is None
        assert "in=x" in plan.find("self_attn.q_proj").flags
        # Only the value residual separates `v_proj`'s input from q/k/g's.
        v_shares_input = any(flag.startswith("in=") for flag in plan.find("self_attn.v_proj").flags)
        assert v_shares_input == (not valres)
        assert ("nosrc" in plan.find("input_layernorm").flags) == valres
    assert RicursiveQModel.fusable is None
    assert not RicursiveQModel.require_trust_remote_code


def test_default_registered_module_plan_is_unchanged():
    from pathlib import Path
    inventory = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    expected = inventory["cases"]["RicursiveQModel/gptq"]
    assert RicursiveQModel.extract_layers_node() == expected["layers"]
    assert RicursiveQModel.build_layer_modules(RicursiveQModel.module_tree, include_capture_only=True) == expected["blocks"]


def _awq_neutral_groups(**overrides):
    from awq_neutrality import assert_awq_folds_neutral
    from ricursive_test_utils import make_model

    model = make_model(**overrides)
    tree = RicursiveQModel._resolve_effective_module_tree(model, SimpleNamespace(method="awq"))
    ids = torch.randint(2, 60, (1, 8), generator=torch.Generator().manual_seed(3))

    def forward():
        # The whole model: `value_residual` is seeded from the embeddings.
        return model(input_ids=ids, use_cache=False).logits

    # Both a sliding (RoPE) and a full-attention (NoPE) layer.
    return model, tree, forward, [
        assert_awq_folds_neutral(RicursiveQModel, model, layer, forward, tree=tree) for layer in model.model.layers
    ]


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("valres", [True, False])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_scale_folds_are_neutral(gated, valres, kv_heads):
    _model, _tree, _forward, per_layer = _awq_neutral_groups(
        use_gated_attention=gated, use_valres=valres, num_key_value_heads=kv_heads
    )
    qkv = ["self_attn.q_proj", "self_attn.k_proj"] + (["self_attn.g_proj"] if gated else []) + ["self_attn.v_proj"]
    expected = {
        ("pre_feedforward_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ("mlp.up_proj", ("mlp.down_proj",)),
    }
    if not valres:
        # The lerp with the value residual is the only thing between the
        # input norm and `v_proj`.
        expected.add(("input_layernorm", tuple(qkv)))
    if kv_heads == 4:
        # Without GQA `v_proj`'s output channels are `o_proj`'s input channels;
        # the sigmoid gate multiplies per channel and commutes with the scale.
        expected.add(("self_attn.v_proj", ("self_attn.o_proj",)))
    for checked in per_layer:
        assert {(source, tuple(targets)) for source, targets, _consumers in checked} == expected
        for source, _targets, _consumers in checked:
            assert source not in {"post_attention_layernorm", "post_feedforward_layernorm"}
            assert source not in RicursiveQModel._AWQ_NON_PRODUCERS


def test_awq_drops_up_down_fold_under_ffn_clamp():
    from awq_neutrality import _capture, _quantized_features, _wrapper
    from test_awq_scaling_groups import _foldable_groups

    from gptqmodel.models.base import BaseQModel
    from gptqmodel.quantization.awq.quantize.scale import apply_scale
    from gptqmodel.quantization.awq.utils.module import get_op_name

    clamp = 0.02
    model, tree, forward, per_layer = _awq_neutral_groups(
        use_valres=False, num_key_value_heads=4, ffn_clamp_threshold=clamp
    )
    for checked in per_layer:
        sources = {source for source, _targets, _consumers in checked}
        assert sources == {"input_layernorm", "self_attn.v_proj", "pre_feedforward_layernorm"}

    # The generic planner still proposes `up_proj -> down_proj`; the clamp on
    # `up_proj`'s output makes that fold change the model output.
    layer = model.model.layers[0]
    wrapper = _wrapper(RicursiveQModel, model, tree)
    reference, features = _capture(layer, forward)
    features = _quantized_features(layer, features, tree)
    with torch.inference_mode():
        up = layer.mlp.up_proj(layer.pre_feedforward_layernorm(torch.randn(4, 32)))
    assert clamp < float(up.abs().max())
    generic = _foldable_groups(layer, BaseQModel.awq_get_modules_for_scaling(wrapper, layer, features, {}))
    node = next(node for node in generic if get_op_name(layer, node["prev_op"]) == "mlp.up_proj")
    snapshot = {name: value.clone() for name, value in layer.state_dict().items()}
    try:
        scales = torch.linspace(0.55, 1.45, node["prev_op"].weight.shape[0])
        apply_scale(layer, [("mlp.up_proj", ["mlp.down_proj"], scales, None)], consumer_map={"mlp.up_proj": []})
        with torch.inference_mode():
            changed = forward()
        assert float((changed - reference).norm() / reference.norm()) > 1e-4
    finally:
        layer.load_state_dict(snapshot)
