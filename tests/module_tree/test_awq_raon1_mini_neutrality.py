# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Raon-1 mini AWQ scale folds must leave the full mHC model output unchanged."""

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.raon1_mini import Raon1MiniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from tests.test_raon1_mini_local_port import _model


IDS = torch.tensor([[1, 7, 4, 11, 3, 9, 5, 2]])

RDA_INPUTS = [f"self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "rda_w_proj", "f_a_proj")]
MLA_INPUTS = ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.g_proj"]
MOE_GATE_UP = [f"moe.experts.{e}.{leaf}" for e in range(3) for leaf in ("gate_proj", "up_proj")]
MOE_GATE_UP += ["moe.shared_experts.gate_proj", "moe.shared_experts.up_proj"]


def _tiny(layout, **kwargs):
    model = _model(torch.float64, layout, **kwargs)
    _convert_model_with_defuser(Raon1MiniQModel, model, cleanup_original=False)

    def forward():
        return model(input_ids=IDS, use_cache=False).logits

    return model, forward


@pytest.mark.parametrize(
    "layout,settings,expected",
    [
        # Layer 0 is dense (first_k_dense_replace=1), layer 1 is MoE.
        ("hybrid", {}, [("rda", "dense"), ("mla", "moe")]),
        ("reverse", {}, [("mla", "dense"), ("rda", "moe")]),
        ("hybrid", {"num_experts": None}, [("rda", "dense"), ("mla", "dense")]),
    ],
)
def test_raon_awq_folds_preserve_rda_mla_dense_and_moe_output(layout, settings, expected):
    model, forward = _tiny(layout, **settings)
    for layer, (mixer, mlp) in zip(model.model.layers, expected):
        assert ("rda" if layer.is_rda else "mla", "moe" if layer.is_moe else "dense") == (mixer, mlp)
        checked = assert_awq_folds_neutral(Raon1MiniQModel, model, layer, forward)
        attention = RDA_INPUTS if layer.is_rda else MLA_INPUTS
        if layer.is_moe:
            ffn = ("post_attention_layernorm", MOE_GATE_UP, ["moe.gate"])
        else:
            ffn = ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], [])
        assert checked == [("input_layernorm", attention, []), ffn]


@pytest.mark.parametrize(
    "layer_index,target,dropped",
    [
        # The generic planner falls back to the input norm for `o_proj`.
        (0, "o_proj", r"input_layernorm -> \[.*'self_attn\.o_proj'\]"),
        # SiTU-GLU bounds the value branch with tanh: not positively homogeneous.
        (0, "up_proj", r"mlp\.up_proj -> \['mlp\.down_proj'\]"),
        (1, "up_proj", r"moe\.experts\.\d\.up_proj -> \['moe\.experts\.\d\.down_proj'\]"),
    ],
)
def test_raon_dropped_generic_folds_are_not_neutral(layer_index, target, dropped):
    """Pin why the definition drops these generic groups."""

    model, forward = _tiny("hybrid")
    layer = model.model.layers[layer_index]

    def generic(wrapper, module, input_feat, module_kwargs):
        nodes = BaseQModel.awq_get_modules_for_scaling(wrapper, module, input_feat, module_kwargs)
        if target == "o_proj":
            return [node for node in nodes if any(m is module.self_attn.o_proj for m in node["layers"])]
        up_projs = [m for name, m in module.named_modules() if name.endswith("up_proj")]
        return [node for node in nodes if any(node["prev_op"] is m for m in up_projs)]

    with pytest.raises(AssertionError, match=dropped):
        assert_awq_folds_neutral(Raon1MiniQModel, model, layer, forward, scaling_provider=generic)
