# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Nanbeige AWQ scale folds must leave the full looped model output unchanged."""

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.nanbeige.configuration_nanbeige import NanbeigeConfig
from gptqmodel.models.ports.nanbeige.modeling_nanbeige import (
    NanbeigeForCausalLM,
    NanbeigeHyperConnectionModule,
    NanbeigeRMSNorm,
)


IDS = torch.tensor([[1, 7, 4, 11, 3, 9], [1, 8, 5, 6, 10, 12]])

QKV = ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
VO = ("self_attn.v_proj", ["self_attn.o_proj"])
GATE_UP = ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"])
UP_DOWN = ("mlp.up_proj", ["mlp.down_proj"])


def _model(**kwargs):
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "num_loops": 2,
        "pad_token_id": 0,
        "max_position_embeddings": 32,
    }
    values.update(kwargs)
    torch.manual_seed(918)
    model = NanbeigeForCausalLM(NanbeigeConfig(**values)).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NanbeigeRMSNorm):
                module.weight.copy_(torch.linspace(0.73, 1.19, module.weight.numel()))
            if isinstance(module, NanbeigeHyperConnectionModule):
                module.mapping_proj.weight.normal_(std=0.13)
                for name in ("alpha_pre", "alpha_post", "alpha_res"):
                    getattr(module, name).fill_(0.17)
    _convert_model_with_defuser(NanbeigeQModel, model, cleanup_original=False)
    return model


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, [QKV, VO, GATE_UP, UP_DOWN]),
        # GQA: `v_proj` is narrower than `o_proj`'s input, so it is no source.
        ({"num_key_value_heads": 2}, [QKV, GATE_UP, UP_DOWN]),
        ({"attention_bias": True, "mlp_bias": True}, [QKV, VO, GATE_UP, UP_DOWN]),
        (
            {"num_hidden_layers": 4, "enable_double_loop_split": True, "loop_middle_layers": 2, "loop_share_kv": True},
            [QKV, VO, GATE_UP, UP_DOWN],
        ),
        ({"enable_hyper_connection": True, "enable_mhc": True, "num_residual_streams": 2}, [QKV, VO, GATE_UP, UP_DOWN]),
        # The planner does not emit `v_proj -> o_proj` once Q/K norms are declared.
        ({"qk_layernorm": True}, [QKV, GATE_UP, UP_DOWN]),
        # Depth-Attention mixes other layers' cached values into this layer's
        # values, so `v_proj -> o_proj` is dropped by the definition.
        ({"enable_depth_attention": True}, [QKV, GATE_UP, UP_DOWN]),
    ],
    ids=["mha", "gqa", "bias", "split_shared_kv", "mhc", "qk_layernorm", "depth_attention"],
)
def test_nanbeige_awq_folds_preserve_looped_model_output(options, expected):
    model = _model(**options)
    tree = NanbeigeQModel._resolve_effective_module_tree(model, None)

    def forward():
        return model(IDS, use_cache=False).logits

    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(NanbeigeQModel, model, layer, forward, tree=tree)
        assert [(source, targets) for source, targets, _ in checked] == expected


def test_nanbeige_depth_attention_value_fold_is_not_neutral():
    """Pin the reason the definition drops `v_proj -> o_proj` under Depth-Attention."""

    model = _model(enable_depth_attention=True)
    tree = NanbeigeQModel._resolve_effective_module_tree(model, None)

    def forward():
        return model(IDS, use_cache=False).logits

    with pytest.raises(AssertionError, match=r"self_attn\.v_proj -> \['self_attn\.o_proj'\]"):
        assert_awq_folds_neutral(
            NanbeigeQModel,
            model,
            model.model.layers[1],
            forward,
            tree=tree,
            scaling_provider=BaseQModel.awq_get_modules_for_scaling,
        )
