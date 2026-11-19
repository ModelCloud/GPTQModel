# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Falcon's two-norm parallel decoder must not fold across the branches (#5414).

With ``new_decoder_architecture=True`` (or ``num_ln_in_parallel_attn=2``) the
decoder normalizes each branch on its own: ``ln_attn`` feeds ``self_attention``
and ``ln_mlp`` feeds ``mlp``. The Falcon module plan declared the older
``input_layernorm``/``post_attention_layernorm`` names, neither of which exists
there, so the AWQ scaling builder fell back to the previous module and folded
``self_attention.dense`` into ``mlp.dense_h_to_4h``. Those projections are on
separate branches, so the fold changed the model (relative logit error ~0.2).
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.utils.module import get_op_name
from tests.test_awq_scaling_groups import (
    _features,
    _foldable_groups,
    _worst_relative_change,
)


BASE = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}


def _tiny_falcon(**overrides):
    qmodel_cls = MODEL_MAP["falcon"]
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(
        AutoConfig.for_model("falcon", **BASE, **overrides)
    ).eval()
    tree = qmodel_cls._resolve_effective_module_tree(hf_model, None)
    qmodel = qmodel_cls.__new__(qmodel_cls)
    torch.nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    layer = hf_model.transformer.h[0]
    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(layer, ModulePlan.from_tree(tree)), {})
    return hf_model, layer, nodes


def _groups_by_source(layer, nodes):
    groups = {}
    for node in nodes:
        prev_op = node.get("prev_op")
        if prev_op is None:
            continue
        groups[get_op_name(layer, prev_op)] = {
            get_op_name(layer, module) for module in (node.get("layers") or [])
        }
    return groups


def test_two_norm_parallel_groups_use_their_branch_norm():
    _, layer, nodes = _tiny_falcon(
        new_decoder_architecture=True,
        parallel_attn=True,
        num_ln_in_parallel_attn=2,
    )
    groups = _groups_by_source(layer, nodes)

    assert groups == {
        "ln_attn": {"self_attention.query_key_value"},
        "self_attention.query_key_value": {"self_attention.dense"},
        "ln_mlp": {"mlp.dense_h_to_4h"},
    }
    # The attention output projection must never scale the MLP input.
    assert "self_attention.dense" not in groups


def test_two_norm_parallel_fold_preserves_logits():
    hf_model, layer, nodes = _tiny_falcon(
        new_decoder_architecture=True,
        parallel_attn=True,
        num_ln_in_parallel_attn=2,
    )
    groups = _foldable_groups(layer, nodes)
    assert groups, "the two-norm decoder produced no foldable group"

    torch.manual_seed(5414)
    ids = torch.tensor([[1, 2, 3, 4]])
    worst = _worst_relative_change(hf_model, ids, layer, groups)
    assert worst < 1e-5, f"a fold changed the two-norm parallel model by {worst:.3f}"


def test_single_norm_parallel_control_keeps_the_shared_group():
    """`parallel_attn` without the second norm still shares `input_layernorm`."""

    _, layer, nodes = _tiny_falcon(parallel_attn=True, new_decoder_architecture=False)
    groups = _groups_by_source(layer, nodes)

    assert groups["input_layernorm"] == {
        "self_attention.query_key_value",
        "mlp.dense_h_to_4h",
    }
    assert "self_attention.dense" not in groups


def test_sequential_control_keeps_the_post_attention_norm_group():
    _, layer, nodes = _tiny_falcon(parallel_attn=False, new_decoder_architecture=False)
    groups = _groups_by_source(layer, nodes)

    assert groups == {
        "input_layernorm": {"self_attention.query_key_value"},
        "self_attention.query_key_value": {"self_attention.dense"},
        "post_attention_layernorm": {"mlp.dense_h_to_4h"},
    }
