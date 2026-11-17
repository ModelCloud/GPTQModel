# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hymba AWQ: `**kwargs` replay and a tree that names a norm the layer has (#4891).

`HymbaDecoderLayer` applies `pre_moe_layernorm(residual + mamba_out)` in front
of the MoE. The definition declared a `post_attention_layernorm` that does not
exist, so the experts block had no producer of its own and its `gate_proj`/
`up_proj` were folded into `input_layernorm` -- which is not output-preserving.

`HymbaBlock.forward(self, hidden_states, past_key_value=None, **kwargs)` reads
`kwargs["attention_mask"]` and friends, so the AWQ replay must not filter the
captured kwargs down to the named parameters.

The real model needs remote code with `mamba_ssm`/`causal_conv1d`/`flash_attn`,
so the fold invariant runs on a synthetic decoder layer with the same module
layout and forward flow.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


class _Mamba(nn.Module):
    """`in_proj -> relu -> out_proj`, the shape of the SSM path for a fold check."""

    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden, inner, bias=False)
        self.out_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, hidden_states):
        return self.out_proj(torch.relu(self.in_proj(hidden_states)))


class _Expert(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.relu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _MoE(nn.Module):
    """`HymbaSparseMoeBlock` with one expert, the layout the definition declares.

    `HymbaSparseMoeBlock` builds a router only for `num_experts > 1`; the Hymba
    checkpoints have a single expert per layer, which is why the module tree
    declares only `experts.0`. A second expert would read the same norm without
    being in the fold group, so it could not be part of this invariant.
    """

    def __init__(self, hidden: int, inner: int, experts: int = 1):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(hidden, inner) for _ in range(experts)])

    def forward(self, hidden_states):
        return sum(expert(hidden_states) for expert in self.experts) / len(self.experts)


class _HymbaShapedLayer(nn.Module):
    """`HymbaDecoderLayer`'s module names and residual flow."""

    def __init__(self, hidden: int = 16, inner: int = 32):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.mamba = _Mamba(hidden, inner)
        self.pre_moe_layernorm = nn.LayerNorm(hidden)
        self.moe = _MoE(hidden, inner)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.mamba(self.input_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.moe(self.pre_moe_layernorm(hidden_states))
        return residual + hidden_states


def _layer_config():
    return SimpleNamespace(
        num_experts=1,
        num_experts_per_tok=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        mamba_expand=2,
    )


def _qmodel():
    """A `HymbaQModel` with only the state the tree walk reads."""

    model = HymbaQModel.__new__(HymbaQModel)
    model.model = SimpleNamespace(config=_layer_config())
    model.dynamic_expert_index = None
    return model


def _features(layer):
    """Captured activations keyed by quantized module name, as AWQ records them."""

    features = {}
    for name in (
        "mamba.in_proj",
        "mamba.out_proj",
        "moe.experts.0.gate_proj",
        "moe.experts.0.up_proj",
        "moe.experts.0.down_proj",
    ):
        in_features = layer.get_submodule(name).in_features
        features[name] = torch.zeros(4, in_features)
    return features


def _groups(layer):
    nodes = _qmodel().awq_get_modules_for_scaling(layer, _features(layer), {})
    return [
        (
            get_op_name(layer, node["prev_op"]) if node.get("prev_op") is not None else None,
            [get_op_name(layer, module) for module in node.get("layers") or []],
        )
        for node in nodes
        if node.get("layers")
    ]


def test_tree_declares_the_norm_the_decoder_layer_actually_has():
    tree = repr(HymbaQModel.module_tree)

    assert "post_attention_layernorm" not in tree
    assert "pre_moe_layernorm" in tree


def test_mamba_in_proj_is_not_an_awq_scale_source():
    """`in_proj` feeds conv/SSM/attention, which is not positively homogeneous."""

    tree = repr(HymbaQModel.module_tree)

    assert "in_proj:0:nosrc" in tree


def test_scaling_groups_follow_the_real_forward():
    layer = _HymbaShapedLayer()
    groups = _groups(layer)

    assert ("input_layernorm", ["mamba.in_proj"]) in groups
    assert (
        "pre_moe_layernorm",
        ["moe.experts.0.gate_proj", "moe.experts.0.up_proj"],
    ) in groups
    assert ("moe.experts.0.up_proj", ["moe.experts.0.down_proj"]) in groups
    # The defect: expert gate/up folded into the pre-mamba norm.
    for prev_op, consumers in groups:
        assert not (
            prev_op == "input_layernorm"
            and any("moe.experts" in consumer for consumer in consumers)
        ), groups
        assert not (
            prev_op == "mamba.in_proj" and "mamba.out_proj" in consumers
        ), groups


@pytest.mark.parametrize("group_index", [0, 1, 2])
def test_folding_a_scale_is_output_preserving(group_index):
    torch.manual_seed(4891)
    layer = _HymbaShapedLayer().eval()
    hidden_states = torch.randn(2, 8, 16)

    groups = _groups(layer)
    prev_op, consumers = groups[group_index]
    prev_module = layer.get_submodule(prev_op)
    width = (
        prev_module.out_features
        if isinstance(prev_module, nn.Linear)
        else prev_module.weight.numel()
    )
    scales = torch.rand(width) + 0.5

    with torch.no_grad():
        before = layer(hidden_states).clone()

    apply_scale(layer, [(prev_op, consumers, scales, None)])
    with torch.no_grad():
        after = layer(hidden_states)

    relative = float((after - before).norm() / before.norm())
    assert relative < 1e-5, f"{prev_op}: folding changed the layer output by {relative:.3e}"


class _VarKwargsModule(nn.Module):
    """`HymbaBlock`-shaped forward: reads named entries out of `**kwargs`."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        self.seen = dict(kwargs)
        return hidden_states


class _ExplicitSignatureModule(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


def test_sanitize_kwargs_passes_everything_to_a_var_keyword_forward():
    processor = AWQProcessor.__new__(AWQProcessor)
    module = _VarKwargsModule()
    kwargs = {
        "attention_mask": torch.ones(1, 8),
        "position_ids": torch.arange(8).unsqueeze(0),
        "kv_last_layer": None,
        "use_cache": False,
        "use_swa": False,
        "unsupported_extra": 3,
    }

    assert processor._sanitize_kwargs(kwargs, module) == kwargs


def test_sanitize_kwargs_still_filters_an_explicit_signature():
    processor = AWQProcessor.__new__(AWQProcessor)
    module = _ExplicitSignatureModule()

    sanitized = processor._sanitize_kwargs(
        {"attention_mask": torch.ones(1, 8), "unsupported_extra": 3}, module
    )

    assert set(sanitized) == {"attention_mask"}
