# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5722: AWQ clip search decides by projection role, not by substrings.

The skip test used to be ``any(part in name for part in ["q_", "k_", ...])``
over the whole dotted path. ``block_sparse_moe`` contains ``k_``, so every
routed expert of Granite-MoE/GRIN/Kimi silently skipped clip search, while the
fused projections that really do carry q and k (``qkv_proj``, ``c_attn``,
``wqkv``, ``W_pack``) were clipped against the documented policy.

These tests drive the real ``AWQProcessor._search_best_clip`` and read which
modules it returns, so the decision is observed where quantization consumes it.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import types

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.definitions import granitemoe
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.config import FORMAT, METHOD, CalibrationConfig, QuantizeConfig


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 32
OUT_FEATURES = 16


class _ClipProbeAWQProcessor(AWQProcessor):
    """Minimal processor: clip search needs no model forward for these cases."""

    def __init__(self, qcfg: QuantizeConfig, gptq_model=None) -> None:
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=CalibrationConfig(),
            gptq_model=gptq_model,
            model=None,
            calculate_w_wq_diff=False,
        )

    def _module_forward(self, x, module, module_kwargs):
        return module(x)


def _qcfg() -> QuantizeConfig:
    return QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=IN_FEATURES)


def _layer_and_features(names):
    """Build a layer with one small Linear per dotted name, plus its features."""

    layer = nn.Module()
    linears = {}
    features = {}
    generator = torch.Generator().manual_seed(5722)
    for name in names:
        *parents, leaf = name.split(".")
        parent = layer
        for part in parents:
            child = getattr(parent, part, None)
            if child is None:
                child = nn.Module()
                setattr(parent, part, child)
            parent = child
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        setattr(parent, leaf, linear)
        linears[name] = linear
        features[name] = torch.randn(8, IN_FEATURES, generator=generator)
    return layer, linears, features


def _clipped_names(processor, layer, linears, features):
    clip_list = processor._search_best_clip(layer, linears, features)
    for name, bound in clip_list:
        assert name in linears
        assert bound.numel() > 0 and torch.isfinite(bound).all()
    return {name for name, _ in clip_list}


def test_routed_experts_are_clipped_and_q_k_are_not():
    """Regression: ``block_sparse_moe`` matched ``k_`` and lost every expert.

    Roles come from the model's module plan, the same plan that already tags
    these leaves ``gate``/``up``/``down`` for every other consumer.
    """

    names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "block_sparse_moe.experts.0.gate_proj",
        "block_sparse_moe.experts.0.up_proj",
        "block_sparse_moe.experts.0.down_proj",
        "block_sparse_moe.experts.63.down_proj",
    ]
    layer, linears, features = _layer_and_features(names)
    plan = ModulePlan.from_tree(granitemoe.GraniteMoeQModel.module_tree)
    gptq_model = types.SimpleNamespace(module_plan=lambda: plan)

    clipped = _clipped_names(
        _ClipProbeAWQProcessor(_qcfg(), gptq_model=gptq_model), layer, linears, features
    )

    assert clipped == {
        "self_attn.v_proj",
        "self_attn.o_proj",
        "block_sparse_moe.experts.0.gate_proj",
        "block_sparse_moe.experts.0.up_proj",
        "block_sparse_moe.experts.0.down_proj",
        "block_sparse_moe.experts.63.down_proj",
    }


def test_a_leaf_outside_the_plan_is_decided_by_its_own_name():
    """A fused q/k output is skipped by its leaf, not by its parent's name.

    These leaves carry q and k among their outputs, so the documented ``qk
    bmm`` policy applies to them; before the fix they were clipped while
    ``block_sparse_moe`` experts were not.
    """

    names = [
        "self_attn.qkv_proj",
        "attn.c_attn",
        "attention.wqkv",
        "self_attn.W_pack",
        "self_attn.v_proj",
        "block_sparse_moe.experts.0.down_proj",
    ]
    layer, linears, features = _layer_and_features(names)

    clipped = _clipped_names(_ClipProbeAWQProcessor(_qcfg()), layer, linears, features)

    assert clipped == {"self_attn.v_proj", "block_sparse_moe.experts.0.down_proj"}


def test_a_model_without_a_plan_still_clips_experts():
    """A partial model has no plan; the container name must not decide alone."""

    names = [
        "block_sparse_moe.experts.0.gate_proj",
        "block_sparse_moe.experts.1.down_proj",
        "self_attn.k_proj",
    ]
    layer, linears, features = _layer_and_features(names)

    clipped = _clipped_names(_ClipProbeAWQProcessor(_qcfg()), layer, linears, features)

    assert clipped == {
        "block_sparse_moe.experts.0.gate_proj",
        "block_sparse_moe.experts.1.down_proj",
    }
