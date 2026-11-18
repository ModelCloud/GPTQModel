# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared AWQ qualification: every emitted scale fold must leave a real layer's output unchanged.

Build a tiny CPU model, pick a decoder layer and a ``forward`` closure that runs it,
then call ``assert_awq_folds_neutral(Definition, model, layer, forward)``. Each fold
group the definition emits is applied once with a non-uniform scale (no
quantization) and the full layer output must stay within ``TOLERANCE``.
"""

import torch
from test_awq_scaling_groups import _foldable_groups
from torch import nn

from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


TOLERANCE = 1e-5


def _wrapper(definition, model, tree):
    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["effective_module_tree"] = tree
    wrapper.__dict__["module_tree"] = tree
    return wrapper


def _capture(layer, forward):
    features = {}
    handles = []

    def record(name):
        def hook(module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if isinstance(value, torch.Tensor):
                features[name] = value.detach().clone()

        return hook

    try:
        for name, module in layer.named_modules():
            handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))
        with torch.inference_mode():
            reference = forward().clone()
    finally:
        for handle in handles:
            handle.remove()
    return reference, features


def _quantized_features(layer, features, tree):
    """Keep only the captured inputs of quantized 2-D weight modules.

    The AWQ processor's ``input_feat`` only contains the modules it will
    quantize. Capturing every module would let a non-quantized norm inside a
    stage win ``_select_feature_name`` and change the measured input width.
    """

    plan = ModulePlan.from_tree(tree)
    kept = {}
    for name, value in features.items():
        spec = plan.find(name)
        if spec is None or not spec.quantize:
            continue
        weight = getattr(layer.get_submodule(name), "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            kept[name] = value
    return kept


def _assert_neutral(definition, model, layer, forward, tree=None, *, scaling_provider=None):
    """Apply every emitted fold once and require the full output to be unchanged."""

    tree = tree if tree is not None else definition.module_tree
    wrapper = _wrapper(definition, model, tree)
    reference, features = _capture(layer, forward)
    features = _quantized_features(layer, features, tree)
    if scaling_provider is None:
        scaling_provider = definition.awq_get_modules_for_scaling
    nodes = scaling_provider(wrapper, layer, features, {})
    groups = _foldable_groups(layer, nodes)
    assert groups, f"{type(definition).__name__}: no foldable AWQ group was emitted"

    snapshot = {name: value.clone() for name, value in layer.state_dict().items()}
    checked = []
    try:
        for node in groups:
            source = get_op_name(layer, node["prev_op"])
            targets = [get_op_name(layer, module) for module in node["layers"]]
            if isinstance(node["prev_op"], nn.Linear):
                width = node["prev_op"].weight.shape[0]
            else:
                width = node["prev_op"].weight.numel()
            scales = torch.linspace(0.55, 1.45, width)
            consumers = list(node.get("_scale_consumers") or [])
            apply_scale(layer, [(source, targets, scales, None)], consumer_map={source: consumers})
            with torch.inference_mode():
                actual = forward()
            relative = float((actual - reference).norm() / reference.norm())
            assert relative < TOLERANCE, (
                f"{type(definition).__name__}: {source} -> {targets} changed the layer output by {relative:.3e}"
            )
            checked.append((source, targets, consumers))
            layer.load_state_dict(snapshot)
    finally:
        layer.load_state_dict(snapshot)
    return checked


assert_awq_folds_neutral = _assert_neutral
