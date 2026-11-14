# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scaling groups must be output-preserving for every registered family.

Each case builds a tiny Hugging Face model, asks the definition for its scaling
groups, folds a random scale with the library's own ``apply_scale``, and checks
that the decoder layer's logits are unchanged. A group that crosses a residual
add, a parallel branch, or a non-homogeneous activation shows up here as a
large relative change even though the folded group still reproduces its own
consumer.
"""

import importlib

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


# name -> (definition module, definition class, config class, extra config)
FAMILIES = {
    "llama": ("gptqmodel.models.definitions.llama", "LlamaQModel", "LlamaConfig", {"head_dim": 8}),
    "olmo2": ("gptqmodel.models.definitions.olmo2", "Olmo2QModel", "Olmo2Config", {"head_dim": 8}),
    "olmo3": ("gptqmodel.models.definitions.olmo3", "Olmo3QModel", "Olmo3Config", {"head_dim": 8}),
    "exaone4": (
        "gptqmodel.models.definitions.exaone4",
        "Exaone4QModel",
        "Exaone4Config",
        {"head_dim": 8},
    ),
    "opt": ("gptqmodel.models.definitions.opt", "OptQModel", "OPTConfig", {}),
    "opt350": (
        "gptqmodel.models.definitions.opt",
        "OptQModel",
        "OPTConfig",
        {"do_layer_norm_before": False},
    ),
    "phi": ("gptqmodel.models.definitions.phi", "PhiQModel", "PhiConfig", {"head_dim": 8}),
    "gptj": (
        "gptqmodel.models.definitions.gptj",
        "GptJQModel",
        "GPTJConfig",
        {"head_dim": 8, "rotary_dim": 8},
    ),
    "starcoder2": (
        "gptqmodel.models.definitions.starcoder2",
        "Starcoder2QModel",
        "Starcoder2Config",
        {"head_dim": 8},
    ),
    "phi3": ("gptqmodel.models.definitions.phi3", "Phi3QModel", "Phi3Config", {"head_dim": 8}),
    "apertus": (
        "gptqmodel.models.definitions.apertus",
        "ApertusQModel",
        "ApertusConfig",
        {"head_dim": 8},
    ),
    "codegen": (
        "gptqmodel.models.definitions.codegen",
        "CodeGenQModel",
        "CodeGenConfig",
        {"rotary_dim": 8},
    ),
    "mpt": ("gptqmodel.models.definitions.mpt", "MptQModel", "MptConfig", {}),
}

_BASE_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 0,
}


def _tiny_config(config_name, extra):
    import transformers

    kwargs = dict(_BASE_CONFIG)
    kwargs.update(extra)
    config_cls = getattr(transformers, config_name)
    while True:
        try:
            return config_cls(**kwargs)
        except TypeError as exc:
            removed = False
            for key in list(kwargs):
                if key in str(exc):
                    kwargs.pop(key)
                    removed = True
                    break
            if not removed:
                raise


def _in_features(module):
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    # transformers Conv1D stores [in, out].
    return weight.shape[0] if type(module).__name__ == "Conv1D" else weight.shape[1]


def _out_features(module):
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    return weight.shape[1] if type(module).__name__ == "Conv1D" else weight.shape[0]


def _features(layer, plan):
    features = {}
    for spec in plan.specs:
        if not spec.quantize or "{" in spec.template:
            continue
        try:
            module = layer.get_submodule(spec.template)
        except AttributeError:
            continue
        width = _in_features(module)
        if width is not None:
            features[spec.template] = torch.randn(4, width)
    return features


def _tiny_layer(family):
    module_path, class_name, config_name, extra = FAMILIES[family]
    definition = importlib.import_module(module_path)
    qmodel_cls = getattr(definition, class_name)
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(_tiny_config(config_name, extra)).eval()
    qmodel = qmodel_cls.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    tree = qmodel_cls._resolve_effective_module_tree(hf_model, None)
    qmodel.__dict__["effective_module_tree"] = tree
    qmodel.__dict__["module_tree"] = tree
    layer_paths = qmodel_cls.extract_layers_node()
    if isinstance(layer_paths, str):
        layer_paths = [layer_paths]
    layer = hf_model.get_submodule(layer_paths[0] + ".0")
    return hf_model, qmodel, layer, tree


def _foldable_groups(layer, nodes):
    """Drop the groups ``AWQProcessor._quantize_layer`` drops on a width mismatch."""

    groups = []
    for node in nodes:
        layers = node.get("layers") or []
        prev_op = node.get("prev_op")
        if not layers or prev_op is None:
            continue
        first_in = _in_features(layers[0])
        if first_in is None:
            continue
        if isinstance(prev_op, nn.Linear) and _out_features(prev_op) != first_in:
            continue
        groups.append(node)
    return groups


def _fold_scale_width(prev_op):
    if isinstance(prev_op, nn.Linear):
        return _out_features(prev_op)
    return prev_op.weight.numel()


def _worst_relative_change(hf_model, ids, layer, groups):
    with torch.no_grad():
        before = hf_model(ids).logits.clone()
    worst = 0.0
    for node in groups:
        prev_op = node["prev_op"]
        prev_name = get_op_name(layer, prev_op)
        layer_names = [get_op_name(layer, module) for module in node["layers"]]
        scales = torch.rand(_fold_scale_width(prev_op)) + 0.5
        apply_scale(layer, [(prev_name, layer_names, scales, None)])
        with torch.no_grad():
            after = hf_model(ids).logits
        worst = max(worst, float((after - before).norm() / before.norm()))
        apply_scale(layer, [(prev_name, layer_names, 1.0 / scales, None)])
    return worst


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_scaling_groups_preserve_decoder_layer_output(family):
    hf_model, qmodel, layer, tree = _tiny_layer(family)
    features = _features(layer, ModulePlan.from_tree(tree))
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    groups = _foldable_groups(layer, nodes)

    if family == "codegen":
        # CodeGen is a parallel block with a fused QKV and a GELU MLP: no group
        # can be folded without changing the block output, so the layer falls
        # back to plain quantization.
        assert groups == []
        return

    assert groups, f"{family}: no AWQ scaling groups were built"
    ids = torch.randint(0, hf_model.config.vocab_size, (1, 8))
    worst = _worst_relative_change(hf_model, ids, layer, groups)
    assert worst < 1e-5, f"{family}: AWQ scaling changed the layer output by {worst:.3e}"


@pytest.mark.parametrize("family", ["phi3", "apertus", "codegen", "mpt"])
def test_families_without_declared_norms_still_produce_a_scaling_map(family):
    """The issue's repro: these trees built an empty AWQMAP and crashed at pack time."""

    _, qmodel, layer, tree = _tiny_layer(family)
    features = _features(layer, ModulePlan.from_tree(tree))
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    assert nodes, f"{family}: awq_get_modules_for_scaling returned no configuration"


@pytest.mark.parametrize(
    "family,forbidden",
    [
        ("olmo2", ["post_attention_layernorm"]),
        ("olmo3", ["post_attention_layernorm"]),
        ("exaone4", ["post_attention_layernorm"]),
        ("phi", ["input_layernorm", "self_attn.dense", "mlp.fc1"]),
        ("gptj", ["ln_1", "attn.out_proj", "mlp.fc_in"]),
        ("starcoder2", ["mlp.c_fc"]),
        ("apertus", ["mlp.up_proj"]),
        ("mpt", ["ffn.up_proj"]),
        ("opt350", ["self_attn_layer_norm"]),
    ],
)
def test_declared_non_sources_never_become_prev_ops(family, forbidden):
    _, qmodel, layer, tree = _tiny_layer(family)
    features = _features(layer, ModulePlan.from_tree(tree))
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    used = {get_op_name(layer, node["prev_op"]) for node in nodes}
    assert not (used & set(forbidden)), f"{family}: folded a scale into {used & set(forbidden)}"
