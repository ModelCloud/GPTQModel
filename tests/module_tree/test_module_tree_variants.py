# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import METHOD
from gptqmodel.utils.model import get_layers_with_prefixes


class _BranchALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4)
        self.self_attn.o_proj = nn.Linear(4, 4)


class _BranchBLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = nn.Module()
        self.mixer.in_proj = nn.Linear(4, 4)
        self.mixer.out_proj = nn.Linear(4, 4)


class _VariantTreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.shared = nn.Identity()

        self.model.A_module = nn.Module()
        self.model.A_module.layers = nn.ModuleList([_BranchALayer()])
        self.model.A_module.a_norm = nn.Identity()

        self.model.B_module = nn.Module()
        self.model.B_module.layers = nn.ModuleList([_BranchBLayer()])
        self.model.B_module.b_norm = nn.Identity()


class _VariantTreeQModel(BaseQModel):
    layer_modules_strict = False
    module_tree = [
        [
            "model",
            "A_module",
            "layers",
            "#",
            {
                "self_attn": ("q_proj:0", "o_proj:1"),
            },
        ],
        [
            "model",
            "B_module",
            "layers",
            "#",
            {
                "mixer": ("in_proj:0", "out_proj:1"),
            },
        ],
    ]


class _SingleTreeQModel(BaseQModel):
    module_tree = [
        "model",
        "A_module",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "o_proj:1"),
        },
    ]


class _LegacyPipePrefixQModel(BaseQModel):
    module_tree = [
        "model",
        "A_module|B_module",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "o_proj:1"),
        },
    ]


class _AutoDetectedTreeQModel(BaseQModel):
    """Tiny unknown-architecture definition used to verify instance isolation."""

    module_tree = None

    def _auto_detect_module_tree(self, model, quant_method):
        return model.config.selected_tree

    def _configure_modelopt_runtime(self):
        pass


class _MethodOverrideTreeQModel(BaseQModel):
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0",),
            "dense_mlp": ("up_proj:0", "down_proj:1"),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts:0": {"#": ("up_proj:0",)},
            },
        },
    ]
    module_tree_overrides = {
        METHOD.AWQ: [{"mlp:moe": {"gate": ("gate",)}}],
    }
    dynamic_expert_index = "num_experts"

    def _configure_modelopt_runtime(self):
        pass


def _init_tree_qmodel(qmodel_cls, selected_tree, method=METHOD.GPTQ):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="unknown", selected_tree=selected_tree, num_experts=2)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    qcfg = SimpleNamespace(method=method, adapter=None)
    return qmodel_cls(model, quantized=False, quantize_config=qcfg)


def test_single_module_tree_expands_layer_path_base_modules_and_layer_modules():
    model = _VariantTreeModel()

    assert _SingleTreeQModel.extract_layers_node() == ["model.A_module.layers"]
    assert _SingleTreeQModel.get_base_modules(model) == [
        "model.shared",
        "model.B_module",
        "model.A_module.a_norm",
    ]
    assert _SingleTreeQModel.build_layer_modules(_SingleTreeQModel.module_tree) == [
        ["self_attn.q_proj"],
        ["self_attn.o_proj"],
    ]


def test_module_tree_variant_normalization_accepts_single_and_multiple_trees():
    assert _SingleTreeQModel._iter_module_tree_variants(_SingleTreeQModel.module_tree) == [
        _SingleTreeQModel.module_tree,
    ]
    assert _VariantTreeQModel._iter_module_tree_variants(
        _VariantTreeQModel.module_tree
    ) == _VariantTreeQModel.module_tree


def test_variant_module_tree_expands_layer_paths_and_base_modules():
    model = _VariantTreeModel()

    assert _VariantTreeQModel.extract_layers_node() == [
        "model.A_module.layers",
        "model.B_module.layers",
    ]
    assert _VariantTreeQModel.get_base_modules(model) == [
        "model.shared",
        "model.A_module.a_norm",
        "model.B_module.b_norm",
    ]


def test_variant_module_tree_merges_branch_specific_layer_modules():
    model = _VariantTreeModel()

    layers, layer_names = get_layers_with_prefixes(model, _VariantTreeQModel.extract_layers_node())

    assert len(layers) == 2
    assert layer_names == [
        "model.A_module.layers.0",
        "model.B_module.layers.0",
    ]
    assert _VariantTreeQModel.build_layer_modules(_VariantTreeQModel.module_tree) == [
        ["self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mixer.in_proj"],
        ["mixer.out_proj"],
    ]


def test_pipe_separated_module_tree_prefix_is_not_expanded():
    assert _LegacyPipePrefixQModel.extract_layers_node() == [
        "model.A_module|B_module.layers",
    ]


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_effective_module_tree_is_instance_local_and_deeply_immutable(order):
    tree_a = ["model", "layers", "#", {"mlp": ("a",)}]
    tree_b = ["other", "blocks", "#", {"mixer": {"proj": ("b",)}}]
    trees = {"a": tree_a, "b": tree_b}
    first_key, second_key = order
    first = _init_tree_qmodel(_AutoDetectedTreeQModel, trees[first_key])
    second = _init_tree_qmodel(_AutoDetectedTreeQModel, trees[second_key])

    assert _AutoDetectedTreeQModel.module_tree is None
    expected_paths = {"a": "model.layers", "b": "other.blocks"}
    expected_modules = {"a": [["mlp.a"]], "b": [["mixer.proj.b"]]}
    assert first.extract_layers_node() == [expected_paths[first_key]]
    assert second.extract_layers_node() == [expected_paths[second_key]]
    assert first.simple_layer_modules(first.model.config, SimpleNamespace(dynamic=None)) == expected_modules[first_key]
    assert second.full_layer_modules(second.model.config) == expected_modules[second_key]
    assert first.effective_module_tree[0] == trees[first_key][0]
    try:
        first.effective_module_tree[3]["mlp"] = ("changed",)
    except TypeError:
        pass
    else:
        raise AssertionError("effective_module_tree must be deeply immutable")
    assert first.extract_layers_node() == [expected_paths[first_key]]


def test_fresh_quantized_load_planning_binds_auto_detected_tree():
    selected_tree = [
        "model",
        "layers",
        "#",
        {"self_attn": ("q_proj:0", "o_proj:1")},
    ]
    model = nn.Module()
    model.config = SimpleNamespace(model_type="unknown", selected_tree=selected_tree)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([_BranchALayer()])
    qcfg = SimpleNamespace(method=METHOD.GPTQ, dynamic=None, adapter=None)

    effective_tree = _AutoDetectedTreeQModel._resolve_effective_module_tree(model, qcfg)

    assert _AutoDetectedTreeQModel.module_tree is None
    with _AutoDetectedTreeQModel._module_tree_context(effective_tree):
        assert _AutoDetectedTreeQModel.extract_layers_node() == ["model.layers"]
        assert _AutoDetectedTreeQModel.simple_layer_modules(model.config, qcfg) == [
            ["self_attn.q_proj"],
            ["self_attn.o_proj"],
        ]

    # The loader passes this same tree into the final instance, so detection
    # does not need to run again after quantized modules replace the linears.
    instance = _AutoDetectedTreeQModel(
        model,
        quantized=True,
        quantize_config=qcfg,
        effective_module_tree=effective_tree,
    )
    assert instance.extract_layers_node() == ["model.layers"]


def test_module_tree_method_override_is_copy_on_write_for_dense_moe_tree():
    gptq = _init_tree_qmodel(_MethodOverrideTreeQModel, None, METHOD.GPTQ)
    awq = _init_tree_qmodel(_MethodOverrideTreeQModel, None, METHOD.AWQ)

    assert _MethodOverrideTreeQModel.module_tree[3]["mlp:moe"]["gate"] == ("gate:!",)
    assert gptq.effective_module_tree[3]["mlp:moe"]["gate"] == ("gate:!",)
    assert awq.effective_module_tree[3]["mlp:moe"]["gate"] == ("gate",)

    qcfg = SimpleNamespace(dynamic=None)
    gptq_modules = gptq.simple_layer_modules(gptq.model.config, qcfg)
    awq_modules = awq.simple_layer_modules(gptq.model.config, qcfg, is_awq_quantize=True)
    assert ["dense_mlp.up_proj"] in gptq_modules
    assert ["dense_mlp.down_proj"] in gptq_modules
    assert ["mlp.experts.0.up_proj", "mlp.experts.1.up_proj"] in gptq_modules
    assert all("mlp.gate" not in block for block in gptq_modules)
    assert ["mlp.gate"] in awq_modules
    assert ["mlp.experts.0.up_proj", "mlp.experts.1.up_proj"] in awq_modules
