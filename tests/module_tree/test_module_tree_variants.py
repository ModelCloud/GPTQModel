# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from gptqmodel.models.base import BaseQModel
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
