# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Auto-detection must return every concurrent decoder branch, not only the first."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import METHOD
from gptqmodel.utils.model import get_layers_with_prefixes


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = DummyMLP()


class DummyModel:
    def __init__(self):
        self.layers = nn.ModuleList([DummyBlock()])


class ParallelTextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4)
        self.self_attn.o_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(4, 4)
        self.mlp.down_proj = nn.Linear(4, 4)


class ParallelVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4)
        self.self_attn.o_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 4)
        self.mlp.fc2 = nn.Linear(4, 4)


class ParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ParallelTextBlock(), ParallelTextBlock()])
        self.model.embed = nn.Embedding(4, 4)
        self.model.norm = nn.LayerNorm(4)
        self.vision_tower = nn.Module()
        self.vision_tower.layers = nn.ModuleList([ParallelVisionBlock()])
        self.vision_tower.patch = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)


class ParallelQModel(BaseQModel):
    layer_modules_strict = False
    module_tree = None


def _wrapper(model, tree):
    wrapper = ParallelQModel.__new__(ParallelQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.effective_module_tree = tree
    wrapper.quantize_config = SimpleNamespace(dynamic=None, method=METHOD.GPTQ)
    return wrapper


def _detect(model):
    return BaseQModel.__new__(BaseQModel)._auto_detect_module_tree(model, quant_method="gptq")


def test_diagnostic_tree_detection_remains_available():
    model = DummyModel()
    tree = _detect(model)
    assert tree[0] == "layers"
    assert tree[1] == "#"
    mapping = tree[2]
    assert "self_attn" in mapping
    assert "mlp" in mapping
    assert set(mapping["self_attn"]) == {"q_proj", "k_proj"}
    assert set(mapping["mlp"]) == {"fc1", "fc2"}


def test_auto_detect_discovers_every_concurrent_branch():
    model = ParallelModel()
    tree = _detect(model)

    assert isinstance(tree, list)
    assert [variant[:2] for variant in tree] == [["model", "layers"], ["vision_tower", "layers"]]
    assert set(tree[0][3]["self_attn"]) == {"q_proj", "o_proj"}
    assert set(tree[0][3]["mlp"]) == {"up_proj", "down_proj"}
    assert set(tree[1][3]["self_attn"]) == {"q_proj", "o_proj"}
    assert set(tree[1][3]["mlp"]) == {"fc1", "fc2"}


def test_auto_detect_finds_a_non_whitelisted_transformer_stack():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([ParallelTextBlock()])
    model.vision_tower = nn.Module()
    model.vision_tower.stages = nn.ModuleList([ParallelVisionBlock()])

    tree = _detect(model)

    assert [variant[:2] for variant in tree] == [
        ["model", "layers"],
        ["vision_tower", "stages"],
    ]


def test_auto_detected_parallel_tree_resolves_every_prefixed_layer():
    model = ParallelModel()
    tree = _detect(model)
    wrapper = _wrapper(model, tree)

    assert wrapper.extract_layers_node() == ["model.layers", "vision_tower.layers"]
    assert wrapper.get_base_modules(model) == [
        "model.embed",
        "model.norm",
        "vision_tower.patch",
    ]

    layers, layer_names = get_layers_with_prefixes(model, wrapper.extract_layers_node())
    assert len(layers) == 3
    assert layer_names == [
        "model.layers.0",
        "model.layers.1",
        "vision_tower.layers.0",
    ]

    modules = wrapper.simple_layer_modules(SimpleNamespace(), wrapper.quantize_config)
    flat_modules = {name for block in modules for name in block}
    assert {
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.fc1",
        "mlp.fc2",
    } <= flat_modules


def test_each_branch_keeps_its_own_leaf_declaration():
    model = ParallelModel()
    tree = _detect(model)
    wrapper = _wrapper(model, tree)

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    looper.processors = []
    looper._layer_modules_by_prefix = {}

    text_modules = looper.layer_modules_for_prefix("model.layers.0")
    vision_modules = looper.layer_modules_for_prefix("vision_tower.layers.0")

    assert text_modules == [["self_attn.q_proj", "self_attn.o_proj"], ["mlp.up_proj", "mlp.down_proj"]]
    assert vision_modules == [["self_attn.q_proj", "self_attn.o_proj"], ["mlp.fc1", "mlp.fc2"]]
    assert ["mlp.fc1"] not in text_modules
    assert ["mlp.up_proj"] not in vision_modules


def test_auto_detect_returns_none_when_no_branch_is_resolvable():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.LayerNorm(4)])
    assert _detect(model) is None


def test_apply_requested_module_tree_auto_attaches_every_branch():
    from gptqmodel.models.loader import _apply_requested_module_tree

    model = ParallelModel()
    model.config.model_type = "unknown_parallel_test"
    _apply_requested_module_tree(
        ParallelQModel,
        model,
        "auto",
        SimpleNamespace(method=METHOD.GPTQ),
        allow_auto=True,
    )
    assert model.gptqmodel_effective_module_tree == _detect(model)


def test_auto_module_tree_requires_an_unquantized_source_and_resolvable_branch():
    from gptqmodel.models.loader import _apply_requested_module_tree

    model = nn.Module()
    model.config = SimpleNamespace(model_type="unknown_parallel_test")
    model.layers = nn.ModuleList([DummyBlock()])
    quantize_config = SimpleNamespace(method=METHOD.GPTQ)

    with pytest.raises(ValueError, match="requires an unquantized source"):
        _apply_requested_module_tree(
            BaseQModel, model, "auto", quantize_config, allow_auto=False
        )

    model.layers = nn.ModuleList([nn.LayerNorm(4)])
    with pytest.raises(ValueError, match="could not resolve any quantizable decoder branch"):
        _apply_requested_module_tree(
            BaseQModel, model, "auto", quantize_config, allow_auto=True
        )
