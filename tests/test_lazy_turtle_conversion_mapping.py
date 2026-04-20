# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.models.definitions.mixtral import MixtralQModel
from gptqmodel.utils import structure as structure_module
from gptqmodel.utils.structure import LazyTurtle


def _write_checkpoint_index(path: Path, shard_name: str, state_dict: dict[str, torch.Tensor]) -> None:
    weight_map = dict.fromkeys(state_dict, shard_name)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def _build_lazy_turtle(
    tmp_path: Path,
    checkpoint_tensors: dict[str, torch.Tensor],
    *,
    module_tree=None,
    hf_conversion_map_reversed=None,
    target_model: nn.Module | None = None,
) -> LazyTurtle:
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=module_tree,
        hf_conversion_map_reversed=hf_conversion_map_reversed,
        target_model=target_model,
    )
    assert turtle is not None
    return turtle


class _Gemma3DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="gemma3")


class _LegacyGemma3DummyModel(nn.Module):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="gemma3")


class _WeightRenamingStub:
    def __init__(self, source_pattern: str, target_pattern: str):
        self.source_patterns = [source_pattern]
        self.target_patterns = [target_pattern]
        self.operations = []


def _gemma3_weight_renamings():
    return [
        _WeightRenamingStub(r"^language_model.model", "model.language_model"),
        _WeightRenamingStub(r"^language_model.lm_head", "lm_head"),
        _WeightRenamingStub(r"^vision_tower", "model.vision_tower"),
        _WeightRenamingStub(r"^multi_modal_projector", "model.multi_modal_projector"),
    ]


def _assert_gemma3_alias_resolution(turtle: LazyTurtle) -> None:
    assert turtle._resolve_checkpoint_module_path("model.language_model") == "language_model.model"
    assert turtle._resolve_checkpoint_module_path("model.vision_tower") == "vision_tower"
    assert turtle._resolve_checkpoint_module_path("model.multi_modal_projector") == "multi_modal_projector"
    assert turtle._resolve_checkpoint_module_path("lm_head") == "language_model.lm_head"

    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.language_model.layers.0.mlp",
            "gate_proj.weight",
        )
        == "language_model.model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "wrapper.model.language_model.layers.0.mlp",
            "gate_proj.weight",
        )
        == "language_model.model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.vision_tower.vision_model.head",
            "weight",
        )
        == "vision_tower.vision_model.head.weight"
    )
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.multi_modal_projector",
            "mm_input_projection_weight",
        )
        == "multi_modal_projector.mm_input_projection_weight"
    )
    assert turtle._resolve_checkpoint_tensor_name("lm_head", "weight") == "language_model.lm_head.weight"


def test_lazy_turtle_reverses_transformers_weight_renaming_list():
    reversed_map = LazyTurtle.reverse_hf_conversion_map(_gemma3_weight_renamings())

    assert reversed_map == {
        "model.language_model": "language_model.model",
        "lm_head": "language_model.lm_head",
        "model.vision_tower": "vision_tower",
        "model.multi_modal_projector": "multi_modal_projector",
    }


def test_lazy_turtle_runtime_to_checkpoint_alias_candidates_do_not_expand_infinitely(tmp_path):
    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "language_model.model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        },
        hf_conversion_map_reversed={
            "language_model": "language_model.model",
            "lm_head": "language_model.lm_head",
        },
    )

    assert turtle._runtime_to_checkpoint_alias_candidates("language_model.layers.0") == [
        "language_model.layers.0",
        "language_model.model.layers.0",
    ]


def test_lazy_turtle_uses_transformers_checkpoint_conversion_mapping_for_gemma3(tmp_path, monkeypatch):
    conversion_mapping_module = SimpleNamespace(
        get_checkpoint_conversion_mapping=lambda model_type: _gemma3_weight_renamings()
        if model_type == "gemma3"
        else None
    )
    monkeypatch.setattr(structure_module, "import_module", lambda name: conversion_mapping_module)

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "vision_tower.vision_model.head.weight": torch.zeros(2, 2),
            "multi_modal_projector.mm_input_projection_weight": torch.zeros(2, 2),
            "language_model.lm_head.weight": torch.zeros(2, 2),
        },
        target_model=_Gemma3DummyModel(),
    )

    _assert_gemma3_alias_resolution(turtle)


def test_lazy_turtle_falls_back_to_legacy_checkpoint_conversion_mapping(tmp_path, monkeypatch):
    def _raise_import_error(_name: str):
        raise ImportError("transformers.conversion_mapping is unavailable")

    monkeypatch.setattr(structure_module, "import_module", _raise_import_error)

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "vision_tower.vision_model.head.weight": torch.zeros(2, 2),
            "multi_modal_projector.mm_input_projection_weight": torch.zeros(2, 2),
            "language_model.lm_head.weight": torch.zeros(2, 2),
        },
        target_model=_LegacyGemma3DummyModel(),
    )

    _assert_gemma3_alias_resolution(turtle)


def test_lazy_turtle_keeps_module_tree_alias_resolution_for_mixtral(tmp_path):
    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(2, 2),
        },
        module_tree=MixtralQModel.module_tree,
    )

    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.layers.0.mlp.experts.0",
            "gate_proj.weight",
        )
        == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
    )
