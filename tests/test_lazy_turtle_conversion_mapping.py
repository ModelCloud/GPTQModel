# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
from gptqmodel.models.definitions.mixtral import MixtralQModel
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
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


class _Qwen2VLDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2_vl")


class _Qwen2_5_VLDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2_5_vl")


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


def _qwen2_vl_weight_renamings():
    return [
        _WeightRenamingStub(
            r"(?<!_)model(?!\.(language_model|visual))",
            "model.language_model",
        ),
        _WeightRenamingStub(r"^visual", "model.visual"),
    ]


def _renaming_pairs(renamings) -> list[tuple[str, str]]:
    assert renamings is not None
    return [(entry.source_patterns[0], entry.target_patterns[0]) for entry in renamings]


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


def _assert_qwen2_vl_alias_resolution(turtle: LazyTurtle) -> None:
    assert turtle._resolve_checkpoint_module_path("model.language_model") == "model"
    assert turtle._resolve_checkpoint_module_path("model.visual") == "visual"

    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.language_model.layers.0.mlp",
            "gate_proj.weight",
        )
        == "model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "wrapper.model.language_model.layers.0.mlp",
            "gate_proj.weight",
        )
        == "model.layers.0.mlp.gate_proj.weight"
    )
    assert (
        turtle._resolve_checkpoint_tensor_name(
            "model.visual.blocks.0.attn",
            "weight",
        )
        == "visual.blocks.0.attn.weight"
    )


def test_lazy_turtle_reverses_transformers_weight_renaming_list():
    reversed_map = LazyTurtle.reverse_hf_conversion_map(_gemma3_weight_renamings())

    assert _renaming_pairs(reversed_map) == [
        ("model.language_model", r"^language_model.model"),
        ("lm_head", r"^language_model.lm_head"),
        ("model.vision_tower", r"^vision_tower"),
        ("model.multi_modal_projector", r"^multi_modal_projector"),
    ]


def test_lazy_turtle_runtime_to_checkpoint_alias_candidates_do_not_expand_infinitely(tmp_path):
    reversed_map = LazyTurtle.reverse_hf_conversion_map(
        {
            "language_model.model": "language_model",
            "language_model.lm_head": "lm_head",
        }
    )

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "language_model.model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        },
        hf_conversion_map_reversed=reversed_map,
    )

    assert turtle._runtime_to_checkpoint_alias_candidates("language_model.layers.0") == [
        "language_model.layers.0",
        "language_model.model.layers.0",
    ]


def test_lazy_turtle_applies_reversed_weight_renamings_with_capturing_groups(tmp_path):
    reversed_map = LazyTurtle.reverse_hf_conversion_map(
        [_WeightRenamingStub(r"(.+)", r"timm_model.\1")]
    )

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "backbone.conv.weight": torch.zeros(2, 2),
        },
        hf_conversion_map_reversed=reversed_map,
    )

    assert turtle._resolve_checkpoint_tensor_name("timm_model.backbone.conv", "weight") == "backbone.conv.weight"


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


def test_lazy_turtle_uses_transformers_checkpoint_conversion_mapping_for_qwen2_vl(tmp_path, monkeypatch):
    conversion_mapping_module = SimpleNamespace(
        get_checkpoint_conversion_mapping=lambda model_type: _qwen2_vl_weight_renamings()
        if model_type == "qwen2_vl"
        else None
    )
    monkeypatch.setattr(structure_module, "import_module", lambda name: conversion_mapping_module)

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "visual.blocks.0.attn.weight": torch.zeros(2, 2),
        },
        module_tree=Qwen2VLQModel.module_tree,
        target_model=_Qwen2VLDummyModel(),
    )

    _assert_qwen2_vl_alias_resolution(turtle)


def test_lazy_turtle_uses_transformers_checkpoint_conversion_mapping_for_qwen2_5_vl(tmp_path, monkeypatch):
    observed_model_types: list[str] = []

    def _get_checkpoint_conversion_mapping(model_type: str):
        observed_model_types.append(model_type)
        if model_type == "qwen2_5_vl":
            return _qwen2_vl_weight_renamings()
        return None

    conversion_mapping_module = SimpleNamespace(get_checkpoint_conversion_mapping=_get_checkpoint_conversion_mapping)
    monkeypatch.setattr(structure_module, "import_module", lambda name: conversion_mapping_module)

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "visual.blocks.0.attn.weight": torch.zeros(2, 2),
        },
        module_tree=Qwen2_5_VLQModel.module_tree,
        target_model=_Qwen2_5_VLDummyModel(),
    )

    assert observed_model_types == ["qwen2_5_vl"]
    _assert_qwen2_vl_alias_resolution(turtle)


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


def test_base_qmodel_prefers_manual_hf_conversion_map_reversed(tmp_path, monkeypatch):
    manual_renamings = LazyTurtle.reverse_hf_conversion_map(_gemma3_weight_renamings())
    assert manual_renamings is not None
    monkeypatch.setattr(
        Gemma3ForConditionalGenerationGPTQ,
        "HF_CONVERSION_MAP_REVERSED",
        manual_renamings,
        raising=False,
    )

    def _unexpected_import(_name: str):
        raise AssertionError("manual HF_CONVERSION_MAP_REVERSED should bypass inferred transformers mappings")

    monkeypatch.setattr(structure_module, "import_module", _unexpected_import)

    resolved = Gemma3ForConditionalGenerationGPTQ.resolve_hf_conversion_map_reversed(target_model=_Gemma3DummyModel())
    assert _renaming_pairs(resolved) == _renaming_pairs(manual_renamings)

    resolved[0].source_patterns[0] = "mutated.runtime"
    resolved_again = Gemma3ForConditionalGenerationGPTQ.resolve_hf_conversion_map_reversed(
        target_model=_Gemma3DummyModel()
    )
    assert _renaming_pairs(resolved_again) == _renaming_pairs(manual_renamings)

    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 2),
            "vision_tower.vision_model.head.weight": torch.zeros(2, 2),
            "multi_modal_projector.mm_input_projection_weight": torch.zeros(2, 2),
            "language_model.lm_head.weight": torch.zeros(2, 2),
        },
        module_tree=Gemma3ForConditionalGenerationGPTQ.module_tree,
        hf_conversion_map_reversed=resolved_again,
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
