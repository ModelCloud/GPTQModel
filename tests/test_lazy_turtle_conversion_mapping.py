# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


class _ConvertibleCheckpointModel(nn.Module):
    from_pretrained_calls = []
    save_pretrained_calls = []

    def __init__(self, checkpoint_tensors: dict[str, torch.Tensor] | None = None):
        super().__init__()
        self.config = SimpleNamespace(model_type="dummy")
        self._checkpoint_tensors = {
            name: tensor.clone()
            for name, tensor in (checkpoint_tensors or {}).items()
        }

    @classmethod
    def reset_tracking(cls) -> None:
        cls.from_pretrained_calls = []
        cls.save_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, model_local_path: str, config=None, **kwargs):
        cls.from_pretrained_calls.append(
            {
                "model_local_path": str(model_local_path),
                "config": config,
                "kwargs": dict(kwargs),
            }
        )
        model_dir = Path(model_local_path)
        for filename in (
            "model.bin",
            "pytorch_model.bin",
            "model.pt",
            "pytorch_model.pt",
            "model.pth",
            "pytorch_model.pth",
            "model.ckpt",
            "pytorch_model.ckpt",
        ):
            checkpoint_path = model_dir / filename
            if not checkpoint_path.is_file():
                continue
            payload = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            model = cls(payload)
            if config is not None:
                model.config = config
            return model
        raise FileNotFoundError(f"No supported checkpoint file found in {model_local_path}")

    def save_pretrained(self, save_dir: str, safe_serialization: bool = False, **kwargs) -> None:
        type(self).save_pretrained_calls.append(
            {
                "save_dir": str(save_dir),
                "safe_serialization": safe_serialization,
                "kwargs": dict(kwargs),
            }
        )
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        shard_name = "model.safetensors"
        save_file(self._checkpoint_tensors, str(save_path / shard_name))
        _write_checkpoint_index(save_path, shard_name, self._checkpoint_tensors)


class _FailingConvertibleCheckpointModel(_ConvertibleCheckpointModel):
    @classmethod
    def from_pretrained(cls, model_local_path: str, config=None, **kwargs):
        cls.from_pretrained_calls.append(
            {
                "model_local_path": str(model_local_path),
                "config": config,
                "kwargs": dict(kwargs),
            }
        )
        raise RuntimeError("full model reload is unavailable in this test")


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


@pytest.mark.parametrize(
    ("extension", "payload"),
    [
        (".bin", lambda tensors: tensors),
        (".pt", lambda tensors: {"state_dict": tensors}),
    ],
)
def test_lazy_turtle_converts_non_safetensors_checkpoints_via_full_model_reload(tmp_path, extension, payload):
    checkpoint_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
    }
    model_dir = tmp_path / f"source_model_{extension[1:]}"
    model_dir.mkdir()
    torch.save(payload(checkpoint_tensors), model_dir / f"pytorch_model{extension}")

    _ConvertibleCheckpointModel.reset_tracking()
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None, model_type="dummy"),
        model_init_kwargs={"device_map": {"": "cpu"}, "trust_remote_code": False},
        hf_conversion_map_reversed={},
        target_model=_ConvertibleCheckpointModel(),
    )

    assert turtle is not None
    assert Path(turtle.model_local_path) != model_dir
    assert (Path(turtle.model_local_path) / "model.safetensors").is_file()
    assert len(_ConvertibleCheckpointModel.from_pretrained_calls) == 1
    assert len(_ConvertibleCheckpointModel.save_pretrained_calls) == 1
    assert _ConvertibleCheckpointModel.save_pretrained_calls[0]["safe_serialization"] is True
    assert "device_map" not in _ConvertibleCheckpointModel.from_pretrained_calls[0]["kwargs"]
    assert _ConvertibleCheckpointModel.from_pretrained_calls[0]["kwargs"]["low_cpu_mem_usage"] is False

    tensors = turtle._load_checkpoint_tensors_for_module_path(
        module_path="model.layers.0.self_attn",
        recurse=True,
    )
    assert torch.equal(
        tensors["q_proj.weight"],
        checkpoint_tensors["model.layers.0.self_attn.q_proj.weight"],
    )


def test_lazy_turtle_disables_itself_when_transformers_full_reload_fails(tmp_path):
    checkpoint_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
    }
    model_dir = tmp_path / "source_model_pt_fallback"
    model_dir.mkdir()
    torch.save({"state_dict": checkpoint_tensors}, model_dir / "pytorch_model.pt")

    _FailingConvertibleCheckpointModel.reset_tracking()
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None, model_type="dummy"),
        model_init_kwargs={"device_map": {"": "cpu"}},
        hf_conversion_map_reversed={},
        target_model=_FailingConvertibleCheckpointModel(),
    )

    assert turtle is None
    assert len(_FailingConvertibleCheckpointModel.from_pretrained_calls) == 1
    assert _FailingConvertibleCheckpointModel.save_pretrained_calls == []


def test_lazy_turtle_registers_atexit_cleanup_for_temporary_checkpoint_dir(tmp_path, monkeypatch):
    checkpoint_tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
    }
    model_dir = tmp_path / "source_model_close"
    model_dir.mkdir()
    torch.save(checkpoint_tensors, model_dir / "pytorch_model.bin")

    registered = []

    def _register(func, *args, **kwargs):
        registered.append((func, args, kwargs))
        return (func, args, kwargs)

    monkeypatch.setattr(structure_module.atexit, "register", _register)

    _ConvertibleCheckpointModel.reset_tracking()
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None, model_type="dummy"),
        model_init_kwargs={"device_map": {"": "cpu"}},
        hf_conversion_map_reversed={},
        target_model=_ConvertibleCheckpointModel(),
    )

    assert turtle is not None
    temp_path = Path(turtle.model_local_path)
    assert temp_path.exists()
    assert registered == [
        (structure_module.shutil.rmtree, (str(temp_path),), {"ignore_errors": True}),
    ]
