# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import types

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.models.base import _QuantizedCheckpointSource
from gptqmodel.models.writer import _save_module_replacement_safetensors
from gptqmodel.utils.checkpoint_paths import (
    CheckpointPathError,
    assert_auxiliary_copy_allowed,
    resolve_checkpoint_path,
    validate_relative_checkpoint_path,
)
from gptqmodel.utils.model import copy_py_files
from gptqmodel.utils.structure import LazyTurtle


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/absolute/model.safetensors",
        "../outside.safetensors",
        "model.layers.0/../../outside.safetensors",
        "model.layers.0/./layer.safetensors",
        "model.layers.0//layer.safetensors",
        "model\\layers\\0\\layer.safetensors",
        "model.layers.0/\x00layer.safetensors",
    ],
)
def test_validate_relative_checkpoint_path_rejects_unsafe_names(name):
    with pytest.raises(CheckpointPathError):
        validate_relative_checkpoint_path(name)


def test_validate_relative_checkpoint_path_accepts_nested_shard():
    assert validate_relative_checkpoint_path("model.layers.0/layer.safetensors") == (
        "model.layers.0/layer.safetensors"
    )


def test_resolve_checkpoint_path_rejects_escape(tmp_path):
    with pytest.raises(CheckpointPathError):
        resolve_checkpoint_path(tmp_path, "../outside.safetensors")


def _write_index(root, weight_map):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "shard_name",
    ["../outside.safetensors", "/tmp/gptqmodel-victim/outside.safetensors"],
)
def test_quantized_checkpoint_source_rejects_unsafe_weight_map(tmp_path, shard_name):
    source = tmp_path / "source"
    _write_index(source, {"model.layers.0.weight": shard_name})

    with pytest.raises(CheckpointPathError):
        _QuantizedCheckpointSource(str(source))


@pytest.mark.parametrize(
    "shard_name",
    ["../outside.safetensors", "/tmp/gptqmodel-victim/outside.safetensors"],
)
def test_lazy_turtle_weight_map_rejects_unsafe_shard(tmp_path, shard_name):
    source = tmp_path / "source"
    _write_index(source, {"model.layers.0.weight": shard_name})

    with pytest.raises(CheckpointPathError):
        LazyTurtle.__new__(LazyTurtle)._load_weight_map(str(source))


@pytest.mark.parametrize(
    "shard_name",
    ["../outside.safetensors", "/tmp/gptqmodel-victim/outside.safetensors"],
)
def test_module_replacement_rejects_unsafe_weight_map(tmp_path, shard_name):
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {"model.layers.0.weight": torch.ones(2, 2)},
        str(source / "model-00001-of-00001.safetensors"),
    )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    model.config = types.SimpleNamespace(tie_word_embeddings=False)
    turtle_model = types.SimpleNamespace(
        _weight_map={"model.layers.0.weight": shard_name},
        model_local_path=str(source),
        _index_filename="model.safetensors.index.json",
    )
    save_dir = tmp_path / "out"
    save_dir.mkdir()

    with pytest.raises(CheckpointPathError):
        _save_module_replacement_safetensors(
            model,
            turtle_model,
            ["model.layers.0"],
            save_dir=str(save_dir),
            metadata={"format": "pt"},
        )

    assert list(save_dir.iterdir()) == []
    assert not (tmp_path / "outside.safetensors").exists()


def test_auxiliary_symlink_escape_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "id_rsa"
    secret.write_text("private key", encoding="utf-8")
    os.symlink(secret, source / "README.md")

    with pytest.raises(CheckpointPathError):
        assert_auxiliary_copy_allowed(str(source), str(source / "README.md"))


def test_auxiliary_hf_blob_symlink_allowed(tmp_path):
    hub = tmp_path / "hub"
    blob = hub / "blobs" / "abc"
    snapshot = hub / "snapshots" / "rev"
    blob.parent.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob.write_text("readme", encoding="utf-8")
    os.symlink(os.path.join("..", "..", "blobs", "abc"), snapshot / "README.md")

    assert_auxiliary_copy_allowed(str(snapshot), str(snapshot / "README.md"))


def test_copy_py_files_rejects_symlink_escape(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("TOKEN = 'secret'\n", encoding="utf-8")
    os.symlink(secret, source / "custom.py")
    save_dir = tmp_path / "out"

    with pytest.raises(CheckpointPathError):
        copy_py_files(str(save_dir), model_id_or_path=str(source))

    assert not (save_dir / "custom.py").exists()
