# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Nested quantized checkpoints resolve every artifact under ``subfolder``.

`from_quantized(..., subfolder="nested")` used to forward ``subfolder`` only to
``AutoConfig.from_pretrained``. The quantize config, the checkpoint shards, and
a relative adapter manifest were all resolved at the repository root, so a
nested checkpoint either failed to load or mixed metadata from two directories.
"""

import json
from pathlib import Path

import pytest

import gptqmodel.utils.model as model_utils
from gptqmodel.models import loader as loader_module
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.hf import resolve_checkpoint_subfolder


def _write_nested_quantize_config(root: Path, bits: int = 4) -> Path:
    nested = root / "nested"
    nested.mkdir()
    QuantizeConfig(bits=bits).save_pretrained(nested)
    return nested


@pytest.mark.parametrize("filename", ["quantize_config.json", "quant_config.json", "config.json"])
def test_quantize_config_from_pretrained_honors_subfolder(tmp_path, filename):
    nested = tmp_path / "nested"
    nested.mkdir()
    payload = {"bits": 8, "group_size": 64, "sym": True}
    body = {"quantization_config": payload} if filename == "config.json" else payload
    (nested / filename).write_text(json.dumps(body), encoding="utf-8")

    # The repository root has no config, so success proves the subfolder was used.
    assert not (tmp_path / filename).exists()
    config = QuantizeConfig.from_pretrained(str(tmp_path), subfolder="nested")

    assert config.bits == 8
    assert config.group_size == 64


def test_quantize_config_from_pretrained_rejects_escaping_subfolder(tmp_path):
    (tmp_path / "checkpoint").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    QuantizeConfig(bits=4).save_pretrained(outside)

    for subfolder in ("../outside", "nested/../../outside", "/etc", "C:\\checkpoint", "nested\\..\\.."):
        with pytest.raises(ValueError, match="subfolder"):
            QuantizeConfig.from_pretrained(str(tmp_path / "checkpoint"), subfolder=subfolder)


def test_quantize_config_from_pretrained_rejects_symlinked_subfolder(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    QuantizeConfig(bits=4).save_pretrained(outside)
    (checkpoint / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="inside the checkpoint"):
        QuantizeConfig.from_pretrained(str(checkpoint), subfolder="nested")


def test_quantize_config_from_pretrained_without_subfolder_is_unchanged(tmp_path):
    nested = _write_nested_quantize_config(tmp_path, bits=3)

    assert QuantizeConfig.from_pretrained(str(nested)).bits == 3
    assert QuantizeConfig.from_pretrained(str(nested), subfolder="").bits == 3
    assert QuantizeConfig.from_pretrained(str(nested), subfolder=".").bits == 3


def test_get_checkpoints_resolves_local_subfolder(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "model.safetensors").write_bytes(b"")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}), encoding="utf-8"
    )

    is_sharded, archive, basename = model_utils.get_checkpoints(
        str(tmp_path), [".safetensors"], ["model"], subfolder="nested"
    )
    assert is_sharded is True
    assert archive == str(nested / "model.safetensors.index.json")
    assert basename == str(nested / "model")

    (nested / "model.safetensors.index.json").unlink()
    is_sharded, archive, basename = model_utils.get_checkpoints(
        str(tmp_path), [".safetensors"], ["model"], subfolder="nested"
    )
    assert is_sharded is False
    assert archive == str(nested / "model.safetensors")
    assert basename == "model"


def test_get_checkpoints_does_not_fall_back_to_the_repository_root(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="nested"):
        model_utils.get_checkpoints(str(tmp_path), [".safetensors"], ["model"], subfolder="nested")


def test_get_checkpoints_hub_branch_forwards_subfolder(tmp_path, monkeypatch):
    calls = []
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"")

    def fake_cached_file(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs.get("subfolder")))
        if filename == "model.safetensors":
            return str(weights)
        return None

    monkeypatch.setattr(model_utils, "cached_file", fake_cached_file)

    is_sharded, archive, _ = model_utils.get_checkpoints(
        "org/repo", [".safetensors"], ["model"], subfolder="nested", revision="abc"
    )

    assert is_sharded is False
    assert archive == str(weights)
    assert calls == [
        ("org/repo", "model.safetensors.index.json", "nested"),
        ("org/repo", "model.safetensors", "nested"),
    ]


def test_resolve_checkpoint_subfolder_narrows_without_normalizing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()

    assert resolve_checkpoint_subfolder(str(tmp_path), None) == str(tmp_path)
    assert resolve_checkpoint_subfolder(str(tmp_path), "") == str(tmp_path)
    assert resolve_checkpoint_subfolder(str(tmp_path), "a/b") == str(tmp_path / "a" / "b")
    assert resolve_checkpoint_subfolder(str(tmp_path), "a//b/") == str(tmp_path / "a" / "b")


def test_loader_resolves_adapter_manifest_under_subfolder(tmp_path, monkeypatch):
    """`from_quantized` must hand the subfolder root to the manifest resolver."""
    from gptqmodel.models.base import BaseQModel

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "adapter").mkdir()
    (nested / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "sym": True,
                    "adapter": {"name": "lora", "rank": 2, "path": "adapter"},
                },
            }
        ),
        encoding="utf-8",
    )
    seen = {}

    class _Stop(Exception):
        """Abort once the loader has resolved the manifest root."""

    def fake_resolve(path, checkpoint_path):
        seen.update(path=path, checkpoint=checkpoint_path)
        raise _Stop

    monkeypatch.setattr(loader_module, "resolve_checkpoint_adapter_path", fake_resolve)

    with pytest.raises(_Stop):
        BaseQModel.from_quantized(str(tmp_path), subfolder="nested", device="cpu")

    assert seen == {"path": "adapter", "checkpoint": str(nested)}
