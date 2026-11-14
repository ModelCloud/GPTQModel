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


def _write_nested_llama_checkpoint(root: Path) -> Path:
    """Write a nested checkpoint whose config is only readable via ``subfolder``."""
    nested = root / "nested"
    nested.mkdir()
    (nested / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "vocab_size": 8,
                "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
            }
        ),
        encoding="utf-8",
    )
    return nested


def test_config_load_kwargs_forward_subfolder_and_revision():
    """Model-type detection must read the same config file the loader reads."""
    from gptqmodel.models import auto

    assert auto._get_config_load_kwargs({"subfolder": "nested", "revision": "abc123"}) == {
        "subfolder": "nested",
        "revision": "abc123",
    }
    # An unset option keeps the AutoConfig default instead of forcing None.
    assert auto._get_config_load_kwargs({}) == {}
    assert auto._get_config_load_kwargs({"revision": None}) == {}
    # An explicit empty subfolder means "the repository root" and is preserved.
    assert auto._get_config_load_kwargs({"subfolder": ""}) == {"subfolder": ""}
    # The GGUF selector keeps working alongside the checkpoint selectors.
    assert auto._get_config_load_kwargs(
        {loader_module.INTERNAL_HF_GGUF_FILE_KWARG: "m.gguf", "subfolder": "nested"}
    ) == {"subfolder": "nested", "gguf_file": "m.gguf"}


def test_tokenizer_load_kwargs_forward_subfolder_and_revision():
    from gptqmodel.models import loader as loader_module

    assert loader_module._get_tokenizer_load_kwargs({"subfolder": "nested", "revision": "abc123"}) == {
        "subfolder": "nested",
        "revision": "abc123",
    }
    assert loader_module._get_tokenizer_load_kwargs({}) == {}


@pytest.mark.parametrize("entry_point", ["from_quantized", "load"])
def test_public_dispatch_resolves_a_nested_checkpoint_config(tmp_path, entry_point):
    """The reported failure: public dispatch dropped ``subfolder`` before the loader.

    With only ``nested/config.json`` present, dispatch raised
    ``ValueError: Unrecognized model in <root>`` because ``_get_config_load_kwargs``
    returned ``{}`` for ``{"subfolder": "nested"}``. The assertion is on that
    specific failure mode: any later error (missing weights or tokenizer files)
    is outside the dispatch contract being pinned here.
    """
    from gptqmodel import GPTQModel

    _write_nested_llama_checkpoint(tmp_path)
    assert not (tmp_path / "config.json").exists()

    try:
        getattr(GPTQModel, entry_point)(str(tmp_path), subfolder="nested", device="cpu")
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        assert "Unrecognized model" not in str(exc), exc


def test_loader_forwards_subfolder_and_revision_to_the_tokenizer(tmp_path, monkeypatch):
    """The tokenizer must come from the same directory and revision as the weights."""
    from gptqmodel.models.base import BaseQModel

    _write_nested_llama_checkpoint(tmp_path)
    seen = {}

    class _Stop(Exception):
        """Abort once the loader has reached the tokenizer stage."""

    def fake_tokenizer(path, **kwargs):
        seen.update(path=path, kwargs=kwargs)
        raise _Stop

    monkeypatch.setattr(loader_module, "load_hf_tokenizer", fake_tokenizer)

    with pytest.raises(_Stop):
        BaseQModel.from_quantized(str(tmp_path), subfolder="nested", revision="abc123", device="cpu")

    assert seen["path"] == str(tmp_path)
    assert seen["kwargs"]["subfolder"] == "nested"
    assert seen["kwargs"]["revision"] == "abc123"


def test_public_dispatch_rejects_an_escaping_subfolder(tmp_path):
    """An escaping subfolder is refused before any config read."""
    from gptqmodel import GPTQModel

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}}),
        encoding="utf-8",
    )

    for subfolder in ("../outside", "/etc", "C:\\checkpoint", "nested\\..\\.."):
        for entry_point in ("from_quantized", "load"):
            with pytest.raises(ValueError, match="subfolder"):
                getattr(GPTQModel, entry_point)(str(checkpoint), subfolder=subfolder, device="cpu")


def test_normalize_checkpoint_subfolder_canonicalizes_without_touching_the_root():
    from gptqmodel.utils.hf import normalize_checkpoint_subfolder

    assert normalize_checkpoint_subfolder(None) == ""
    assert normalize_checkpoint_subfolder("") == ""
    assert normalize_checkpoint_subfolder(".") == ""
    assert normalize_checkpoint_subfolder("a/b") == "a/b"
    assert normalize_checkpoint_subfolder("a//b/") == "a/b"
    with pytest.raises(ValueError, match="subfolder"):
        normalize_checkpoint_subfolder("../x")
