# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5379: the legacy sidecar lookup must follow the checkpoint selection.

`GPTQModel.load()` decides between the dense and the quantized loader by looking
for `quantize_config.json` / `quant_config.json` when `config.json` carries no
embedded quantization metadata. That probe ignored `subfolder`/`revision` and
called the Hub listing API even when the caller was offline.
"""

from __future__ import annotations

import json

import pytest

from gptqmodel import GPTQModel
from gptqmodel.models import auto as auto_module
from gptqmodel.models.auto import _has_legacy_quant_sidecar


pytestmark = [pytest.mark.cpu]

_CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}


def _write_checkpoint(path, *, sidecar: bool):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    if sidecar:
        (path / "quantize_config.json").write_text(
            json.dumps({"bits": 4, "group_size": 32}), encoding="utf-8"
        )
    return path


class _Loaded:
    """The minimum `load()` touches on the object a loader returns."""

    def log_budget(self):
        return None


def _capture_dispatch(monkeypatch):
    """Record which loader `load()` picks instead of loading a model."""

    calls = {"quantized": [], "dense": []}

    def _quantized(**kwargs):
        calls["quantized"].append(kwargs)
        return _Loaded()

    def _dense(**kwargs):
        calls["dense"].append(kwargs)
        return _Loaded()

    monkeypatch.setattr(GPTQModel, "from_quantized", staticmethod(_quantized))
    monkeypatch.setattr(GPTQModel, "from_pretrained", staticmethod(_dense))
    return calls


def test_nested_checkpoint_with_legacy_sidecar_dispatches_to_the_quantized_loader(
    tmp_path, monkeypatch
):
    _write_checkpoint(tmp_path / "gptq", sidecar=True)
    calls = _capture_dispatch(monkeypatch)

    GPTQModel.load(str(tmp_path), subfolder="gptq", device="cpu")

    assert calls["quantized"], "a nested sidecar-only checkpoint must use the quantized loader"
    assert calls["dense"] == []
    assert calls["quantized"][0]["subfolder"] == "gptq"


def test_sidecar_at_the_repository_root_does_not_select_a_nested_dense_checkpoint(
    tmp_path, monkeypatch
):
    """The probe must look where the config and weights resolve (#5379)."""

    _write_checkpoint(tmp_path, sidecar=True)
    _write_checkpoint(tmp_path / "dense", sidecar=False)
    calls = _capture_dispatch(monkeypatch)

    GPTQModel.load(str(tmp_path), subfolder="dense", device="cpu")

    assert calls["quantized"] == [], "the root sidecar must not select a nested checkpoint"
    assert calls["dense"], "the nested checkpoint is dense and must use the dense loader"


def test_local_probe_resolves_the_subfolder(tmp_path):
    _write_checkpoint(tmp_path, sidecar=False)
    _write_checkpoint(tmp_path / "gptq", sidecar=True)

    assert _has_legacy_quant_sidecar(
        str(tmp_path), {"subfolder": "gptq"}, treat_as_local_path=True
    ) is True
    assert _has_legacy_quant_sidecar(
        str(tmp_path), {}, treat_as_local_path=True
    ) is False


def test_offline_probe_never_lists_the_repository(monkeypatch, tmp_path):
    from huggingface_hub import constants as hub_constants

    def _offline(*args, **kwargs):
        raise AssertionError("the offline probe must not list the repository")

    monkeypatch.setattr(auto_module, "list_repo_files", _offline)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(auto_module, "_cached_sidecar_exists", lambda *a, **k: False)

    assert (
        _has_legacy_quant_sidecar(
            "org/tiny", {"subfolder": "gptq", "revision": "abc123"},
            treat_as_local_path=False,
        )
        is False
    )


def test_offline_probe_uses_the_local_cache(monkeypatch):
    from huggingface_hub import constants as hub_constants

    seen = []

    def _cached(repo_id, filename, *, subfolder, revision):
        seen.append((repo_id, filename, subfolder, revision))
        return filename == "quant_config.json"

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(auto_module, "_cached_sidecar_exists", _cached)

    assert (
        _has_legacy_quant_sidecar(
            "org/tiny", {"subfolder": "gptq", "revision": "abc123"},
            treat_as_local_path=False,
        )
        is True
    )
    assert seen[0] == ("org/tiny", "quantize_config.json", "gptq", "abc123")


def test_remote_probe_matches_the_subfolder_and_revision(monkeypatch):
    seen = []

    def _list(repo_id, revision=None):
        seen.append((repo_id, revision))
        return ["config.json", "gptq/quantize_config.json", "README.md"]

    monkeypatch.setattr(auto_module, "list_repo_files", _list)
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)

    assert (
        _has_legacy_quant_sidecar(
            "org/tiny",
            {"subfolder": "gptq", "revision": "abc123"},
            treat_as_local_path=False,
        )
        is True
    )
    # The repository-root sidecar must not select a nested checkpoint.
    assert (
        _has_legacy_quant_sidecar(
            "org/tiny", {"subfolder": "dense"}, treat_as_local_path=False
        )
        is False
    )
    assert seen == [("org/tiny", "abc123"), ("org/tiny", None)]
