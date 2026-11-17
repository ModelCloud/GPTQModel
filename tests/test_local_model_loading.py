# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The loading policy applies before config Python, including retries/reloads."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP_CASEFOLD, check_and_get_model_definition
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.utils.hf import (
    has_native_transformers_model_support,
    load_model_config,
    normalize_hf_config_compat,
    prepare_remote_model_init_compat,
    resolve_trust_remote_code,
)
from gptqmodel.utils.security import BYPASS_ENV, RemoteCodeSecurityError


@pytest.fixture(autouse=True)
def security_default(monkeypatch):
    monkeypatch.delenv(BYPASS_ENV, raising=False)


def checkpoint(path, model_type="unported_fixture"):
    path.mkdir()
    marker = path / "EXECUTED"
    (path / "remote_config.py").write_text(
        "from pathlib import Path\nfrom transformers import PreTrainedConfig\n"
        f"Path({str(marker)!r}).write_text('config')\n"
        "class RemoteConfig(PreTrainedConfig):\n    model_type = 'unported_fixture'\n"
    )
    (path / "remote_model.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('model')\n"
        "from transformers import LlamaForCausalLM\n"
        "class RemoteModel(LlamaForCausalLM):\n    pass\n"
    )
    payload = {
        "model_type": model_type,
        "auto_map": {"AutoConfig": "remote_config.RemoteConfig", "AutoModelForCausalLM": "remote_model.RemoteModel"},
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 1,
    }
    (path / "config.json").write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize("entry", [GPTQModel.load, GPTQModel.from_pretrained, GPTQModel.from_quantized])
def test_public_load_requires_security_bypass_before_config_code(tmp_path, entry):
    path = tmp_path / "remote"
    marker = checkpoint(path)
    with pytest.raises(RemoteCodeSecurityError, match=BYPASS_ENV):
        if entry.__name__ == "from_pretrained":
            entry(str(path), quantize_config=None, trust_remote_code=True)
        else:
            entry(str(path), trust_remote_code=True)
    assert not marker.exists()


def test_bypass_requires_explicit_trust_and_is_rechecked_on_retry(tmp_path, monkeypatch):
    path = tmp_path / "remote"
    marker = checkpoint(path)
    monkeypatch.setenv(BYPASS_ENV, "1")  # The existing security flag is the API under test.
    with pytest.raises(ValueError):
        load_model_config(path, trust_remote_code=False)
    assert not marker.exists()
    assert load_model_config(path, trust_remote_code=True).model_type == "unported_fixture"
    assert marker.read_text() == "config"
    marker.unlink()
    monkeypatch.delenv(BYPASS_ENV)
    with pytest.raises(RemoteCodeSecurityError):
        load_model_config(path, trust_remote_code=True)
    assert not marker.exists()


@pytest.mark.parametrize("bypass", [False, True])
def test_native_model_ignores_checkpoint_model_code(tmp_path, monkeypatch, bypass):
    path = tmp_path / "native"
    marker = checkpoint(path, "llama")
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    config = load_model_config(path, trust_remote_code=True)
    assert type(config).__module__.startswith("transformers.models.llama")
    prepare_remote_model_init_compat(path, config, trust_remote_code=True)
    assert not marker.exists()


def test_local_config_and_loader_win_over_auto_map_even_with_bypass(tmp_path, monkeypatch):
    path = tmp_path / "local"
    marker = checkpoint(path, "minimax")
    loader = LocalModelLoader(
        "gptqmodel.hf_minimax_m2.configuration_minimax_m2.MiniMaxM2Config",
        "gptqmodel.hf_minimax_m2.modeling_minimax_m2.MiniMaxM2ForCausalLM",
    )
    definition = SimpleNamespace(loader=loader)
    monkeypatch.setitem(MODEL_MAP_CASEFOLD, "minimax", definition)
    monkeypatch.setenv(BYPASS_ENV, "1")
    assert check_and_get_model_definition(str(path), trust_remote_code=True) is definition
    assert resolve_trust_remote_code(path, trust_remote_code=True) is False
    config = load_model_config(path, trust_remote_code=True, use_mtp=False, num_mtp_modules=0,
                               num_local_experts=2, num_experts_per_tok=1, head_dim=8, rotary_dim=8)
    assert type(config).__module__.startswith("gptqmodel.")
    prepare_remote_model_init_compat(path, config, trust_remote_code=True)
    normalize_hf_config_compat(config)
    with torch.device("meta"):
        model = loader.from_config(config, trust_remote_code=True)
    assert type(model).__module__.startswith("gptqmodel.")
    assert next(model.parameters()).device.type == "meta"
    assert not marker.exists()
    del model


def test_config_selection_uses_subfolder_for_policy_and_dispatch(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    marker = checkpoint(root / "nested", "llama")
    assert load_model_config(root, subfolder="nested", trust_remote_code=True).model_type == "llama"
    assert check_and_get_model_definition(str(root), trust_remote_code=True, subfolder="nested").__name__ == "LlamaQModel"
    assert not marker.exists()


def _gguf_config(path):
    """Tiny metadata-only GGUF, parsed by Transformers' actual reader."""
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    fields = [("general.architecture", 8, string("llama"))]
    for key, value in {"context_length": 32, "embedding_length": 16,
                       "feed_forward_length": 32, "block_count": 1,
                       "attention.head_count": 2, "attention.head_count_kv": 1}.items():
        fields.append(("llama." + key, 4, struct.pack("<I", value)))
    payload = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for key, tag, value in fields:
        payload += string(key) + struct.pack("<I", tag) + value
    path.write_bytes(payload + b"\0" * (-len(payload) % 32))


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("decoy_config", [False, True])
@pytest.mark.parametrize("subfolder", ["", "nested"])
def test_gguf_policy_reads_selected_metadata_instead_of_json(tmp_path, monkeypatch, trust, decoy_config, subfolder):
    from gptqmodel.utils.hf import _patch_transformers_prism_gguf_compat

    # This is the runtime installed by the normal public GGUF path; do not
    # replace the decoder or stub the metadata returned by Transformers.
    _patch_transformers_prism_gguf_compat(api_name="test")
    root = tmp_path / "checkpoint"
    path = root / subfolder
    if decoy_config:
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = checkpoint(path, "bailing_hybrid")
        decoy = json.loads((path / "config.json").read_text())
        decoy["q_lora_rank"] = 8
        (path / "config.json").write_text(json.dumps(decoy))
    else:
        path.mkdir(parents=True)
        marker = path / "EXECUTED"
    _gguf_config(path / "tiny.gguf")
    selected = {"gguf_file": "tiny.gguf", "subfolder": subfolder, "local_files_only": True}
    assert check_and_get_model_definition(str(root), trust_remote_code=trust, **selected).__name__ == "LlamaQModel"
    config = load_model_config(root, trust_remote_code=trust, **selected)
    assert config.model_type == "llama"
    assert config.hidden_size == 16
    assert config.num_hidden_layers == 1
    # Permission changes must not make an unrelated JSON config authoritative.
    monkeypatch.setenv(BYPASS_ENV, "1")
    assert load_model_config(root, trust_remote_code=trust, **selected).hidden_size == 16
    assert check_and_get_model_definition(str(root), trust_remote_code=trust, **selected).__name__ == "LlamaQModel"
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["load", "from_pretrained", "from_quantized"])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_dispatch_uses_local_definition_and_disables_checkpoint_model_code(tmp_path, monkeypatch, entry, bypass):
    path = tmp_path / "local"
    marker = checkpoint(path, "minimax")
    loader = LocalModelLoader(
        "gptqmodel.hf_minimax_m2.configuration_minimax_m2.MiniMaxM2Config",
        "gptqmodel.hf_minimax_m2.modeling_minimax_m2.MiniMaxM2ForCausalLM",
    )
    captured = []
    sentinel = SimpleNamespace(log_kv_cache_estimate=lambda: None)

    def consume(**kwargs):
        captured.append(kwargs)
        config = load_model_config(path, trust_remote_code=kwargs["trust_remote_code"])
        assert type(config).__module__.startswith("gptqmodel.")
        return sentinel

    monkeypatch.setitem(MODEL_MAP_CASEFOLD, "minimax", SimpleNamespace(
        loader=loader, from_pretrained=consume, from_quantized=consume,
    ))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    kwargs = {"quantize_config": None} if entry == "from_pretrained" else {}
    assert getattr(GPTQModel, entry)(str(path), trust_remote_code=True, **kwargs) is sentinel
    assert captured and captured[0]["trust_remote_code"] is False
    assert not marker.exists()


def test_native_multimodal_support_uses_definition_auto_class(tmp_path):
    from transformers import AutoModelForImageTextToText

    path = tmp_path / "vl"
    marker = checkpoint(path, "qwen2_vl")
    assert has_native_transformers_model_support(path, AutoModelForImageTextToText, proxies={})
    assert not marker.exists()


def test_remote_model_import_requires_both_permissions(tmp_path, monkeypatch):
    path = tmp_path / "remote"
    marker = checkpoint(path)
    config = SimpleNamespace(model_type="unported_fixture", auto_map={"AutoModelForCausalLM": "remote_model.RemoteModel"})
    with pytest.raises(RemoteCodeSecurityError):
        prepare_remote_model_init_compat(path, config, trust_remote_code=True)
    assert not marker.exists()
    monkeypatch.setenv(BYPASS_ENV, "1")
    prepare_remote_model_init_compat(path, config, trust_remote_code=True)
    assert marker.read_text() == "model"


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_external_engine_helpers_require_security_bypass(monkeypatch, backend):
    from gptqmodel.utils.sglang import load_model_by_sglang
    from gptqmodel.utils.vllm import load_model_by_vllm

    load = load_model_by_sglang if backend == "sglang" else load_model_by_vllm
    with pytest.raises(RemoteCodeSecurityError, match=BYPASS_ENV):
        load("unported-model", trust_remote_code=True)


def test_unported_layout_fails_instead_of_loading_incompatible_native_model(tmp_path):
    path = tmp_path / "ling"
    marker = checkpoint(path, "bailing_hybrid")
    payload = json.loads((path / "config.json").read_text())
    payload["q_lora_rank"] = 16
    (path / "config.json").write_text(json.dumps(payload))
    for trust in (False, True):
        with pytest.raises(RemoteCodeSecurityError):
            load_model_config(path, trust_remote_code=trust)
    assert not marker.exists()


def test_local_loader_rejects_checkpoint_import_paths():
    with pytest.raises(ValueError, match="must belong to gptqmodel"):
        LocalModelLoader("transformers_modules.repo.Config", "gptqmodel.models.Model")


def test_native_config_is_not_changed_by_local_definition_selection(tmp_path, monkeypatch):
    path = tmp_path / "native"
    checkpoint(path, "minimax")
    original = type(AutoConfig.from_pretrained(path, trust_remote_code=False))
    monkeypatch.setitem(MODEL_MAP_CASEFOLD, "minimax", SimpleNamespace(loader=LocalModelLoader(
        "gptqmodel.hf_minimax_m2.configuration_minimax_m2.MiniMaxM2Config",
        "gptqmodel.hf_minimax_m2.modeling_minimax_m2.MiniMaxM2ForCausalLM",
    )))
    assert type(load_model_config(path)) is not original
    assert type(AutoConfig.from_pretrained(path, trust_remote_code=False)) is original
