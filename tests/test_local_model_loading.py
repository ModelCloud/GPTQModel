# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The loading policy applies before config Python, including retries/reloads."""

import json
import struct
from dataclasses import dataclass
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


@pytest.mark.parametrize("model_type", ["llama", "internlm"])
def test_disabled_trust_defers_config_errors_and_allows_repair(tmp_path, model_type):
    path = tmp_path / "repair"
    path.mkdir()
    config_file = path / "config.json"
    config_file.write_text("{invalid JSON")
    # Permission resolution needs no checkpoint data when Python is disabled.
    assert resolve_trust_remote_code(path, trust_remote_code=False) is False
    with pytest.raises(OSError):
        load_model_config(path, trust_remote_code=False)
    config_file.write_text(json.dumps({
        "model_type": model_type, "hidden_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 1, "intermediate_size": 32,
    }))
    config = load_model_config(path, trust_remote_code=False)
    assert config.model_type == model_type
    assert type(config).__module__.startswith(("transformers.", "gptqmodel."))


@pytest.mark.parametrize("subfolder", ["", "selected"])
@pytest.mark.parametrize("trust", [False, True])
def test_owned_config_precedes_optional_qwen_drive_registration(tmp_path, monkeypatch, subfolder, trust):
    import builtins

    from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
    from gptqmodel.utils import hf

    # Exercise the generic boundary using a real existing owned config class;
    # the Qwen-Drive port separately qualifies its actual public model loads.
    owned = MODEL_MAP_CASEFOLD["internlm"].loader
    monkeypatch.setattr(QwenDriveQModel, "loader", owned)
    selected = tmp_path / subfolder
    selected.mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"model_type":"qwen_drive"}')
    (selected / "config.json").write_text(json.dumps({
        "model_type": "qwen_drive", "hidden_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 1, "intermediate_size": 32,
    }))
    original_import = builtins.__import__

    def forbidden_optional_package(name, *args, **kwargs):
        if name == "qwen_drive":
            raise AssertionError("Owned loading imported the optional official package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_optional_package)
    kwargs = {"subfolder": subfolder} if subfolder else {}
    config = hf.load_model_config(tmp_path, trust_remote_code=trust, **kwargs)
    assert type(config).__module__ == "gptqmodel.models.ports.internlm.configuration_internlm"
    assert hf.resolve_trust_remote_code(tmp_path, trust_remote_code=trust, **kwargs) is False


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


def test_public_load_preserves_native_max_seq_len(tmp_path):
    from transformers import MptConfig, MptForCausalLM

    from gptqmodel.quantization import QuantizeConfig
    from tests.test_internlm_local_port import _write_tokenizer

    torch.manual_seed(31)
    config = MptConfig(
        vocab_size=32, d_model=16, n_heads=2, n_layers=1, max_seq_len=8192,
        expansion_ratio=2,
    )
    model = MptForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    assert "max_position_embeddings" not in config.to_dict()
    assert config.to_dict()["max_seq_len"] == 8192

    loaded = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32,
        quantize_config=QuantizeConfig(bits=4, group_size=16, offload_to_disk=False),
    )
    assert loaded.model.seqlen == 8192
    ids = torch.tensor([[1, 7, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits)


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


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_local_mla_layout_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel.models.ports.bailing_moe.configuration_bailing import BailingConfig

    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    path = tmp_path / "ling"
    marker = checkpoint(path, "bailing_hybrid")
    payload = json.loads((path / "config.json").read_text())
    payload["q_lora_rank"] = 16
    (path / "config.json").write_text(json.dumps(payload))
    config = load_model_config(path, trust_remote_code=trust)
    assert isinstance(config, BailingConfig)
    assert config.q_lora_rank == 16
    assert resolve_trust_remote_code(path, trust_remote_code=trust) is False
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


@pytest.fixture
def legacy_owned_checkpoint(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models
    from transformers import TokenizersBackend

    from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
    from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM

    # Package declarations exist independently of checkpoint metadata. The
    # subclass lets the pre-fix loader reach the actual missing-type behavior.
    @dataclass(frozen=True)
    class DeclaredLoader(LocalModelLoader):
        config_architectures: tuple[str, ...] = ("InternLMForCausalLM",)

    definition = MODEL_MAP_CASEFOLD["internlm"]
    monkeypatch.setattr(definition, "loader", DeclaredLoader(
        definition.loader.config_class, definition.loader.model_class,
    ))
    model = InternLMForCausalLM(InternLMConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attn_implementation="eager",
    )).eval()
    root = tmp_path / "neutral"
    path = root / "nested"
    model.save_pretrained(path)
    TokenizersBackend(tokenizer_object=Tokenizer(models.WordLevel(
        {f"t{i}": i for i in range(32)}, unk_token="t0",
    )), unk_token="t0").save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload.pop("model_type")
    payload["auto_map"] = {
        "AutoConfig": "remote_config.Config", "AutoModelForCausalLM": "remote_model.Model",
    }
    marker = path / "EXECUTED"
    for name in ("remote_config.py", "remote_model.py"):
        (path / name).write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint Python')\n")
    (path / "config.json").write_text(json.dumps(payload))
    return root, path, payload, marker, model


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_missing_model_type_uses_only_declared_architecture_and_retries(legacy_owned_checkpoint, monkeypatch, trust, bypass):
    root, path, payload, marker, reference = legacy_owned_checkpoint
    if bypass is not None:
        monkeypatch.setenv(BYPASS_ENV, bypass)
    options = {
        "subfolder": "nested", "local_files_only": True, "trust_remote_code": trust,
        "device": "cpu", "dtype": torch.float32, "attn_implementation": "eager",
    }
    selected = load_model_config(root, subfolder="nested", trust_remote_code=trust)
    assert type(selected) is type(reference.config)
    assert resolve_trust_remote_code(root, subfolder="nested", trust_remote_code=trust) is False
    loaded = GPTQModel.load(str(root), **options)
    inputs = torch.tensor([[1, 7, 13, 19]])
    with torch.no_grad():
        expected = reference(inputs).logits
        torch.testing.assert_close(loaded.model(inputs).logits, expected, atol=0, rtol=0)

    # Conflicting metadata must fail even after a successful load, and repairing
    # it must retry normally without a cached architectural guess.
    for conflict in ("UnknownModel", None, []):
        (path / "config.json").write_text(json.dumps({**payload, "architectures": ["InternLMForCausalLM", conflict]}))
        with pytest.raises(ValueError, match="package.*architectures"):
            GPTQModel.load(str(root), **options)
    assert not marker.exists()
    (path / "config.json").write_text(json.dumps(payload))
    repaired = GPTQModel.load(str(root), **options)
    with torch.no_grad():
        torch.testing.assert_close(repaired.model(inputs).logits, expected, atol=0, rtol=0)
    assert not marker.exists()


@pytest.mark.parametrize("model_type", [None, "", "unknown_explicit_type", "llama"])
def test_architecture_inference_never_overrides_explicit_type(legacy_owned_checkpoint, model_type):
    root, path, payload, marker, _ = legacy_owned_checkpoint
    (path / "config.json").write_text(json.dumps({**payload, "model_type": model_type}))
    if model_type == "llama":
        assert load_model_config(root, subfolder="nested").model_type == "llama"
    else:
        with pytest.raises(ValueError):
            load_model_config(root, subfolder="nested")
    assert not marker.exists()


def test_ambiguous_package_architecture_rejects_before_checkpoint_code(legacy_owned_checkpoint, monkeypatch):
    root, _, _, marker, _ = legacy_owned_checkpoint
    loader = MODEL_MAP_CASEFOLD["internlm"].loader
    monkeypatch.setitem(MODEL_MAP_CASEFOLD, "architecture_conflict_fixture", SimpleNamespace(loader=type(loader)(
        "gptqmodel.hf_minimax_m2.configuration_minimax_m2.MiniMaxM2Config",
        "gptqmodel.hf_minimax_m2.modeling_minimax_m2.MiniMaxM2ForCausalLM",
    )))
    monkeypatch.setenv(BYPASS_ENV, "1")
    with pytest.raises(ValueError, match="package.*architectures"):
        load_model_config(root, subfolder="nested", trust_remote_code=True)
    with pytest.raises(ValueError, match="package.*architectures"):
        resolve_trust_remote_code(root, subfolder="nested", trust_remote_code=True)
    assert not marker.exists()
