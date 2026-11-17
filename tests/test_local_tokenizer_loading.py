# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned tokenizers are selected before Tokenicer normalization."""

import inspect
import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tokenicer import Tokenicer
from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    LlamaTokenizer,
    PreTrainedConfig,
    PreTrainedTokenizerFast,
)

from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP_CASEFOLD
from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
from gptqmodel.utils.hf import load_hf_tokenizer
from gptqmodel.utils.security import BYPASS_ENV


VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3, "science": 4, "<unk>": 5, "<other>": 6}


def _write_tokenizer(path):
    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)


def _config():
    return InternLMConfig(
        vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, pad_token_id=None,
    )


def _hostile_metadata(path):
    marker = path / "CHECKPOINT_PYTHON_EXECUTED"
    (path / "checkpoint_python.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise AssertionError('checkpoint tokenizer imported')\n"
    )
    mapping = {
        "AutoConfig": "checkpoint_python.RemoteConfig",
        "AutoTokenizer": ["checkpoint_python.InternS1Tokenizer", None],
    }
    config_file = path / "config.json"
    config = json.loads(config_file.read_text())
    config["auto_map"] = mapping
    config["tokenizer_class"] = "InternS1Tokenizer"
    config["nested_config"] = {"auto_map": mapping, "tokenizer_class": "InternS1Tokenizer"}
    config_file.write_text(json.dumps(config))
    tokenizer_file = path / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_file.read_text())
    tokenizer_config.update({"tokenizer_class": "InternS1Tokenizer", "auto_map": mapping})
    tokenizer_file.write_text(json.dumps(tokenizer_config))
    return marker


@pytest.fixture
def owned(monkeypatch):
    """Use a real HF tokenizer backend, recording only its construction boundary."""
    module_name = "gptqmodel._local_tokenizer_contract_fixture"
    module = ModuleType(module_name)
    calls = []

    class OwnedTokenizer(PreTrainedTokenizerFast):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, dict(kwargs)))
            return super().from_pretrained(path, **kwargs)

    module.OwnedTokenizer = OwnedTokenizer
    monkeypatch.setitem(sys.modules, module_name, module)
    loader = replace(InternLMQModel.loader, tokenizer_class=f"{module_name}.OwnedTokenizer")
    monkeypatch.setattr(InternLMQModel, "loader", loader)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    return loader, OwnedTokenizer, calls


def _reject_auto(*args, **kwargs):
    raise AssertionError("definition-owned tokenizer re-entered checkpoint Auto discovery")


def _ban_auto(monkeypatch):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _reject_auto)
    monkeypatch.setattr(AutoConfig, "from_pretrained", _reject_auto)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("path_kind", [str, Path])
def test_public_tokenizer_uses_owned_class_and_normalizes(tmp_path, monkeypatch, owned, trust, bypass, path_kind):
    _, tokenizer_class, calls = owned
    config = _config()
    config.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    _ban_auto(monkeypatch)
    selected = InternLMConfig.from_pretrained(tmp_path)
    # The selected outer definition, not nested tokenizer hints, owns dispatch.
    selected.text_config = PreTrainedConfig(
        bos_token_id=1, eos_token_id=2, pad_token_id=None,
        auto_map={"AutoConfig": "checkpoint_python.RemoteConfig"},
    )
    loaded = load_hf_tokenizer(
        path_kind(tmp_path), model_config=selected,
        trust_remote_code=trust, code_revision="untrusted",
        config=PreTrainedConfig(model_type="untrusted"),
        strict=True, pad_tokens=["<other>"], prefer_serialized_tokenizer=False,
        local_files_only=True, padding_side="left", model_max_length=73,
    )
    assert type(loaded) is tokenizer_class
    assert loaded("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert loaded.decode([3, 4]) == "hello science"
    assert loaded.pad_token_id == selected.text_config.pad_token_id == 6
    assert selected.pad_token_id is None
    assert loaded.padding_side == "left" and loaded.model_max_length == 73
    assert len(calls) == 1
    assert calls[0][1] == {"local_files_only": True, "padding_side": "left", "model_max_length": 73}
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_local_model_without_owned_tokenizer_cannot_import_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    # Exercise the loader's native-tokenizer branch independently of which
    # release tokenizer InternLM itself declares.
    monkeypatch.setattr(InternLMQModel, "loader", replace(InternLMQModel.loader, tokenizer_class=None))
    reference = InternLMForCausalLM(_config()).eval()
    reference.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    # Preserve the actual native tokenizer class while adding a hostile map.
    path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(path.read_text())
    metadata["tokenizer_class"] = "TokenizersBackend"
    path.write_text(json.dumps(metadata))
    assert InternLMQModel.loader.tokenizer_class is None
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert loaded.tokenizer("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert loaded.tokenizer.decode([3, 4]) == "hello science"
    assert not marker.exists()


def test_owned_constructor_receives_preloaded_legacy_tokens(tmp_path, monkeypatch, owned):
    _, tokenizer_class, calls = owned
    config = _config()
    config.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    metadata = json.loads(path.read_text())
    metadata["additional_special_tokens"] = [
        {"content": "<legacy>", "single_word": True, "lstrip": True, "rstrip": False, "normalized": False},
    ]
    path.write_text(json.dumps(metadata))
    original_assets = {asset.name: asset.read_bytes() for asset in tmp_path.iterdir() if asset.is_file()}
    _ban_auto(monkeypatch)

    loaded = load_hf_tokenizer(tmp_path, model_config=config, local_files_only=True)
    assert type(loaded) is tokenizer_class
    token = calls[0][1]["additional_special_tokens"][0]
    assert token.content == "<legacy>"
    assert token.special and token.single_word and token.lstrip
    assert not token.rstrip and not token.normalized
    token_id = loaded.convert_tokens_to_ids("<legacy>")
    assert token_id >= len(VOCAB)
    assert loaded.decode([token_id], skip_special_tokens=False) == "<legacy>"
    assert loaded.decode([token_id], skip_special_tokens=True) == ""
    assert not marker.exists()
    assert original_assets == {asset.name: asset.read_bytes() for asset in tmp_path.iterdir() if asset.is_file()}


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_model_load_uses_owned_tokenizer(tmp_path, monkeypatch, owned, trust, bypass):
    _, tokenizer_class, _ = owned
    reference = InternLMForCausalLM(_config()).eval()
    reference.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    _ban_auto(monkeypatch)
    normalized = []
    original = Tokenicer.load.__func__

    def record_normalization(cls, value, *args, **kwargs):
        result = original(cls, value, *args, **kwargs)
        normalized.append(result)
        return result

    monkeypatch.setattr(Tokenicer, "load", classmethod(record_normalization))
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert len(normalized) == 1
    assert loaded.tokenizer is normalized[0]
    assert loaded.model.tokenizer is normalized[0].tokenizer
    assert type(loaded.tokenizer.tokenizer) is tokenizer_class
    ids = loaded.tokenizer("hello science", add_special_tokens=False, return_tensors="pt").input_ids
    assert ids.tolist() == [[3, 4]]
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


def test_caller_tokenizer_keeps_identity_and_normalization_overrides(tmp_path, monkeypatch, owned):
    _, tokenizer_class, calls = owned
    _write_tokenizer(tmp_path)
    supplied = PreTrainedTokenizerFast.from_pretrained(tmp_path, padding_side="left", model_max_length=79)
    config = _config()
    _ban_auto(monkeypatch)
    loaded = load_hf_tokenizer(
        supplied, model_config=config, trust_remote_code=True,
        pad_tokens=["<other>"], strict=True, prefer_serialized_tokenizer=True,
    )
    assert loaded is supplied
    assert not isinstance(loaded, tokenizer_class)
    assert calls == []
    assert loaded.pad_token_id == config.pad_token_id == 6
    assert loaded.padding_side == "left" and loaded.model_max_length == 79
    assert loaded("hello science", add_special_tokens=False).input_ids == [3, 4]


def test_quantized_reload_retains_once_normalized_wrapper(tmp_path, monkeypatch, owned):
    from gptqmodel import BACKEND
    from gptqmodel.quantization.config import AWQConfig

    config = _config()
    config.intermediate_size = 32
    reference = InternLMForCausalLM(config).eval()
    reference.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    monkeypatch.setenv(BYPASS_ENV, "1")
    normalized = []
    original = Tokenicer.load.__func__

    def record_normalization(cls, value, *args, **kwargs):
        result = original(cls, value, *args, **kwargs)
        normalized.append(result)
        return result

    monkeypatch.setattr(Tokenicer, "load", classmethod(record_normalization))
    loaded = GPTQModel.load(
        str(tmp_path), dtype=torch.float16, trust_remote_code=True,
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=False),
        backend=BACKEND.TORCH,
    )
    assert len(normalized) == 1 and loaded.tokenizer is normalized[0]
    sample = {"input_ids": torch.tensor([[1, 3, 4, 3, 4, 3, 4, 2]]),
              "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    loaded.quantize([sample, sample], batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    with torch.inference_mode():
        expected = loaded.model(**sample, use_cache=False).logits
    for cycle in range(2):
        saved = tmp_path / f"quantized-{cycle}"
        loaded.save_quantized(str(saved))
        normalized.clear()
        loaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float16, backend=BACKEND.TORCH)
        assert len(normalized) == 1
        assert loaded.tokenizer is normalized[0]
        assert loaded.model.tokenizer is normalized[0].tokenizer
        assert loaded.tokenizer("hello science", add_special_tokens=False).input_ids == [3, 4]
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(**sample, use_cache=False).logits, expected, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("broken_asset", ["missing", "malformed", "config"])
def test_broken_owned_tokenizer_fails_then_repairs_without_auto_fallback(tmp_path, monkeypatch, owned, broken_asset):
    _, tokenizer_class, calls = owned
    _config().save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    monkeypatch.setenv(BYPASS_ENV, "1")
    _ban_auto(monkeypatch)
    selected = InternLMConfig.from_pretrained(tmp_path)
    asset = tmp_path / ("tokenizer_config.json" if broken_asset == "config" else "tokenizer.json")
    good = asset.read_bytes()
    if broken_asset == "missing":
        asset.unlink()
    else:
        asset.write_text("{invalid tokenizer payload")
    # The tokenizer backend raises a plain Exception for malformed Rust data.
    with pytest.raises(Exception) as error:
        load_hf_tokenizer(str(tmp_path), model_config=selected, trust_remote_code=True)
    assert not isinstance(error.value, AssertionError), "must fail in owned asset loading, never Auto discovery"
    failed_constructions = 0 if broken_asset == "config" else 1
    assert len(calls) == failed_constructions
    assert selected.pad_token_id is None
    assert not marker.exists()
    asset.write_bytes(good)
    repaired = load_hf_tokenizer(str(tmp_path), model_config=selected, trust_remote_code=True, pad_tokens=["<pad>"])
    assert type(repaired) is tokenizer_class
    assert repaired("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert selected.pad_token_id == repaired.pad_token_id == 0
    assert len(calls) == failed_constructions + 1
    assert not marker.exists()


def test_local_tokenizer_metadata_is_frozen_and_package_owned():
    with pytest.raises(ValueError, match="must belong to gptqmodel"):
        replace(InternLMQModel.loader, tokenizer_class="checkpoint_python.InternS1Tokenizer")
    loader = replace(InternLMQModel.loader, tokenizer_class="gptqmodel.fixture.Tokenizer")
    with pytest.raises(FrozenInstanceError):
        loader.tokenizer_class = "checkpoint_python.Tokenizer"
    with pytest.raises(ValueError, match="does not declare a tokenizer"):
        LocalModelLoader(loader.config_class, loader.model_class).load_tokenizer("checkpoint")


def test_local_tokenizer_direct_dispatch_strips_code_options(tmp_path, owned):
    loader, tokenizer_class, calls = owned
    _write_tokenizer(tmp_path)
    result = loader.load_tokenizer(
        str(tmp_path), trust_remote_code=True, code_revision="checkpoint", local_files_only=True,
    )
    assert type(result) is tokenizer_class
    assert calls[0][1] == {"local_files_only": True}


def test_same_checkpoint_class_name_uses_each_definitions_full_path(tmp_path, monkeypatch, owned):
    first_loader, first_class, _ = owned
    _config().save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = _hostile_metadata(tmp_path)
    module_name = "gptqmodel._second_scientific_tokenizer_fixture"
    module = ModuleType(module_name)

    class OwnedTokenizer(PreTrainedTokenizerFast):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            result = super().from_pretrained(path, **kwargs)
            # Same declared class name, different scientific vocabulary ABI.
            backend = Tokenizer(WordLevel({**VOCAB, "hello": 4, "science": 3}, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            result._tokenizer = backend
            return result

    module.OwnedTokenizer = OwnedTokenizer
    monkeypatch.setitem(sys.modules, module_name, module)
    second_loader = replace(first_loader, tokenizer_class=f"{module_name}.OwnedTokenizer")
    monkeypatch.setitem(MODEL_MAP_CASEFOLD, "second_scientific_fixture", SimpleNamespace(loader=second_loader))
    _ban_auto(monkeypatch)
    first_config, second_config = _config(), _config()
    second_config.model_type = "second_scientific_fixture"
    first = load_hf_tokenizer(str(tmp_path), model_config=first_config, pad_tokens=["<pad>"])
    second = load_hf_tokenizer(str(tmp_path), model_config=second_config, pad_tokens=["<pad>"])
    assert first_class.__name__ == OwnedTokenizer.__name__
    assert type(first) is first_class and type(second) is OwnedTokenizer
    assert first("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert second("hello science", add_special_tokens=False).input_ids == [4, 3]
    assert not marker.exists()


def test_owned_tokenizer_retains_checkpoint_selection_kwargs(tmp_path, monkeypatch, owned):
    _, tokenizer_class, calls = owned
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_tokenizer(nested)
    # A decoy root must not replace the explicitly selected subfolder.
    (tmp_path / "tokenizer.json").write_text("malformed root decoy")
    _ban_auto(monkeypatch)
    selection = {
        "subfolder": "nested", "revision": "pinned-assets", "local_files_only": True,
        "cache_dir": str(tmp_path / "cache"), "force_download": False,
    }
    loaded = load_hf_tokenizer(str(tmp_path), model_config=_config(), pad_tokens=["<pad>"], **selection)
    assert type(loaded) is tokenizer_class
    assert loaded("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert calls[0] == (str(tmp_path), selection)


@pytest.mark.parametrize("local_config", [False, True])
def test_ordinary_models_keep_auto_tokenizer_loading(tmp_path, monkeypatch, local_config):
    if local_config:
        monkeypatch.setattr(InternLMQModel, "loader", replace(InternLMQModel.loader, tokenizer_class=None))
    _write_tokenizer(tmp_path)
    config = _config() if local_config else LlamaConfig(bos_token_id=1, eos_token_id=2, pad_token_id=None)
    monkeypatch.setattr(AutoConfig, "from_pretrained", _reject_auto)
    # A definition without tokenizer metadata keeps the normal Tokenicer path.
    loaded = load_hf_tokenizer(str(tmp_path), model_config=config, pad_tokens=["<pad>"])
    assert type(loaded) is PreTrainedTokenizerFast
    assert loaded("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert loaded.pad_token_id == config.pad_token_id == 0


@pytest.mark.skipif(
    "prefer_serialized_tokenizer" not in inspect.signature(Tokenicer.load).parameters,
    reason="Serialized-tokenizer preference requires Tokenicer >=0.0.16 (installed minimum 0.0.14 lacks it)",
)
@pytest.mark.parametrize("prefer_serialized", [None, False], ids=["default", "optout"])
def test_ordinary_models_preserve_tokenicers_serialized_preference(tmp_path, monkeypatch, prefer_serialized):
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Hello world, this is a scientific test.", "  leading and repeated whitespace ", "Unicode café 中文"],
        trainers.BpeTrainer(
            vocab_size=350, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        ),
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "add_bos_token": True, "add_eos_token": False,
    }))
    config = LlamaConfig(bos_token_id=1, eos_token_id=2, pad_token_id=None)
    serialized = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    legacy = LlamaTokenizer.from_pretrained(tmp_path)
    probes = ["Hello world, this is a scientific test.", "  leading and repeated whitespace "]

    def ids(tokenizer):
        return [tokenizer(text, add_special_tokens=False).input_ids for text in probes]

    assert ids(serialized) != ids(legacy), "fixture must exercise class/serialized tokenization disagreement"
    # Some Transformers versions recognize this simple ByteLevel fixture in
    # AutoTokenizer already. Inject only its legacy class choice, keeping both
    # real tokenizers and Tokenicer's preference/normalization consumer intact.
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: legacy)
    options = {} if prefer_serialized is None else {"prefer_serialized_tokenizer": prefer_serialized}
    loaded = load_hf_tokenizer(str(tmp_path), model_config=config, pad_tokens=["<pad>"], **options)
    assert ids(loaded) == ids(serialized if prefer_serialized is None else legacy)
