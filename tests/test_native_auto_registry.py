# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The package-local registry resolves only vendored or framework classes."""

import json

import pytest

from gptqmodel.models.ports.native import auto


def _checkpoint(path, model_type, tokenizer_class=None, processor_class=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    if tokenizer_class is not None:
        (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": tokenizer_class}))
    if processor_class is not None:
        (path / "processor_config.json").write_text(json.dumps({"processor_class": processor_class}))
    return path


def _is_vendored(cls):
    return cls.__module__.startswith("gptqmodel.models.ports.native.")


@pytest.mark.parametrize("declared", ["Qwen2Tokenizer", "Qwen2TokenizerFast"])
def test_declared_tokenizer_class_resolves_to_vendored_class(tmp_path, declared):
    cls = auto.NativeTokenizer.tokenizer_class(str(_checkpoint(tmp_path, "qwen3", tokenizer_class=declared)))
    assert cls.__name__ == "Qwen2Tokenizer" and _is_vendored(cls)


def test_declared_framework_tokenizer_resolves_to_framework_class(tmp_path):
    cls = auto.NativeTokenizer.tokenizer_class(str(_checkpoint(tmp_path, "qwen3", tokenizer_class="PreTrainedTokenizerFast")))
    assert cls.__module__.startswith("transformers.") and not cls.__module__.startswith("transformers.models.")


def test_unknown_or_model_family_tokenizer_name_never_resolves_to_transformers_models(tmp_path):
    # An unknown name falls back to the model_type default; a non-vendored model-family
    # name is never imported from transformers.models.
    for declared in ("EvilTokenizer", "BertTokenizer"):
        cls = auto.NativeTokenizer.tokenizer_class(str(_checkpoint(tmp_path / declared, "qwen3", tokenizer_class=declared)))
        assert not cls.__module__.startswith("transformers.models.")


def test_missing_tokenizer_config_uses_model_type_default(tmp_path):
    cls = auto.NativeTokenizer.tokenizer_class(str(_checkpoint(tmp_path, "qwen3")))
    assert not cls.__module__.startswith("transformers.models.")


def test_declared_processor_class_is_resolved_from_vendored_classes():
    cls = auto._by_name("Glm46VProcessor")
    assert cls is not None and _is_vendored(cls)
    assert auto._by_name("BertModel") is None


def test_loaders_reject_native_transformers_configs():
    from transformers import Qwen2Config

    with pytest.raises(TypeError, match="not a vendored GPT-QModel config"):
        auto.NativeForCausalLM._from_config(Qwen2Config(num_hidden_layers=1))
