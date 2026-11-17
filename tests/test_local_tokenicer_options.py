# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package profiles reach Tokenicer without changing tokenizer constructors."""

from dataclasses import replace

import pytest
from tokenicer import Tokenicer

from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.utils.hf import load_hf_tokenizer
from tests.test_local_tokenizer_loading import (
    _ban_auto,
    _config,
    _hostile_metadata,
    _write_tokenizer,
)
from tests.test_local_tokenizer_loading import owned as owned_fixture


# Re-export the pytest fixture without shadowing an imported name in tests.
owned = owned_fixture


@pytest.mark.parametrize("caller", [{}, {"normalize_sentencepiece": False},
                                   {"normalize_sentencepiece": True}, {"normalize_sentencepiece": None}])
@pytest.mark.parametrize("provider", ["owned", "native", "instance"])
def test_public_profile_preserves_caller_and_provider(tmp_path, monkeypatch, owned, caller, provider):
    loader, tokenizer_class, constructors = owned
    config = _config()
    config.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    if provider == "owned":
        marker = _hostile_metadata(tmp_path)
        _ban_auto(monkeypatch)
    else:
        marker = tmp_path / "CHECKPOINT_PYTHON_EXECUTED"
    selected = replace(loader, tokenicer_options=(("normalize_sentencepiece", True),),
                       **({"tokenizer_class": None} if provider == "native" else {}))
    monkeypatch.setattr(InternLMQModel, "loader", selected)
    source = tokenizer_class.from_pretrained(tmp_path) if provider == "instance" else tmp_path
    constructors.clear()
    calls = []
    original = Tokenicer.load

    def record(value, **kwargs):
        calls.append((value, dict(kwargs)))
        return original(value, **kwargs)

    monkeypatch.setattr(Tokenicer, "load", record)
    result = load_hf_tokenizer(source, model_config=config, return_tokenicer=True,
                               local_files_only=True, **caller)
    assert len(calls) == 1
    assert calls[0][1]["normalize_sentencepiece"] is caller.get("normalize_sentencepiece", True)
    assert result("hello science", add_special_tokens=False).input_ids == [3, 4]
    assert result.decode([3, 4]) == "hello science"
    assert not marker.exists()
    if provider == "owned":
        assert type(result.tokenizer) is tokenizer_class
        assert len(constructors) == 1
        assert "normalize_sentencepiece" not in constructors[0][1]
    elif provider == "instance":
        assert result.tokenizer is source
        assert not constructors


@pytest.mark.parametrize("options", [[], (("normalize_sentencepiece", True), ("normalize_sentencepiece", False)),
                                     (("trust_remote_code", True),), (("config", None),),
                                     (("model_config", None),), (("code_revision", "other"),)])
def test_profile_cannot_override_class_selection_or_security(options):
    with pytest.raises(ValueError, match="tokenicer_options"):
        replace(InternLMQModel.loader, tokenicer_options=options)
