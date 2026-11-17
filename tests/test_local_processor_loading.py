# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-owned processors obey the local model selection contract."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoProcessor, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.internlm import InternLMQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig
from gptqmodel.models.ports.internlm.modeling_internlm import InternLMForCausalLM
from gptqmodel.utils.security import BYPASS_ENV


@pytest.mark.parametrize("field", ["config_class", "model_class", "processor_class"])
def test_local_processor_metadata_rejects_external_classes(field):
    paths = {
        "config_class": InternLMQModel.loader.config_class,
        "model_class": InternLMQModel.loader.model_class,
        "processor_class": "gptqmodel.processor.LocalProcessor",
    }
    paths[field] = "checkpoint_python.RemoteClass"
    with pytest.raises(ValueError, match="must belong to gptqmodel"):
        LocalModelLoader(**paths)


def test_local_processor_requires_explicit_definition_metadata():
    with pytest.raises(ValueError, match="does not declare a processor"):
        InternLMQModel.loader.load_processor("checkpoint", trust_remote_code=True)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("subfolder", ["", "nested"])
def test_public_load_selects_definition_owned_processor(tmp_path, monkeypatch, trust, bypass, subfolder):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    marker = tmp_path / "CHECKPOINT_PROCESSOR_EXECUTED"
    config = InternLMConfig(
        vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2,
    )
    checkpoint = tmp_path / subfolder
    checkpoint.mkdir(exist_ok=True)
    InternLMForCausalLM(config).eval().save_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3})),
        pad_token="<pad>", bos_token="<s>", eos_token="</s>",
    )
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "processor_config.json").write_text(json.dumps({
        "processor_class": "UntrustedProcessor",
        "auto_map": {"AutoProcessor": "checkpoint_processor.UntrustedProcessor"},
    }))
    (checkpoint / "checkpoint_processor.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('checkpoint processor executed')\n"
    )
    module_name = "gptqmodel._processor_contract_fixture"
    module = ModuleType(module_name)
    calls = []
    asset = checkpoint / "processor_asset.json"
    asset.write_text(json.dumps({"pixel_scale": 0.375, "source": "selected-checkpoint"}))

    class LocalProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            result = cls()
            result.tokenizer = kwargs["tokenizer"]
            result.asset = json.loads((Path(path) / kwargs.get("subfolder", "") / "processor_asset.json").read_text())
            return result

    module.LocalProcessor = LocalProcessor
    # This contract fixture owns a generic tokenizer; release-specific InternLM
    # SentencePiece behavior is covered by test_internlm_tokenizers_local_port.
    module.LocalTokenizer = PreTrainedTokenizerFast
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(InternLMQModel, "loader", replace(
        InternLMQModel.loader, processor_class=f"{module_name}.LocalProcessor",
        tokenizer_class=f"{module_name}.LocalTokenizer",
    ))
    monkeypatch.setattr(InternLMQModel, "require_load_processor", True)
    # A processor may never fall back to the checkpoint-controlled auto class.
    def reject_auto_processor(*args, **kwargs):
        raise AssertionError("definition-owned processor re-entered AutoProcessor")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", reject_auto_processor)
    options = {
        "subfolder": subfolder, "revision": "processor-source",
        "local_files_only": True, "cache_dir": str(tmp_path / "cache"),
    }
    loaded = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, **options)
    assert isinstance(loaded.processor, LocalProcessor)
    assert len(calls) == 1
    assert calls[0][1]["config"] is loaded.model.config
    assert calls[0][1]["tokenizer"] is loaded.tokenizer.tokenizer
    assert all(calls[0][1][key] == value for key, value in options.items())
    assert loaded.processor.asset == {"pixel_scale": 0.375, "source": "selected-checkpoint"}
    assert not {"trust_remote_code", "code_revision"}.intersection(calls[0][1])
    assert not marker.exists()
    inputs = torch.tensor([[1, 3]])
    expected = loaded.model(inputs).logits
    assert expected.shape == (1, 2, 8)
    asset.unlink()
    with pytest.raises(FileNotFoundError, match="processor_asset.json"):
        GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, **options)
    asset.write_text(json.dumps(loaded.processor.asset))
    retried = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, **options)
    assert retried.processor.asset == loaded.processor.asset
    torch.testing.assert_close(retried.model(inputs).logits, expected, rtol=0, atol=0)
    assert not marker.exists()


def test_local_processor_ignores_checkpoint_code_options(monkeypatch):
    calls = []
    module_name = "gptqmodel._processor_options_fixture"
    module = ModuleType(module_name)

    class LocalProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

    module.LocalProcessor = LocalProcessor
    monkeypatch.setitem(sys.modules, module_name, module)
    loader = replace(InternLMQModel.loader, processor_class=f"{module_name}.LocalProcessor")
    loader.load_processor("checkpoint", trust_remote_code=True, code_revision="untrusted", local_files_only=True)
    assert calls == [{"local_files_only": True}]
