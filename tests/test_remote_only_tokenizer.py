# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned models whose released tokenizer exists only as checkpoint Python.

Original Yi (``YiTokenizer``) and StableLM Epoch (``Arcade100kTokenizer``) keep the
package-owned model, but their tokenizer can only come from ``auto_map``. Loading it
needs both ``trust_remote_code=True`` and ``GPTQ_BYPASS_SAFETY_CHECKS=1``, and the
load always logs that requirement.
"""

import json
import sys

import pytest

from gptqmodel.models.ports.llama.dispatch import StableLMEpochConfig, YiConfig
from gptqmodel.utils import hf
from gptqmodel.utils.security import RemoteCodeSecurityError


_TOKENIZER_SOURCE = '''
from transformers import PreTrainedTokenizer


class RemoteOnlyTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        self._vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4}
        super().__init__(unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="</s>", **kwargs)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return [char for char in text if char in self._vocab]

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return {v: k for k, v in self._vocab.items()}.get(index, "<unk>")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()
'''


def _checkpoint(tmp_path, config, module="tokenization_remote"):
    config.save_pretrained(tmp_path)
    (tmp_path / f"{module}.py").write_text(_TOKENIZER_SOURCE)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "RemoteOnlyTokenizer",
        "auto_map": {"AutoTokenizer": [f"{module}.RemoteOnlyTokenizer", None]},
    }))
    return tmp_path


def _configs():
    yi = YiConfig(vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=1)
    epoch = StableLMEpochConfig(vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                num_attention_heads=2, num_key_value_heads=1)
    return [pytest.param(yi, id="yi"), pytest.param(epoch, id="stablelm_epoch")]


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(hf, "_REMOTE_ONLY_TOKENIZER_WARNED", set())
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("trust", [False, True])
def test_remote_only_tokenizer_needs_trust_and_bypass(tmp_path, monkeypatch, config, trust):
    warnings = []
    monkeypatch.setattr(hf.log, "warning", lambda message, *args, **kwargs: warnings.append(message))
    path = _checkpoint(tmp_path, config)

    with pytest.raises(RemoteCodeSecurityError, match="GPTQ_BYPASS_SAFETY_CHECKS=1"):
        hf.load_hf_tokenizer(str(path), model_config=config, trust_remote_code=trust)

    assert len(warnings) == 1
    assert config.model_type in warnings[0] and "trust_remote_code=True" in warnings[0]


@pytest.mark.parametrize("config", _configs())
def test_remote_only_tokenizer_loads_with_both_consents(tmp_path, monkeypatch, config):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    path = _checkpoint(tmp_path, config)

    tokenizer = hf.load_hf_tokenizer(str(path), model_config=config, trust_remote_code=True)

    assert type(tokenizer).__name__ == "RemoteOnlyTokenizer"
    assert tokenizer("ab")["input_ids"][-2:] == [3, 4]


def test_other_package_owned_models_never_use_checkpoint_tokenizer_code(tmp_path, monkeypatch):
    from transformers import LlamaConfig

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    config = LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1)
    path = _checkpoint(tmp_path, config, module="tokenization_never_imported")

    with pytest.raises(Exception) as error:
        hf.load_hf_tokenizer(str(path), model_config=config, trust_remote_code=True)
    assert not isinstance(error.value, RemoteCodeSecurityError)
    assert not any(name.endswith("tokenization_never_imported") for name in sys.modules)
