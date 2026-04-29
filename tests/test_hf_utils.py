# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
import tempfile

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import _get_tied_weight_keys

from gptqmodel.utils import hf as _hf_utils  # noqa: F401


class _DummyConfig(PretrainedConfig):
    model_type = "dummy_hf_compat"

    def __init__(self):
        super().__init__(tie_word_embeddings=True)
        self.vocab_size = 8
        self.hidden_size = 4


class _LegacyTiedWeightsModel(PreTrainedModel):
    config_class = _DummyConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_legacy_list_tied_weights_are_normalized_to_input_embeddings():
    model = _LegacyTiedWeightsModel(_DummyConfig())

    assert model.get_expanded_tied_weights_keys(all_submodels=False) == {
        "lm_head.weight": "embed_tokens.weight"
    }
    assert model._tied_weights_keys == {"lm_head.weight": "embed_tokens.weight"}
    assert _get_tied_weight_keys(model) == ["lm_head.weight"]


def test_legacy_list_tied_weights_allow_save_pretrained():
    model = _LegacyTiedWeightsModel(_DummyConfig())

    with tempfile.TemporaryDirectory() as tmp_dir:
        model._tied_weights_keys = ["lm_head.weight"]
        model.get_expanded_tied_weights_keys(all_submodels=False)
        model._tied_weights_keys = ["lm_head.weight"]
        _hf_utils._normalize_legacy_tied_weights_keys(model)
        model.save_pretrained(tmp_dir, state_dict={}, is_main_process=True)


def test_normalize_legacy_tied_weights_keys_ignores_non_module_stubs():
    class _NoModulesDummy:
        pass

    _hf_utils._normalize_legacy_tied_weights_keys(_NoModulesDummy())
