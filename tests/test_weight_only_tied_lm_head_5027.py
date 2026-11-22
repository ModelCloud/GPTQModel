# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Weight-only `lm_head=True` must refuse a tied head up front (#5027).

`WeightOnlyLooper` guarded the tied case with `find_tied_parameters`, which only
compares materialized parameters. With the default `offload_to_disk=True` the
model is a lazy shell whose `lm_head.weight` and `embed_tokens.weight` are
independent META parameters, so the guard never fired: the run quantized every
layer and then died on the head's meta tensor with an unrelated error. The
module looper had already been fixed in #4034; both now share one check.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import CalibrationConfig, RTNConfig
from gptqmodel.utils.tied_weights import lm_head_quantization_is_tied


VOCAB = 64
HIDDEN = 32


class _TinyTiedModel(nn.Module):
    """The shapes `lm_head_quantization_is_tied` has to tell apart.

    Mirrors a decoder: `model.embed_tokens` plus a root-level `lm_head`, which is
    the path `gptq_model.lm_head` names.
    """

    def __init__(self, *, materialized: bool, tie: bool):
        super().__init__()
        device = "cpu" if materialized else "meta"
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN, device=device)
        head = nn.Linear(HIDDEN, VOCAB, bias=False, device=device)
        if tie:
            head.weight = self.model.embed_tokens.weight
        self.lm_head = head

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _fake_qmodel(model, *, checkpoint_has_head: bool):
    class _Turtle:
        def has_checkpoint_tensor(self, *, target_model, target_submodule):
            return checkpoint_has_head

    return SimpleNamespace(model=model, lm_head="lm_head", turtle_model=_Turtle())


def test_materialized_tied_pair_is_detected():
    model = _TinyTiedModel(materialized=True, tie=True)

    assert lm_head_quantization_is_tied(_fake_qmodel(model, checkpoint_has_head=True)) is True


def test_materialized_untied_pair_is_not_tied():
    model = _TinyTiedModel(materialized=True, tie=False)

    assert lm_head_quantization_is_tied(_fake_qmodel(model, checkpoint_has_head=True)) is False


def test_lazy_shell_without_a_head_tensor_is_tied():
    """The #5027 shape: both sides META, the checkpoint omits `lm_head.weight`."""

    model = _TinyTiedModel(materialized=False, tie=False)

    assert model.lm_head.weight.is_meta
    assert lm_head_quantization_is_tied(_fake_qmodel(model, checkpoint_has_head=False)) is True


def test_lazy_shell_with_an_independent_head_tensor_is_not_tied():
    model = _TinyTiedModel(materialized=False, tie=False)

    assert lm_head_quantization_is_tied(_fake_qmodel(model, checkpoint_has_head=True)) is False


def _save_tiny_tokenizer(directory) -> None:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["the quick brown fox", "a tiny tied llama checkpoint"],
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


def _save_tiny_tied_model(directory) -> None:
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    LlamaForCausalLM(config).eval().save_pretrained(directory)
    _save_tiny_tokenizer(directory)


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_rtn_lm_head_on_a_tied_model_is_refused_before_any_layer(tmp_path, offload_to_disk):
    """The default offload path must fail up front, not after the layer loop."""

    _save_tiny_tied_model(tmp_path)
    qcfg = RTNConfig(bits=4, group_size=32, lm_head=True, device="cpu")
    qcfg.offload_to_disk = offload_to_disk
    model = GPTQModel.load(str(tmp_path), qcfg, dtype=torch.float16)

    first_proj = model.model.model.layers[0].self_attn.q_proj
    assert isinstance(first_proj, nn.Linear)

    with pytest.raises(NotImplementedError, match="tied weights is not supported"):
        model.quantize(
            CalibrationConfig(dataset=["the quick brown fox jumps over the lazy dog"] * 8)
        )

    # The refusal happens before quantization, so no layer was replaced.
    assert isinstance(model.model.model.layers[0].self_attn.q_proj, nn.Linear)
