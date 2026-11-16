# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A tied head is tied even when the checkpoint stores `lm_head.weight`."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from gptqmodel.looper.module_looper import ModuleLooper


class _TiedModel(nn.Module):
    def __init__(self, *, tie_word_embeddings: bool, declared: bool):
        super().__init__()
        self.config = types.SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if declared:
            self._tied_weights_keys = ["lm_head.weight"]
        else:
            self._tied_weights_keys = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _looper(model, *, checkpoint_has_head: bool):
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = types.SimpleNamespace(
        model=model,
        lm_head="lm_head",
        turtle_model=types.SimpleNamespace(
            has_checkpoint_tensor=lambda **_kwargs: checkpoint_has_head
        ),
    )
    return looper


def _meta_head(model):
    model.lm_head.weight = nn.Parameter(torch.empty(8, 4, device="meta"))


def test_declared_tied_head_is_rejected_even_when_checkpoint_stores_it():
    model = _TiedModel(tie_word_embeddings=True, declared=True)
    _meta_head(model)
    # Qwen3 checkpoints contain `lm_head.weight` even though the head is tied.
    assert ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))


def test_declared_tied_head_without_checkpoint_tensor_is_rejected():
    model = _TiedModel(tie_word_embeddings=True, declared=True)
    _meta_head(model)
    assert ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=False))


def test_untied_config_keeps_an_independent_head_quantizable():
    model = _TiedModel(tie_word_embeddings=False, declared=True)
    _meta_head(model)
    assert not ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))


def test_materialized_shared_parameter_is_detected_without_declaration():
    model = _TiedModel(tie_word_embeddings=True, declared=False)
    model.lm_head.weight = model.embed_tokens.weight
    assert ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))


def test_nested_text_config_does_not_imply_tied_embeddings():
    """A nested text_config flag is not proof of the loaded storage (#4369)."""

    model = _TiedModel(tie_word_embeddings=False, declared=True)
    model.config.text_config = types.SimpleNamespace(tie_word_embeddings=True)
    _meta_head(model)
    assert not ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))


def test_nested_text_config_with_independent_materialized_embeddings_is_untied():
    """Qwen3-VL keeps independent input/output embeddings despite text_config."""

    model = _TiedModel(tie_word_embeddings=False, declared=True)
    model.config.text_config = types.SimpleNamespace(tie_word_embeddings=True)
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    assert not ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))


def test_nested_text_config_does_not_hide_genuine_shared_storage():
    """Real sharing still wins over an untied root/nested config declaration."""

    model = _TiedModel(tie_word_embeddings=False, declared=True)
    model.config.text_config = types.SimpleNamespace(tie_word_embeddings=True)
    model.lm_head.weight = model.embed_tokens.weight
    assert ModuleLooper._lm_head_is_tied(_looper(model, checkpoint_has_head=True))
