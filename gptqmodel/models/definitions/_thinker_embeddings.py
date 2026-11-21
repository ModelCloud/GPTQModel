# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Embedding endpoints of the Qwen Omni wrappers (issue #4816).

Transformers 5.23 (vendored unchanged) leaves ``PreTrainedModel.get_input_embeddings``
and ``get_output_embeddings`` in place on ``Qwen2_5OmniForConditionalGeneration``
and ``Qwen3OmniMoeForConditionalGeneration``. The token embedding and the
language head live on the nested ``thinker``, so the input getter raises
``NotImplementedError`` and the output getter returns ``None``. The quantized
loader (and writer, RCO, lm_head handling) discovers those endpoints on the
loaded model before or outside our wrapper, so the loaded *instance* gets
thinker-resolving getters. No class is modified: the vendored classes stay
exactly as Transformers ships them.
"""

from __future__ import annotations

import types
from typing import Optional

from torch import nn
from transformers import PreTrainedModel


def _thinker_input_embeddings(model: nn.Module) -> Optional[nn.Module]:
    return model.thinker.get_input_embeddings()


def _thinker_output_embeddings(model: nn.Module) -> Optional[nn.Module]:
    return model.thinker.lm_head


class ThinkerEmbeddingsMixin:
    """Resolve the Omni embedding endpoints through the ``thinker`` submodule."""

    #: Wrapper attribute that owns ``embed_tokens`` and ``lm_head``.
    THINKER_ROOT = "thinker"

    @classmethod
    def after_defuser_conversion(cls, model):
        parent = getattr(super(), "after_defuser_conversion", None)
        if callable(parent):
            parent(model)
        if not isinstance(model, PreTrainedModel) or not hasattr(model, cls.THINKER_ROOT):
            return
        model_cls = type(model)
        # Instance-level getters only; a class that grows real getters keeps them.
        if model_cls.get_input_embeddings is PreTrainedModel.get_input_embeddings:
            model.__dict__["get_input_embeddings"] = types.MethodType(_thinker_input_embeddings, model)
        if model_cls.get_output_embeddings is PreTrainedModel.get_output_embeddings:
            model.__dict__["get_output_embeddings"] = types.MethodType(_thinker_output_embeddings, model)

    def _thinker_embeddings_root(self) -> nn.Module:
        return getattr(self.model, self.THINKER_ROOT)

    def get_input_embeddings(self) -> Optional[nn.Module]:
        return self._thinker_embeddings_root().get_input_embeddings()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self._thinker_embeddings_root().lm_head
