# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Embedding endpoints of the Qwen Omni wrappers (issue #4816).

transformers 5.22 leaves ``PreTrainedModel.get_input_embeddings`` and
``get_output_embeddings`` in place on ``Qwen2_5OmniForConditionalGeneration``
and ``Qwen3OmniMoeForConditionalGeneration``. The token embedding and the
language head live on the nested ``thinker``, so the input getter raises
``NotImplementedError`` and the output getter returns ``None`` (it delegates to
the input getter). The quantized loader discovers those endpoints before it
builds our wrapper, so the HF class getters have to be supplied as well as the
wrapper ones.
"""

from __future__ import annotations

from typing import Optional

from torch import nn

from ..local import LocalModelLoader


class ThinkerEmbeddingsMixin:
    """Resolve the Omni embedding endpoints through the ``thinker`` submodule."""

    #: Wrapper attribute that owns ``embed_tokens`` and ``lm_head``.
    THINKER_ROOT = "thinker"
    #: Transformers wrapper whose missing getters must be supplied.
    HF_WRAPPER_CLASS_NAME: Optional[str] = None

    @classmethod
    def _hf_wrapper_class(cls):
        if cls.HF_WRAPPER_CLASS_NAME is None:
            raise NotImplementedError(
                f"{cls.__name__} must declare HF_WRAPPER_CLASS_NAME to patch the Omni getters"
            )
        if not isinstance(getattr(cls, "loader", None), LocalModelLoader):
            raise TypeError(f"{cls.__name__} must load through a package-owned LocalModelLoader")
        # The definition loads the vendored Transformers 5.23 wrapper.
        from ..ports.native.auto import _by_name

        wrapper = _by_name(cls.HF_WRAPPER_CLASS_NAME)
        if wrapper is None:
            raise ValueError(f"no vendored class named {cls.HF_WRAPPER_CLASS_NAME}")
        return wrapper

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        # The loader passes the definition *class* as the receiver: no model
        # instance exists yet.
        from transformers import PreTrainedModel

        model_cls = self._hf_wrapper_class()
        root = self.THINKER_ROOT
        # Only replace the inherited default: a transformers release that grows
        # real getters keeps them.
        if model_cls.get_input_embeddings is PreTrainedModel.get_input_embeddings:
            model_cls.get_input_embeddings = lambda model: getattr(model, root).get_input_embeddings()
        if model_cls.get_output_embeddings is PreTrainedModel.get_output_embeddings:
            model_cls.get_output_embeddings = lambda model: getattr(model, root).lm_head

    def _thinker_embeddings_root(self) -> nn.Module:
        return getattr(self.model, self.THINKER_ROOT)

    def get_input_embeddings(self) -> Optional[nn.Module]:
        return self._thinker_embeddings_root().get_input_embeddings()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self._thinker_embeddings_root().lm_head
