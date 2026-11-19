# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tied-head Gemma-3 text MLX exports must run mlx-lm's ``Model.sanitize`` (#4808).

``mlx_lm.models.gemma3_text.Model`` always builds ``lm_head`` and only drops it
in ``Model.sanitize`` when ``lm_head.weight`` is absent; it does not read
``tie_word_embeddings`` from its args. The packed export skipped the tied head
in ``_packed_mlx_weights`` but loaded the model directly, so the strict load
still demanded ``lm_head.weight`` and every tied Gemma-3 text checkpoint failed
to export.

These tests do not need an MLX runtime: they drive the sanitize step with
faithful copies of the mlx-lm families' ``sanitize`` contracts. The
``importorskip`` test below checks the same contract against a real
``mlx_lm`` installation when one is available.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gptqmodel.utils import mlx as mlx_module


PACKED = {
    "model.embed_tokens.weight": "embed",
    "model.layers.0.self_attn.q_proj.weight": "q",
}


class _Gemma3TextModel:
    """Mirror ``mlx_lm.models.gemma3_text.Model.sanitize``'s observable contract."""

    def __init__(self):
        self.lm_head = object()
        self.tie_word_embeddings = False
        self.loaded_keys = None

    def sanitize(self, weights):
        # mlx_lm/models/gemma3_text.py: sanitize()
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
            self.pop("lm_head")
        return weights

    def pop(self, name):
        if name == "lm_head":
            self.lm_head = None

    def load_weights(self, items):
        keys = {key for key, _ in items}
        if self.lm_head is not None and "lm_head.weight" not in keys:
            raise ValueError("Missing 1 parameters: lm_head.weight.")
        self.loaded_keys = keys


class _LlamaModel:
    """Mirror ``mlx_lm.models.llama.Model.sanitize``'s observable contract."""

    def __init__(self, *, tie_word_embeddings):
        self.args = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.lm_head = None if tie_word_embeddings else object()
        self.loaded_keys = None

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def load_weights(self, items):
        self.loaded_keys = {key for key, _ in items}


# ---------------------------------------------------------------------------
# Gemma-3 text: sanitize drops the always-built tied head.
# ---------------------------------------------------------------------------


def test_gemma3_text_tied_head_is_dropped_before_the_strict_load():
    model = _Gemma3TextModel()

    # gemma3_text's sanitize returns the key set unchanged; the observable
    # effect is the dropped `lm_head` module plus the tie flag.
    expected_keys = set(PACKED)
    sanitized = mlx_module._sanitize_mlx_weights(model, PACKED)

    assert set(sanitized) == expected_keys
    assert model.lm_head is None
    assert model.tie_word_embeddings is True

    model.load_weights(list(sanitized.items()))
    assert model.loaded_keys == expected_keys


def test_gemma3_text_strict_load_rejects_the_unsanitized_map():
    """The pre-fix export path: load the tied checkpoint without sanitize."""

    model = _Gemma3TextModel()

    with pytest.raises(ValueError, match=r"Missing 1 parameters: lm_head\.weight"):
        model.load_weights(list(PACKED.items()))


# ---------------------------------------------------------------------------
# Control: a family that keeps its (untied) head.
# ---------------------------------------------------------------------------


def test_untied_llama_head_is_kept_and_rotary_freqs_are_dropped():
    model = _LlamaModel(tie_word_embeddings=False)
    packed = {
        **PACKED,
        "lm_head.weight": "head",
        "model.layers.0.self_attn.rotary_emb.inv_freq": "unused",
    }
    # Expected sanitized key set from llama's own sanitize contract: keep the
    # untied head, drop the precomputed rotary frequencies.
    expected_keys = set(PACKED) | {"lm_head.weight"}

    sanitized = mlx_module._sanitize_mlx_weights(model, packed)

    assert set(sanitized) == expected_keys
    assert model.lm_head is not None
    model.load_weights(list(sanitized.items()))
    assert model.loaded_keys == expected_keys


def test_tied_llama_head_is_dropped_from_the_weight_map():
    model = _LlamaModel(tie_word_embeddings=True)
    packed = {**PACKED, "lm_head.weight": "head"}

    sanitized = mlx_module._sanitize_mlx_weights(model, packed)

    assert set(sanitized) == set(PACKED)
    model.load_weights(list(sanitized.items()))
    assert "lm_head.weight" not in model.loaded_keys


# ---------------------------------------------------------------------------
# Fallbacks: models without a sanitizer, or one that returns nothing.
# ---------------------------------------------------------------------------


def test_model_without_sanitize_returns_the_weights_unchanged():
    class _PlainModel:
        pass

    assert mlx_module._sanitize_mlx_weights(_PlainModel(), PACKED) == PACKED


def test_sanitize_returning_none_keeps_the_original_weights():
    class _MutatingModel:
        def __init__(self):
            self.called = False

        def sanitize(self, weights):
            self.called = True
            return None

    model = _MutatingModel()
    assert mlx_module._sanitize_mlx_weights(model, PACKED) == PACKED
    assert model.called is True


# ---------------------------------------------------------------------------
# Real mlx-lm contract, when the runtime is installed.
# ---------------------------------------------------------------------------


def test_real_gemma3_text_model_drops_the_tied_head():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import gemma3_text

    args = gemma3_text.ModelArgs(
        model_type="gemma3_text",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=16,
        vocab_size=64,
        num_key_value_heads=1,
        sliding_window=8,
        sliding_window_pattern=1,
    )

    tied = gemma3_text.Model(args)
    sanitized = mlx_module._sanitize_mlx_weights(tied, {})
    assert sanitized == {}
    assert tied.tie_word_embeddings is True
    assert not hasattr(tied, "lm_head")

    untied = gemma3_text.Model(args)
    weights = {"lm_head.weight": "head"}
    assert mlx_module._sanitize_mlx_weights(untied, weights) == weights
    assert untied.tie_word_embeddings is False
    assert hasattr(untied, "lm_head")
