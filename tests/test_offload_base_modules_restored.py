# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4203.

Quantization offloads the embedding/lm_head base modules to disk. They must be
materialized again before ``quantize()`` returns: leaving them on meta makes the
in-session model unusable because ``tie_weights()`` re-points ``lm_head.weight``
at the meta embedding parameter, which has no offload hook.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.utils.offload import (
    capture_offloaded_base_modules,
    is_meta_module,
    offload_to_disk,
    restore_offloaded_modules,
)


BASE_MODULES = ("model.embed_tokens", "model.rotary_emb", "lm_head")


def _tiny_tied_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        # Embedding payload must exceed the 4 KiB small-module offload cutoff.
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config).eval()


def _offload_tied_base_modules(model, tmp_path):
    entries = capture_offloaded_base_modules(model, BASE_MODULES)
    offload_to_disk(model=model, module=list(BASE_MODULES), disk_path=str(tmp_path))
    # Mirrors ``LazyTurtle.materialize_submodule``: every layer materialization
    # re-ties the output head to the (now meta) input embedding parameter.
    model.tie_weights()
    assert is_meta_module(model.get_input_embeddings())
    assert model.device.type == "meta"
    return entries


def _assert_runnable(model):
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    assert not embed.is_meta
    assert not head.is_meta
    assert head is embed
    assert model.device.type == "cpu"
    ids = torch.tensor([[1, 3, 5]])
    with torch.inference_mode():
        logits = model(input_ids=ids).logits
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, 3, 32)


def _bare_looper(model, entries):
    """A ``ModuleLooper`` built without ``__init__`` but with the loop's state.

    ``loop()`` reads ``_inflight_layer`` on the failure path (#4546) and drains
    ``_dangling_threads`` before restoring anything (#4595); a double without
    them raises ``AttributeError`` while handling the injected failure.
    """
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(model=model)
    looper._offloaded_base_modules = entries
    looper._inflight_layer = None
    looper._dangling_threads_lock = threading.Lock()
    looper._dangling_threads = []
    return looper


def test_restore_offloaded_modules_materializes_tied_embeddings(tmp_path):
    model = _tiny_tied_model()
    entries = _offload_tied_base_modules(model, tmp_path)

    assert restore_offloaded_modules(model, entries, strict=True) == []

    _assert_runnable(model)


def test_restore_offloaded_modules_accepts_already_materialized_modules(tmp_path):
    """Non-offloaded base modules (rotary embeddings, norm) must not fail restore."""
    model = _tiny_tied_model()
    entries = _offload_tied_base_modules(model, tmp_path)
    assert not is_meta_module(model.model.rotary_emb)

    restore_offloaded_modules(model, entries, strict=True)

    assert not is_meta_module(model.model.rotary_emb)


def test_loop_restores_base_modules_on_success(monkeypatch, tmp_path):
    model = _tiny_tied_model()
    entries = _offload_tied_base_modules(model, tmp_path)

    looper = _bare_looper(model, entries)
    monkeypatch.setattr(looper, "_loop_impl", lambda **_kwargs: "finished")

    assert looper.loop() == "finished"
    _assert_runnable(model)
    assert looper._offloaded_base_modules == []


def test_loop_restores_base_modules_when_quantize_fails(monkeypatch, tmp_path):
    model = _tiny_tied_model()
    entries = _offload_tied_base_modules(model, tmp_path)

    looper = _bare_looper(model, entries)

    def _boom(**_kwargs):
        raise RuntimeError("quantize failed")

    monkeypatch.setattr(looper, "_loop_impl", _boom)

    with pytest.raises(RuntimeError, match="quantize failed"):
        looper.loop()

    # The original failure is preserved, but the model is not left on meta.
    assert not is_meta_module(model.get_input_embeddings())
    assert model.device.type == "cpu"
