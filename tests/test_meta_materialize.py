# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, LlamaConfig

from gptqmodel.utils.model import materialize_meta_tensors


def _tiny_llama():
    return LlamaConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_materialize_meta_tensors_materializes_rope_buffers():
    """Meta-device shells created by init_empty_weights leave non-persistent
    buffers (e.g. RoPE inv_freq) on meta. materialize_meta_tensors should move
    them to the target device and initialize their values."""
    config = _tiny_llama()

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    assert model.model.rotary_emb.inv_freq.device.type == "meta"
    assert model.model.rotary_emb.original_inv_freq.device.type == "meta"

    materialize_meta_tensors(model, {"": "cpu"})

    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.model.rotary_emb.original_inv_freq.device.type == "cpu"
    assert not model.model.rotary_emb.inv_freq.is_meta
    assert not model.model.rotary_emb.original_inv_freq.is_meta
    assert torch.isfinite(model.model.rotary_emb.inv_freq).all()
    assert torch.isfinite(model.model.rotary_emb.original_inv_freq).all()


class _FakeLog:
    def __init__(self):
        self.messages = []

    def warn(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warn(msg, *args)


def test_materialize_meta_tensors_keeps_loaded_weights_intact(monkeypatch):
    """If a parameter was already loaded into the model before materialize,
    _init_weights must not overwrite it, even when _init_weights calls
    torch.nn.init.* directly (guard_torch_init_functions patches those calls)."""
    config = _tiny_llama()

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.dummy = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    # Pretend dummy.weight was loaded from the checkpoint while dummy.bias
    # is still a meta placeholder.
    original_weight = torch.empty(model.dummy.weight.shape, dtype=model.dummy.weight.dtype).normal_(0, 0.02)
    model.dummy.weight = nn.Parameter(original_weight.clone())

    orig_init = model._init_weights

    def _init_weights_wrapped(module):
        orig_init(module)
        if isinstance(module, nn.Linear):
            # This call is patched by guard_torch_init_functions and must be
            # skipped because dummy.weight is already loaded.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model._init_weights = _init_weights_wrapped

    materialize_meta_tensors(model, {"": "cpu"})

    assert torch.equal(model.dummy.weight, original_weight)
    assert model.dummy.bias.device.type == "cpu"
    assert not model.dummy.bias.is_meta
    assert torch.all(model.dummy.bias == 0)


def test_materialize_meta_tensors_warns_on_failed_parent_init(monkeypatch):
    """A computed non-persistent buffer whose parent _init_weights fails must be
    warned about instead of silently left with uninitialized memory."""
    config = _tiny_llama()

    class BadBufferModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("computed", torch.empty(2), persistent=False)

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.bad = BadBufferModule()

    orig_init = model._init_weights

    def _init_weights_wrapped(module):
        if isinstance(module, BadBufferModule):
            raise RuntimeError("cannot init bad buffer")
        orig_init(module)

    model._init_weights = _init_weights_wrapped

    fake_log = _FakeLog()
    monkeypatch.setattr("gptqmodel.utils.model.log", fake_log)

    materialize_meta_tensors(model, {"": "cpu"})

    all_text = " ".join(fake_log.messages)
    assert "cannot init bad buffer" in all_text
    assert "not re-initialized" in all_text


def test_materialize_meta_tensors_reinitializes_top_level_buffer():
    """A non-persistent buffer attached to the root model should not be lost; the
    empty parent name from rpartition('.') must resolve to the root module."""
    config = _tiny_llama()

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.register_buffer("top_buffer", torch.empty(2), persistent=False)

    orig_init = model._init_weights

    def _init_weights_wrapped(module):
        if module is model and hasattr(module, "top_buffer"):
            module.top_buffer.copy_(torch.ones_like(module.top_buffer))
        else:
            orig_init(module)

    model._init_weights = _init_weights_wrapped

    materialize_meta_tensors(model, {"": "cpu"})

    assert model.top_buffer.device.type == "cpu"
    assert not model.top_buffer.is_meta
    assert torch.all(model.top_buffer == 1)


def test_materialize_meta_tensors_skips_offloaded_modules(monkeypatch):
    """Tensors whose parent module is disk-offloaded by accelerate must not be
    overwritten with empty tensors; their real data lives in offload hooks."""
    config = _tiny_llama()

    class OffloadedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("computed", torch.empty(2), persistent=False)

    def fake_has_offloaded_params(module):
        return getattr(module, "_offloaded", False)

    monkeypatch.setattr("accelerate.utils.has_offloaded_params", fake_has_offloaded_params)

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.offloaded = OffloadedModule()
        model.offloaded._offloaded = True

    materialize_meta_tensors(model, {"": "cpu"})

    # The offloaded buffer should remain untouched (still meta) because we do not
    # know how to restore its real data here.
    assert model.offloaded.computed.device.type == "meta"
    assert model.offloaded.computed.is_meta


def test_materialize_meta_tensors_only_non_persistent_buffers():
    """With only_non_persistent_buffers=True, persistent parameters and buffers
    must stay on meta so a later loader (e.g. GGUF) can allocate them directly on
    the target device; only computed non-persistent buffers are materialized."""
    config = _tiny_llama()

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    materialize_meta_tensors(model, {"": "cpu"}, only_non_persistent_buffers=True)

    # Non-persistent RoPE buffers should be real on the target device.
    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    assert torch.isfinite(model.model.rotary_emb.inv_freq).all()

    # Persistent weights must remain meta to be filled by the checkpoint loader.
    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.lm_head.weight.device.type == "meta"
    assert model.model.layers[0].self_attn.q_proj.weight.device.type == "meta"
