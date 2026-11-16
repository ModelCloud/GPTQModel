# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for issue #4093 item 10 (native remote-code RoPE)."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers import PreTrainedModel

from gptqmodel.utils.structure import restore_init_owned_rotary_buffers


class _LegacyRotaryEmbedding(nn.Module):
    """Pre-5.x remote-code rotary module (Instella / EXAONE-3.0 / DeciLM shape).

    It owns ``inv_freq`` (and, for DeciLM, the cos/sin caches) as non-persistent
    buffers computed in ``__init__`` and has no ``original_inv_freq`` marker, so
    transformers 5's default ``_init_weights`` never re-derives it.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class _LegacyRemoteModel(nn.Module):
    """Remote-code model whose ``_init_weights`` override skips rotary restore."""

    def __init__(self, *, dim=8, base=10000.0, seq=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim * 2, num_attention_heads=2)
        self.rotary_emb = _LegacyRotaryEmbedding(
            dim,
            max_position_embeddings=seq,
            base=base,
        )

    def _init_weights(self, module):  # remote-code override: no rotary restore
        pass


class _DefaultInitModel(_LegacyRemoteModel):
    """Native-style model: transformers' default ``_init_weights`` owns rotary."""

    _init_weights = PreTrainedModel._init_weights


def _oracle_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Independent float64 reference for the default RoPE inverse frequencies."""

    exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    return 1.0 / (base ** exponents)


def _corrupt(module: nn.Module) -> None:
    """Reproduce transformers' ``torch.empty_like`` materialization of meta buffers."""

    for _name, buffer in module.named_buffers():
        buffer.fill_(float("nan"))


def test_restore_rebuilds_legacy_rotary_inv_freq():
    model = _LegacyRemoteModel(dim=16, base=10000.0)
    _corrupt(model.rotary_emb)
    assert torch.isnan(model.rotary_emb.inv_freq).all()

    restored = restore_init_owned_rotary_buffers(model)

    expected = _oracle_inv_freq(16, 10000.0).to(torch.float32)
    assert restored == 3
    assert torch.allclose(model.rotary_emb.inv_freq, expected, rtol=0.0, atol=1e-7)


def test_restore_rebuilds_legacy_cos_sin_caches():
    model = _LegacyRemoteModel(dim=16, base=10000.0, seq=32)
    _corrupt(model.rotary_emb)

    restored = restore_init_owned_rotary_buffers(model)

    inv_freq = _oracle_inv_freq(16, 10000.0)
    positions = torch.arange(32, dtype=torch.float64)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert restored == 3
    assert torch.allclose(model.rotary_emb.cos_cached, emb.cos().to(torch.float32), atol=1e-7)
    assert torch.allclose(model.rotary_emb.sin_cached, emb.sin().to(torch.float32), atol=1e-7)


def test_restore_keeps_nonpersistent_buffers_out_of_state_dict():
    model = _LegacyRemoteModel(dim=16, base=10000.0)
    _corrupt(model.rotary_emb)

    restore_init_owned_rotary_buffers(model)

    keys = set(model.state_dict())
    assert keys.isdisjoint(
        {"rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached"}
    )


def test_restore_skips_native_rotary_owned_by_default_init_weights():
    model = _DefaultInitModel(dim=16, base=10000.0)
    model.rotary_emb.original_inv_freq = model.rotary_emb.inv_freq.clone()
    _corrupt(model.rotary_emb)

    restored = restore_init_owned_rotary_buffers(model)

    assert restored == 0
    assert torch.isnan(model.rotary_emb.inv_freq).all()


def test_restore_skips_lazy_meta_shells():
    with torch.device("meta"):
        model = _LegacyRemoteModel(dim=16, base=10000.0)

    restored = restore_init_owned_rotary_buffers(model)

    assert restored == 0
    assert model.rotary_emb.inv_freq.is_meta


def test_restore_is_idempotent():
    model = _LegacyRemoteModel(dim=16, base=10000.0)
    _corrupt(model.rotary_emb)

    first_count = restore_init_owned_rotary_buffers(model)
    first_inv_freq = model.rotary_emb.inv_freq.clone()
    second_count = restore_init_owned_rotary_buffers(model)

    assert first_count == 3
    assert second_count == 3
    assert torch.equal(model.rotary_emb.inv_freq, first_inv_freq)
