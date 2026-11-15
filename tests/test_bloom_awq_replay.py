# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bloom must supply residual, mask, and alibi scaffolding during AWQ replay."""

import torch
import torch.nn as nn
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

from gptqmodel.models.definitions.bloom import BloomQModel


def test_bloom_awq_replay_supplies_residual_for_mlp():
    """AWQ block replay must pass the residual argument BloomMLP.forward requires."""

    class _MLP(nn.Module):
        def forward(self, hidden_states, residual):
            raise AssertionError("replay kwargs are what this test exercises")

    instance = BloomQModel.__new__(BloomQModel)
    x = torch.randn(4, 16)
    kwargs = {}

    new_x, new_kwargs = BloomQModel.awq_module_replay_inputs(
        instance, _MLP(), x, kwargs
    )

    assert new_x.dim() == 3
    assert new_x.shape == (1, 4, 16)
    assert "residual" in new_kwargs
    assert torch.equal(new_kwargs["residual"], new_x)


def test_bloom_awq_replay_supplies_attention_scaffolding():
    """AWQ block replay must pass residual, attention_mask, and alibi for attention."""

    class _Attention(nn.Module):
        num_heads = 4

        def forward(self, hidden_states, residual, alibi, attention_mask):
            raise AssertionError("replay kwargs are what this test exercises")

    instance = BloomQModel.__new__(BloomQModel)
    x = torch.randn(2, 8, 32)
    kwargs = {}

    new_x, new_kwargs = BloomQModel.awq_module_replay_inputs(
        instance, _Attention(), x, kwargs
    )

    assert new_x.shape == (2, 8, 32)
    assert torch.equal(new_kwargs["residual"], new_x)
    assert "attention_mask" in new_kwargs
    assert new_kwargs["attention_mask"].shape == (2, 1, 8, 8)
    assert "alibi" in new_kwargs
    # alibi shape: (batch_size * num_heads, 1, seq_len)
    assert new_kwargs["alibi"].shape == (2 * 4, 1, 8)


def test_bloom_real_submodules_execute_cleanly_with_replay_inputs():
    """Real BloomMLP and BloomAttention must execute without TypeError under replay inputs."""

    config = BloomConfig(
        hidden_size=64,
        n_layer=2,
        n_head=4,
        vocab_size=128,
    )
    model = BloomForCausalLM(config).eval()
    instance = BloomQModel.__new__(BloomQModel)
    instance.__dict__["model"] = model

    block = model.transformer.h[0]
    mlp = block.mlp
    attn = block.self_attention

    x = torch.randn(1, 16, 64)

    # 1. Real BloomMLP forward
    new_x_mlp, kwargs_mlp = BloomQModel.awq_module_replay_inputs(
        instance, mlp, x, {}
    )
    out_mlp = mlp(new_x_mlp, **kwargs_mlp)
    assert out_mlp.shape == (1, 16, 64)

    # 2. Real BloomAttention forward
    new_x_attn, kwargs_attn = BloomQModel.awq_module_replay_inputs(
        instance, attn, x, {}
    )
    out_attn, _ = attn(new_x_attn, **kwargs_attn)
    assert out_attn.shape == (1, 16, 64)
