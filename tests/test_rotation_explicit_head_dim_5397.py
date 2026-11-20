# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Rotation must use a model's declared ``head_dim`` (#5397).

Transformers Llama accepts an explicit ``head_dim`` that differs from
``hidden_size // num_attention_heads``. The per-head V/O Hadamard transform
used the ratio instead, so a valid configuration reshaped the V projection with
the wrong chunk size and raised after embeddings and norms were already rotated.
"""

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.quantization.rotation import rotation as rotation_module
from gptqmodel.quantization.rotation.rotation import rotate_model, validate_rotation_dimensions


def _config(*, head_dim):
    return LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        max_position_embeddings=64,
    )


def test_resolve_head_dim_prefers_the_declared_value():
    assert rotation_module.resolve_head_dim(_config(head_dim=16)) == 16


def test_resolve_head_dim_falls_back_to_the_hidden_size_ratio():
    class LegacyConfig:
        hidden_size = 64
        num_attention_heads = 2

    assert rotation_module.resolve_head_dim(LegacyConfig()) == 32


def test_rotation_accepts_a_declared_head_dim_that_differs_from_the_ratio():
    config = _config(head_dim=16)
    model = LlamaForCausalLM(config).eval()
    assert model.model.layers[0].self_attn.v_proj.weight.shape == (16, 64)

    embeddings_before = model.model.embed_tokens.weight.detach().clone()

    validate_rotation_dimensions(config, "hadamard")
    model, _ = rotate_model(
        model,
        "hadamard",
        torch.device("cpu"),
        "lm_head",
        "model.layers",
        embedding_name="model.embed_tokens",
    )

    assert not torch.equal(embeddings_before, model.model.embed_tokens.weight)
    ids = torch.arange(16).unsqueeze(0) % 60 + 1
    with torch.inference_mode():
        logits = model(ids).logits
    assert logits.shape == (1, 16, 64)
    assert torch.isfinite(logits).all()


def test_a_non_power_of_two_head_dim_is_refused_before_any_weight_changes():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=96,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=24,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="head_dim"):
        validate_rotation_dimensions(config, "hadamard")

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
