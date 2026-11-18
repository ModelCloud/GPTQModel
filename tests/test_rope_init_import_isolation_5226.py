# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression for #5226: importing gptqmodel must not change native RoPE init.

The former remote-code compat shim published a generic
``ROPE_INIT_FUNCTIONS["default"]``. transformers 5.x builds ``_init_weights``'
lookup as
``{"default": module.compute_default_rope_parameters, **ROPE_INIT_FUNCTIONS}``,
so that global entry replaced every native family's own initializer:
ERNIE-4.5-VL pre-rotates its h/w sections, and its logits changed as soon as
``gptqmodel`` was imported.

GPT-QModel targets the Transformers 5.23 ABI and no longer patches
Transformers at all, so the table and ``_init_weights`` stay exactly as
Transformers ships them and a module-owned initializer always wins.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import gptqmodel  # noqa: F401  - the import itself is what used to break models


def _rope_config():
    return PretrainedConfig(
        hidden_size=16,
        num_attention_heads=2,
        head_dim=8,
        rope_theta=10_000.0,
    )


class _OwnedRotaryEmbedding(nn.Module):
    """A native family that defines its own ``compute_default_rope_parameters``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
        self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)

    def compute_default_rope_parameters(self, config, device=None, seq_len=None, layer_type=None):
        return torch.tensor([1.0, 2.0, 3.0, 4.0]), 1.0


class _Host(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.owned = _OwnedRotaryEmbedding(config)
        self.post_init()


def test_import_keeps_a_model_owned_default_rope_initializer():
    model = _Host(_rope_config())

    # The module's own initializer is what Transformers selects for "default".
    assert model.owned.inv_freq.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_import_publishes_no_rope_initializer():
    # Nothing GPT-QModel defines may sit in the global table or wrap
    # `_init_weights` (the old shim did both).
    owned = {
        name: fn.__module__
        for name, fn in ROPE_INIT_FUNCTIONS.items()
        if getattr(fn, "__module__", "").startswith("gptqmodel")
    }
    assert owned == {}
    assert PreTrainedModel._init_weights.__module__.startswith("transformers.")


def test_ernie_vl_keeps_its_pre_rotated_inv_freq_after_post_init():
    """The real GPT-QModel-supported family the issue reproduced with."""

    from gptqmodel.models.ports.native.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeTextConfig as config_cls,
    )
    from gptqmodel.models.ports.native.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeTextRotaryEmbedding as rotary_cls,
    )

    config = config_cls(
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "mrope_section": [3, 3, 2]},
    )

    class _ErnieHost(PreTrainedModel):
        config_class = config_cls

        def __init__(self, cfg):
            super().__init__(cfg)
            self.rotary = rotary_cls(cfg)
            self.post_init()

    model = _ErnieHost(config)
    expected, _ = rotary_cls.compute_default_rope_parameters(config)

    assert torch.equal(model.rotary.inv_freq, expected)
    assert torch.equal(model.rotary.original_inv_freq, expected)
