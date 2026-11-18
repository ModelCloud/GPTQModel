# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression for #5226: importing gptqmodel must not change native RoPE init.

The remote-code compat shim publishes a generic
``ROPE_INIT_FUNCTIONS["default"]`` so legacy ``trust_remote_code`` modules keep
working. transformers 5.x builds ``_init_weights``' lookup as
``{"default": module.compute_default_rope_parameters, **ROPE_INIT_FUNCTIONS}``,
so the global entry used to replace every native family's own initializer.
ERNIE-4.5-VL pre-rotates its h/w sections, and its logits changed as soon as
``gptqmodel`` was imported.

``gptqmodel`` now keeps the module-owned initializer ahead of the shim while
leaving the shim in the table for remote code.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

import gptqmodel  # noqa: F401  - the import itself is what used to break models
from gptqmodel.models.definitions.ouro import OuroQModel


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


class _ShimRotaryEmbedding(nn.Module):
    """A legacy module that relies on ``ROPE_INIT_FUNCTIONS["default"]``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
        self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)


class _Host(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.owned = _OwnedRotaryEmbedding(config)
        self.shim = _ShimRotaryEmbedding(config)
        self.post_init()


def test_import_keeps_a_model_owned_default_rope_initializer():
    model = _Host(_rope_config())

    # The module's own initializer must win over the global shim.
    assert model.owned.inv_freq.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_import_keeps_the_remote_code_shim_available():
    model = _Host(_rope_config())

    # Legacy remote modules have no own initializer and must still resolve the
    # shim from the global table.
    assert "default" in ROPE_INIT_FUNCTIONS
    expected, _ = ROPE_INIT_FUNCTIONS["default"](_rope_config())
    assert torch.equal(model.shim.inv_freq, expected)
    # Sanity: the shim and the model-owned function disagree here, so the two
    # tests above really pin both sides of the precedence rule.
    assert not torch.equal(expected, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_ernie_vl_keeps_its_pre_rotated_inv_freq_after_post_init():
    """The real GPT-QModel-supported family the issue reproduced with."""

    modeling = pytest.importorskip(
        "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe"
    )
    configuration = pytest.importorskip(
        "transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe"
    )
    rotary_cls = modeling.Ernie4_5_VLMoeTextRotaryEmbedding
    config_cls = configuration.Ernie4_5_VLMoeTextConfig

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


class _DelegatingRotaryEmbedding(nn.Module):
    """A remote rotary whose own initializer delegates to the global shim."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = "default"
        self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
        self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)

    def compute_default_rope_parameters(self, config):
        return ROPE_INIT_FUNCTIONS["default"](config)


class _BridgeHost(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.rotary = _DelegatingRotaryEmbedding(config)
        self.post_init()


def _install_fake_ouro_rotary(monkeypatch):
    """Install Ouro's real bridge helper on a fake remote rotary class."""

    class FakeOuroRotaryEmbedding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.rope_type = "default"
            self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
            self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)

    remote_module = types.ModuleType("fake_ouro_rope_remote")
    FakeOuroRotaryEmbedding.__module__ = remote_module.__name__
    monkeypatch.setitem(sys.modules, remote_module.__name__, remote_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_local_path: FakeOuroRotaryEmbedding,
    )
    OuroQModel.before_model_load(OuroQModel, "/unused/Ouro-1.4B", False)
    return FakeOuroRotaryEmbedding


def test_ouro_bridge_survives_hf_weight_init(monkeypatch):
    """#5259: the Ouro bridge must resolve the shim during ``post_init``."""

    rotary_cls = _install_fake_ouro_rotary(monkeypatch)

    installed = ROPE_INIT_FUNCTIONS["default"]
    expected, _ = installed(_rope_config())

    class _OuroHost(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.rotary = rotary_cls(config)
            self.post_init()

    model = _OuroHost(_rope_config())

    assert getattr(rotary_cls.compute_default_rope_parameters, "_gptqmodel_rope_bridge", False)
    assert torch.equal(model.rotary.inv_freq, expected)
    assert torch.equal(model.rotary.original_inv_freq, expected)
    assert ROPE_INIT_FUNCTIONS["default"] is installed


def test_unmarked_delegating_bridge_survives_hf_weight_init():
    """A bridge without the explicit marker is handled the same way."""

    installed = ROPE_INIT_FUNCTIONS["default"]
    expected, _ = installed(_rope_config())

    model = _BridgeHost(_rope_config())

    assert torch.equal(model.rotary.inv_freq, expected)
    assert ROPE_INIT_FUNCTIONS["default"] is installed


def test_rope_init_precedence_leaves_global_table_intact():
    """The owned initializer wins and never sees the shim removed (#5226/#5259)."""

    installed = ROPE_INIT_FUNCTIONS["default"]
    observed = []

    class _WatchingOwnedRotaryEmbedding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.rope_type = "default"
            self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
            self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)

        def compute_default_rope_parameters(self, config):
            observed.append(("default" in ROPE_INIT_FUNCTIONS, ROPE_INIT_FUNCTIONS.get("default")))
            return torch.tensor([1.0, 2.0, 3.0, 4.0]), 1.0

    class _WatchingHost(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.rotary = _WatchingOwnedRotaryEmbedding(config)
            self.post_init()

    model = _WatchingHost(_rope_config())

    assert model.rotary.inv_freq.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert observed == [(True, installed)]
    assert ROPE_INIT_FUNCTIONS["default"] is installed


def test_rope_init_bridge_detection(monkeypatch):
    from gptqmodel.utils.hf import _is_rope_init_bridge

    rotary_cls = _install_fake_ouro_rotary(monkeypatch)

    assert _is_rope_init_bridge(rotary_cls.compute_default_rope_parameters) is True
    assert _is_rope_init_bridge(_DelegatingRotaryEmbedding.compute_default_rope_parameters) is True
    assert _is_rope_init_bridge(_OwnedRotaryEmbedding.compute_default_rope_parameters) is False

    class _LinearDelegatingRotaryEmbedding(nn.Module):
        def compute_default_rope_parameters(self, config):
            return ROPE_INIT_FUNCTIONS["linear"](config)

    assert _is_rope_init_bridge(_LinearDelegatingRotaryEmbedding().compute_default_rope_parameters) is False


def test_rope_init_precedence_restores_table_when_initializer_raises():
    """The table stays intact and is restored even when init fails."""

    installed = ROPE_INIT_FUNCTIONS["default"]
    observed = []

    class _ExplodingRotaryEmbedding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.rope_type = "default"
            self.inv_freq = nn.Buffer(torch.zeros(4), persistent=False)
            self.original_inv_freq = nn.Buffer(torch.zeros(4), persistent=False)

        def compute_default_rope_parameters(self, config):
            observed.append("default" in ROPE_INIT_FUNCTIONS)
            raise RuntimeError("rope init boom")

    class _ExplodingHost(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.rotary = _ExplodingRotaryEmbedding(config)
            self.post_init()

    with pytest.raises(RuntimeError, match="rope init boom"):
        _ExplodingHost(_rope_config())

    assert observed == [True]
    assert ROPE_INIT_FUNCTIONS["default"] is installed


def test_rope_init_fallback_handles_bridges_and_owned_initializers(monkeypatch):
    """The pop fallback never breaks a bridge and still favors an owned init."""

    import gptqmodel.utils.hf as hf_utils

    monkeypatch.setattr(hf_utils, "_rope_init_lookup_globals", lambda init_fn: None)

    installed = ROPE_INIT_FUNCTIONS["default"]
    expected, _ = installed(_rope_config())

    bridge_model = _BridgeHost(_rope_config())
    assert torch.equal(bridge_model.rotary.inv_freq, expected)
    assert ROPE_INIT_FUNCTIONS["default"] is installed

    owned_model = _Host(_rope_config())
    assert owned_model.owned.inv_freq.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert ROPE_INIT_FUNCTIONS["default"] is installed
