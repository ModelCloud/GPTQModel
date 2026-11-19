# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5411: untied DiffusionGemma parameters must keep their own values.

``DiffusionGemmaConfig.tie_word_embeddings=False`` is supported and stores two
independent copies of the text parameters. The definition aliased them anyway:
``_sync_decoder_tied_parameters`` matched encoder and decoder parameters by
name and shape without consulting the config, so quantization replaced the
decoder's trained values with the encoder's, and
``_refresh_tied_weight_metadata`` still published the class-level tie map so a
save would drop the decoder copies.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions import diffusion_gemma as definition
from gptqmodel.models.definitions.diffusion_gemma import (
    DiffusionGemmaQModel,
    _encoder_decoder_tied,
    _refresh_tied_weight_metadata,
    _sync_decoder_quantized_modules,
    _sync_decoder_tied_parameters,
)


pytestmark = [pytest.mark.cpu]


class _Stack(nn.Module):
    """The minimal encoder/decoder layer tree the finalizer walks."""

    def __init__(self, layers, *, bias: float):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.Parameter(torch.full((4,), bias))


class _Core(nn.Module):
    def __init__(self, *, bias: float):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.language_model = _Stack([_Layer(bias)], bias=bias)
        self.decoder = _Stack([_Layer(bias)], bias=bias)
        self._tied_weights_keys = {
            "encoder.language_model.norm.weight": "decoder.norm.weight",
        }

    def get_expanded_tied_weights_keys(self, all_submodels=False):
        del all_submodels
        return dict(self._tied_weights_keys)


class _Layer(nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.input_layernorm = nn.Parameter(torch.full((4,), bias))


class _FakeModel(nn.Module):
    def __init__(self, *, tied: bool, encoder_bias: float, decoder_bias: float):
        super().__init__()
        self.config = SimpleNamespace(
            tie_word_embeddings=tied,
            text_config=SimpleNamespace(tie_word_embeddings=tied),
        )
        self.model = _Core(bias=encoder_bias)
        # Give the decoder deliberately different values from the encoder.
        with torch.no_grad():
            self.model.decoder.layers[0].input_layernorm.fill_(decoder_bias)
            self.model.decoder.norm.fill_(decoder_bias)

    def get_expanded_tied_weights_keys(self, all_submodels=False):
        del all_submodels
        return dict(self.model._tied_weights_keys)


def _decoder_values(model):
    return (
        model.model.decoder.layers[0].input_layernorm.detach().clone(),
        model.model.decoder.norm.detach().clone(),
    )


def test_untied_config_is_detected():
    assert _encoder_decoder_tied(_FakeModel(tied=True, encoder_bias=0.0, decoder_bias=0.5)) is True
    assert _encoder_decoder_tied(_FakeModel(tied=False, encoder_bias=0.0, decoder_bias=0.5)) is False


def test_untied_checkpoint_keeps_its_decoder_values():
    model = _FakeModel(tied=False, encoder_bias=0.0, decoder_bias=0.5)
    before = _decoder_values(model)

    assert _sync_decoder_tied_parameters(model) == 0

    for kept, expected in zip(_decoder_values(model), before):
        torch.testing.assert_close(kept, expected, rtol=0.0, atol=0.0)
    # Still independent storage, not a shared Parameter.
    assert (
        model.model.decoder.layers[0].input_layernorm is not
        model.model.encoder.language_model.layers[0].input_layernorm
    )


def test_untied_checkpoint_publishes_no_tie_map():
    model = _FakeModel(tied=False, encoder_bias=0.0, decoder_bias=0.5)

    _refresh_tied_weight_metadata(model, include_layer_ties=True)

    assert model.model._tied_weights_keys == {}
    assert model.model.all_tied_weights_keys == {}
    assert model.all_tied_weights_keys == {}


def test_tied_checkpoint_still_aliases_and_records_the_tie():
    model = _FakeModel(tied=True, encoder_bias=0.25, decoder_bias=0.5)
    encoder_norm = model.model.encoder.language_model.layers[0].input_layernorm

    assert _sync_decoder_tied_parameters(model) > 0

    # The alias is restored and the encoder value wins, as before.
    assert model.model.decoder.layers[0].input_layernorm is encoder_norm
    _refresh_tied_weight_metadata(model, include_layer_ties=True)
    assert (
        model.model._tied_weights_keys["decoder.layers.0.input_layernorm"]
        == "encoder.language_model.layers.0.input_layernorm"
    )


def test_untied_checkpoint_does_not_share_quantized_modules():
    model = _FakeModel(tied=False, encoder_bias=0.0, decoder_bias=0.5)
    decoder_module = nn.Linear(2, 2)
    model.model.decoder.layers[0].proj = decoder_module
    quant_module = nn.Linear(2, 2)
    model.model.encoder.language_model.layers[0].proj = quant_module

    # Pretend the encoder projection is a quantized leaf.
    original = definition.BaseQuantLinear
    definition.BaseQuantLinear = nn.Linear
    try:
        assert _sync_decoder_quantized_modules(model) == 0
    finally:
        definition.BaseQuantLinear = original

    assert model.model.decoder.layers[0].proj is decoder_module


def test_quantizing_an_untied_checkpoint_is_refused_before_any_weight_changes():
    model = _FakeModel(tied=False, encoder_bias=0.0, decoder_bias=0.5)
    before = _decoder_values(model)
    stub = SimpleNamespace(model=model)

    with pytest.raises(NotImplementedError, match="tie_word_embeddings=False"):
        DiffusionGemmaQModel.pre_quantize_generate_hook_start(stub)

    for kept, expected in zip(_decoder_values(model), before):
        torch.testing.assert_close(kept, expected, rtol=0.0, atol=0.0)
