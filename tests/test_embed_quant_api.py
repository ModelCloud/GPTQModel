# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch import nn

from gptqmodel import QuantizeEmbed, QuantizeEmbedConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization import QuantizeEmbedConfig as QuantizationEmbedConfig


def _bare_model(*, quantized: bool):
    model = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(model)
    model.quantized = quantized
    return model


def test_quantize_embed_config_is_exported_from_public_packages():
    assert QuantizeEmbedConfig is QuantizationEmbedConfig


def test_normalize_embed_quant_config_accepts_config_and_compatibility_mode():
    config = QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.INPUT, embed_only=False)
    assert BaseQModel._normalize_embed_quant_config(config, None) is config
    assert BaseQModel._normalize_embed_quant_config(QuantizeEmbed.BOTH, None) == QuantizeEmbedConfig(
        embed_quant_mode=QuantizeEmbed.BOTH
    )
    assert BaseQModel._normalize_embed_quant_config(None, QuantizeEmbed.OUTPUT) == QuantizeEmbedConfig(
        embed_quant_mode=QuantizeEmbed.OUTPUT
    )


def test_normalize_embed_quant_config_rejects_conflicting_arguments():
    with pytest.raises(ValueError, match="Pass only one"):
        BaseQModel._normalize_embed_quant_config(
            QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.INPUT),
            QuantizeEmbed.OUTPUT,
        )


def test_requantize_forwards_canonical_embedding_config():
    model = _bare_model(quantized=True)
    captured = {}

    def quantize(**kwargs):
        captured.update(kwargs)
        return {"ok": []}

    model.quantize = quantize
    config = QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.BOTH, embed_only=False)
    assert model.requantize(calibration=["sample"], embed_quant_config=config) == {"ok": []}
    assert captured["embed_quant_config"] is config
    assert "embed_quant_mode" not in captured


def test_requantize_keeps_embedding_mode_compatibility():
    model = _bare_model(quantized=True)
    captured = {}
    model.quantize = lambda **kwargs: captured.update(kwargs)

    model.requantize(calibration=["sample"], embed_quant_mode=QuantizeEmbed.INPUT)

    assert captured["embed_quant_config"] == QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.INPUT)


def test_requantize_requires_an_embedding_target():
    model = _bare_model(quantized=True)
    with pytest.raises(ValueError, match="requires"):
        model.requantize(calibration=["sample"])
