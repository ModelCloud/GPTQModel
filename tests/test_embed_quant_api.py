# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch import nn
from safetensors.torch import save_file

from gptqmodel import QuantizeEmbed, QuantizeEmbedConfig
from gptqmodel.models.base import BaseQModel, _QuantizedCheckpointSource
from gptqmodel.quantization import QuantizeEmbedConfig as QuantizationEmbedConfig


def _bare_model(*, quantized: bool):
    model = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(model)
    model.quantized = quantized
    model.load_quantized_model = False
    model.get_input_embeddings_name = lambda: "model.embed_tokens"
    model.get_output_embeddings_name = lambda: "lm_head"
    return model


def test_quantize_embed_config_is_exported_from_public_packages():
    assert QuantizeEmbedConfig is QuantizationEmbedConfig


def test_normalize_embed_quant_config_accepts_config_and_compatibility_mode():
    config = QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.INPUT, embed_only=False)
    assert BaseQModel._normalize_embed_quant_config(config, None) is config
    assert BaseQModel._normalize_embed_quant_config(
        QuantizeEmbed.BOTH, None
    ) == QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.BOTH)
    assert BaseQModel._normalize_embed_quant_config(
        None, QuantizeEmbed.OUTPUT
    ) == QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.OUTPUT)


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
    assert model.requantize(calibration=["sample"], embed_quant_config=config) == {
        "ok": []
    }
    assert captured["embed_quant_config"] is config
    assert "embed_quant_mode" not in captured
    assert model._embedding_replacement_prefixes == {"model.embed_tokens", "lm_head"}


def test_requantize_keeps_embedding_mode_compatibility():
    model = _bare_model(quantized=True)
    captured = {}
    model.quantize = lambda **kwargs: captured.update(kwargs)

    model.requantize(calibration=["sample"], embed_quant_mode=QuantizeEmbed.INPUT)

    assert captured["embed_quant_config"] == QuantizeEmbedConfig(
        embed_quant_mode=QuantizeEmbed.INPUT
    )


def test_requantize_requires_an_embedding_target():
    model = _bare_model(quantized=True)
    with pytest.raises(ValueError, match="requires"):
        model.requantize(calibration=["sample"])


def test_quantized_checkpoint_source_reads_single_safetensors_weight_map(tmp_path):
    save_file(
        {
            "model.embed_tokens.qweight": nn.Parameter().new_zeros((2, 2)),
            "model.layers.0.qweight": nn.Parameter().new_zeros((2, 2)),
        },
        str(tmp_path / "model.safetensors"),
    )

    source = _QuantizedCheckpointSource(str(tmp_path))

    assert source.model_local_path == str(tmp_path)
    assert source._weight_map == {
        "model.embed_tokens.qweight": "model.safetensors",
        "model.layers.0.qweight": "model.safetensors",
    }


def test_requantize_loaded_checkpoint_configures_embedding_only_save_source(tmp_path):
    save_file(
        {"lm_head.qweight": nn.Parameter().new_zeros((2, 2))},
        str(tmp_path / "model.safetensors"),
    )
    model = _bare_model(quantized=True)
    model.load_quantized_model = True
    model.model_local_path = str(tmp_path)
    model.turtle_model = None
    model.quantize = lambda **_kwargs: {"ok": []}

    result = model.requantize(
        calibration=["sample"],
        embed_quant_mode=QuantizeEmbed.OUTPUT,
    )

    assert result == {"ok": []}
    assert model._embedding_replacement_prefixes == {"lm_head"}
    assert model._model_free_weight_only_embeddings_only is True
    assert isinstance(model._embedding_replacement_source, _QuantizedCheckpointSource)
    assert model._embedding_replacement_source._weight_map == {
        "lm_head.qweight": "model.safetensors"
    }
