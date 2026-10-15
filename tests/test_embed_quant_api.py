# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from safetensors.torch import save_file
from torch import nn

from gptqmodel import NormConfig, QuantizeConfig, QuantizeEmbed, QuantizeEmbedConfig
from gptqmodel.models.base import BaseQModel, _QuantizedCheckpointSource
from gptqmodel.quantization import FORMAT
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
    assert model._module_replacement_prefixes == {"model.embed_tokens", "lm_head"}


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
    assert model._module_replacement_prefixes == {"lm_head"}
    assert model._model_free_weight_only_replacements_only is True
    assert isinstance(model._module_replacement_source, _QuantizedCheckpointSource)
    assert model._module_replacement_source._weight_map == {
        "lm_head.qweight": "model.safetensors"
    }


@pytest.mark.parametrize("sharded", [False, True])
def test_checkpoint_source_uses_loader_selection_over_canonical_files(tmp_path, sharded):
    canonical = "model.safetensors"
    legacy = "gptq_model-4bit-16g.safetensors"
    save_file({"wrong.weight": nn.Parameter().new_zeros(2)}, str(tmp_path / canonical))
    (tmp_path / f"{canonical}.index.json").write_text(json.dumps({"weight_map": {"wrong.weight": canonical}}))
    save_file({"norm.weight": nn.Parameter().new_ones(2)}, str(tmp_path / legacy))
    selected = tmp_path / legacy
    if sharded:
        selected = tmp_path / f"{legacy}.index.json"
        selected.write_text(json.dumps({"weight_map": {"norm.weight": legacy}}))
    source = _QuantizedCheckpointSource(str(tmp_path), checkpoint_file_name=str(selected))
    assert source._weight_map == {"norm.weight": legacy}
    assert source._index_filename == (selected.name if sharded else None)


@pytest.mark.parametrize("operation", ["norm", "embedding", "combined"])
@pytest.mark.parametrize("broken", ["selected-file-missing", "shard-missing", "invalid-index"])
def test_requantize_checks_persistence_before_dispatch(tmp_path, operation, broken):
    model = _bare_model(quantized=True)
    model.load_quantized_model = True
    model.model_local_path = str(tmp_path)
    model.model = nn.Module()
    model.turtle_model = None
    model.quantize_config = QuantizeConfig(format=FORMAT.GPTQ_V2, offload_to_disk=False)
    selected = tmp_path / "gptq_model-4bit-16g.safetensors.index.json"
    model.model.checkpoint_file_name = str(selected)
    if broken == "shard-missing":
        selected.write_text(json.dumps({"weight_map": {"norm.weight": "missing.safetensors"}}))
    elif broken == "invalid-index":
        selected.write_text('{"weight_map": {}}')
    # A different valid checkpoint must not hide a missing selected source.
    save_file({"norm.weight": nn.Parameter().new_ones(2)}, str(tmp_path / "model.safetensors"))
    model._module_replacement_source = _QuantizedCheckpointSource(str(tmp_path))
    model._requantize_norm = lambda **kwargs: pytest.fail("must fail before tuning")
    model._requantize_embeddings = lambda **kwargs: pytest.fail("must fail before embedding packing")
    kwargs = {}
    if operation != "embedding":
        kwargs["norm"] = NormConfig(teacher="must-not-load")
    if operation != "norm":
        kwargs["embed_quant_mode"] = QuantizeEmbed.OUTPUT
    with pytest.raises(ValueError if broken == "invalid-index" else FileNotFoundError):
        model.requantize(calibration=["unused"], **kwargs)
    assert not getattr(model, "_module_replacement_prefixes", set())
