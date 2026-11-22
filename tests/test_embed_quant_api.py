# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel import CalibrationConfig, NormConfig, QuantizeConfig, QuantizeEmbed, QuantizeEmbedConfig
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


@pytest.mark.parametrize("bad", ["inptu", "outputs", "Both ", "", None, 1, ["input"]])
def test_quantize_embed_config_rejects_unknown_mode(bad):
    """A mode that matches no consumer branch must not reach the looper (#5124)."""

    with pytest.raises(
        ValueError,
        match=r"`embed_quant_mode` must be one of 'input', 'output', 'both'",
    ):
        QuantizeEmbedConfig(embed_quant_mode=bad)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("input", QuantizeEmbed.INPUT),
        ("output", QuantizeEmbed.OUTPUT),
        ("both", QuantizeEmbed.BOTH),
        (QuantizeEmbed.INPUT, QuantizeEmbed.INPUT),
    ],
)
def test_quantize_embed_config_coerces_known_modes(value, expected):
    assert QuantizeEmbedConfig(embed_quant_mode=value).embed_quant_mode is expected


@pytest.mark.parametrize("bad", ["False", "True", 1, 0, None])
def test_quantize_embed_config_rejects_non_bool_embed_only(bad):
    """``bool("False")`` is True; #4668 rejects that coercion for config bools."""

    with pytest.raises(TypeError, match=r"`embed_only` must be a bool"):
        QuantizeEmbedConfig(embed_only=bad)


def _mutated_embed_config(**fields):
    """Build a valid config, then mutate it past ``__post_init__``."""

    config = QuantizeEmbedConfig()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def test_normalize_embed_quant_config_revalidates_a_mutated_instance():
    config = _mutated_embed_config(embed_quant_mode="inptu")
    with pytest.raises(ValueError, match=r"`embed_quant_mode` must be one of"):
        BaseQModel._normalize_embed_quant_config(config, None)


def test_requantize_rejects_a_mutated_embed_config_before_quantizing():
    model = _bare_model(quantized=True)
    model.quantize = lambda **_kwargs: pytest.fail("must reject before the layer replay")

    with pytest.raises(ValueError, match=r"`embed_quant_mode` must be one of"):
        model.requantize(
            CalibrationConfig(dataset=["sample"]),
            embed_quant_config=_mutated_embed_config(embed_quant_mode="inptu"),
        )


def test_requantize_rejects_a_mutated_embed_only_before_quantizing():
    model = _bare_model(quantized=True)
    model.quantize = lambda **_kwargs: pytest.fail("must reject before the layer replay")

    with pytest.raises(TypeError, match=r"`embed_only` must be a bool"):
        model.requantize(
            CalibrationConfig(dataset=["sample"]),
            embed_quant_config=_mutated_embed_config(embed_only="False"),
        )


def test_quantize_rejects_a_mutated_embed_config_before_any_layer_work():
    model = _bare_model(quantized=False)
    model.quantize_config = QuantizeConfig(bits=4, group_size=128, device="cpu")
    model.prepare_dataset = lambda **_kwargs: pytest.fail("must reject before dataset preparation")
    model._quantize_with_calibration = lambda **_kwargs: pytest.fail("must reject before replay")

    with pytest.raises(ValueError, match=r"`embed_quant_mode` must be one of"):
        model.quantize(
            CalibrationConfig(dataset=["sample"]),
            embed_quant_config=_mutated_embed_config(embed_quant_mode="inptu"),
        )


def _cached_gptq_checkpoint():
    """Local snapshot of the cached GPTQ checkpoint, or ``None``."""

    from huggingface_hub.constants import HF_HUB_CACHE

    root = (
        Path(HF_HUB_CACHE)
        / "models--Qwen--Qwen2.5-0.5B-Instruct-GPTQ-Int4"
        / "snapshots"
    )
    if not root.is_dir():
        return None
    snapshots = sorted(path for path in root.iterdir() if (path / "config.json").is_file())
    return str(snapshots[-1]) if snapshots else None


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the real-checkpoint requantize entry point",
)
def test_requantize_rejects_a_bad_mode_before_replaying_a_real_checkpoint():
    """The real entry point must fail before ``quantize()`` replays any layer."""

    checkpoint = _cached_gptq_checkpoint()
    if checkpoint is None:
        pytest.skip("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 is not in the local HF cache")

    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND

    model = GPTQModel.load(checkpoint, device_map={"": "cuda:0"}, backend=BACKEND.AUTO)
    model.quantize = lambda **_kwargs: pytest.fail("must reject before the layer replay")
    embed_name = model.get_input_embeddings_name()
    before = type(dict(model.model.named_modules())[embed_name])
    try:
        with pytest.raises(ValueError, match=r"`embed_quant_mode` must be one of"):
            model.requantize(
                CalibrationConfig(dataset=["The capital of France is Paris."]),
                embed_quant_config=_mutated_embed_config(embed_quant_mode="inptu"),
            )
        assert type(dict(model.model.named_modules())[embed_name]) is before
    finally:
        del model


def test_requantize_forwards_canonical_embedding_config():
    model = _bare_model(quantized=True)
    captured = {}

    def quantize(**kwargs):
        captured.update(kwargs)
        return {"ok": []}

    model.quantize = quantize
    config = QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.BOTH, embed_only=False)
    assert model.requantize(CalibrationConfig(dataset=["sample"]), embed_quant_config=config) == {
        "ok": []
    }
    assert captured["embed_quant_config"] is config
    assert "embed_quant_mode" not in captured
    assert model._module_replacement_prefixes == {"model.embed_tokens", "lm_head"}


def test_requantize_keeps_embedding_mode_compatibility():
    model = _bare_model(quantized=True)
    captured = {}
    model.quantize = lambda **kwargs: captured.update(kwargs)

    model.requantize(CalibrationConfig(dataset=["sample"]), embed_quant_mode=QuantizeEmbed.INPUT)

    assert captured["embed_quant_config"] == QuantizeEmbedConfig(
        embed_quant_mode=QuantizeEmbed.INPUT
    )


def test_requantize_requires_an_embedding_target():
    model = _bare_model(quantized=True)
    with pytest.raises(ValueError, match="requires"):
        model.requantize(CalibrationConfig(dataset=["sample"]))


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
        CalibrationConfig(
            dataset=["sample"],
        ),
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
        model.requantize(CalibrationConfig(dataset=["unused"]), **kwargs)
    assert not getattr(model, "_module_replacement_prefixes", set())
