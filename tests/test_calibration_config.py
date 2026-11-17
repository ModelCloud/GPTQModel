# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`quantize(CalibrationConfig(...))` is the only calibration API.

Every calibration choice of a run lives in one frozen `CalibrationConfig`:
the samples, concatenation, sort order, minimum length, batch size, the
attention-sink row policy and the EoRA adapter samples. `quantize()`,
`requantize()` and `GPTQModel.adapter.generate()` take that object; the old
`calibration_*` keyword arguments and a bare dataset fail with `TypeError`.
"""

import dataclasses
import inspect

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import gptqmodel
from gptqmodel import CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS, GPTQConfig


OLD_QUANTIZE_KWARGS = {
    "calibration_concat_size": 64,
    "calibration_sort": "asc",
    "batch_size": 2,
    "calibration_data_min_length": 1,
    "calibration_concat_separator": "\n",
    "calibration_exclude_attention_sinks": True,
    "adapter_calibration_dataset": ["text"],
}


# --------------------------------------------------------------------------
# The dataclass
# --------------------------------------------------------------------------


def test_one_public_definition():
    from gptqmodel import quantization
    from gptqmodel.looper import loop_processor

    assert gptqmodel.CalibrationConfig is quantization.CalibrationConfig
    assert quantization.CalibrationConfig is loop_processor.CalibrationConfig
    assert CalibrationConfig.__module__ == "gptqmodel.quantization.config"


def test_defaults_match_the_previous_quantize_defaults():
    config = CalibrationConfig()
    assert config.dataset is None
    assert config.concat_size is None
    assert config.sort == "desc"
    assert config.batch_size == 1
    assert config.min_length == 10
    assert config.concat_separator is None
    assert config.exclude_attention_sinks is False
    assert config.adapter_dataset is None
    assert config.prepare_dataset is None


@pytest.mark.parametrize("sort", [None, "asc", "desc", "shuffle"])
def test_valid_sort_orders(sort):
    assert CalibrationConfig(sort=sort).sort == sort


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"sort": "descending"}, ValueError, "sort"),
        ({"sort": 1}, ValueError, "sort"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"batch_size": -1}, ValueError, "batch_size"),
        ({"batch_size": 2.0}, TypeError, "batch_size"),
        ({"batch_size": True}, TypeError, "batch_size"),
        ({"concat_size": 0}, ValueError, "concat_size"),
        ({"concat_size": "2048"}, TypeError, "concat_size"),
        ({"concat_size": False}, TypeError, "concat_size"),
        ({"min_length": -1}, ValueError, "min_length"),
        ({"min_length": 1.5}, TypeError, "min_length"),
        ({"concat_separator": 1}, TypeError, "concat_separator"),
        ({"exclude_attention_sinks": 1}, TypeError, "exclude_attention_sinks"),
        ({"exclude_attention_sinks": None}, TypeError, "exclude_attention_sinks"),
        ({"dataset": "one string"}, TypeError, "dataset"),
        ({"adapter_dataset": b"bytes"}, TypeError, "adapter_dataset"),
        ({"dataset": CalibrationConfig()}, TypeError, "dataset"),
    ],
)
def test_invalid_fields_are_rejected(kwargs, error, match):
    with pytest.raises(error, match=match):
        CalibrationConfig(**kwargs)


def test_min_length_zero_and_concat_size_none_are_valid():
    config = CalibrationConfig(min_length=0, concat_size=None)
    assert config.min_length == 0


def test_frozen_and_preparer_is_not_a_constructor_argument():
    config = CalibrationConfig(dataset=["a"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.batch_size = 4
    with pytest.raises(TypeError, match="prepare_dataset"):
        CalibrationConfig(dataset=["a"], prepare_dataset=lambda **kwargs: [])


def test_bind_returns_a_validated_copy():
    def prepare(**kwargs):
        return []

    config = CalibrationConfig(dataset=["a"], sort="asc")
    bound = config.bind(prepare, batch_size=4)
    assert bound is not config
    assert bound.prepare_dataset is prepare and config.prepare_dataset is None
    assert (bound.dataset, bound.sort, bound.batch_size) == (["a"], "asc", 4)
    with pytest.raises(ValueError, match="batch_size"):
        config.bind(prepare, batch_size=0)
    with pytest.raises(TypeError, match="callable"):
        config.bind(object())
    # A copy of a bound config drops the binding; only bind() sets it.
    assert dataclasses.replace(bound, sort="desc").prepare_dataset is None


# --------------------------------------------------------------------------
# The public signatures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [BaseQModel.quantize, BaseQModel.requantize, auto.GPTQModel.adapter.generate],
    ids=["quantize", "requantize", "adapter.generate"],
)
def test_signatures_drop_every_calibration_kwarg(fn):
    parameters = inspect.signature(fn).parameters
    assert "calibration" in parameters
    for name in (
        *OLD_QUANTIZE_KWARGS,
        "calibration_dataset",
        "calibration_dataset_concat_size",
        "calibration_dataset_sort",
    ):
        assert name not in parameters, name
    # Everything after the calibration config is keyword-only, so an old
    # positional call (`quantize(data, 2048, "desc")`) cannot bind.
    names = list(parameters)
    after = names[names.index("calibration") + 1:]
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in after)


def _bare_model(*, quantized=False):
    model = object.__new__(BaseQModel)
    model.quantize_config = GPTQConfig(bits=4, group_size=128)
    model.quantized = quantized
    return model


@pytest.mark.parametrize("dataset", [["text"], [{"input_ids": [1, 2, 3]}], ()])
def test_quantize_rejects_a_bare_dataset(dataset):
    with pytest.raises(TypeError, match="CalibrationConfig"):
        _bare_model().quantize(dataset)
    with pytest.raises(TypeError, match="CalibrationConfig"):
        _bare_model().quantize(calibration=dataset)


@pytest.mark.parametrize("name", sorted(OLD_QUANTIZE_KWARGS))
def test_quantize_rejects_every_old_kwarg(name):
    with pytest.raises(TypeError, match=name):
        _bare_model().quantize(CalibrationConfig(dataset=["text"]), **{name: OLD_QUANTIZE_KWARGS[name]})


def test_quantize_rejects_old_positional_arguments():
    with pytest.raises(TypeError, match="positional"):
        _bare_model().quantize(CalibrationConfig(dataset=["text"]), 2048)


def test_quantize_rejects_a_bound_config():
    bound = CalibrationConfig(dataset=["text"]).bind(lambda **kwargs: [])
    with pytest.raises(ValueError, match="unbound CalibrationConfig"):
        _bare_model().quantize(bound)


@pytest.mark.parametrize("name", sorted(OLD_QUANTIZE_KWARGS))
def test_requantize_rejects_a_bare_dataset_and_old_kwargs(name):
    model = _bare_model(quantized=True)
    with pytest.raises(TypeError, match="CalibrationConfig"):
        model.requantize(["text"], embed_quant_mode="output")
    with pytest.raises(TypeError, match=name):
        model.requantize(CalibrationConfig(dataset=["text"]), **{name: OLD_QUANTIZE_KWARGS[name]})


def test_adapter_generate_rejects_a_bare_dataset_and_old_kwargs(tmp_path):
    from gptqmodel.adapter.adapter import Lora

    lora = Lora(path=str(tmp_path / "adapter"), rank=4)
    with pytest.raises(TypeError, match="CalibrationConfig"):
        auto.GPTQModel.adapter.generate(lora, "native", "quantized", ["text"])
    with pytest.raises(TypeError, match="calibration_dataset"):
        auto.GPTQModel.adapter.generate(
            adapter=lora, model_id_or_path="native", quantized_model_id_or_path="quantized",
            calibration_dataset=["text"],
        )


# --------------------------------------------------------------------------
# End to end: the config drives the real quantize() pipeline
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory) -> str:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    path = tmp_path_factory.mktemp("tiny_llama_calibration_config")
    LlamaForCausalLM(config).save_pretrained(path)

    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    tok.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 20,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=128),
    )
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>").save_pretrained(path)
    return str(path)


def _texts(count=12):
    return [
        " ".join(["the quick brown fox jumps over the lazy dog"] * (1 + index % 3))
        for index in range(count)
    ]


def test_quantize_forwards_every_field_to_dataset_preparation(tiny_llama_dir, monkeypatch):
    """The processor prepares exactly the policy the caller put in the config."""

    seen = []
    original = BaseQModel.prepare_dataset

    def recording_prepare(self, **kwargs):
        seen.append({key: value for key, value in kwargs.items() if key != "calibration_dataset"})
        return original(self, **kwargs)

    monkeypatch.setattr(BaseQModel, "prepare_dataset", recording_prepare)
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=32), device="cpu")
    model.quantize(
        CalibrationConfig(
            dataset=_texts(),
            concat_size=24,
            sort="asc",
            batch_size=2,
            min_length=3,
            concat_separator=" ",
        )
    )

    assert model.quantized
    assert seen == [
        {
            "calibration_dataset_concat_size": 24,
            "calibration_dataset_sort": "asc",
            "batch_size": 2,
            "calibration_data_min_length": 3,
            "calibration_concat_separator": " ",
        }
    ]
    assert META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS not in (model.quantize_config.meta or {})


def test_quantize_end_to_end_with_sink_provenance(tiny_llama_dir, tmp_path):
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=32), device="cpu")
    model.quantize(CalibrationConfig(dataset=_texts(), batch_size=4, exclude_attention_sinks=True))
    assert model.quantized
    assert model.quantize_config.meta_get(META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS) is True

    model.save(str(tmp_path))
    reloaded = GPTQModel.load(str(tmp_path), device="cpu")
    assert reloaded.quantize_config.meta_get(META_FIELD_CALIBRATION_EXCLUDE_ATTENTION_SINKS) is True
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        assert torch.isfinite(reloaded.model(input_ids=ids).logits).all()


def test_adapter_dataset_requires_a_lora_adapter(tiny_llama_dir):
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=32), device="cpu")
    with pytest.raises(ValueError, match="adapter_dataset requires a `Lora` adapter"):
        model.quantize(CalibrationConfig(dataset=_texts(), adapter_dataset=_texts(4)))
    assert not model.quantized


def test_adapter_dataset_feeds_only_the_eora_pass(tiny_llama_dir, tmp_path, monkeypatch):
    from gptqmodel.adapter.adapter import Lora

    prepared = []
    original = BaseQModel.prepare_dataset

    def recording_prepare(self, **kwargs):
        prepared.append(list(kwargs["calibration_dataset"]))
        return original(self, **kwargs)

    monkeypatch.setattr(BaseQModel, "prepare_dataset", recording_prepare)
    main_texts = _texts(8)
    adapter_texts = [text.upper() for text in _texts(4)]
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=32), device="cpu")
    model.quantize(
        CalibrationConfig(dataset=main_texts, batch_size=4, adapter_dataset=adapter_texts),
        adapter=Lora(rank=4, path=str(tmp_path / "lora")),
    )

    assert model.quantized
    assert prepared == [main_texts, adapter_texts]


def test_weight_only_quantize_takes_no_calibration(tmp_path, tiny_llama_dir):
    from gptqmodel.quantization.config import RTNConfig

    model = GPTQModel.load(tiny_llama_dir, RTNConfig(bits=4, group_size=32), device="cpu")
    model.quantize()
    assert model.quantized

    calibrated = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=32), device="cpu")
    with pytest.raises(ValueError, match="CalibrationConfig.dataset is required"):
        calibrated.quantize()
