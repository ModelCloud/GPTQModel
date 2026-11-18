# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fresh `quantize()` must not start embedding-endpoint work it cannot finish.

Issue #5111: on an unquantized (LazyTurtle) model, `QuantizeEmbed.INPUT/BOTH`
raised ``RuntimeError: Tensor.item() cannot be called on meta tensors`` inside
input capture, and ``OUTPUT`` on a tied model quantized every decoder layer
before failing to re-tie ``lm_head``, leaving ``model.quantized=True`` on an
unusable model. Both paths are now refused before calibration; requantize on a
loaded quantized checkpoint (including a tied one) keeps working (#3050/#5124).
"""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeEmbed
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig
from gptqmodel.utils.tied_weights import lm_head_quantization_is_tied


def _texts(count=8):
    return [
        " ".join(["the quick brown fox jumps over the lazy dog"] * (1 + index % 3))
        for index in range(count)
    ]


def _build_tiny_llama(root, *, tied: bool) -> None:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    LlamaForCausalLM(config).save_pretrained(root)

    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(specials)
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 20,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=128),
    )
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>").save_pretrained(root)


@pytest.fixture(scope="module")
def tiny_untied_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama_untied_5111")
    _build_tiny_llama(path, tied=False)
    return str(path)


@pytest.fixture(scope="module")
def tiny_tied_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama_tied_5111")
    _build_tiny_llama(path, tied=True)
    return str(path)


@pytest.fixture(scope="module")
def tied_quantized_dir(tiny_tied_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_llama_tied_q4_5111")
    model = _load(tiny_tied_dir)
    model.quantize(CalibrationConfig(dataset=_texts(), batch_size=2))
    model.save(str(out))
    return str(out)


def _load(path):
    # The bug is dtype- and device-independent; CPU float32 keeps the test
    # runnable without a GPU and without the packed-FP16 endpoint dtype mismatch
    # that affects a dense BF16 decoder.
    return GPTQModel.load(
        path,
        GPTQConfig(bits=4, group_size=32),
        device="cpu",
        dtype=torch.float32,
    )


def _quantized_module_names(model):
    return [
        name
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear)
    ]


def _forbid_calibration(monkeypatch):
    calls = []

    def cache_inputs(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("calibration capture must not start")

    monkeypatch.setattr(StageInputsCapture, "cache_inputs", cache_inputs)
    return calls


@pytest.mark.parametrize("mode", [QuantizeEmbed.INPUT, QuantizeEmbed.BOTH])
def test_fresh_input_endpoint_is_refused_before_calibration(
    tiny_untied_dir, monkeypatch, mode
):
    model = _load(tiny_untied_dir)
    calls = _forbid_calibration(monkeypatch)

    with pytest.raises(NotImplementedError, match="requantize"):
        model.quantize(
            CalibrationConfig(dataset=_texts(), batch_size=2),
            embed_quant_mode=mode,
        )

    assert calls == []
    assert model.quantized is False
    assert _quantized_module_names(model) == []


@pytest.mark.parametrize("mode", [QuantizeEmbed.OUTPUT, QuantizeEmbed.BOTH])
def test_fresh_output_on_tied_model_is_refused_before_calibration(
    tiny_tied_dir, monkeypatch, mode
):
    model = _load(tiny_tied_dir)
    assert lm_head_quantization_is_tied(model) is True
    calls = _forbid_calibration(monkeypatch)

    with pytest.raises(NotImplementedError, match="tied"):
        model.quantize(
            CalibrationConfig(dataset=_texts(), batch_size=2),
            embed_quant_mode=mode,
        )

    # The refusal is before the looper mutates the run config or replaces a
    # module: no layer was quantized and the model still reports itself dense.
    assert calls == []
    assert model.quantized is False
    assert _quantized_module_names(model) == []
    assert isinstance(model.get_input_embeddings(), nn.Embedding)
    assert isinstance(model.get_output_embeddings(), nn.Linear)
    assert model.quantize_config.dynamic is None


def test_fresh_output_on_untied_model_still_quantizes_and_forwards(tiny_untied_dir):
    model = _load(tiny_untied_dir)
    model.quantize(
        CalibrationConfig(dataset=_texts(), batch_size=2),
        embed_quant_mode=QuantizeEmbed.OUTPUT,
    )

    assert model.quantized is True
    assert isinstance(model.get_output_embeddings(), BaseQuantLinear)
    assert isinstance(model.get_input_embeddings(), nn.Embedding)

    with torch.inference_mode():
        out = model.model(input_ids=torch.tensor([[4, 5, 6, 7]]))
    assert out.logits.shape == (1, 4, 128)
    assert torch.isfinite(out.logits).all()


def test_requantize_both_on_tied_quantized_checkpoint_still_works(tied_quantized_dir):
    model = GPTQModel.load(tied_quantized_dir, device="cpu", dtype=torch.float32)
    assert model.quantized is True
    # The loaded checkpoint is the #3050/#5124 shape: decoder layers packed,
    # `lm_head` still tied to `embed_tokens`. The fresh-model guard must not
    # apply here.
    assert lm_head_quantization_is_tied(model) is True

    result = model.requantize(
        CalibrationConfig(dataset=_texts(), batch_size=2),
        embed_quant_mode=QuantizeEmbed.BOTH,
    )

    assert isinstance(result, dict)
    assert model.quantized is True
    assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
    assert isinstance(model.get_output_embeddings(), BaseQuantLinear)

    with torch.inference_mode():
        out = model.model(input_ids=torch.tensor([[4, 5, 6, 7]]))
    assert out.logits.shape == (1, 4, 128)
    assert torch.isfinite(out.logits).all()
