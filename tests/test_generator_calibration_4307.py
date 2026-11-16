# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Regression tests for #4307: a one-shot calibration iterable must survive the
native and EoRA preparation passes.

``quantize()`` builds a separate ``CalibrationConfig`` for the native and EoRA
steps and each ``LoopProcessor`` re-runs ``prepare_dataset``. A generator is
exhausted by the first pass, so the second one used to raise
``ValueError: Quantize: calibration dataset is empty.``
"""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.adapter.adapter import Lora
from gptqmodel.looper.loop_processor import _reusable_calibration_dataset
from gptqmodel.quantization.config import GPTAQConfig


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
    path = tmp_path_factory.mktemp("tiny_llama_4307")
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


def _texts():
    return [f"The quick brown fox number {i} jumps over the lazy dog." for i in range(16)]


def _generator():
    # A fresh generator per run: iterating the list would hide the bug.
    return (text for text in _texts())


def test_generator_calibration_with_gptaq(tiny_llama_dir):
    """GPTAQ runs NATIVE + EORA, so both processors prepare the dataset."""
    model = GPTQModel.load(
        tiny_llama_dir,
        QuantizeConfig(bits=4, group_size=128, gptaq=GPTAQConfig()),
    )

    model.quantize(_generator(), batch_size=4)

    assert model.quantized


def test_generator_calibration_with_lora_adapter(tiny_llama_dir, tmp_path):
    """A LoRA adapter adds the EoRA step, which needs the same calibration data."""
    lora_dir = tmp_path / "lora"
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=128))

    model.quantize(_generator(), batch_size=4, adapter=Lora(rank=4, path=str(lora_dir)))

    assert model.quantized
    # EoRA adapters are generated during quantize() and persisted by save().
    assert model.lora_results


def test_generator_adapter_calibration_dataset(tiny_llama_dir, tmp_path):
    """``adapter_calibration_dataset`` is shared the same way."""
    lora_dir = tmp_path / "lora"
    model = GPTQModel.load(tiny_llama_dir, QuantizeConfig(bits=4, group_size=128))

    model.quantize(
        _texts(),
        batch_size=4,
        adapter=Lora(rank=4, path=str(lora_dir)),
        adapter_calibration_dataset=(text for text in _texts()),
    )

    assert model.lora_results


def test_reusable_calibration_dataset_materializes_only_one_shot_inputs():
    texts = _texts()

    # A sized container is returned untouched: no unnecessary copy.
    assert _reusable_calibration_dataset(texts) is texts

    materialized = _reusable_calibration_dataset(text for text in texts)
    assert isinstance(materialized, list)
    assert materialized == texts

    assert _reusable_calibration_dataset(None) is None


def test_reusable_calibration_dataset_keeps_hf_iterable_dataset_lazy():
    datasets = pytest.importorskip("datasets")

    stream = datasets.IterableDataset.from_generator(
        lambda: ({"text": text} for text in ("a", "b"))
    )

    # Re-iterable streams stay lazy; only plain iterators are materialised.
    assert _reusable_calibration_dataset(stream) is stream
