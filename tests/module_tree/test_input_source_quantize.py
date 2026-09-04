# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import QuantizeConfig


def _write_tiny_llama(model_dir):
    texts = [
        "tiny input source validation sample one with enough tokens for calibration data",
        "tiny input source validation sample two with enough tokens for calibration data",
        "tiny input source validation sample three with enough tokens for calibration data",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        texts,
        trainer=WordLevelTrainer(
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        ),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
            use_cache=False,
            attn_implementation="eager",
        )
    )
    model.save_pretrained(model_dir)
    return texts


@pytest.mark.slow
def test_cpu_quantize_validates_input_sources_and_logs_summary(tmp_path):
    model_dir = tmp_path / "native"
    model_dir.mkdir()
    texts = _write_tiny_llama(model_dir)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=16,
        validate_input_sources=True,
        device="cpu",
    )
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
    )
    import gptqmodel.looper.module_looper as module_looper

    logger = module_looper.log
    logger_info = Mock(wraps=logger.info)
    logger_spy = Mock(wraps=logger)
    logger_spy.info = logger_info
    with patch.object(module_looper, "log", logger_spy):
        model.quantize(
            calibration=texts,
            batch_size=1,
            backend=BACKEND.TORCH,
            calibration_data_min_length=1,
        )
    assert any(
        call.args and str(call.args[0]).startswith("Input-source validation:")
        for call in logger_info.call_args_list
    )
