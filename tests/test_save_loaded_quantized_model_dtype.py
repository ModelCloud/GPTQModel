# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Re-saving a loaded quantized model must preserve non-quantized BF16 tensors."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.hf import get_hf_config_dtype


def _bf16_bits(tensor):
    return tensor.contiguous().view(torch.uint16).cpu()


def _save_tokenizer(directory):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    tokenizer.train_from_iterator(
        ["tiny calibration text", "another dtype round trip"],
        trainer=trainers.WordLevelTrainer(
            special_tokens=specials,
            vocab_size=64,
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(directory)


def test_loaded_quantized_resave_preserves_bf16_norm_bits(tmp_path):
    source_dir = tmp_path / "source"
    quantized_dir = tmp_path / "quantized"
    resaved_dir = tmp_path / "resaved"
    source_dir.mkdir()

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        dtype=torch.bfloat16,
    )
    source = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        source.model.norm.weight.fill_(70144.0)
        source.model.layers[0].input_layernorm.weight.fill_(9.98e-7)
    source.save_pretrained(source_dir)
    _save_tokenizer(source_dir)

    expected = {
        "norm": _bf16_bits(source.model.norm.weight),
        "input_layernorm": _bf16_bits(source.model.layers[0].input_layernorm.weight),
    }

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
    )
    model = GPTQModel.load(
        str(source_dir),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
    )
    model.quantize(
        [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
                "attention_mask": torch.ones((1, 8), dtype=torch.long),
            }
        ],
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
        calibration_data_min_length=1,
    )
    model.save(str(quantized_dir))
    del model

    loaded = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
    )
    loaded.save(str(resaved_dir))
    del loaded

    resaved = GPTQModel.load(
        str(resaved_dir),
        backend=BACKEND.TORCH,
        device="cpu",
    )
    assert get_hf_config_dtype(resaved.model.config) is torch.bfloat16
    assert resaved.model.model.norm.weight.dtype is torch.bfloat16
    assert (
        resaved.model.model.layers[0].input_layernorm.weight.dtype
        is torch.bfloat16
    )
    assert torch.equal(
        _bf16_bits(resaved.model.model.norm.weight),
        expected["norm"],
    )
    assert torch.equal(
        _bf16_bits(resaved.model.model.layers[0].input_layernorm.weight),
        expected["input_layernorm"],
    )
