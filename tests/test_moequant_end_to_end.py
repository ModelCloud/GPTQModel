# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Tiny Qwen3-MoE end-to-end coverage for MoEQuant AWQ-EBSS."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models.ports.native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from gptqmodel.models.ports.native.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import (
    CalibrationConfig,
    EBSSConfig,
    MoEQuantConfig,
)


pytestmark = [pytest.mark.cpu]


_TEXTS = [
    "tiny moe calibration sample one with enough tokens to survive minimum length filtering and trigger expert routing",
    "tiny moe calibration sample two with repeated expert words to exercise the moe quantization smoke path cleanly",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2


def _build_tiny_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3MoeForCausalLM(config).save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _TEXTS,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return fast_tokenizer


def _calibration(tokenizer: PreTrainedTokenizerFast):
    return [
        {
            "input_ids": tokenizer(text, return_tensors="pt")["input_ids"],
            "attention_mask": tokenizer(text, return_tensors="pt")["attention_mask"],
        }
        for text in _TEXTS
    ]


def _assert_generates(model: GPTQModel) -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=2,
        do_sample=False,
    )
    assert output.shape == (1, input_ids.shape[1] + 2)
    assert torch.isfinite(output.float()).all()


@pytest.mark.slow
def test_awq_ebss_tiny_moe_roundtrip(tmp_path: Path):
    model_dir = tmp_path / "native"
    _build_tiny_fixture(model_dir)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
        offload_to_disk=False,
        moequant=MoEQuantConfig(
            agq=False,
            ebss=EBSSConfig(num_sentences=2, seq_len=4, width=2, seed=0),
        ),
    )
    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(
        CalibrationConfig(dataset=None, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert model.quantized
    assert model.quantize_config.moequant.ebss is not None

    with pytest.raises(ValueError, match="requantize"):
        model.requantize(
            CalibrationConfig(dataset=None, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )

    saved = tmp_path / "ebss"
    model.save(str(saved))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu")
    assert reloaded.quantize_config.moequant.ebss.seed == 0
    _assert_generates(reloaded)


def test_ebss_rejects_caller_dataset(tmp_path: Path):
    model_dir = tmp_path / "native"
    tokenizer = _build_tiny_fixture(model_dir)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        moequant=MoEQuantConfig(
            agq=False,
            ebss=EBSSConfig(num_sentences=1, seq_len=2, width=1),
        ),
    )
    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    with pytest.raises(ValueError, match="do not pass"):
        model.quantize(
            CalibrationConfig(
                dataset=_calibration(tokenizer), batch_size=1, min_length=1
            ),
            backend=BACKEND.TORCH,
        )


def test_ebss_rejects_lazy_checkpoint_shell(tmp_path: Path):
    model_dir = tmp_path / "native"
    _build_tiny_fixture(model_dir)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        moequant=MoEQuantConfig(
            agq=False,
            ebss=EBSSConfig(num_sentences=1, seq_len=2, width=1),
        ),
    )
    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    with pytest.raises(ValueError, match="materialized"):
        model.quantize(
            CalibrationConfig(dataset=None, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )

