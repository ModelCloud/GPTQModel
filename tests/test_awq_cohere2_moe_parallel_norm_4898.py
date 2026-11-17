# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Cohere2-MoE's parallel norm group must keep AWQ, router consumer included (#4898).

Cohere2-MoE reads one ``input_layernorm`` from two parallel branches
(``self_attn`` and the MoE block) and the MoE router consumes the same normed
input. #4772 let the group fold a member that lives outside the replay
boundary, but the router is a *scale consumer*, and the containment check in
``_validate_scale_group`` still required every consumer to be inside
``self_attn``. The group was rejected for every layer, so an "AWQ"
Cohere2-MoE checkpoint had plain-quantized q/k/v and plain-quantized gate/up
for every routed and shared expert.

The fold is output-preserving: the router reads the same tensor, so multiplying
its weight columns by the folded scale is exact. The consumer stays in the
``apply_scale`` consumer map even when it cannot be replayed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    Cohere2MoeConfig,
    Cohere2MoeForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import CalibrationConfig, QuantizeConfig


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytestmark = [pytest.mark.cpu]


_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a second calibration sample with enough tokens to survive the minimum length filter",
    "yet another synthetic calibration sentence so every expert receives some tokens",
    "final calibration row with repeated words words words to exercise the router paths",
    "one more sample sentence about nothing in particular but long enough to be kept",
    "a closing calibration example that mentions rivers mountains and distant cities",
    "a seventh row so the routed experts see enough tokens to be scored reliably",
    "the last calibration sentence keeps the batch count even for the replay path",
]


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return fast_tokenizer


def _build_tiny_cohere2_moe(model_dir: Path):
    config = Cohere2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    model = Cohere2MoeForCausalLM(config)
    model.save_pretrained(model_dir)
    tokenizer = _build_local_tokenizer(model_dir)
    return config, tokenizer


def _calibration_rows(tokenizer) -> list[dict]:
    rows = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
    return rows


def test_cohere2_moe_parallel_norm_group_keeps_awq(tmp_path: Path, caplog):
    model_dir = tmp_path / "cohere2_moe"
    config, tokenizer = _build_tiny_cohere2_moe(model_dir)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
    )
    model = GPTQModel.load(
        str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH
    )
    with caplog.at_level(logging.WARNING):
        model.quantize(
            CalibrationConfig(
                dataset=_calibration_rows(tokenizer), batch_size=1, min_length=1
            ),
            backend=BACKEND.TORCH,
        )

    messages = [record.getMessage() for record in caplog.records]
    fallbacks = [
        message for message in messages if "plain-quantization fallback" in message
    ]
    assert not fallbacks, fallbacks

    rows = {(int(row["layer"]), str(row["module"])): row for row in model.quant_log}
    for layer in range(config.num_hidden_layers):
        for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            key = (layer, module)
            assert key in rows, sorted(rows)
            assert rows[key]["loss"] != "not applicable", rows[key]

        expert_gate_up = [
            row
            for (row_layer, module), row in rows.items()
            if row_layer == layer
            and module.startswith("mlp.experts.")
            and module.endswith(("gate_proj", "up_proj"))
        ]
        assert expert_gate_up, sorted(rows)
        assert any(
            row["loss"] != "not applicable" for row in expert_gate_up
        ), expert_gate_up
