# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""A hybrid MoE must keep its dense-prefix norm -> gate/up AWQ group (#4903).

ERNIE-4.5-MoE mixes dense decoder layers (before ``moe_layer_start_index``)
with MoE layers, and declares no routing lifecycle hooks. The AWQ root policy
seeds a capture task for the declared MoE root (``mlp``) on *every* layer, but
the root hook only runs on a layer that executes an expert. On a dense layer
the seeded ``mlp`` feature therefore exists with zero elements, and
``awq_get_modules_for_scaling`` preferred it over the dense block's own
captured feature. The ``post_attention_layernorm -> [mlp.gate_proj,
mlp.up_proj]`` group was then dropped without a log line, so those projections
were quantized with plain RTN inside an "AWQ" checkpoint.

The same mechanism is why #4697 (a definition without lifecycle hooks silently
losing the *routed-expert* group) mattered: the root capture now comes from the
module plan (#4723). This file guards both: the dense prefix keeps its own
group, and a layer where every expert is routed keeps AWQ for all of them.
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
    Ernie4_5_MoeConfig,
    Ernie4_5_MoeForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

pytestmark = [pytest.mark.cpu]


_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a second calibration sample with enough tokens to survive the minimum length filter",
    "yet another synthetic calibration sentence so every expert receives some tokens",
    "final calibration row with repeated words words words to exercise the router paths",
    "one more sample sentence about nothing in particular but long enough to be kept",
    "a closing calibration example that mentions rivers mountains and distant cities",
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


def _build_tiny_ernie(
    model_dir: Path,
    *,
    num_layers: int,
    moe_layer_start_index: int,
    num_experts: int,
    experts_per_tok: int,
):
    """A tiny ``ernie4_5_moe``: dense layers first, then MoE layers.

    ``ernie4_5_moe`` is one of the definitions with a declared MoE root but no
    ``moe_lifecycle_hooks``, so the root capture has to come from the module
    plan (#4697). ``moe_layer_start_index`` creates the dense prefix that used
    to lose its own group (#4903).
    """

    config = Ernie4_5_MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        moe_num_experts=num_experts,
        moe_k=experts_per_tok,
        moe_num_shared_experts=1,
        moe_layer_start_index=moe_layer_start_index,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(11)
    model = Ernie4_5_MoeForCausalLM(config)
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


def _quantize(model_dir: Path, rows: list[dict]):
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
    model.quantize(
        CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    return model


def _rows_by_layer_module(model) -> dict[tuple[int, str], dict]:
    return {
        (int(row["layer"]), str(row["module"])): row
        for row in model.quant_log
    }


def test_dense_prefix_layer_keeps_its_norm_gate_up_group(tmp_path: Path, caplog):
    """Layer 0 is dense: its ``mlp.gate_proj/up_proj`` must keep an AWQ scale."""

    model_dir = tmp_path / "ernie_dense_prefix"
    config, tokenizer = _build_tiny_ernie(
        model_dir,
        num_layers=2,
        moe_layer_start_index=1,
        num_experts=4,
        experts_per_tok=2,
    )
    assert config.moe_layer_start_index == 1

    with caplog.at_level(logging.WARNING):
        model = _quantize(model_dir, _calibration_rows(tokenizer))

    rows = _rows_by_layer_module(model)
    for suffix in ("gate_proj", "up_proj"):
        key = (0, f"mlp.{suffix}")
        assert key in rows, sorted(rows)
        # "not applicable" is AWQ's marker for a plain-quantized (unscaled)
        # module: the group was dropped before the scale search ran.
        assert rows[key]["loss"] != "not applicable", rows[key]

    # The dense layer's group is real, not a leftover from the MoE layer.
    assert (1, "mlp.experts.0.gate_proj") in rows or (
        1,
        "mlp.shared_experts.gate_proj",
    ) in rows, sorted(rows)

    messages = [record.getMessage() for record in caplog.records]
    assert not [
        message for message in messages if "captured no activations" in message
    ], messages
    assert not [
        message for message in messages if "plain-quantization fallback" in message
    ], messages


def test_moe_layer_keeps_awq_without_lifecycle_hooks(tmp_path: Path, caplog):
    """#4697: the declared MoE root is captured from the module plan.

    Two of the four experts are never routed, so the group has to drop exactly
    those members (#4720/#4760) and keep AWQ for everyone else. Before #4723
    the root feature stayed empty, the group anchored on the shared expert (a
    member of the group), and the shared expert fell back to plain
    quantization along with the starved experts. Only the MoE root capture
    makes the search input a non-member again.
    """

    model_dir = tmp_path / "ernie_moe_layer"
    config, tokenizer = _build_tiny_ernie(
        model_dir,
        num_layers=1,
        moe_layer_start_index=0,
        num_experts=4,
        experts_per_tok=2,
    )
    assert config.moe_layer_start_index == 0

    with caplog.at_level(logging.WARNING):
        model = _quantize(model_dir, _calibration_rows(tokenizer))

    rows = _rows_by_layer_module(model)
    for suffix in ("gate_proj", "up_proj"):
        key = (0, f"mlp.shared_experts.{suffix}")
        assert key in rows, sorted(rows)
        assert rows[key]["loss"] != "not applicable", rows[key]

    routed = [
        expert
        for expert in range(config.moe_num_experts)
        if rows[(0, f"mlp.experts.{expert}.gate_proj")]["loss"] != "not applicable"
    ]
    assert routed, "no routed expert kept an AWQ scale"

    messages = [record.getMessage() for record in caplog.records]
    assert not [
        message for message in messages if "captured no activations" in message
    ], messages
    assert not [
        message for message in messages if "plain-quantization fallback" in message
    ], messages
