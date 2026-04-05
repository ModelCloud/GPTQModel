# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tiny end-to-end MoE quantization smoke coverage.

This test intentionally avoids the large on-disk MoE fixtures used elsewhere in
the suite. It builds a one-layer Qwen3 MoE model and a tiny local tokenizer in
temporary directories, then runs the real GPTQ save/load/quantize/save/reload
flow against that fixture.

The goal is not kernel benchmarking or quality evaluation. The goal is a cheap
regression guard for the MoE lifecycle:
1. native HF MoE model can be loaded through GPT-QModel
2. MoE routing override can drive expert quantization
3. the quantized checkpoint can be reloaded
4. every expert gate/up/down projection is exported as a quantized linear
"""

import os
from pathlib import Path


# Keep the smoke test on CPU so it stays small and works on CPU-only runners.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


_CALIBRATION_TEXTS = [
    "tiny moe calibration sample one with enough tokens to survive minimum length filtering and trigger expert routing",
    "tiny moe calibration sample two with repeated expert words to exercise the moe quantization smoke path cleanly",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    """Persist a minimal tokenizer because the GPT-QModel `load()` path expects local tokenizer files."""

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
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


def _build_tiny_qwen3_moe_fixture(model_dir: Path) -> tuple[Qwen3MoeConfig, PreTrainedTokenizerFast]:
    """Save a tiny native HF MoE checkpoint that still exercises the real qwen3_moe path.

    Qwen3Moe is the lightest native MoE family available in this repo's test
    dependencies because it does not carry Qwen2 MoE's large shared-expert
    default weights.
    """

    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(model_dir)
    tokenizer = _build_local_tokenizer(model_dir)
    return config, tokenizer


def _build_calibration_dataset(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    """Return the exact calibration shape accepted by prepare_calibration_dataset()."""

    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return dataset


def test_tiny_qwen3_moe_quantization_smoke(tmp_path: Path):
    """Quantize and reload a tiny local MoE model, then assert all expert projections are quantized."""

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"

    config, tokenizer = _build_tiny_qwen3_moe_fixture(model_dir)
    calibration = _build_calibration_dataset(tokenizer)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(quantized_dir)

    quantized_model = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
    )

    # Assert the full expert set was quantized, not just whichever experts the
    # natural router happened to hit in this tiny calibration sample.
    modules = dict(quantized_model.named_modules())
    expected_quantized = config.num_experts * 3
    quantized_expert_modules = []

    for expert_index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            module_name = f"model.model.layers.0.mlp.experts.{expert_index}.{suffix}"
            module = modules[module_name]
            assert isinstance(module, TorchLinear), module_name
            quantized_expert_modules.append(module_name)

    assert len(quantized_expert_modules) == expected_quantized
    assert quantized_model.quantize_config.meta_get("moe")["routing"]["class"] == "ExpertsRoutingOverride"
