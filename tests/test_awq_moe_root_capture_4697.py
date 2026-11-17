# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4697: AWQ must capture a declared MoE root without hooks.

Definitions such as Granite-MoE and gpt-oss declare a ``:moe`` root but have no
``moe_lifecycle_hooks``. The AWQ gate/up scaling group reads the root's
pre-router input, so a missing capture hook left a ``(0,)`` feature and the
whole group was dropped without a log line. These fixtures run a real
``quantize()`` and assert the group was applied to every routed expert.
"""

import logging

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import FORMAT, AWQConfig


CALIBRATION_ROWS = 32
CALIBRATION_LENGTH = 16
TOTAL_TOKENS = CALIBRATION_ROWS * CALIBRATION_LENGTH


def _write_tokenizer(model_dir) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    texts = ["the quick brown fox jumps over the lazy dog"] * CALIBRATION_ROWS
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        texts,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(model_dir)


def _tiny_config(model_type: str):
    import transformers

    common = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
    }
    if model_type == "granitemoe":
        config_cls = getattr(transformers, "GraniteMoeConfig", None)
        if config_cls is None:
            pytest.skip("transformers does not provide GraniteMoeConfig")
        return config_cls(
            **common,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
    if model_type == "gpt_oss":
        config_cls = getattr(transformers, "GptOssConfig", None)
        if config_cls is None:
            pytest.skip("transformers does not provide GptOssConfig")
        return config_cls(
            **common,
            head_dim=16,
            num_local_experts=8,
            num_experts_per_tok=2,
            layer_types=["full_attention", "full_attention"],
            sliding_window=128,
        )
    raise AssertionError(f"unknown fixture {model_type}")


def _build_tiny_model(model_type: str, model_dir):
    from transformers import AutoModelForCausalLM

    config = _tiny_config(model_type)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    _write_tokenizer(model_dir)
    return config


def _calibration():
    generator = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(CALIBRATION_ROWS):
        input_ids = torch.randint(5, 128, (1, CALIBRATION_LENGTH), generator=generator)
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return rows


@pytest.mark.parametrize(
    ("model_type", "moe_root", "num_experts"),
    [
        ("granitemoe", "block_sparse_moe", 8),
        ("gpt_oss", "mlp", 8),
    ],
)
def test_declared_moe_root_capture_drives_gate_up_group(
    tmp_path, model_type, moe_root, num_experts
):
    model_dir = tmp_path / model_type
    config = _build_tiny_model(model_type, model_dir)

    quantize_config = AWQConfig(
        bits=4,
        group_size=32,
        format=FORMAT.GEMM,
        device="cpu",
    )
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        dtype=torch.float16,
    )
    model.quantize(_calibration(), batch_size=4, calibration_data_min_length=1)

    rows = {
        (int(row["layer"]), str(row["module"])): row
        for row in model.quant_log
    }
    for layer_index in range(config.num_hidden_layers):
        for expert_index in range(num_experts):
            for projection in ("gate_proj", "up_proj"):
                module_name = f"{moe_root}.experts.{expert_index}.{projection}"
                row = rows[(layer_index, module_name)]
                # The group exists only when the declared root feature was
                # captured. A missing hook leaves ``scale_feature`` pointing at
                # the module itself and drops the group.
                assert row["scale_feature"] == moe_root
                assert row["scale_aggregation"] == "token_rows"
                assert int(row["scale_raw_tokens"]) == TOTAL_TOKENS


def test_empty_declared_root_warns_instead_of_silent_drop(tmp_path, caplog):
    model_dir = tmp_path / "granitemoe"
    _build_tiny_model("granitemoe", model_dir)
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(
            bits=4,
            group_size=32,
            format=FORMAT.GEMM,
            device="cpu",
        ),
        dtype=torch.float16,
    )
    layer = model.model.model.layers[0]
    input_feat = {
        f"block_sparse_moe.experts.{expert}.{projection}": torch.randn(1, 4, 64)
        for expert in range(8)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    # Simulate a missing capture hook: the declared root feature stays empty
    # while every routed projection still has its own (non-empty) capture.
    input_feat["block_sparse_moe"] = torch.empty(0)

    with caplog.at_level(logging.WARNING):
        nodes = model.awq_get_modules_for_scaling(layer, input_feat, {})

    assert "declared MoE root `block_sparse_moe` captured no activations" in caplog.text
    grouped = {id(module) for node in nodes for module in node["layers"]}
    for expert in range(8):
        for projection in ("gate_proj", "up_proj"):
            module = layer.get_submodule(
                f"block_sparse_moe.experts.{expert}.{projection}"
            )
            assert id(module) not in grouped
