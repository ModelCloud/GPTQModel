# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-checkpoint coverage for AWQ variable-length MoE feature aggregation."""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.models.writer import QUANT_LOG_NSAMPLES
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_PATH = Path(
    os.environ.get(
        "GPTQMODEL_AWQ_MOE_TEST_MODEL",
        "/monster/data/model/Qwen3-30B-A3B-layers-1",
    )
)
CALIBRATION_LENGTHS = (17, 31, 53)


def _variable_length_calibration(tokenizer) -> list[dict[str, torch.Tensor]]:
    source = tokenizer(
        "AWQ mixture of experts calibration needs several differently sized token sequences. " * 32,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    assert source.numel() >= max(CALIBRATION_LENGTHS)

    return [
        {
            "input_ids": source[:length].clone().unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }
        for length in CALIBRATION_LENGTHS
    ]


def _log_by_module(model) -> dict[str, dict]:
    return {str(row["module"]): row for row in model.quant_log}


def test_qwen3_moe_awq_aggregates_variable_length_features_across_batches():
    if not MODEL_PATH.is_dir():
        pytest.skip(f"local Qwen3-MoE fixture is unavailable: {MODEL_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-checkpoint AWQ integration test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    calibration = _variable_length_calibration(tokenizer)
    raw_tokens = sum(CALIBRATION_LENGTHS)
    retained_tokens = max(CALIBRATION_LENGTHS)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
        sym=False,
        device="cuda:0",
        calibration_data_device="cuda:0",
        offload_to_disk=False,
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )
    model = GPTQModel.load(str(MODEL_PATH), quantize_config=quantize_config)
    assert model.__class__.awq_input_feature_aggregation("mlp") == {
        "mode": "token_rows",
        "capture_root": True,
    }

    model.quantize(
        calibration,
        batch_size=1,
        calibration_concat_size=0,
        calibration_sort=None,
        calibration_data_min_length=1,
    )
    rows = _log_by_module(model)

    expert_modules = tuple(
        f"mlp.experts.{expert_index}.{projection}"
        for expert_index in range(128)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    assert set(expert_modules).issubset(rows)

    for module_name in expert_modules:
        row = rows[module_name]
        assert row["activation_aggregation"] == "token_rows"
        assert int(row["activation_raw_tokens"]) == raw_tokens
        assert int(row["activation_retained_tokens"]) == retained_tokens
        assert int(row["activation_batches"]) == len(CALIBRATION_LENGTHS)
        assert int(row[QUANT_LOG_NSAMPLES]) == retained_tokens

        assert row["scale_aggregation"] == "token_rows"
        assert int(row["scale_raw_tokens"]) == raw_tokens
        assert int(row["scale_retained_tokens"]) == retained_tokens
        assert int(row["scale_batches"]) == len(CALIBRATION_LENGTHS)

        if module_name.endswith(("gate_proj", "up_proj")):
            assert row["scale_feature"] == "mlp"

    attention_row = rows["self_attn.q_proj"]
    assert attention_row["activation_aggregation"] == "latest_batch"
    assert int(attention_row["activation_raw_tokens"]) == raw_tokens
    assert int(attention_row["activation_retained_tokens"]) == CALIBRATION_LENGTHS[-1]

    print(
        {
            "model": str(MODEL_PATH),
            "calibration_lengths": CALIBRATION_LENGTHS,
            "raw_tokens": raw_tokens,
            "retained_tokens": retained_tokens,
            "expert_telemetry": {
                name: rows[name]
                for name in (
                    "mlp.experts.0.gate_proj",
                    "mlp.experts.0.up_proj",
                    "mlp.experts.0.down_proj",
                    "mlp.experts.127.gate_proj",
                    "mlp.experts.127.up_proj",
                    "mlp.experts.127.down_proj",
                )
            },
            "attention_telemetry": attention_row,
        }
    )
