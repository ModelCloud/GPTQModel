# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model GPU E2E coverage for shared-input Hessian deduplication.

The tiny CPU A/B test proves that deduplicated and independently accumulated
Hessians produce bit-identical weights.  These cases exercise the complete
quantize/save/reload/generate flow with the repository's smallest dense and
MoE checkpoints used by model tests.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper import stage_subset
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import (
    ExpertsRoutingOverride,
    HessianConfig,
    MoEConfig,
    QuantizeConfig,
)


pytestmark = [pytest.mark.cuda, pytest.mark.model, pytest.mark.slow]


@dataclass(frozen=True)
class RealModelCase:
    model_path: Path
    model_type: str
    is_moe: bool


CASES = (
    pytest.param(
        RealModelCase(
            model_path=Path(
                os.environ.get(
                    "GPTQMODEL_SHARED_INPUT_LLAMA_MODEL",
                    "/monster/data/model/Llama-3.2-1B-Instruct",
                )
            ),
            model_type="llama",
            is_moe=False,
        ),
        id="llama-3.2-1b-instruct",
    ),
    pytest.param(
        RealModelCase(
            model_path=Path(
                os.environ.get(
                    "GPTQMODEL_SHARED_INPUT_QWEN_MOE_MODEL",
                    "/monster/data/model/Qwen1.5-MoE-A2.7B",
                )
            ),
            model_type="qwen2_moe",
            is_moe=True,
        ),
        id="qwen1.5-moe-a2.7b",
    ),
)


def _calibration(tokenizer) -> list[dict[str, torch.Tensor]]:
    texts = (
        "Shared input Hessian calibration checks attention projections and feed forward projections. " * 6,
        "A second deterministic sample exercises complete model quantization and checkpoint reload. " * 6,
    )
    return [
        dict(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
        )
        for text in texts
    ]


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", CASES)
def test_shared_input_dedup_real_model_quantize_save_reload_generate(
    case: RealModelCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    if not case.model_path.is_dir():
        pytest.skip(f"local model fixture is unavailable: {case.model_path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-model shared-input E2E test")

    dedup_telemetry: list[dict[str, object]] = []
    original_emit_telemetry = stage_subset.emit_device_telemetry

    def spy_emit_telemetry(event: str, **fields):
        if event == "hessian_input_collection_dedup":
            dedup_telemetry.append(fields)
        return original_emit_telemetry(event, **fields)

    monkeypatch.setattr(stage_subset, "emit_device_telemetry", spy_emit_telemetry)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        device="cuda:0",
        calibration_data_device="cuda:0",
        offload_to_disk=False,
        hessian=HessianConfig(dedup_shared_inputs=True),
        moe=MoEConfig(routing=ExpertsRoutingOverride()) if case.is_moe else None,
    )
    model = GPTQModel.load(
        str(case.model_path),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
        local_files_only=True,
    )
    assert model.model.config.model_type == case.model_type

    plan = model.shared_input_plan(
        model_config=model.model.config,
        quantize_config=model.quantize_config,
    )
    layer_count = model.model.config.num_hidden_layers
    expected_dedup_count = plan.dedup_count * layer_count
    assert plan.dedup_count > 0

    calibration = _calibration(model.tokenizer)
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_concat_size=0,
        calibration_sort=None,
        calibration_data_min_length=1,
    )

    assert sum(event["adopted_followers"] for event in dedup_telemetry) == expected_dedup_count
    assert sum(event["expected_followers"] for event in dedup_telemetry) == expected_dedup_count
    assert all(event["status"] == "verified" for event in dedup_telemetry)
    assert all(event["lifecycle_stage"] == "forward_capture_complete" for event in dedup_telemetry)
    assert model.quantize_config.hessian.dedup_shared_inputs is True

    quantized_dir = tmp_path / "quantized"
    model.save(quantized_dir)
    with open(quantized_dir / "config.json") as config_file:
        saved_config = json.load(config_file)
    saved_hessian = saved_config["quantization_config"]["meta"]["hessian"]
    assert saved_hessian["dedup_shared_inputs"] is True
    del model
    _release_cuda_memory()

    reloaded = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cuda:0",
        local_files_only=True,
    )
    assert any(isinstance(module, BaseQuantLinear) for _, module in reloaded.named_modules())
    assert reloaded.quantize_config.meta_get("hessian")["dedup_shared_inputs"] is True

    encoded = reloaded.tokenizer("The capital city of France is", return_tensors="pt").to("cuda:0")
    output = reloaded.generate(**encoded, max_new_tokens=2, do_sample=False)
    assert output.shape == (1, encoded["input_ids"].shape[1] + 2)

    del reloaded
    _release_cuda_memory()
