# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from gptqmodel.utils.paroquant_benchmark import run_paroquant_first_layer_case


@pytest.mark.cuda
def test_llama3_2_paroquant_first_4_layers_full_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for prefix-layer ParoQuant test")

    if os.environ.get("GPTQMODEL_RUN_PAROQUANT_FIRST_LAYER_TEST") != "1":
        pytest.skip("Set GPTQMODEL_RUN_PAROQUANT_FIRST_LAYER_TEST=1 to run this prefix-layer integration test.")

    eval_max_rows = os.environ.get("GPTQMODEL_PAROQUANT_TEST_MAX_ROWS")
    result = run_paroquant_first_layer_case(
        num_quant_layers=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_NUM_LAYERS", "4")),
        calibration_rows=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_CAL_ROWS", "32")),
        eval_batch_size=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_EVAL_BATCH", "32")),
        eval_max_rows=None if eval_max_rows in {None, ""} else int(eval_max_rows),
        sym=True,
        fused_opt_rotation=os.environ.get("GPTQMODEL_PAROQUANT_TEST_FUSED", "1") != "0",
        opt_rotation_epochs=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_ROT_EPOCHS", "4")),
        opt_finetune_epochs=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_FT_EPOCHS", "4")),
        opt_train_samples=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_TRAIN_ROWS", "512")),
        opt_validation_samples=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_VAL_ROWS", "64")),
        opt_batch_size=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_OPT_BATCH", "16")),
    )

    assert result["module_time_rows"], "expected per-module quantization timings"
    assert result["num_quant_layers"] == int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_NUM_LAYERS", "4"))
    assert "gsm8k_platinum_cot" in result["eval_metrics"], "expected gsm8k platinum metrics"
