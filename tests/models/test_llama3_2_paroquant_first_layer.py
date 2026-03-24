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
    eval_batch = os.environ.get("GPTQMODEL_PAROQUANT_TEST_EVAL_BATCH", "auto").strip().lower()
    eval_batch_size: int | str = "auto" if eval_batch == "auto" else int(eval_batch)
    resolved_eval_max_rows = 128 if eval_max_rows in {None, ""} else int(eval_max_rows)

    result = run_paroquant_first_layer_case(
        num_quant_layers=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_NUM_LAYERS", "4")),
        calibration_rows=int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_CAL_ROWS", "32")),
        eval_batch_size=eval_batch_size,
        eval_max_rows=resolved_eval_max_rows,
        eval_model_args={
            "dtype": os.environ.get("GPTQMODEL_PAROQUANT_TEST_EVAL_DTYPE", "bfloat16"),
            "attn_implementation": os.environ.get(
                "GPTQMODEL_PAROQUANT_TEST_ATTN_IMPL",
                "paged|flash_attention_2",
            ),
            "device": os.environ.get("GPTQMODEL_PAROQUANT_TEST_EVAL_DEVICE", "cuda:0"),
        },
        eval_suite_kwargs={
            "batch_size": int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_SUITE_BATCH", "24")),
            "max_new_tokens": int(os.environ.get("GPTQMODEL_PAROQUANT_TEST_MAX_NEW_TOKENS", "96")),
            "stream": os.environ.get("GPTQMODEL_PAROQUANT_TEST_STREAMING", "1") != "0",
            "max_rows": resolved_eval_max_rows,
        },
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
    gsm8k_metrics = result["eval_metrics"]["gsm8k_platinum_cot"]
    assert "acc,num" in gsm8k_metrics, "expected gsm8k_platinum_cot acc,num metric"
