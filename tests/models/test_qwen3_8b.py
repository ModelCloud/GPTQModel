# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from model_test import ModelTest


class TestQwen3_8B(ModelTest):
    """GPTQ regression gate for the Qwen3-8B (post-trained) dense model.

    Uses the same calibration/evaluation pipeline as Llama-3.2-1B-Instruct so
    results stay comparable across model sizes. Thresholds are intentionally
    conservative for the default 4-bit group-128 config and should be tightened
    once a stable A100 baseline is measured.
    """

    SAVE_PATH = os.environ.get(
        "GPTQMODEL_QWEN3_8B_SAVE_PATH",
        "/tmp/qwen3_8b_gptq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = True
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-8B"  # Qwen/Qwen3-8B
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.80,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.48,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.49,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_qwen3_8b(self):
        self.quantize_and_evaluate()
