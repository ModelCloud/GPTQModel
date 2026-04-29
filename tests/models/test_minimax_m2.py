# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestMinimaxM2(ModelTest):

    NATIVE_MODEL_ID = "/monster/data/model/MiniMax-M2-bf16"
    USE_FLASH_ATTN = False
    TRUST_REMOTE_CODE = True
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 1024
    GROUP_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5026, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5171, "floor_pct": 0.04},
        },
        "mmlu_stem": {
            "acc": {"value": 0.6362, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    def test_minimax_m2(self):
        self.quantize_and_evaluate()
