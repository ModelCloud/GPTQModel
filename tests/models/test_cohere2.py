# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestCohere2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/c4ai-command-r7b-12-2024"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.4680, "floor_pct": 0.15},
            "acc_norm": {"value": 0.4693, "floor_pct": 0.15},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 4
    USE_FLASH_ATTN = False

    def test_cohere2(self):
        self.quantize_and_evaluate()
