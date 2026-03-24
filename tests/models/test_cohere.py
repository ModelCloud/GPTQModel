# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/aya-expanse-8b" # "CohereForAI/aya-expanse-8b"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5401, "floor_pct": 0.20},
            "acc_norm": {"value": 0.5640, "floor_pct": 0.20},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 4

    def test_cohere(self):
        self.quant_lm_eval()
