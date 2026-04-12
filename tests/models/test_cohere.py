# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestCohere(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/aya-expanse-8b" # "CohereForAI/aya-expanse-8b"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": {"A100": 0.5546, "RTX4090": 0.5520}, "floor_pct": 0.20},
            "acc_norm": {"value": {"A100": 0.5802, "RTX4090": 0.5802}, "floor_pct": 0.20},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 4

    def test_cohere(self):
        self.quantize_and_evaluate()
