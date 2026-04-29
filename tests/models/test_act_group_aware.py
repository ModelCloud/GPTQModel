# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestHybridActOrder(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3140, "floor_pct": 0.05},
            "acc_norm": {"value": 0.3439, "floor_pct": 0.05},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    GPTAQ = None
    ACT_GROUP_AWARE = True

    def test_llama3_2(self):
        self.quantize_and_evaluate()
