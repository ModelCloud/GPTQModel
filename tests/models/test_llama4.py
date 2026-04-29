# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestLlama4(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-4-Scout-17B-16E-Instruct" # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3567, "floor_pct": 0.36},
            "acc_norm": {"value": 0.3805, "floor_pct": 0.36},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False

    def test_llama4(self):
        self.quantize_and_evaluate()
