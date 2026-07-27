# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


class TestNanbeige(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Nanbeige4.2-3B"
    TRUST_REMOTE_CODE = True

    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3268,
                "floor_pct": 0.4,
            },
            "acc_norm": {
                "value": 0.3541,
                "floor_pct": 0.4,
            },
        },
    }
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def test_nanbeige(self):
        self.quantize_and_evaluate()
