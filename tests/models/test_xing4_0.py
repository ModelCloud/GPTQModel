# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


class TestXing4_0(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Xing4.0-29B-A4B" # XingChen-AGI/Xing4.0-29B-A4B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": "0.4462", "floor_pct": 0.04},
            "acc_norm": {"value": "0.4497", "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_xing4_0(self):
        self.quantize_and_evaluate()
