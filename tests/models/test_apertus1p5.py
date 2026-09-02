# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


class TestApertusV15(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Apertus-v1.5-8B" # swiss-ai/Apertus-v1.5-8B
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "evalution_model_args": {"device_map": "auto", "attn_implementation": "eager"},
            "acc": {"value": 0.5597269624573379, "floor_pct": 0.20},
            "acc_norm": {"value": 0.5503412969283277, "floor_pct": 0.20},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_BATCH_SIZE = 8

    def test_apertus_v15(self):
        self.quantize_and_evaluate()
