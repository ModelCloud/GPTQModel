# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestJamba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/AI21-Jamba2-3B" # ai21labs/AI21-Jamba2-3B
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3916, "floor_pct": 0.04},
            "acc_norm": {"value": 0.3797, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_BATCH_SIZE = 32

    def test_jamba(self):
        self.quantize_and_evaluate()
