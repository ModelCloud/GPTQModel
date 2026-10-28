# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

from model_test import ModelTest


class TestRaon1Mini(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "RAON1_MINI_MODEL_ID", "KRAFTON/Raon-1-mini-preview"
    )
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.32337883959044367, "floor_pct": 0.04},
            "acc_norm": {"value": 0.34982935153583616, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_raon1_mini(self):
        self.quantize_and_evaluate()
