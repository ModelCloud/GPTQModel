# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestExaone4_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/EXAONE-4.5-33B" # LGAI-EXAONE/EXAONE-4.5-33B
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 16

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5478, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5990, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_exaone4_5(self):
        self.quantize_and_evaluate()
