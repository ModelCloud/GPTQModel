# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

from model_test import ModelTest


class TestFlint(ModelTest):
    NATIVE_MODEL_ID = os.getenv("FLINT_MODEL_ID", "RecursiveSI/flint")
    TRUST_REMOTE_CODE = False
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.32337883959044367, "floor_pct": 0.04},
            "acc_norm": {"value": 0.34982935153583616, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_flint(self):
        self.quantize_and_evaluate()
